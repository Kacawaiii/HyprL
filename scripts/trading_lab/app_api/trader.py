"""Read-only, bounded trader views. No runtime initialization or dispatch here."""
import json
from pathlib import Path

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.platform.contracts import timestamp
from scripts.trading_lab.research.store import IntegrityError, ResearchStore, now
from scripts.trading_lab.trader_agent.scoring import rows, scorecard


def scan(store, kind, after=0):
    """Lazy, ordered pass over one record kind, resumable after a sequence number."""
    while True:
        page = store.records(kind, after=after, limit=1000)
        if not page:
            return
        yield from page
        after = page[-1]["sequence"]



class TraderViews:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None

    def dispatch(self, path, query):
        if self.root is None:
            raise AppApiError("trader runtime not configured", 503)
        if set(query) - {"after", "limit", "date", "synthetic"} or any(len(v) != 1 for v in query.values()):
            raise AppApiError("invalid trader query")
        try:
            limit = int(query.get("limit", ["100"])[0])
            after = int(query.get("after", ["0"])[0])
            if not 1 <= limit <= 200 or after < 0:
                raise ValueError()
            day = query.get("date", [now()[:10]])[0]
            timestamp(day + "T00:00:00Z")
            if path not in ("/api/v1/trader/today", "/api/v1/trader/context") and "date" in query:
                raise ValueError()
            # Dry-run runtimes hold only synthetic records; a real one holds none, so this never mixes the two.
            if "synthetic" in query and (path != "/api/v1/trader/scorecard" or query["synthetic"][0] not in ("0", "1")):
                raise ValueError()
            store = ResearchStore(self.root / "evidence", read_only=True)
            try:
                if path == "/api/v1/trader/ledger":
                    page = store.records("prediction", after=after, limit=limit + 1)
                    return {"schema": "trader-ledger-page-v1", "records": page[:limit],
                            "next_after": page[limit - 1]["sequence"] if len(page) > limit else None,
                            "label_state": "labels_append_separately"}
                if path == "/api/v1/trader/today":
                    runs = [r for r in rows(store, "replay-summary") if r["payload"].get("schema") == "trader-run-v1"
                            and r["payload"]["at"][:10] == day]
                    return {"schema": "trader-today-v1", "date": day, "runs": runs[-limit:]}
                if path == "/api/v1/trader/runs":
                    return self.runs(store, after, limit)
                if path == "/api/v1/trader/labels":
                    page = store.records("label", after=after, limit=limit + 1)
                    return {"schema": "trader-labels-page-v1", "records": page[:limit],
                            "next_after": page[limit - 1]["sequence"] if len(page) > limit else None}
                if path == "/api/v1/trader/context":
                    return self.context(store, day)
                if path == "/api/v1/trader/series":
                    return self.series(store, limit)
                if path == "/api/v1/trader/health":
                    return self.health()
                if path == "/api/v1/trader/scorecard":
                    return scorecard(store, synthetic=query.get("synthetic", ["0"])[0] == "1")
                if path == "/api/v1/trader/alerts":
                    alerts = self.root / "alerts.jsonl"
                    if not alerts.is_file():
                        return {"schema": "trader-alerts-v1", "alerts": []}
                    with alerts.open("rb") as stream:
                        size = stream.seek(0, 2)
                        start = max(0, size - 1024 * 1024)
                        stream.seek(start)
                        if start:
                            stream.readline()
                        records = [json.loads(line) for line in stream.read().splitlines()]
                    return {"schema": "trader-alerts-v1", "alerts": records[-limit:]}
            finally:
                store.close()
        except (FileNotFoundError, OSError):
            raise AppApiError("trader evidence unavailable", 503) from None
        except IntegrityError:
            raise AppApiError("trader evidence integrity failure", 409) from None
        except (ValueError, TypeError, KeyError):
            raise AppApiError("invalid trader evidence or selection", 400) from None
        raise AppApiError("no such trader endpoint", 404)

    @staticmethod
    def contexts(store):
        return [r for r in rows(store, "replay-summary") if r["payload"].get("schema") == "trader-context-evidence-v1"]

    def runs(self, store, after, limit):
        """Run summaries without their (large) decision body: status, reason and budgets only."""
        page = []
        for r in scan(store, "replay-summary", after):
            if r["payload"].get("schema") == "trader-run-v1":
                page.append(r)
                if len(page) > limit:
                    break
        keep = ("run_id", "status", "at", "error", "synthetic", "budget_counts")
        records = [{"sequence": r["sequence"], "recorded_at": r["recorded_at"],
                    **{k: r["payload"][k] for k in keep if k in r["payload"]}} for r in page[:limit]]
        return {"schema": "trader-runs-page-v1", "records": records,
                "next_after": page[limit - 1]["sequence"] if len(page) > limit else None}

    def context(self, store, day):
        """Prices, source identities and exclusions of one day's context. No headlines text, no archive content."""
        found = [r for r in self.contexts(store) if r["payload"]["context"]["decision_time"][:10] == day]
        if not found:
            return {"schema": "trader-context-view-v1", "date": day, "context": None}
        c = found[-1]["payload"]["context"]
        return {"schema": "trader-context-view-v1", "date": day, "context": {
            "decision_time": c["decision_time"], "price_cutoff": c["price_cutoff"], "synthetic": c["synthetic"],
            "universe": c["universe"], "prices": c["prices"], "exclusions": c["exclusions"],
            "limitations": c["limitations"],
            "sources": [{k: s.get(k) for k in ("url", "received_at", "digest")} for s in c["sources"]]}}

    def series(self, store, limit):
        """One timestamped reference price per asset per run: the last close known at the decision."""
        out = {}
        for record in self.contexts(store)[-limit:]:
            c = record["payload"]["context"]
            for asset, price in c["prices"].items():
                closes = price.get("recent_closes") or []
                if price.get("state") != "AVAILABLE" or not closes:
                    continue
                out.setdefault(asset, []).append({"decision_time": c["decision_time"], "price_at": price["last_price_at"],
                    "close": closes[-1], "adjustment": price.get("adjustment"), "synthetic": c["synthetic"]})
        return {"schema": "trader-series-v1", "series": out,
                "note": "reference prices only: the last close available at each decision, not a full price history"}

    def health(self):
        """Files the supervisor writes. Missing means never written, not healthy."""
        out = {"schema": "trader-health-view-v1"}
        for key, name in (("health", "health.json"), ("last_label", "last-label.json")):
            path = self.root / name
            try:
                out[key] = json.loads(path.read_text()) if path.is_file() and path.stat().st_size <= 65536 else None
            except ValueError:
                out[key] = None
        out["paused"] = (self.root / "PAUSED").exists()
        return out

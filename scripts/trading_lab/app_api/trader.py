"""Read-only, bounded trader views. No runtime initialization or dispatch here."""
import json
from pathlib import Path

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.platform.contracts import timestamp
from scripts.trading_lab.research.store import IntegrityError, ResearchStore, now
from scripts.trading_lab.trader_agent.scoring import rows, scorecard


class TraderViews:
    def __init__(self, root=None):
        self.root = Path(root).resolve() if root else None

    def dispatch(self, path, query):
        if self.root is None:
            raise AppApiError("trader runtime not configured", 503)
        if set(query) - {"after", "limit", "date"} or any(len(v) != 1 for v in query.values()):
            raise AppApiError("invalid trader query")
        try:
            limit = int(query.get("limit", ["100"])[0])
            after = int(query.get("after", ["0"])[0])
            if not 1 <= limit <= 200 or after < 0:
                raise ValueError()
            day = query.get("date", [now()[:10]])[0]
            timestamp(day + "T00:00:00Z")
            if path != "/api/v1/trader/today" and "date" in query:
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
                if path == "/api/v1/trader/scorecard":
                    return scorecard(store)
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

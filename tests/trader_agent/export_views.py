"""Real-shaped trader API responses from the SYNTHETIC runner, for the cockpit's Vitest fixtures.

Run `python -m tests.trader_agent.export_views apps/web/src/test/traderFixtures.json` to regenerate;
tests/trader_agent/test_views.py fails when the committed file drifts from what the API serves.
Nothing here is a market result: SyntheticData/SyntheticRunner only, with a mixed set of verdicts so the
UI sees KEEP, DOWNGRADE, REJECT and disagreement.
"""
from datetime import timedelta
import json
import sys
import tempfile
from pathlib import Path

from scripts.trading_lab.app_api.trader import TraderViews
from scripts.trading_lab.trader_agent.config import Authorization, instant, iso
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.scoring import realize
from scripts.trading_lab.trader_agent.service import TraderService
from scripts.trading_lab.trader_agent.synthetic import Clock, SyntheticData, SyntheticRunner


PAGE = 40


def grant_payload():
    models = {role: {"cli": "codex" if role == "analyst_gpt" else "claude",
              "model": "gpt-synthetic-test" if role == "analyst_gpt" else "sonnet" if role == "reviewer" else "opus",
              "tools": ["WebSearch", "WebFetch"], "calls_per_day": 1, "retries_per_day": 1, "timeout_minutes": 1}
              for role in ("analyst_claude", "analyst_gpt", "reviewer")}
    return {"mode": "PAPER_SHADOW_ONLY", "granted_at": "2026-01-01T00:00:00Z", "not_after": "2027-02-01T00:00:00Z",
        "external_models": models,
        "data_sources": {"yahoo_chart": {"host": "query1.finance.yahoo.com", "max_requests_per_day": 80},
                         "coinbase_exchange_public": {"host": "api.exchange.coinbase.com", "max_requests_per_day": 20},
                         "gdelt_doc_api": {"host": "api.gdeltproject.org", "max_requests_per_day": 80, "min_spacing_seconds": 6}},
        "universe": {"stocks": ["AAPL", "MSFT"], "sector_etfs": ["XLK"], "crypto": ["BTC-USD", "ETH-USD"],
                     "benchmarks_not_predicted": ["SPY", "QQQ"]},
        "budgets": {"max_runs_per_day": 1, "max_llm_calls_per_day": 6, "paper_capital_usd": 100000},
        "cadence": {"runs_per_us_trading_day": 1}}


class MixedRunner(SyntheticRunner):
    """The synthetic fake with deliberate disagreement, so every verdict state exists in the fixture."""

    def once(self, role, text):
        out = super().once(role, text)
        if role == "analyst_gpt":
            for v in out["views"]:
                if v["asset"] in ("MSFT", "ETH-USD"):
                    v.update(view="DOWN", p_outperform=.44)
                if v["asset"] == "XLK" and v["horizon"] == "5d":
                    v.update(view="ABSTAIN", p_outperform=.5)
        if role == "analyst_claude":
            for v in out["views"]:
                if v["asset"] == "BTC-USD":
                    v.update(p_outperform=.62)
        if role == "reviewer":
            for r in out["verdicts"]:
                if r["analyst"] == "analyst_gpt" and r["asset"] == "MSFT" and r["horizon"] == "5d":
                    r.update(verdict="REJECT", reason_code="unsupported", note="Synthetic: the cited fact does not support the claim.")
                if r["analyst"] == "analyst_claude" and r["asset"] == "XLK" and r["horizon"] == "1d":
                    r.update(verdict="DOWNGRADE", reason_code="priced_in", adjusted_p=.53, note="Synthetic: largely priced in.")
                if r["analyst"] == "analyst_gpt" and r["asset"] == "XLK" and r["horizon"] == "5d":
                    r.update(reason_code="abstained")
        return out


def build(root):
    grant_file = Path(root) / "grant.json"
    grant_file.write_text(json.dumps(grant_payload()))
    clock = Clock(instant("2026-10-06T12:00:00Z"))
    ledger = Ledger(Path(root) / "private", Authorization.load(grant_file), clock=clock)
    data = SyntheticData(ledger, clock)
    service = TraderService(ledger, data, MixedRunner(ledger, clock=clock), clock=clock)
    api = TraderViews(ledger.root)

    def get(path, **query):
        return api.dispatch("/api/v1/trader/" + path, {k: [str(v)] for k, v in query.items()})

    def state():
        # Labels and scores are the only things that change as outcomes arrive; the predictions never do.
        return {"labels": get("labels", limit=PAGE), "scorecard": get("scorecard", synthetic=1)}

    # Scenario "empty": a configured runtime nobody has run yet.
    empty_root = Path(root) / "empty"
    empty = Ledger(empty_root, Authorization.load(grant_file), clock=clock)
    TraderService(empty, data, MixedRunner(empty, clock=clock), clock=clock).store.close()
    empty_api = TraderViews(empty.root)
    empty_view = {p: empty_api.dispatch("/api/v1/trader/" + p, {}) for p in ("runs", "ledger", "labels", "scorecard", "alerts", "series", "health")}
    empty_view["today"] = empty_api.dispatch("/api/v1/trader/today", {"date": ["2026-10-07"]})
    empty_view["context"] = empty_api.dispatch("/api/v1/trader/context", {"date": ["2026-10-07"]})

    service.run()
    clock.at = instant("2026-10-07T12:00:00Z")
    service.run()
    # "issued": two runs, nothing realized yet.
    issued = state()
    # "partial": the first day's 1d labels exist; 5d and the second day's are pending.
    clock.at = instant("2026-10-07T21:30:00Z")
    realize(service.store, ledger, data, at=clock())
    partial = state()
    # "realized": every endpoint passed.
    clock.at = instant("2026-10-16T21:30:00Z")
    realize(service.store, ledger, data, at=clock())
    realized = state()
    # A skipped day: Saturday has no session.
    clock.at = instant("2026-10-10T12:00:00Z")
    service.run()
    skipped = get("today", date="2026-10-10")
    # The supervisor files, as the health timer writes them.
    (ledger.root / "health.json").write_text(json.dumps({"schema": "trader-health-v1", "at": iso(clock()), "state": "HEALTHY",
                                                         "budget_counts": ledger.counts()}))
    (ledger.root / "last-label.json").write_text(json.dumps({"at": iso(clock()), "state": "COMPLETE"}))
    health = get("health")
    service.store.close()
    shared = {"today": get("today", date="2026-10-07"), "runs": get("runs"), "ledger": get("ledger", limit=PAGE),
              "alerts": get("alerts"), "context": get("context", date="2026-10-07"), "series": get("series"),
              "skipped_day": skipped, "health": health}
    return {"synthetic": True, "note": "SYNTHETIC runner and data; the shapes are the live API's. Not a market result.",
            "empty": empty_view, "shared": shared, "states": {"issued": issued, "partial": partial, "realized": realized}}


def main(argv=None):
    target = Path((argv or sys.argv[1:])[0])
    with tempfile.TemporaryDirectory() as root:
        target.write_text(json.dumps(build(root), separators=(",", ":"), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

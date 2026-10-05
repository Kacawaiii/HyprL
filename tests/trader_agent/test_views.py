"""The cockpit's read-only trader views: bounded, non-mutating, and the committed web fixtures match the API."""
import json
from pathlib import Path

import pytest

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.app_api.trader import TraderViews
from scripts.trading_lab.trader_agent.scoring import realize

from tests.trader_agent.export_views import build

FIXTURE = Path(__file__).resolve().parents[2] / "apps/web/src/test/traderFixtures.json"


def get(api, leaf, **query):
    return api.dispatch("/api/v1/trader/" + leaf, {k: [str(v)] for k, v in query.items()})


def test_runs_context_series_health_are_bounded_and_do_not_mutate(service, clock):
    service.run()
    before = service.store.verify()
    api = TraderViews(service.ledger.root)
    runs = get(api, "runs")
    assert [r["status"] for r in runs["records"]] == ["RUNNING", "COMPLETE"]
    assert all("decision" not in r for r in runs["records"])  # the large body stays in /today
    first = get(api, "runs", limit=1)
    assert len(first["records"]) == 1 and first["next_after"]
    assert get(api, "runs", limit=1, after=first["next_after"])["records"][0]["status"] == "COMPLETE"
    context = get(api, "context", date="2026-10-06")["context"]
    assert set(context["universe"]) <= set(context["prices"]) and context["sources"]
    assert all(set(s) == {"url", "received_at", "digest"} for s in context["sources"])
    assert "archives" not in context and "headlines" not in context
    assert get(api, "context", date="2026-10-05")["context"] is None
    series = get(api, "series")["series"]
    assert series["AAPL"][0]["close"] == context["prices"]["AAPL"]["recent_closes"][-1]
    assert get(api, "health") == {"schema": "trader-health-view-v1", "health": None, "last_label": None, "paused": False}
    assert get(api, "labels")["records"] == []  # pending labels are absent, never fabricated
    assert service.store.verify() == before


def test_labels_append_without_rewriting_predictions(service, clock, tmp_path):
    from datetime import timedelta
    service.run()
    api = TraderViews(service.ledger.root)
    predictions = get(api, "ledger", limit=200)["records"]
    clock.at += timedelta(days=9, hours=10)
    realize(service.store, service.ledger, service.data, at=clock())
    assert get(api, "ledger", limit=200)["records"] == predictions
    page = get(api, "labels", limit=5)
    assert len(page["records"]) == 5 and page["next_after"]
    hashes = {p["identity"] for p in predictions}
    assert all(r["payload"]["prediction_hash"] in hashes for r in page["records"])
    assert get(api, "scorecard", synthetic=1)["scores"] and get(api, "scorecard")["scores"] == {}


@pytest.mark.parametrize("leaf,query", [
    ("labels", {"date": "2026-10-06"}), ("series", {"date": "2026-10-06"}), ("health", {"date": "2026-10-06"}),
    ("scorecard", {"synthetic": "2"}), ("today", {"synthetic": "1"}), ("runs", {"limit": "201"}),
    ("labels", {"after": "-1"}), ("context", {"date": "not-a-day"})])
def test_views_reject_invalid_queries(service, leaf, query):
    service.run()
    with pytest.raises(AppApiError):
        get(TraderViews(service.ledger.root), leaf, **query)


def test_committed_web_fixtures_match_the_api(tmp_path):
    """Regenerate with `python -m tests.trader_agent.export_views apps/web/src/test/traderFixtures.json`."""
    built = json.loads(json.dumps(build(tmp_path), separators=(",", ":"), sort_keys=True))
    assert built == json.loads(FIXTURE.read_text()), "traderFixtures.json drifted from the trader API"
    assert built["synthetic"] is True
    assert built["states"]["issued"]["labels"]["records"] == []

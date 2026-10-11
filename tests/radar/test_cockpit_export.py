"""Synthetic inputs only: no store, broker, network or private path is read."""
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.radar import cockpit_export as export
from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.app_api.radar import RadarViews
from tests.crypto import loopback

HASH = "a" * 64


def story(url="https://example.invalid/a?utm=1&apikey=SECRETVALUE"):
    return {"id": HASH, "source": "alpaca:news", "publisher": "benzinga", "url": url, "source_url": url,
            "published_at": "2026-10-10T10:00:00Z", "received_at": "2026-10-10T10:05:00Z",
            "headline": "Synthetic headline", "summary": "RAW ARTICLE TEXT must never be exported",
            "headline_hash": HASH, "summary_hash": HASH, "symbols": ["BTC/USD"], "retail": False, "primary": False}


def event(i, importance, symbol="BTC/USD", scenario=True):
    out = {"id": f"{i:064x}", "headline": f"Event {i}", "first_received_at": "2026-10-10T10:05:00Z",
           "stories": [story()], "entities": {"symbols": [symbol], "countries": ["Iran"], "themes": ["geopolitics"]},
           "evidence": {"score": 40 + i, "status": "single_source", "publishers": ["benzinga"], "primary": False,
                        "independence": "x"},
           "novelty": "new", "importance": importance, "retail_hype": {"score": 0}, "published_at": "2026-10-10T10:00:00Z",
           "transmission_hypotheses": [{"symbol": symbol, "role": "direct", "mechanism": "Exposition directe."},
                                       {"symbol": "COIN", "role": "sector", "mechanism": "Commissions."}],
           "conviction": None, "expectations": "Consensus inconnu.",
           "priced_in": {"status": "MEASURED", "measurements": [
               {"symbol": symbol, "baseline": 1.0, "price": 1.1, "baseline_at": "2026-10-10T10:00:00Z",
                "price_at": "2026-10-10T11:00:00Z", "move_atr": 0.9, "return_pct": 10.0,
                "interpretation": "observed movement, not proof of causation"}]}}
    if scenario:
        out["scenario"] = {"id": out["id"], "summary": "s", "changed_expectations": "c", "impact": "i", "horizon": "h",
                           "priced_in": "p", "invalidation": "v", "conviction": "faible",
                           "exposures": [{"symbol": symbol, "direction": "down", "role": "direct", "mechanism": "m"}]}
    return out


RADAR = {"schema": "news-radar-v1", "status": "PARTIAL", "date": "2026-10-10", "slot": "morning",
         "cutoff": "2026-10-10T11:32:02Z", "sources": [{"source": "a", "status": "LIVE"}, {"source": "b", "status": "DEAD"}],
         "events": [event(1, 40), event(2, 80, "ETH/USD"), event(3, 60, "XLE", scenario=False)],
         "regime": {"BTC": {"symbol": "BTC/USD", "last": 82000.5, "at": "2026-10-10T11:31:00Z", "status": "OBSERVED",
                            "returns_pct": {"1d": 1.2}}},
         "limitations": ["Prix: cours clôturés."]}


def view(analyst, asset, horizon, v, p, verdict="KEEP"):
    return {"analyst": analyst, "asset": asset, "horizon": horizon, "view": v, "p_outperform": p, "verdict": verdict,
            "raw_view": {"confidence_reason": "reason", "falsifier": "f", "catalysts": [
                {"url": "https://example.invalid/c?token=zzz", "published_at": "2026-10-10T09:00:00Z", "fact": "raw fact"}]},
            "review": {"note": "ok"}}


def run(run_id, at, p_claude, p_gpt):
    return {"run_id": run_id, "at": at, "status": "COMPLETE", "synthetic": False, "recorded_at": at,
            "models": {}, "views": [view("analyst_claude", "BTC-USD", "1d", "UP", p_claude),
                                    view("analyst_gpt", "BTC-USD", "1d", "DOWN", p_gpt),
                                    view("consensus", "BTC-USD", "1d", "UP", (p_claude + p_gpt) / 2)]}


TRADER = {"runs": [run("trader:2026-10-09:real", "2026-10-09T12:00:00Z", 0.6, 0.4),
                   run("trader:2026-10-10:real", "2026-10-10T12:00:00Z", 0.7, 0.45)],
          "labels": [{"model_id": "trader:analyst_claude", "asset": "BTC-USD", "horizon": "1d", "run_id": "trader:2026-10-09:real",
                      "view": "UP", "realized_at": "2026-10-09T20:00:00Z", "available_at": "2026-10-09T21:30:00Z",
                      "raw_return": 0.01, "spy_relative_return": 0.005, "net_unit_pnl": 0.004, "cost_roundtrip": 0.002}],
          "scorecard": {"scores": {"analyst_claude/crypto/1d/raw": {"issued": 10, "realized": 4, "hit_rate": 0.5, "brier": 0.25,
                                                      "climatology_brier": 0.25, "ic": 0.1, "days": 3,
                                                      "mean_unit_pnl_after_costs": 0.001}},
                        "hypothesis_state": "NOT_CONFIRMED"}}

PAPER_REPORT = {"at": "2026-10-10T17:00:00Z", "accounts": [
    {"account": "ia_actions", "account_suffix": "GAHT", "equity": "100000", "pnl": "0", "day_pnl": "0", "peak": "100000",
     "open_lots": 0, "halted": False, "planned_orders": [], "return_since_paper_start": "0"}],
    "benchmarks": {"SPY": {"price": "778.5", "return_since_paper_start": "0.0047", "base_at": "2026-10-08T13:30:00Z"}},
    "limitations": ["After-hours fills are optimistic."]}
BOOK_STATUS = {"equity": "107303.58", "start_equity": 107292.42, "cash": "88536.36", "peak": "107303.58",
               "positions": [{"symbol": "BTCUSD", "asset_class": "crypto", "qty": "0.04", "avg_entry_price": "82783.9",
                              "current_price": "82971.9", "unrealized_pl": "8.2", "unrealized_plpc": "0.0022"},
                             {"symbol": "GS", "asset_class": "us_equity", "qty": "9", "avg_entry_price": "850",
                              "current_price": "855", "unrealized_pl": "45", "unrealized_plpc": "0.006"}],
               "open_orders": [{"id": "SHOULD-NOT-LEAK", "symbol": "BTC/USD", "side": "buy", "qty": "0.066", "type": "limit",
                                "limit": "80600", "status": "new", "submitted": "2026-10-10T14:06", "legs": []}]}
JOURNAL = [{"at": "2026-10-10T17:07:55+00:00", "action": "buy_intent", "symbol": "BTC/USD", "qty": 0.04, "limit": 82783.9,
            "stop": 78000, "target": 90000, "risk_usd": 190.0,
            "journal": {"engine": "momo_v0", "event": "breakout", "mechanism": "momentum", "priced_in": "no",
                        "scenario": "trend", "invalidation": "close below 78000"}},
           {"at": "2026-10-10T17:08:00+00:00", "action": "submitted", "symbol": "BTC/USD", "order_id": "x"}]


def test_home_ranks_by_importance_and_keeps_five_separate_badges():
    home = export.build_home(RADAR, TRADER, None, generated_at="2026-10-10T12:00:00Z")
    assert [e["rank"] for e in home["events"]] == [1, 2, 3]
    assert [e["badges"]["event_importance"]["value"] for e in home["events"]] == [80, 60, 40]
    badges = home["events"][0]["badges"]
    assert set(badges) == {"event_importance", "evidence_strength", "model_conviction", "predictive_quality",
                           "after_cost_performance"}
    assert "score" not in badges and "combined" not in json.dumps(badges)
    assert home["radar"]["sources_by_status"] == {"LIVE": 1, "DEAD": 1}
    assert home["radar"]["report_hash"] == export.digest(RADAR)


def test_home_joins_model_views_timeline_and_outcome_by_normalised_symbol():
    home = export.build_home(RADAR, TRADER, None)
    btc = next(e for e in home["events"] if e["headline"] == "Event 1")
    asset = btc["assets"][0]
    assert asset["symbol"] == "BTC/USD" and asset["anticipation"]["state"] == "COVERED"
    assert [t["run_id"] for t in asset["anticipation"]["timeline"]] == ["trader:2026-10-09:real", "trader:2026-10-10:real"]
    assert asset["anticipation"]["timeline"][0]["by_analyst"]["analyst_claude"]["p_outperform"] == 0.6
    assert asset["anticipation"]["latest"]["views"][0]["analyst"] in ("analyst_claude", "analyst_gpt", "consensus")
    assert asset["outcomes"][0]["net_unit_pnl"] == 0.004
    assert btc["badges"]["after_cost_performance"]["event_labels"]["n"] == 1
    assert btc["badges"]["predictive_quality"]["by_analyst"]["analyst_claude/1d"]["hit_rate"] == 0.5
    assert btc["badges"]["predictive_quality"]["basis"]
    coin = btc["assets"][1]
    assert coin["symbol"] == "COIN" and coin["anticipation"]["state"] == "NOT_COVERED" and coin["outcomes"] == []


def test_home_keeps_point_in_time_stamps_and_no_article_text():
    home = export.build_home(RADAR, TRADER, None)
    first = home["events"][0]
    assert first["available_at"] == "2026-10-10T10:05:00Z" and first["published_at"] == "2026-10-10T10:00:00Z"
    blob = json.dumps(home)
    assert "RAW ARTICLE TEXT" not in blob
    # Phase 2 retains the model's short cited fact inside its decision chain;
    # it still never exports a captured story body/summary.
    btc = next(e for e in home["events"] if e["headline"] == "Event 1")
    assert btc["assets"][0]["anticipation"]["latest"]["views"][0]["decision"]["fact"] == "raw fact"
    assert "SECRETVALUE" not in blob and "token=zzz" not in blob
    assert first["link"] == "https://example.invalid/a?utm=1"


def test_home_without_trader_or_scenario_says_so_instead_of_inventing():
    home = export.build_home(RADAR, None, None)
    plain = next(e for e in home["events"] if e["headline"] == "Event 3")
    assert plain["what_changed"]["source"] == "rules_only" and plain["what_changed"]["summary"] is None
    assert plain["assets"][0]["anticipation"] == {"state": "NOT_COVERED", "latest": None, "timeline": []}
    assert plain["badges"]["predictive_quality"]["by_analyst"] is None


def test_paper_snapshot_tags_positions_and_keeps_suffix_only():
    paper = export.build_paper(PAPER_REPORT, BOOK_STATUS, JOURNAL, sleeve={"BTCUSD": 0.04}, radar=RADAR,
                               generated_at="2026-10-10T17:10:00Z")
    book = next(a for a in paper["accounts"] if a["account"] == "claude_book")
    assert book["suffix"] == "2EQN"
    by_symbol = {p["symbol"]: p for p in book["positions"]}
    assert by_symbol["BTCUSD"]["tag"] == "sleeve" and by_symbol["BTCUSD"]["stop"] == 78000
    assert by_symbol["GS"]["tag"] == "discretionary" and by_symbol["GS"]["protection"] == "none_defined"
    assert paper["accounts"][0]["suffix"] == "GAHT" and paper["benchmarks"]["BTC"]["price"] == 82000.5
    assert "SHOULD-NOT-LEAK" not in json.dumps(paper)
    assert book["journal"][0]["engine"] == "momo_v0"


def test_momo_tag_comes_from_the_journal_engine():
    book = export.build_claude_book(BOOK_STATUS, JOURNAL)
    assert {p["symbol"]: p["tag"] for p in book["positions"]}["BTCUSD"] == "momo_v0"


@pytest.mark.parametrize("poison", ["someone@example.invalid", "PKABCDEFGHIJKLMNOPQRS", "Bearer abcdef", "api_key=zzz",
                                    "/home/agent/private"])
def test_secret_shaped_text_is_refused(poison):
    radar = json.loads(json.dumps(RADAR))
    radar["events"][0]["headline"] = "ok " + poison
    with pytest.raises(export.ExportError):
        export.build_home(radar, None, None)


def test_forbidden_field_is_refused():
    with pytest.raises(export.ExportError):
        export.assert_clean({"a": [{"account_number": "123"}]})


def test_history_appends_once_per_timestamp(tmp_path):
    path = tmp_path / "history.jsonl"
    point = {"at": "2026-10-10T17:10:00Z", "equity": {"claude_book": 1.0}, "spy": 1.0, "btc": 2.0}
    export.append_history(path, point)
    rows = export.append_history(path, point)
    assert len(rows) == 1
    rows = export.append_history(path, {**point, "at": "2026-10-10T18:00:00Z"})
    assert [r["at"] for r in rows] == ["2026-10-10T17:10:00Z", "2026-10-10T18:00:00Z"]


class FakeStore:
    def __init__(self, tables):
        self.tables = tables

    def records(self, kind, after=0, limit=1000):
        return [r for r in self.tables.get(kind, []) if r["sequence"] > after][:limit]


def test_load_trader_joins_labels_to_predictions_and_drops_synthetic():
    def chained(seq, payload):
        return {"sequence": seq, "recorded_at": "2026-10-10T12:00:00Z", "payload": payload}
    real = {"schema": "trader-run-v1", "run_id": "r1", "at": "2026-10-10T12:00:00Z", "status": "COMPLETE",
            "synthetic": False, "decision": {"views": [view("analyst_claude", "BTC-USD", "1d", "UP", 0.6)]}}
    fake = {**real, "run_id": "r2", "synthetic": True}
    pred = {"prediction_id": "p1", "model_id": "trader:analyst_claude", "product": "BTC-USD", "synthetic": False,
            "decision_at": "2026-10-10T12:00:00Z",
            "signal": {"run_id": "r1", "view": {"view": "UP"}, "label_definition": {"horizon": "1d"}}}
    label = {"prediction_id": "p1", "realized_at": "2026-10-10T20:00:00Z", "available_at": "2026-10-10T21:00:00Z",
             "value": {"raw_return": 0.01, "spy_relative_return": None, "net_unit_pnl": 0.002, "cost_roundtrip": 0.001}}
    store = FakeStore({"replay-summary": [chained(1, real), chained(2, fake)], "prediction": [chained(1, pred)],
                       "label": [chained(1, label)]})
    out = export.load_trader(store)
    assert [r["run_id"] for r in out["runs"]] == ["r1"]
    assert out["labels"][0]["asset"] == "BTC-USD" and out["labels"][0]["net_unit_pnl"] == 0.002
    assert len(export.load_trader(store, include_synthetic=True)["runs"]) == 2


def test_snapshot_write_is_atomic_and_cli_refuses_the_repository(tmp_path):
    target = export.write_snapshot(tmp_path / "out", "radar-home.json", {"schema": export.HOME_SCHEMA})
    assert json.loads(target.read_text())["schema"] == export.HOME_SCHEMA
    assert not list((tmp_path / "out").glob("*.tmp"))
    radar = tmp_path / "radar.json"
    radar.write_text(json.dumps(RADAR))
    with pytest.raises(SystemExit):
        export.main(["--radar", str(radar), "--out", str(export.Path(export.__file__).parents[2] / "var-snapshots")])


# ---- the API view

def test_views_serve_only_the_two_snapshots_and_validate_schema(tmp_path):
    views = RadarViews(tmp_path)
    with pytest.raises(AppApiError) as missing:
        views.dispatch("/api/v1/radar/home", {})
    assert missing.value.status == 503
    export.write_snapshot(tmp_path, "radar-home.json", export.build_home(RADAR, TRADER, None))
    assert views.dispatch("/api/v1/radar/home", {})["schema"] == export.HOME_SCHEMA
    (tmp_path / "paper.json").write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(AppApiError) as wrong:
        views.dispatch("/api/v1/radar/paper", {})
    assert wrong.value.status == 409
    with pytest.raises(AppApiError):
        views.dispatch("/api/v1/radar/home", {"x": ["1"]})
    with pytest.raises(AppApiError) as unknown:
        views.dispatch("/api/v1/radar/../secrets", {})
    assert unknown.value.status == 404
    with pytest.raises(AppApiError):
        RadarViews().dispatch("/api/v1/radar/home", {})


def test_http_serves_snapshot_and_refuses_writes(tmp_path):
    from scripts.trading_lab.app_api.server import make_server
    export.write_snapshot(tmp_path, "radar-home.json", export.build_home(RADAR, TRADER, None))
    server = make_server("tests/fixtures/crypto", host=loopback.host(), port=0, radar_root=tmp_path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = loopback.url(server.server_address[1])
    try:
        with urlopen(base + "/api/v1/radar/home", timeout=5) as response:
            assert response.status == 200 and json.load(response)["events"][0]["rank"] == 1
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + "/api/v1/radar/home", data=b"{}", method="POST"), timeout=5)
        assert error.value.code == 405
        with pytest.raises(HTTPError) as unavailable:
            urlopen(base + "/api/v1/radar/paper", timeout=5)
        assert unavailable.value.code == 503
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

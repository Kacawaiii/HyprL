"""Read-only application API: bounded, deterministic, and unable to mutate.

The properties worth testing here are architectural. An endpoint that returns
"everything" works on a year of hourly candles and dies on five; a wildcard
CORS header is invisible until the day it matters; a cursor that silently
works across products yields a page that looks perfectly valid and describes
the wrong instrument.
"""

from __future__ import annotations

from decimal import Decimal
import importlib
import json
import pathlib
import threading
import urllib.error
import urllib.request

import pytest

from tests.crypto import loopback


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "crypto"


@pytest.fixture
def contracts():
    return importlib.import_module("scripts.trading_lab.app_api.contracts")


@pytest.fixture
def pagination():
    return importlib.import_module("scripts.trading_lab.app_api.pagination")


@pytest.fixture
def service():
    module = importlib.import_module("scripts.trading_lab.app_api.service")
    if not (DATA_ROOT / "coinbase_history_v1" / "manifest.json").is_file():
        pytest.skip("no market corpus in this checkout")
    return module.AppService(DATA_ROOT)


@pytest.fixture(scope="module")
def live_api():
    server_module = importlib.import_module("scripts.trading_lab.app_api.server")
    if not (DATA_ROOT / "coinbase_history_v1" / "manifest.json").is_file():
        pytest.skip("no market corpus in this checkout")
    server = server_module.make_server(DATA_ROOT, host=loopback.host(), port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield loopback.url(port)
    finally:
        server.shutdown()
        server.server_close()


def _get(base, path, *, headers=None):
    request = urllib.request.Request(f"{base}{path}", headers=headers or {})
    with urllib.request.urlopen(request, timeout=15) as response:
        return response.status, json.loads(response.read().decode("utf-8")), response


# --- system, health, overview ---------------------------------------------


def test_health_reports_status_without_leaking_the_machine(service, contracts):
    payload = service.health()
    assert payload["status"] == "ok"
    assert payload["api_version"] == contracts.APP_API_VERSION
    text = json.dumps(payload)
    for leak in ("/home/", "kyo", "hostname", "token", "secret", "password"):
        assert leak.lower() not in text.lower(), leak


def test_the_system_view_reports_the_frozen_engine_contracts(service):
    signal_engine = importlib.import_module("scripts.trading_lab.signal_engine")
    risk_engine = importlib.import_module("scripts.trading_lab.risk_engine")
    payload = service.system()
    assert payload["signal_engine"]["spec_hash"] == signal_engine.SIGNAL_SPEC_V1.spec_hash
    assert payload["signal_engine"]["frozen"] is True
    assert payload["signal_engine"]["optimized"] is False
    assert payload["signal_engine"]["long_threshold"] == "0.0025"
    assert payload["risk_engine"]["spec_hash"] == risk_engine.RISK_SPEC_V1.risk_spec_hash
    assert payload["risk_engine"]["max_long_exposure"] == "0.25"
    assert payload["risk_engine"]["volatility_scaling_enabled"] is False
    assert payload["risk_engine"]["optimized"] is False


def test_the_capabilities_tell_the_truth_about_what_does_not_exist(service):
    capabilities = service.system()["capabilities"]
    assert capabilities["signal_engine"] is True
    assert capabilities["position_target"] is True
    # Phase 5C added the engine, so the capability is now true. What remains
    # false is what genuinely does not exist: no paper account, no broker.
    assert capabilities["economic_backtest"] is True
    # Phase 5D added shadow trading, so paper is now true. What stays false is
    # the only one that involves money: live trading.
    assert capabilities["paper_trading"] is True
    assert capabilities["live_trading"] is False
    assert capabilities["realtime_stream"] is False


def test_the_confirmatory_holdout_is_reported_as_unobserved(service):
    benchmarks = service.system()["benchmarks"]
    assert benchmarks["v1_available"] is True
    assert benchmarks["v2_exploratory_available"] is True
    assert benchmarks["v2_confirmatory_observed"] is False
    holdout = benchmarks["confirmatory_holdout"]
    assert holdout["range_start"] == "2026-09-01T00:00:00Z"
    assert holdout["captured"] is False


def test_the_overview_never_invents_a_live_price(service):
    """No live feed exists. Reporting null beats reporting a plausible number."""
    payload = service.overview()
    assert payload["products"]
    for product in payload["products"]:
        assert product["latest_price"] is None
        assert product["latest_price_available"] is False
        assert product["rows"] == 8750
        assert product["missing_openings"] == 10
    assert len(json.dumps(payload)) < 20_000     # a first screen, not a data dump


# --- bounded market access -------------------------------------------------


def test_market_candles_are_paginated_by_default(service, contracts):
    page = service.market_candles("BTC-USD")
    assert page["page"]["returned"] == contracts.DEFAULT_PAGE_SIZE
    assert page["page"]["has_more"] is True
    assert page["page"]["next_cursor"]
    assert len(page["candles"]) == contracts.DEFAULT_PAGE_SIZE


def test_no_market_request_can_ask_for_the_whole_corpus(service, contracts):
    """The architectural property: unbounded responses are impossible."""
    with pytest.raises(contracts.AppApiError, match="exceeds the maximum"):
        service.market_candles("BTC-USD", limit=10_000)
    biggest = service.market_candles("BTC-USD", limit=contracts.MAX_PAGE_SIZE)
    assert len(biggest["candles"]) == contracts.MAX_PAGE_SIZE
    assert biggest["page"]["has_more"] is True      # 8750 rows exist; 1000 came back


@pytest.mark.parametrize("limit", [0, -1, "abc", "1e9", 1001])
def test_a_malformed_or_oversized_limit_is_refused(service, contracts, limit):
    with pytest.raises(contracts.AppApiError, match="limit"):
        service.market_candles("BTC-USD", limit=limit)


def test_paging_with_a_cursor_walks_forward_without_gaps_or_repeats(service):
    first = service.market_candles("BTC-USD", limit=50)
    second = service.market_candles("BTC-USD", limit=50,
                                    cursor=first["page"]["next_cursor"])
    stamps = [row["bar_open_at"] for row in first["candles"] + second["candles"]]
    assert len(set(stamps)) == len(stamps) == 100
    assert stamps == sorted(stamps)
    assert second["candles"][0]["bar_open_at"] > first["candles"][-1]["bar_open_at"]


def test_a_cursor_is_bound_to_the_query_that_issued_it(service, contracts):
    """A cursor that silently crossed products would describe the wrong asset."""
    issued = service.market_candles("BTC-USD", limit=50)["page"]["next_cursor"]
    with pytest.raises(contracts.AppApiError, match="different product"):
        service.market_candles("ETH-USD", limit=50, cursor=issued)
    with pytest.raises(contracts.AppApiError, match="different query"):
        service.market_candles("BTC-USD", limit=25, cursor=issued)


@pytest.mark.parametrize("cursor", ["!!!!", "YWJj", "e30", "a" * 200])
def test_a_malformed_cursor_fails_closed(service, contracts, cursor):
    with pytest.raises(contracts.AppApiError):
        service.market_candles("BTC-USD", limit=10, cursor=cursor)


@pytest.mark.parametrize("cursor", [None, ""])
def test_an_absent_cursor_means_the_first_page(service, cursor):
    """`cursor=` with no value is absence, not corruption.

    A client that always sends the parameter -- which is the natural thing for
    a generated query string to do -- would otherwise fail on its very first
    page. Genuinely malformed values still fail closed, as tested above.
    """
    page = service.market_candles("BTC-USD", limit=5, cursor=cursor)
    assert page["page"]["returned"] == 5
    assert page["candles"][0]["bar_open_at"] == "2025-08-01T00:00:00+00:00"


def test_a_time_window_narrows_the_result(service):
    window = service.market_candles(
        "BTC-USD", start="2025-08-01T00:00:00+00:00",
        end="2025-08-01T05:00:00+00:00", limit=100)
    assert window["page"]["returned"] == 6
    assert window["page"]["has_more"] is False
    assert window["candles"][0]["bar_open_at"] == "2025-08-01T00:00:00+00:00"
    assert window["candles"][-1]["bar_open_at"] == "2025-08-01T05:00:00+00:00"


@pytest.mark.parametrize("product", ["SOL-USD", "btc-usd", "", "../../etc/passwd", None])
def test_an_unknown_product_is_refused(service, contracts, product):
    with pytest.raises(contracts.NotFoundError, match="unknown product"):
        service.market_candles(product)


# --- bounded charts --------------------------------------------------------

def test_a_chart_never_returns_more_points_than_requested(service, contracts):
    for requested in (50, 200, contracts.MAX_CHART_POINTS):
        chart = service.market_chart("BTC-USD", max_points=requested)
        assert chart["metadata"]["returned_count"] <= requested
        assert len(chart["series"]) == chart["metadata"]["returned_count"]
    with pytest.raises(contracts.AppApiError, match="exceeds the maximum"):
        service.market_chart("BTC-USD", max_points=contracts.MAX_CHART_POINTS + 1)


def test_a_chart_over_the_full_corpus_is_aggregated_and_says_so(service):
    chart = service.market_chart("BTC-USD", max_points=500)
    metadata = chart["metadata"]
    assert metadata["source_count"] == 8750
    assert metadata["aggregated"] is True
    assert metadata["aggregation"] == "ohlc-bucket"
    assert metadata["bucket_size"] > 1
    assert metadata["source_timeframe"] == "1h"
    # a caller must not be able to mistake this for native 1h candles
    assert metadata["returned_count"] < metadata["source_count"]


def test_aggregated_buckets_keep_ohlc_semantics_rather_than_averaging(service):
    """An averaged "candle" is not a candle."""
    window = {"start": "2025-08-01T00:00:00+00:00", "end": "2025-08-01T23:00:00+00:00"}
    raw = service.market_candles("BTC-USD", limit=100, **window)["candles"]
    assert len(raw) == 24
    chart = service.market_chart("BTC-USD", max_points=4, **window)
    assert chart["metadata"]["bucket_size"] == 6
    for index, bucket in enumerate(chart["series"]):
        chunk = raw[index * 6:(index + 1) * 6]
        assert bucket["bar_open_at"] == chunk[0]["bar_open_at"]
        assert bucket["open"] == chunk[0]["open"]
        assert bucket["close"] == chunk[-1]["close"]
        assert Decimal(bucket["high"]) == max(Decimal(row["high"]) for row in chunk)
        assert Decimal(bucket["low"]) == min(Decimal(row["low"]) for row in chunk)
        assert Decimal(bucket["volume"]) == sum(Decimal(row["volume"]) for row in chunk)


def test_a_small_window_is_returned_exactly_without_aggregation(service):
    chart = service.market_chart("BTC-USD", start="2025-08-01T00:00:00+00:00",
                                 end="2025-08-01T05:00:00+00:00", max_points=100)
    assert chart["metadata"]["aggregated"] is False
    assert chart["metadata"]["aggregation"] == "none"
    assert chart["metadata"]["bucket_size"] == 1
    assert chart["metadata"]["returned_count"] == 6


# --- signals and risk: contracts, never fabricated runs --------------------


def test_the_signals_view_reports_absence_instead_of_inventing_decisions(service):
    payload = service.signals()
    assert payload["available"] is False
    assert payload["decisions"] == []
    assert "no persisted signal run" in payload["reason"]
    signal_engine = importlib.import_module("scripts.trading_lab.signal_engine")
    assert payload["signal_spec"]["spec_hash"] == signal_engine.SIGNAL_SPEC_V1.spec_hash
    assert payload["signal_spec"]["optimized"] is False


def test_the_risk_view_reports_the_contract_and_no_fabricated_targets(service):
    payload = service.risk_targets()
    assert payload["available"] is False
    assert payload["targets"] == []
    risk_engine = importlib.import_module("scripts.trading_lab.risk_engine")
    spec = payload["risk_spec"]
    assert spec["spec_hash"] == risk_engine.RISK_SPEC_V1.risk_spec_hash
    assert spec["max_long_exposure"] == "0.25"
    assert spec["max_short_exposure"] == "0.25"
    assert spec["risk_scale"] == "1"
    assert spec["volatility_scaling_enabled"] is False
    assert spec["optimized"] is False


# --- research: the committed numbers, unchanged ----------------------------


def test_the_benchmark_summaries_match_the_committed_results(service):
    summaries = {entry["version"]: entry for entry in service.benchmark_summaries()}
    assert set(summaries) == {"v1", "v2"}
    assert summaries["v2"]["experiment_type"] == "exploratory"
    assert summaries["v2"]["confirmatory_result"] is False
    for version, product, expected in (("v1", "BTC-USD", "-0.00304"),
                                       ("v1", "ETH-USD", "0.00417"),
                                       ("v2", "BTC-USD", "-0.00797"),
                                       ("v2", "ETH-USD", "-0.01240")):
        entry = next(item for item in summaries[version]["products"]
                     if item["product"] == product)
        assert entry["rank_ic"].startswith(expected)
        # served verbatim from the artefact, never recomputed here
        stored = json.loads((DATA_ROOT / f"benchmark_results_{version}"
                             / f"{product}.json").read_bytes().decode("utf-8"))
        assert entry["rank_ic"] == stored["selection"]["global_test_metrics"]["rank_ic"]
        assert entry["benchmark_results_hash"] == stored["benchmark_results_hash"]


def test_a_benchmark_detail_carries_all_four_quarters_including_negative_ones(service):
    detail = service.benchmark_detail("v2", "ETH-USD")
    assert detail["experiment_type"] == "exploratory"
    assert detail["confirmatory_result"] is False
    assert len(detail["periods"]) == 4
    assert [period["start"] for period in detail["periods"]] == [
        "2025-08-01T00:00:00Z", "2025-11-01T00:00:00Z",
        "2026-02-01T00:00:00Z", "2026-05-01T00:00:00Z"]
    negatives = [p for p in detail["periods"] if p["rank_ic"].startswith("-")]
    assert len(negatives) == 3          # nothing is hidden
    assert len(detail["scenarios"]) == 3
    assert detail["geometry"]["folds"] == 46


@pytest.mark.parametrize("version,product", [
    ("v3", "BTC-USD"), ("v1", "SOL-USD"), ("../v1", "BTC-USD"),
])
def test_an_unknown_benchmark_is_refused(service, contracts, version, product):
    with pytest.raises(contracts.NotFoundError):
        service.benchmark_detail(version, product)


# --- transport: read-only, loopback, strict CORS ---------------------------


def test_every_documented_endpoint_answers(live_api):
    for path in ("/api/v1/health", "/api/v1/system", "/api/v1/overview",
                 "/api/v1/markets", "/api/v1/markets/BTC-USD?limit=5",
                 "/api/v1/markets/BTC-USD/chart?max_points=50",
                 "/api/v1/signals", "/api/v1/risk/targets",
                 "/api/v1/research/benchmarks",
                 "/api/v1/research/benchmarks/v1/BTC-USD"):
        status, payload, _ = _get(live_api, path)
        assert status == 200, path
        assert isinstance(payload, dict)


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE"])
def test_every_mutating_verb_is_refused(live_api, method):
    """The API cannot grow a write path by accident."""
    request = urllib.request.Request(f"{live_api}/api/v1/overview", method=method,
                                     data=b"{}")
    with pytest.raises(urllib.error.HTTPError) as caught:
        urllib.request.urlopen(request, timeout=15)
    assert caught.value.code == 405
    assert "read-only" in json.loads(caught.value.read())["error"]


def test_cors_is_restricted_to_the_expected_dev_origins(live_api):
    server_module = importlib.import_module("scripts.trading_lab.app_api.server")
    assert "*" not in server_module.ALLOWED_ORIGINS
    _, _, allowed = _get(live_api, "/api/v1/health",
                         headers={"Origin": "http://127.0.0.1:5173"})
    assert allowed.headers.get("Access-Control-Allow-Origin") == "http://127.0.0.1:5173"
    _, _, hostile = _get(live_api, "/api/v1/health",
                         headers={"Origin": "https://evil.example"})
    assert hostile.headers.get("Access-Control-Allow-Origin") is None


def test_the_server_binds_loopback_by_default(live_api):
    server_module = importlib.import_module("scripts.trading_lab.app_api.server")
    assert server_module.DEFAULT_HOST == "127.0.0.1"
    assert server_module.DEFAULT_HOST != "0.0.0.0"


def test_errors_never_leak_a_path_or_a_traceback(live_api):
    for path in ("/api/v1/nope", "/api/v1/markets/SOL-USD",
                 "/api/v1/markets/BTC-USD?limit=999999"):
        try:
            _get(live_api, path)
            raise AssertionError(f"{path} should have failed")
        except urllib.error.HTTPError as error:
            body = error.read().decode("utf-8")
            assert error.code in (400, 404)
            for leak in ("/home/", "Traceback", "File \"", "scripts/trading_lab"):
                assert leak not in body, (path, leak)


def test_responses_stay_small_enough_for_a_first_screen(live_api):
    for path, ceiling in (("/api/v1/overview", 20_000), ("/api/v1/system", 8_000),
                          ("/api/v1/markets", 4_000),
                          ("/api/v1/research/benchmarks", 12_000)):
        _, _, response = _get(live_api, path)
        assert int(response.headers["Content-Length"]) < ceiling, path


def test_payloads_are_deterministic_across_repeated_requests(live_api):
    for path in ("/api/v1/system", "/api/v1/overview",
                 "/api/v1/markets/BTC-USD?limit=20"):
        _, first, _ = _get(live_api, path)
        _, second, _ = _get(live_api, path)
        assert first == second, path


def test_decimals_travel_as_strings_and_timestamps_as_utc(live_api):
    _, payload, _ = _get(live_api, "/api/v1/markets/BTC-USD?limit=3")
    for candle in payload["candles"]:
        for field in ("open", "high", "low", "close", "volume"):
            assert isinstance(candle[field], str), field
        assert candle["bar_open_at"].endswith("+00:00")


# --- economic backtests ----------------------------------------------------


def test_the_backtest_index_is_small_and_honest_when_nothing_has_been_run(service):
    """An engine that exists with no persisted run says exactly that."""
    payload = service.backtests()
    assert isinstance(payload["available"], bool)
    assert payload["execution_spec"]["fee_rate"] == "0.0010"
    assert payload["execution_spec"]["slippage_rate"] == "0.0005"
    assert payload["execution_spec"]["initial_equity"] == "100000"
    assert payload["execution_spec"]["cost_model"] == "synthetic"
    assert payload["execution_spec"]["optimized"] is False
    assert payload["execution_spec"]["exchange_account_specific"] is False
    assert len(payload["execution_spec"]["execution_spec_hash"]) == 64
    if not payload["available"]:
        assert payload["runs"] == []
        assert "no persisted" in payload["reason"]
    # the index never carries a curve or a fill
    assert len(json.dumps(payload)) < 20_000
    assert "equity_curve" not in json.dumps(payload)
    assert "fills" not in json.dumps(payload)


def test_the_backtest_capability_reflects_the_engine_not_a_run(service, contracts):
    assert contracts.CAPABILITIES["economic_backtest"] is True
    assert contracts.CAPABILITIES["paper_trading"] is True
    assert contracts.CAPABILITIES["live_trading"] is False


def test_an_unknown_backtest_version_or_product_is_refused(service, contracts):
    with pytest.raises(contracts.NotFoundError):
        service.backtest_detail("v9", "BTC-USD")
    with pytest.raises(contracts.AppApiError):
        service.backtest_detail("v1", "DOGE-USD")
    with pytest.raises(contracts.NotFoundError):
        service.backtest_equity("v9", "BTC-USD")
    with pytest.raises(contracts.NotFoundError):
        service.backtest_fills("v9", "BTC-USD")


def test_the_backtest_endpoints_refuse_an_unbounded_request(service, contracts):
    """Ceilings exist before there is any data big enough to need them."""
    with pytest.raises(contracts.AppApiError):
        service.backtest_equity("v1", "BTC-USD", max_points=contracts.MAX_EQUITY_POINTS + 1)
    with pytest.raises(contracts.AppApiError):
        service.backtest_fills("v1", "BTC-USD", limit=contracts.MAX_FILL_PAGE + 1)
    assert contracts.DEFAULT_EQUITY_POINTS <= contracts.MAX_EQUITY_POINTS
    assert contracts.DEFAULT_FILL_PAGE <= contracts.MAX_FILL_PAGE


def test_the_backtest_routes_are_served_and_remain_read_only(live_api):
    status, payload, _ = _get(live_api, "/api/v1/backtests")
    assert status == 200
    assert "execution_spec" in payload
    request = urllib.request.Request(f"{live_api}/api/v1/backtests", method="POST")
    with pytest.raises(urllib.error.HTTPError) as raised:
        urllib.request.urlopen(request, timeout=15)
    assert raised.value.code in (405, 501)


# --- shadow trading (read-only) --------------------------------------------


def test_the_paper_view_is_honest_when_no_session_is_running(service):
    payload = service.paper_status(now="2026-08-11T03:30:00+00:00")
    assert payload["shadow_mode"] is True
    assert payload["real_money"] is False
    assert payload["broker_connected"] is False
    assert payload["paper_model_optimized"] is False
    assert payload["paper_execution"]["cost_model"] == "synthetic"
    assert payload["paper_execution"]["terminal_liquidation"] is False
    assert payload["paper_execution"]["fill_observation_policy"] == \
        "recorded-when-the-fill-bar-closes-v1"
    if not payload["available"]:
        assert payload["session"] is None
        assert "no shadow session" in payload["reason"]
    assert len(json.dumps(payload)) < 20_000


def test_the_paper_view_reports_the_holdout_and_never_a_protected_price(service):
    payload = service.paper_status(now="2026-10-01T00:00:00+00:00")
    holdout = payload["protected_holdout"]
    assert holdout["start"] == "2026-09-01T00:00:00Z"
    assert holdout["end"] == "2026-11-30T23:00:00Z"
    assert holdout["observed"] is False
    for product in ("BTC-USD", "ETH-USD"):
        state = payload["embargo"][product]
        assert state["embargoed"] is True
        assert "confirmatory research holdout" in state["reason"]
    text = json.dumps(payload["embargo"])
    for forbidden in ("open", "high", "low", "close", "volume"):
        assert f'"{forbidden}"' not in text


def test_the_paper_endpoints_are_bounded(service, contracts):
    with pytest.raises(contracts.AppApiError):
        service.paper_events("BTC-USD", limit=contracts.MAX_PAPER_EVENTS + 1)
    with pytest.raises(contracts.AppApiError):
        service.paper_equity("BTC-USD",
                             max_points=contracts.MAX_PAPER_EQUITY_POINTS + 1)
    with pytest.raises(contracts.AppApiError):
        service.paper_product("DOGE-USD")
    assert contracts.DEFAULT_PAPER_EVENTS <= contracts.MAX_PAPER_EVENTS
    assert contracts.MAX_SSE_REPLAY_EVENTS == 1_000


def test_the_paper_capability_says_shadow_yes_and_live_no(service, contracts):
    capabilities = service.system()["capabilities"]
    assert capabilities["paper_trading"] is True
    assert capabilities["live_trading"] is False
    assert contracts.CAPABILITIES["live_trading"] is False


def test_the_api_offers_no_way_to_control_the_shadow_session(live_api):
    """Control stays on the command line; the cockpit is read-only."""
    status, payload, _ = _get(live_api, "/api/v1/paper/status")
    assert status == 200 and payload["real_money"] is False
    for path in ("/api/v1/paper/start", "/api/v1/paper/stop", "/api/v1/paper/order"):
        with pytest.raises(urllib.error.HTTPError) as raised:
            _get(live_api, path)
        assert raised.value.code in (400, 404)
    for verb in ("POST", "PUT", "DELETE", "PATCH"):
        request = urllib.request.Request(f"{live_api}/api/v1/paper/status", method=verb)
        with pytest.raises(urllib.error.HTTPError) as raised:
            urllib.request.urlopen(request, timeout=15)
        assert raised.value.code in (405, 501)


def test_the_event_stream_is_advertised_without_a_control_channel(live_api):
    request = urllib.request.Request(f"{live_api}/api/v1/paper/events/stream",
                                     method="HEAD")
    with urllib.request.urlopen(request, timeout=15) as response:
        assert response.status == 200

"""The calendar and session endpoints, over real HTTP.

Two things are being checked. First, that the API can answer session questions
at all -- which sessions exist, how long they are, how many bars each holds --
without a browser recomputing any of it. Second, and more important, that the
answers are bounded: a sessions endpoint enumerates real rows, so an unbounded
range is an unbounded response, and the ceiling has to be enforced by the
server rather than trusted to the client.

The equity assertions here are semantic, matching the calendar tests: a short
session is short, a holiday is missing, a weekend is missing. No date table is
copied into the expectations.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from tests.crypto import loopback

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]

pytest.importorskip(
    "pandas_market_calendars",
    reason="session endpoints need the optional [equities] extra")


@pytest.fixture
def server(tmp_path):
    from scripts.trading_lab.app_api.server import make_server

    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=root></div>")
    httpd = make_server(REPO_ROOT / "data/crypto", host=loopback.host(), port=0,
                        dist_root=dist)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield loopback.url(httpd.server_address[1])
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)


def _get(url, *, method="GET"):
    request = urllib.request.Request(url, method=method)
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


def _json(server, path):
    status, _, body = _get(f"{server}{path}")
    return status, json.loads(body)


# --- calendars -------------------------------------------------------------


def test_the_calendar_list_names_both_markets_and_who_uses_them(server):
    status, payload = _json(server, "/api/v1/calendars")
    assert status == 200
    by_id = {item["calendar_id"]: item for item in payload["calendars"]}
    assert set(by_id) == {"CRYPTO_24_7", "US_EQUITY_REGULAR"}
    assert by_id["CRYPTO_24_7"]["instruments"] == [
        "coinbase:BTC-USD", "coinbase:ETH-USD"]
    assert by_id["US_EQUITY_REGULAR"]["instruments"] == [
        "xnas:AAPL", "xnas:MSFT", "xnas:NVDA", "xnas:QQQ"]


def test_the_two_calendars_report_different_annualisation(server):
    """The number a Sharpe ratio is scaled by, per market, from the server."""
    _, payload = _json(server, "/api/v1/calendars")
    by_id = {item["calendar_id"]: item for item in payload["calendars"]}
    crypto = by_id["CRYPTO_24_7"]
    equity = by_id["US_EQUITY_REGULAR"]

    assert crypto["timeframe"] == "1h"
    assert crypto["bars_per_day"] == 24
    assert crypto["annualization_periods"] == 8760

    assert equity["timeframe"] == "30m"
    assert equity["bars_per_day"] == 13
    assert 3200 <= equity["annualization_periods"] <= 3310
    assert equity["annualization_periods"] != 8760


def test_the_equity_calendar_publishes_the_rules_that_produced_it(server):
    status, payload = _json(server, "/api/v1/calendars/US_EQUITY_REGULAR")
    assert status == 200
    assert payload["spec"]["calendar_provider"] == "pandas_market_calendars"
    assert payload["spec"]["calendar_provider_version"] == "5.4.0"
    assert payload["spec"]["timezone"] == "America/New_York"
    assert len(payload["spec_hash"]) == 64
    assert payload["available"] is True


def test_an_unknown_calendar_is_a_404_and_never_a_fallback(server):
    for name in ("XNYS", "NASDAQ", "US_EQUITY", "crypto_24_7"):
        status, _, _ = _get(f"{server}/api/v1/calendars/{name}")
        assert status == 404, name
    # A traversal attempt is refused by the router before it ever reaches the
    # calendar lookup, so it is a 400 rather than a 404. Either is a refusal;
    # what matters is that no spelling returns a calendar.
    assert _get(f"{server}/api/v1/calendars/../etc")[0] in (400, 404)


def test_the_calendar_endpoints_are_read_only(server):
    for path in ("/api/v1/calendars", "/api/v1/calendars/CRYPTO_24_7"):
        status, headers, body = _get(f"{server}{path}")
        assert status == 200
        assert headers["Cache-Control"] == "no-store"
        assert len(body) < 32 * 1024
        assert _get(f"{server}{path}", method="POST")[0] == 405
        assert _get(f"{server}{path}", method="DELETE")[0] == 405


# --- instrument detail -----------------------------------------------------


def test_an_equity_detail_carries_its_venue_calendar_and_provider(server):
    status, payload = _json(server, "/api/v1/instruments/xnas:AAPL")
    assert status == 200
    assert payload["venue"] == "xnas"
    assert payload["symbol"] == "AAPL"
    assert payload["asset_class"] == "EQUITY"
    assert payload["trading_calendar"] == "US_EQUITY_REGULAR"
    assert payload["timezone"] == "America/New_York"
    assert payload["native_timeframes"] == ["30m", "1d"]
    assert payload["providers"] == ["massive-stocks-historical-v1"]
    assert payload["tradable"] is False
    assert payload["legacy_product_id"] is None
    assert payload["calendar"]["bars_per_day"] == 13


def test_the_etf_is_served_as_an_etf(server):
    _, payload = _json(server, "/api/v1/instruments/xnas:QQQ")
    assert payload["asset_class"] == "ETF"


def test_a_percent_encoded_equity_id_resolves_exactly_once(server):
    assert _get(f"{server}/api/v1/instruments/xnas%3AAAPL")[0] == 200
    # Decoding twice would turn this into a valid id, which is the bug.
    assert _get(f"{server}/api/v1/instruments/xnas%253AAAPL")[0] == 404


def test_the_provider_detail_says_it_holds_a_key_without_naming_it(server):
    status, payload = _json(
        server, "/api/v1/providers/massive-stocks-historical-v1")
    assert status == 200
    assert payload["capabilities"]["data_freshness"] == "END_OF_DAY"
    assert payload["capabilities"]["private_account_data"] is False
    assert payload["network_enabled"] is False
    assert payload["brokerage_endpoints"] == []
    assert payload["credential_required"] is True
    rendered = json.dumps(payload).lower()
    for banned in ("api_key", "apikey", "bearer", "authorization", "token",
                   "hyprl_massive", "balance", "wallet"):
        assert banned not in rendered
    # "account" and "order" each appear exactly once, in the flags that deny
    # having either. A negative declaration is the opposite of a leak, so the
    # check is that they occur nowhere else.
    assert rendered.count("account") == 1
    assert rendered.count("order") == 1
    assert payload["capabilities"]["private_account_data"] is False
    assert payload["capabilities"]["order_book"] is False


# --- sessions --------------------------------------------------------------


def test_the_sessions_endpoint_reports_real_sessions_and_bar_counts(server):
    status, payload = _json(
        server, "/api/v1/instruments/xnas:AAPL/sessions"
                "?start=2026-11-23&end=2026-11-30")
    assert status == 200
    assert payload["continuous"] is False
    assert payload["tradable"] is False
    dates = [item["session_date"] for item in payload["sessions"]]

    # Thanksgiving is missing, the weekend is missing, and the day after
    # Thanksgiving is present but short. Asserted as properties of the
    # response, not against a copied holiday list.
    assert "2026-11-26" not in dates
    assert "2026-11-28" not in dates and "2026-11-29" not in dates
    short = [item for item in payload["sessions"] if item["early_close"]]
    assert len(short) == 1
    assert short[0]["expected_bars"] < 13
    assert short[0]["duration_seconds"] < 6 * 3600 + 1800
    full = [item for item in payload["sessions"] if not item["early_close"]]
    assert all(item["expected_bars"] == 13 for item in full)
    assert payload["early_close_count"] == 1


def test_every_session_carries_a_utc_open_and_close_in_order(server):
    _, payload = _json(server, "/api/v1/instruments/xnas:MSFT/sessions"
                               "?start=2026-01-05&end=2026-01-09")
    assert payload["session_count"] == 5
    for item in payload["sessions"]:
        assert item["open_at"].endswith("Z")
        assert item["close_at"].endswith("Z")
        assert item["open_at"] < item["close_at"]
        assert item["session_type"] == "REGULAR"


def test_a_continuous_market_has_no_sessions_to_enumerate(server):
    """Emitting one row per day would be a fabrication, not a convenience."""
    status, payload = _json(server, "/api/v1/instruments/coinbase:BTC-USD"
                                    "/sessions?start=2026-01-05&end=2026-01-09")
    assert status == 200
    assert payload["continuous"] is True
    assert payload["sessions"] == []
    assert payload["session_count"] is None
    assert payload["calendar"]["calendar_id"] == "CRYPTO_24_7"


def test_a_sessions_window_wider_than_the_ceiling_is_refused(server):
    status, _, body = _get(f"{server}/api/v1/instruments/xnas:AAPL/sessions"
                           "?start=2020-01-01&end=2026-01-01")
    assert status == 400
    assert b"at most" in body


def test_a_malformed_or_reversed_window_is_refused(server):
    for query in ("?start=yesterday&end=2026-01-01",
                  "?start=2026-02-01&end=2026-01-01",
                  "?start=2026-13-45&end=2026-12-31"):
        status, _, _ = _get(
            f"{server}/api/v1/instruments/xnas:AAPL/sessions{query}")
        assert status == 400, query


def test_a_timeframe_the_instrument_does_not_publish_is_refused(server):
    """An equity has no hourly grid, so it must not answer as if it had one."""
    status, _, body = _get(f"{server}/api/v1/instruments/xnas:AAPL/sessions"
                           "?start=2026-01-05&end=2026-01-09&timeframe=1h")
    assert status == 400
    assert b"does not publish" in body


def test_sessions_for_an_unknown_instrument_are_a_404(server):
    for name in ("xnas:TSLA", "massive:AAPL", "AAPL", "nyse:AAPL"):
        status, _, _ = _get(f"{server}/api/v1/instruments/{name}/sessions")
        assert status == 404, name


def test_the_sessions_endpoint_is_read_only_and_bounded_in_size(server):
    path = "/api/v1/instruments/xnas:AAPL/sessions?start=2026-01-01&end=2026-03-01"
    status, headers, body = _get(f"{server}{path}")
    assert status == 200
    assert headers["Cache-Control"] == "no-store"
    assert len(body) < 64 * 1024
    assert _get(f"{server}{path}", method="POST")[0] == 405


def test_the_default_window_is_bounded_without_any_parameters(server):
    """No range given must not mean every session that ever happened."""
    status, payload = _json(server, "/api/v1/instruments/xnas:NVDA/sessions")
    assert status == 200
    assert payload["session_count"] is not None
    assert payload["session_count"] <= 30


# --- the boundary the new endpoints must not cross -------------------------


def test_no_equity_endpoint_offers_a_backtest_a_signal_or_a_paper_session(server):
    for path in ("/api/v1/instruments/xnas:AAPL",
                 "/api/v1/instruments/xnas:AAPL/sessions",
                 "/api/v1/calendars/US_EQUITY_REGULAR"):
        rendered = _get(f"{server}{path}")[2].decode().lower()
        for absent in ("prediction", "signal_spec", "backtest", "equity_curve",
                       "fill", "position", "sharpe"):
            assert absent not in rendered, f"{path} mentions {absent}"
    # And the equity products have no runtime endpoints at all.
    for path in ("/api/v1/markets/AAPL", "/api/v1/paper/AAPL",
                 "/api/v1/backtests/v1/AAPL", "/api/v1/markets/xnas:AAPL"):
        assert _get(f"{server}{path}")[0] in (400, 404), path


def test_the_capabilities_still_report_no_live_trading(server):
    _, payload = _json(server, "/api/v1/overview")
    assert payload["capabilities"]["live_trading"] is False

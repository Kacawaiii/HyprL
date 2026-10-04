"""Phase 5E: the production server, over real HTTP.

The static-asset unit tests check the resolver in isolation. These run against
a live socket, because the properties that matter here are properties of the
response -- status, headers, which layer answered -- and those only exist once
a request has gone through the whole handler.

Not marked `ml`: serving an application must not require the model stack.
"""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from tests.crypto import loopback

REPO_ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


@pytest.fixture
def dist(tmp_path):
    root = tmp_path / "dist"
    (root / "assets").mkdir(parents=True)
    (root / "index.html").write_text(
        "<!doctype html><title>HyprL</title><div id=root></div>")
    (root / "assets" / "app-deadbeef.js").write_text("export const x = 1")
    (root / "assets" / "app-deadbeef.css").write_text(":root{}")
    (root / "favicon.ico").write_bytes(b"\x00\x00\x01\x00")
    (tmp_path / "outside-the-root.txt").write_text("NEVER-SERVE-THIS")
    return root


@pytest.fixture
def server(dist):
    from scripts.trading_lab.app_api.server import make_server

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


# --- single origin ---------------------------------------------------------


def test_the_app_and_its_api_are_served_from_one_origin(server):
    status, headers, body = _get(f"{server}/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"<div id=root>" in body

    status, headers, body = _get(f"{server}/api/v1/health")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert json.loads(body)["status"] == "ok"


@pytest.mark.parametrize("route", [
    "/markets", "/signals", "/risk", "/paper", "/backtests", "/research",
    "/system", "/settings", "/backtests/v1/BTC-USD",
])
def test_every_client_route_reloads_into_the_application(server, route):
    """A deep link that 404s on refresh is a broken app, not a broken user."""
    status, headers, body = _get(f"{server}{route}")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"HyprL" in body


def test_an_unknown_api_path_never_falls_through_to_the_document(server):
    """JSON callers must get JSON, including when they are wrong."""
    status, headers, body = _get(f"{server}/api/v1/does-not-exist")
    assert status == 400
    assert headers["Content-Type"].startswith("application/json")
    assert b"<!doctype html>" not in body.lower()

    status, headers, body = _get(f"{server}/api/v1/paper/NOPE-USD")
    assert status == 404
    assert headers["Content-Type"].startswith("application/json")


def test_a_missing_asset_is_a_404_not_a_page_of_html(server):
    status, headers, _ = _get(f"{server}/assets/never-built-abc123.js")
    assert status == 404
    assert headers["Content-Type"].startswith("application/json")


# --- static safety ---------------------------------------------------------


@pytest.mark.parametrize("attack", [
    "/../outside-the-root.txt",
    "/../../etc/passwd",
    "/assets/../../outside-the-root.txt",
    "/%2e%2e/outside-the-root.txt",
    "/..%2foutside-the-root.txt",
    "/assets/%2e%2e/%2e%2e/outside-the-root.txt",
])
def test_no_traversal_reaches_a_file_outside_the_build(server, attack):
    status, _, body = _get(f"{server}{attack}")
    assert status in (403, 404), f"{attack} returned {status}"
    assert b"NEVER-SERVE-THIS" not in body
    assert b"root:" not in body


def test_the_repository_itself_is_not_reachable(server):
    for path in ("/.git/config", "/../.git/config", "/var/trading_lab/paper_v1.sqlite",
                 "/data/crypto", "/scripts/trading_lab/signal_engine.py"):
        status, _, body = _get(f"{server}{path}")
        assert status in (200, 403, 404)
        # a 200 here can only be the SPA document, never repository content
        if status == 200:
            assert b"<div id=root>" in body, path


# --- cache policy ----------------------------------------------------------


def test_hashed_assets_are_cached_forever_and_the_document_is_revalidated(server):
    _, headers, _ = _get(f"{server}/assets/app-deadbeef.js")
    assert "immutable" in headers["Cache-Control"]
    assert "max-age=31536000" in headers["Cache-Control"]

    _, headers, _ = _get(f"{server}/")
    assert headers["Cache-Control"] == "no-cache"


def test_runtime_api_responses_are_never_stored(server):
    """Paper and runtime state are time-sensitive; a cached copy is a lie."""
    for path in ("/api/v1/health", "/api/v1/paper", "/api/v1/ops/runtime",
                 "/api/v1/ops/storage"):
        _, headers, _ = _get(f"{server}{path}")
        assert headers["Cache-Control"] == "no-store", path


def test_responses_carry_the_no_sniff_header(server):
    for path in ("/", "/assets/app-deadbeef.js", "/api/v1/health"):
        _, headers, _ = _get(f"{server}{path}")
        assert headers.get("X-Content-Type-Options") == "nosniff", path


# --- read-only -------------------------------------------------------------


@pytest.mark.parametrize("verb", ["POST", "PUT", "PATCH", "DELETE"])
def test_no_verb_can_change_anything(server, verb):
    status, _, body = _get(f"{server}/api/v1/ops/runtime", method=verb)
    assert status == 405
    assert b"read-only" in body


@pytest.mark.parametrize("path", [
    "/api/v1/paper/start", "/api/v1/paper/stop", "/api/v1/ops/restart",
    "/api/v1/ops/settings/save", "/api/v1/ops/delete",
])
def test_there_is_no_lifecycle_control_over_http(server, path):
    """Lifecycle stays on the command line; a cockpit must not start trading."""
    status, _, _ = _get(f"{server}{path}", method="POST")
    assert status == 405
    status, _, _ = _get(f"{server}{path}")
    assert status in (400, 404)


# --- operations endpoints --------------------------------------------------


def test_the_operations_endpoints_answer_and_stay_bounded(server):
    for path in ("/api/v1/ops/runtime", "/api/v1/ops/recovery",
                 "/api/v1/ops/storage", "/api/v1/ops/settings",
                 "/api/v1/ops/health-history"):
        status, _, body = _get(f"{server}{path}")
        assert status == 200, path
        assert len(body) < 256 * 1024, f"{path} returned {len(body)} bytes"
        json.loads(body)


def test_health_history_refuses_an_unbounded_limit(server):
    status, _, body = _get(f"{server}/api/v1/ops/health-history?limit=1000000")
    assert status == 400
    assert b"limit must sit in" in body


def test_health_history_refuses_an_unknown_component(server):
    status, _, _ = _get(f"{server}/api/v1/ops/health-history?component=nonsense")
    assert status == 404


def test_the_runtime_endpoint_publishes_no_absolute_path(server):
    _, _, body = _get(f"{server}/api/v1/ops/runtime")
    rendered = body.decode()
    assert "/home/" not in rendered
    assert str(REPO_ROOT) not in rendered


def test_the_storage_endpoint_reports_sizes_not_a_filesystem(server):
    _, _, body = _get(f"{server}/api/v1/ops/storage")
    payload = json.loads(body)
    assert isinstance(payload["paper_database_bytes"], int)
    assert payload["log_cap_bytes"] > 0
    assert "append-only" in payload["paper_events_retention"]
    assert "files" not in payload and "path" not in payload


def test_the_settings_endpoint_states_what_it_refuses(server):
    _, _, body = _get(f"{server}/api/v1/ops/settings")
    payload = json.loads(body)
    assert payload["trading_contracts_immutable"] is True
    assert "signal_threshold" in payload["forbidden_trading_fields"]
    assert "theme" in payload["allowed_fields"]


def test_the_api_still_declares_no_live_trading(server):
    _, _, body = _get(f"{server}/api/v1/system")
    capabilities = json.loads(body)["capabilities"]
    assert capabilities["live_trading"] is False
    assert capabilities["paper_trading"] is True


def test_the_server_binds_loopback_by_default():
    """A default of 0.0.0.0 would publish a research runtime to the network."""
    from scripts.trading_lab.app_api import server as server_module

    assert server_module.DEFAULT_HOST == "127.0.0.1"
    assert "0.0.0.0" not in (server_module.DEFAULT_HOST,)
    assert "*" not in server_module.ALLOWED_ORIGINS


def test_cors_never_answers_with_a_wildcard(server):
    request = urllib.request.Request(f"{server}/api/v1/health",
                                     headers={"Origin": "https://evil.example"})
    with urllib.request.urlopen(request, timeout=15) as response:
        assert response.headers.get("Access-Control-Allow-Origin") != "*"
        assert response.headers.get("Access-Control-Allow-Origin") is None


def test_the_dev_origin_is_still_allowed(server):
    request = urllib.request.Request(f"{server}/api/v1/health",
                                     headers={"Origin": "http://127.0.0.1:5173"})
    with urllib.request.urlopen(request, timeout=15) as response:
        assert response.headers.get("Access-Control-Allow-Origin") == \
            "http://127.0.0.1:5173"


def test_the_api_serves_without_any_frontend_build(tmp_path):
    """A missing build must degrade to API-only, not to a broken server."""
    from scripts.trading_lab.app_api.server import make_server

    httpd = make_server(REPO_ROOT / "data/crypto", host=loopback.host(), port=0,
                        dist_root=tmp_path / "no-build-here")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = loopback.url(httpd.server_address[1])
        status, _, body = _get(f"{base}/api/v1/health")
        assert status == 200 and json.loads(body)["status"] == "ok"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)


# --- instruments and providers (Phase 6A) ---------------------------------


def test_the_instrument_registry_is_served_grouped_by_asset_class(server):
    status, _, body = _get(f"{server}/api/v1/instruments")
    assert status == 200
    payload = json.loads(body)
    # The catalogue describes six markets; this build trades two of them, and
    # every entry says which it is. A client that ignored the flag would offer
    # an equity for a backtest that cannot run.
    assert payload["count"] == 6
    assert payload["tradable_count"] == 2
    assert [group["asset_class"] for group in payload["asset_classes"]] == [
        "CRYPTO", "EQUITY", "ETF"]
    ids = [item["instrument_id"] for item in payload["instruments"]]
    assert ids == ["coinbase:BTC-USD", "coinbase:ETH-USD", "xnas:AAPL",
                   "xnas:MSFT", "xnas:NVDA", "xnas:QQQ"]
    tradable = {item["instrument_id"]: item["tradable"]
                for item in payload["instruments"]}
    assert tradable == {"coinbase:BTC-USD": True, "coinbase:ETH-USD": True,
                        "xnas:AAPL": False, "xnas:MSFT": False,
                        "xnas:NVDA": False, "xnas:QQQ": False}
    # Only a tradable instrument has a legacy product id, because only a
    # tradable one appears in the artefacts that use one.
    for item in payload["instruments"]:
        assert (item["legacy_product_id"] is not None) == item["tradable"]


def test_an_instrument_carries_its_identity_and_metadata(server):
    _, _, body = _get(f"{server}/api/v1/instruments")
    btc = json.loads(body)["instruments"][0]
    assert btc["venue"] == "coinbase" and btc["symbol"] == "BTC-USD"
    assert btc["base_asset"] == "BTC" and btc["quote_asset"] == "USD"
    assert btc["trading_calendar"] == "CRYPTO_24_7"
    assert btc["native_timeframes"] == ["1h", "1d"]
    assert len(btc["instrument_spec_hash"]) == 64
    assert btc["providers"] == ["coinbase-public-v1"]
    # the string every committed artefact uses, so the UI can address them
    assert btc["legacy_product_id"] == "BTC-USD"


@pytest.mark.parametrize("spelling", [
    "coinbase:BTC-USD", "COINBASE:BTC-USD", "coinbase:btc-usd", "BTC-USD",
    "btcusd",
])
def test_an_instrument_detail_resolves_every_spelling(server, spelling):
    import urllib.parse

    status, _, body = _get(
        f"{server}/api/v1/instruments/{urllib.parse.quote(spelling, safe='')}")
    assert status == 200, spelling
    assert json.loads(body)["instrument_id"] == "coinbase:BTC-USD"


def test_an_instrument_detail_carries_its_calendar(server):
    _, _, body = _get(f"{server}/api/v1/instruments/coinbase:BTC-USD")
    calendar = json.loads(body)["calendar"]
    assert calendar["calendar_id"] == "CRYPTO_24_7"
    assert calendar["bars_per_day"] == 24
    assert calendar["annualization_periods"] == 8760


@pytest.mark.parametrize("unknown", ["coinbase:SOL-USD", "AAPL", "nonsense",
                                     "nasdaq:AAPL"])
def test_an_unknown_instrument_is_a_404_not_a_guess(server, unknown):
    import urllib.parse

    status, headers, _ = _get(
        f"{server}/api/v1/instruments/{urllib.parse.quote(unknown, safe='')}")
    assert status == 404
    assert headers["Content-Type"].startswith("application/json")


def test_the_providers_endpoint_declares_capabilities_honestly(server):
    status, _, body = _get(f"{server}/api/v1/providers")
    assert status == 200
    provider = json.loads(body)["providers"][0]
    assert provider["provider_id"] == "coinbase-public-v1"
    capabilities = provider["capabilities"]
    assert capabilities["historical_bars"] is True
    assert capabilities["latest_closed_bar"] is True
    assert capabilities["realtime_ticks"] is False
    assert capabilities["order_book"] is False
    assert capabilities["authenticated"] is False
    assert capabilities["private_account_data"] is False


def test_a_provider_detail_lists_the_instruments_it_serves(server):
    status, _, body = _get(f"{server}/api/v1/providers/coinbase-public-v1")
    assert status == 200
    payload = json.loads(body)
    assert payload["instruments"] == ["coinbase:BTC-USD", "coinbase:ETH-USD"]
    assert len(payload["instrument_details"]) == 2


def test_an_unknown_provider_is_a_404(server):
    status, _, _ = _get(f"{server}/api/v1/providers/some-broker")
    assert status == 404


def test_the_registry_endpoints_are_small_and_read_only(server):
    for path in ("/api/v1/instruments", "/api/v1/providers",
                 "/api/v1/instruments/coinbase:BTC-USD"):
        status, headers, body = _get(f"{server}{path}")
        assert status == 200
        assert len(body) < 32 * 1024, f"{path} returned {len(body)} bytes"
        assert headers["Cache-Control"] == "no-store"
        assert _get(f"{server}{path}", method="POST")[0] == 405


def test_no_registry_endpoint_exposes_a_key_or_an_account(server):
    # The Massive provider is the one that actually holds a credential, so it
    # is the one this check exists for. Its detail endpoint may say whether a
    # key is configured; it may not name it, echo it, or offer a field shaped
    # like somewhere to put it.
    for path in ("/api/v1/instruments", "/api/v1/providers",
                 "/api/v1/providers/coinbase-public-v1",
                 "/api/v1/providers/massive-stocks-historical-v1"):
        rendered = _get(f"{server}{path}")[2].decode().lower()
        for secret in ("api_key", "apikey", "secret", "token", "authorization",
                       "bearer", "cookie", "balance", "wallet", "account_id",
                       "hyprl_massive"):
            assert secret not in rendered, f"{path} mentions {secret}"


@pytest.mark.parametrize("encoded,expected", [
    # decoded once -> "coinbase:BTC-USD", a registered id
    ("coinbase%3ABTC-USD", 200),
    # decoded once -> the literal "coinbase%3ABTC-USD", which is not an id.
    # Decoding twice would turn it into one, and a route that decodes twice
    # can be fed a payload that survives the first pass untouched.
    ("coinbase%253ABTC-USD", 404),
    ("coinbase%25253ABTC-USD", 404),
    # a percent-encoded separator must not reopen a path segment either
    ("coinbase%2FBTC-USD", 404),
    ("coinbase%252FBTC-USD", 404),
])
def test_a_route_identifier_is_decoded_exactly_once(server, encoded, expected):
    status, headers, _ = _get(f"{server}/api/v1/instruments/{encoded}")
    assert status == expected, encoded
    assert headers["Content-Type"].startswith("application/json")


def test_a_provider_route_identifier_is_decoded_exactly_once(server):
    assert _get(f"{server}/api/v1/providers/coinbase-public-v1")[0] == 200
    # "%2D" decodes once to "-", so a double decode would resolve this
    assert _get(f"{server}/api/v1/providers/coinbase%252Dpublic%252Dv1")[0] == 404


# --- portfolio (Phase 6B) -------------------------------------------------


def test_the_portfolio_contract_is_served_with_or_without_a_run(server):
    status, _, body = _get(f"{server}/api/v1/portfolio")
    assert status == 200
    payload = json.loads(body)
    contract = payload["portfolio"]
    assert contract["allocation_rule"] == "proportional-gross-cap-v1"
    assert contract["cash_model"] == "shared-cash-v1"
    assert contract["optimized"] is False
    assert contract["max_gross_exposure"] == "0.50"
    assert payload["shared_capital"] is True
    assert payload["real_money"] is False
    assert payload["commercial_edge_established"] is False


def test_the_portfolio_summary_stays_far_inside_its_budget(server):
    """A summary a page renders first must not be a megabyte."""
    for path in ("/api/v1/portfolio", "/api/v1/portfolio/backtests"):
        status, headers, body = _get(f"{server}{path}")
        assert status == 200, path
        assert len(body) < 20 * 1024, f"{path} returned {len(body)} bytes"
        assert headers["Cache-Control"] == "no-store"


def test_an_unknown_portfolio_version_is_a_404(server):
    for path in ("/api/v1/portfolio/backtests/v9",
                 "/api/v1/portfolio/backtests/v9/equity",
                 "/api/v1/portfolio/backtests/v9/attribution"):
        assert _get(f"{server}{path}")[0] == 404, path


def test_an_unknown_portfolio_leaf_is_refused(server):
    status, _, _ = _get(f"{server}/api/v1/portfolio/backtests/v1/nonsense")
    assert status in (400, 404)


@pytest.mark.parametrize("verb", ["POST", "PUT", "PATCH", "DELETE"])
def test_the_portfolio_surface_is_read_only(server, verb):
    assert _get(f"{server}/api/v1/portfolio", method=verb)[0] == 405


# --- shared paper portfolio (Phase 6C) ------------------------------------


def test_the_paper_surface_reports_the_shared_portfolio(server):
    status, _, body = _get(f"{server}/api/v1/paper/portfolio")
    assert status == 200
    payload = json.loads(body)
    assert payload["mode"] == "SHARED_PORTFOLIO"
    assert payload["shared_capital"] is True
    assert payload["real_money"] is False
    assert payload["broker_connected"] is False
    assert payload["commercial_edge_established"] is False
    assert payload["portfolio_spec_hash"]
    assert payload["protected_holdout"]["observed"] is False


def test_every_shared_portfolio_endpoint_answers_and_stays_bounded(server):
    for path in ("/api/v1/paper/portfolio", "/api/v1/paper/portfolio/positions",
                 "/api/v1/paper/portfolio/pending",
                 "/api/v1/paper/portfolio/events",
                 "/api/v1/paper/portfolio/fills",
                 "/api/v1/paper/portfolio/equity", "/api/v1/paper/legacy"):
        status, headers, body = _get(f"{server}{path}")
        assert status == 200, path
        assert len(body) < 256 * 1024, f"{path} returned {len(body)} bytes"
        assert headers["Cache-Control"] == "no-store"
        json.loads(body)


def test_the_legacy_paper_history_is_reported_separately(server):
    _, _, body = _get(f"{server}/api/v1/paper/legacy")
    payload = json.loads(body)
    assert payload["label"] == "PRE-SHARED-PORTFOLIO"
    assert payload["shared_capital"] is False
    assert "never added to it" in payload["note"]


def test_the_portfolio_event_feed_refuses_an_unbounded_limit(server):
    status, _, body = _get(f"{server}/api/v1/paper/portfolio/events?limit=999999")
    assert status == 400
    assert b"limit" in body


@pytest.mark.parametrize("verb", ["POST", "PUT", "PATCH", "DELETE"])
def test_the_shared_portfolio_surface_is_read_only(server, verb):
    assert _get(f"{server}/api/v1/paper/portfolio", method=verb)[0] == 405


def test_no_portfolio_endpoint_can_start_or_stop_a_session(server):
    for path in ("/api/v1/paper/portfolio/start", "/api/v1/paper/portfolio/stop",
                 "/api/v1/paper/start"):
        assert _get(f"{server}{path}", method="POST")[0] == 405
        assert _get(f"{server}{path}")[0] in (400, 404)

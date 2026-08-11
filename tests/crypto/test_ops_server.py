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

    httpd = make_server(REPO_ROOT / "data/crypto", host="127.0.0.1", port=0,
                        dist_root=dist)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}"
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

    httpd = make_server(REPO_ROOT / "data/crypto", host="127.0.0.1", port=0,
                        dist_root=tmp_path / "no-build-here")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        status, _, body = _get(f"{base}/api/v1/health")
        assert status == 200 and json.loads(body)["status"] == "ok"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=10)

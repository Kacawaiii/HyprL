"""Frozen replay endpoints are bounded, cursor-bound, read-only and isolated."""
import hashlib
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import pytest

from scripts.trading_lab.app_api.contracts import AppApiError, ConflictError, NotFoundError
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.app_api.server import make_server
from tests.crypto import loopback

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def service():
    return AppService(ROOT / "data/crypto")


def test_summary_serves_the_exact_frozen_values_without_live_store_access(service, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("offline evidence opened a live database")
    monkeypatch.setattr(service, "_paper_store", forbidden)
    summary = service.replay.summary()
    assert summary["available"] and len(summary["products"]) == 2
    assert len(json.dumps(summary)) < 20000
    for product in summary["products"]:
        stored = json.loads((ROOT / f"data/crypto/paper_replay_v2/{product['product']}.json").read_text())
        assert product == {k: v for k, v in stored.items() if k not in ("fills", "equity_curve")}
    assert summary["confirmatory"] is False and summary["optimized"] is False


@pytest.mark.parametrize("leaf", ["equity", "fills"])
@pytest.mark.parametrize("limit", [0, -1, True, 2.3, "oops", "2.3", 1001, 1000000])
def test_replay_pages_refuse_invalid_or_unbounded_limits(service, leaf, limit):
    with pytest.raises(AppApiError):
        service.replay.page("BTC-USD", leaf, limit=limit)


@pytest.mark.parametrize("product,leaf", [("BTC-USD", "equity"), ("ETH-USD", "fills")])
def test_pages_cover_all_records_once_and_bind_cursors_to_product_endpoint_result(service, product, leaf):
    cursor, collected = None, []
    first = service.replay.page(product, leaf, limit=1)
    field = "series" if leaf == "equity" else "fills"
    if first["page"]["has_more"]:
        other = "ETH-USD" if product == "BTC-USD" else "BTC-USD"
        for candidate, endpoint in ((other, leaf), (product, "fills" if leaf == "equity" else "equity")):
            with pytest.raises(AppApiError, match="different"):
                service.replay.page(candidate, endpoint, cursor=first["page"]["next_cursor"])
    while True:
        page = service.replay.page(product, leaf, limit=1, cursor=cursor)
        assert page["page"]["returned"] <= 1
        collected.extend(page[field])
        cursor = page["page"]["next_cursor"]
        if not cursor:
            break
    assert len(collected) == first["page"]["total"]
    assert len({row["timestamp"] for row in collected}) == len(collected)
    stored = json.loads((ROOT / f"data/crypto/paper_replay_v2/{product}.json").read_text())
    assert collected == stored["equity_curve" if leaf == "equity" else "fills"]
    with pytest.raises(AppApiError, match="malformed"):
        service.replay.page(product, leaf, cursor="garbage")


def test_empty_replay_is_honest_and_unknown_products_cannot_reach_paths(tmp_path):
    service = AppService(tmp_path)
    assert service.replay.summary()["available"] is False
    for product in ("BTC-USD", "../../../private", "DOGE-USD"):
        with pytest.raises(NotFoundError):
            service.replay.page(product, "equity")
    assert list(tmp_path.iterdir()) == []


def test_a_tampered_result_is_not_displayed(tmp_path):
    from scripts.trading_lab.app_api.paper_replay import _digest
    base = tmp_path / "paper_replay_v2"
    base.mkdir()
    payload = {"product": "ETH-USD", "result_hash": "a" * 64}
    raw = json.dumps(payload).encode()
    (base / "BTC-USD.json").write_bytes(raw)
    manifest = {"schema_version": "trading-lab.paper-replay-manifest.v2",
                "products": {"BTC-USD": {"file_sha256": hashlib.sha256(raw).hexdigest(),
                                         "result_hash": "a" * 64}}}
    manifest["manifest_hash"] = _digest(manifest)
    (base / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ConflictError, match="binding mismatch"):
        AppService(tmp_path).replay.page("BTC-USD", "fills")


def test_a_cursor_cannot_continue_across_a_replaced_result(tmp_path):
    from scripts.trading_lab.app_api.paper_replay import _digest
    source = ROOT / "data/crypto/paper_replay_v2"
    base = tmp_path / "paper_replay_v2"
    base.mkdir()
    for name in ("manifest.json", "BTC-USD.json"):
        (base / name).write_bytes((source / name).read_bytes())
    views = AppService(tmp_path).replay
    cursor = views.page("BTC-USD", "equity", limit=1)["page"]["next_cursor"]
    assert cursor
    result = json.loads((base / "BTC-USD.json").read_text())
    result.pop("result_hash")
    result["equity_curve"][0]["equity"] = "99999"
    result["result_hash"] = _digest(result)
    raw = json.dumps(result).encode()
    (base / "BTC-USD.json").write_bytes(raw)
    manifest = json.loads((base / "manifest.json").read_text())
    manifest.pop("manifest_hash")
    manifest["products"]["BTC-USD"].update(result_hash=result["result_hash"],
                                           file_sha256=hashlib.sha256(raw).hexdigest())
    manifest["manifest_hash"] = _digest(manifest)
    (base / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AppApiError, match="different query"):
        views.page("BTC-USD", "equity", cursor=cursor)


def test_routes_and_mutating_verbs(service):
    server = make_server(ROOT / "data/crypto", host=loopback.host(), port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = loopback.url(server.server_address[1])
    try:
        for path in ("/api/v1/paper/replay", "/api/v1/paper/replay/BTC-USD/equity?limit=1",
                     "/api/v1/paper/replay/ETH-USD/fills?limit=1"):
            with urllib.request.urlopen(base + path, timeout=10) as response:
                assert response.status == 200
                payload = json.load(response)
                if "page" in payload:
                    assert payload["page"]["returned"] <= 1
        for verb in ("POST", "PUT", "PATCH", "DELETE"):
            request = urllib.request.Request(base + "/api/v1/paper/replay", method=verb)
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=10)
            assert error.value.code == 405
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(base + "/api/v1/paper/replay/BTC-USD/equity?limit=1001", timeout=10)
        assert error.value.code == 400
    finally:
        server.shutdown()
        server.server_close()

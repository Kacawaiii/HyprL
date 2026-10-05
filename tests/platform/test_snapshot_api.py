import json
import threading
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pytest

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.app_api.server import build_routes, make_server
from scripts.trading_lab.app_api.service import AppService
from scripts.trading_lab.platform.contracts import InformationSnapshot
from scripts.trading_lab.platform.prices import CorpusPrices
from scripts.trading_lab.platform.snapshot import SnapshotBuilder
from scripts.trading_lab.edgar import synthetic as syn
from tests.crypto import loopback
from tests.crypto.test_event_features import A, ACC, EdgarEnv


@pytest.fixture
def env(tmp_path):
    env = EdgarEnv(tmp_path)
    env.serve(A, syn.filing(ACC))
    env.poll()
    env.T = env.settle()
    yield env
    env.close()


@pytest.fixture
def service(env, tmp_path):
    return AppService(tmp_path, edgar_store=env.root)


def test_api_equals_core_and_binds_returned_full_contract(service, env):
    response = service.snapshots.snapshot(visibility_mode="DURABLE_OBSERVED", as_of=env.T.isoformat(), products="AAPL,MSFT")
    with SnapshotBuilder(visibility_mode="DURABLE_OBSERVED", edgar_store=env.root, prices=CorpusPrices(service._root)) as builder:
        expected = builder.build(env.T, ["AAPL", "MSFT"])
    assert response["snapshot"] == expected.to_dict()
    assert response["fingerprint"] == InformationSnapshot.from_dict(response["snapshot"]).identity
    assert response["read_only"] is True
    assert str(env.root) not in json.dumps(response)


@pytest.mark.parametrize("args", [
    {}, {"as_of": "2026-10-04", "products": "AAPL"},
    {"as_of": "bad", "products": "AAPL"}, {"as_of": "2026-10-04T00:00:00Z", "products": "bad"},
    {"as_of": "2026-10-04T00:00:00Z", "products": "AAPL,xnas:AAPL"},
    {"as_of": "2026-10-04T00:00:00Z", "products": "AAPL", "edgar_horizon": "999999"},
    {"as_of": "2026-10-04T00:00:00Z", "products": "AAPL", "edgar_horizon": "-1"},
    {"as_of": "2026-10-04T00:00:00Z", "products": "AAPL", "edgar_horizon": "1.2"},
    {"as_of": "2026-10-04T00:00:00Z", "products": "AAPL", "fomc_horizon": "0"},
])
def test_invalid_queries(service, args):
    with pytest.raises(AppApiError):
        service.snapshots.snapshot(**{"visibility_mode": "DURABLE_OBSERVED", **args})


def test_router_disallows_global_horizon_and_client_paths(service):
    route = build_routes(service)[0]["/api/v1/snapshots"]
    for query in ({"horizon": ["10"]}, {"fomc_store": ["private"]}, {"as_of": ["1", "2"]}):
        with pytest.raises(AppApiError):
            route(query)


def test_read_only_http_contract_and_provider_endpoint(tmp_path, env):
    server = make_server(tmp_path, host=loopback.host(), port=0, edgar_store=env.root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = loopback.url(server.server_address[1])
    try:
        query = urlencode({"as_of": env.T.isoformat(), "products": "AAPL", "visibility_mode": "DURABLE_OBSERVED"})
        with urlopen(base + "/api/v1/snapshots?" + query, timeout=15) as response:
            payload = json.load(response)
            assert payload["snapshot"]["sources"]["edgar"]["state"] == "RESOLVED"
            assert payload["snapshot"]["events"]
            assert response.headers["Cache-Control"] == "no-store"
        with urlopen(base + "/api/v1/contracts/providers", timeout=15) as response:
            assert len(json.load(response)["providers"]) == 8
        with pytest.raises(HTTPError) as err:
            urlopen(Request(base + "/api/v1/snapshots?" + query, method="POST"), timeout=15)
        assert err.value.code == 405
        with pytest.raises(HTTPError) as err:
            urlopen(base + "/api/v1/snapshots?as_of=bad&products=AAPL", timeout=15)
        assert err.value.code == 400
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_api_corruption_remains_explicit_partial_state(service, env):
    response = service.snapshots.snapshot(visibility_mode="DURABLE_OBSERVED", as_of=env.T.isoformat(), products="AAPL")
    digest = response["snapshot"]["sources"]["edgar"]["dependencies"]["raw_sha256"][0]
    (env.root / "raw" / digest[:2] / digest).write_bytes(b"synthetic damage")
    response = service.snapshots.snapshot(visibility_mode="DURABLE_OBSERVED", as_of=env.T.isoformat(), products="AAPL")
    assert response["snapshot"]["sources"]["edgar"]["state"] == "INTEGRITY_ERROR"
    assert response["snapshot"]["events"] == []
    assert response["snapshot"]["quality"]["state"] == "PARTIAL"
    assert str(env.root) not in json.dumps(response)


@pytest.mark.parametrize("mode", [None, "RETROSPECTIVE_SOURCE", "LIVE_OBSERVED"])
def test_mode_is_mandatory_and_bound(service, env, mode):
    with pytest.raises(AppApiError, match="visibility_mode"):
        service.snapshots.snapshot(as_of=env.T.isoformat(), products="AAPL", visibility_mode=mode)

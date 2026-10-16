from pathlib import Path
import pytest

from scripts.trading_lab.app_api.agent_v2 import AgentV2Views
from scripts.trading_lab.app_api.contracts import AppApiError


def test_missing_api_never_creates_runtime(tmp_path):
    target = tmp_path/'absent'
    with pytest.raises(AppApiError) as error:
        AgentV2Views(target).dispatch('/api/v1/agent-v2/summary', {})
    assert error.value.status == 503
    assert not target.exists()


def test_projection_is_read_only_and_sanitized(ledger):
    before = {p.name:(p.stat().st_mtime_ns,p.stat().st_size) for p in ledger.root.iterdir()}
    out = AgentV2Views(ledger.root).dispatch('/api/v1/agent-v2/summary', {})
    assert out['broker_orders'] == 0 and out['state'] == 'NO_QUOTES'
    assert not {'sources','snapshot','authorization','raw','config'} & out.keys()
    after = {p.name:(p.stat().st_mtime_ns,p.stat().st_size) for p in ledger.root.iterdir()}
    assert before == after
    with pytest.raises(AppApiError):
        AgentV2Views(ledger.root).dispatch('/api/v1/agent-v2/summary', {'runtime':['anything']})


def test_http_route_uses_new_root_without_initializing_v1(ledger, tmp_path):
    import json
    from threading import Thread
    from urllib.request import urlopen
    from scripts.trading_lab.app_api.server import make_server
    server = make_server(tmp_path/'data', host='127.0.0.1', port=0, agent_v2_root=ledger.root)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen('http://127.0.0.1:' + str(server.server_port) + '/api/v1/agent-v2/summary') as response:
            body = json.load(response)
        assert body['mode'] == 'SYNTHETIC' and body['events'] == 1
        assert not (tmp_path/'data'/'shadow.sqlite').exists()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

from contextlib import contextmanager
import json
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.trading_lab.app_api.server import make_server
from scripts.trading_lab.research.demo import synthetic_scenarios
from tests.research.conftest import issue, label, AT, RECORDED
from tests.crypto import loopback


@contextmanager
def server_at(tmp_path, root=None):
    server = make_server(tmp_path / 'data', host=loopback.host(), port=0, research_root=root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield loopback.url(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def request(base, path, *, method='GET'):
    req = Request(base + path, method=method, data=b'{}' if method == 'POST' else None)
    try:
        with urlopen(req, timeout=15) as response:
            return response.status, json.load(response) if method != 'HEAD' else response.read()
    except HTTPError as error:
        return error.code, json.load(error)


def test_unconfigured_store_reports_unavailable_and_proposals_comparison_remain_offline(tmp_path):
    with server_at(tmp_path) as base:
        assert request(base, '/api/v1/observability/predictions')[0] == 503
        code, result = request(base, '/api/v1/research/comparison')
        assert code == 200 and result['state'] == 'WAITING_DATA' and not result['execution_enabled']
        code, proposals = request(base, '/api/v1/research/proposals')
        assert code == 200 and proposals['external_model_calls'] == 0
        assert request(base, '/api/v1/observability/predictions', method='POST')[0] == 405
        assert request(base, '/api/v1/research/hypotheses', method='POST')[0] == 405


def test_paged_causal_ledger_head_monitoring_and_integrity_errors(tmp_path, store):
    result = synthetic_scenarios(store, at=RECORDED)
    before = store.verify()
    with server_at(tmp_path, store.root) as base:
        page_path = '/api/v1/observability/predictions?as_of=' + RECORDED.replace('+00:00', 'Z') + '&limit=7'
        code, page = request(base, page_path)
        assert code == 200 and page['page']['returned'] == 7 and page['page']['has_more']
        cursor = page['page']['next_cursor']
        code, next_page = request(base, page_path + '&cursor=' + cursor)
        assert code == 200 and not ({r['identity'] for r in page['records']} & {r['identity'] for r in next_page['records']})
        assert request(base, page_path.replace('limit=7', 'limit=201'))[0] == 400
        assert request(base, page_path + '&product=OTHER&cursor=' + cursor)[0] == 400
        assert request(base, '/api/v1/observability/predictions?cursor=' + cursor)[0] == 400
        assert request(base, '/api/v1/observability/predictions?as_of=bad')[0] == 400
        p = page['records'][0]['identity']
        code, view = request(base, '/api/v1/observability/predictions/' + p + '?as_of=' + RECORDED.replace('+00:00', 'Z'))
        assert code == 200 and view['label_state'] == 'AVAILABLE'
        assert request(base, '/api/v1/observability/predictions/' + p, method='HEAD') == (200, b'')
        monitoring = '/api/v1/observability/monitoring?as_of=' + RECORDED.replace('+00:00', 'Z')
        code, monitored = request(base, monitoring + '&product=SYNTHETIC-DEMO&split=current&reference_hash=' + result['reference_hash'])
        assert code == 200 and len(monitored['classification']) == 4
        assert request(base, '/api/v1/observability/monitoring')[0] == 400
        assert request(base, monitoring + '&as_of=' + RECORDED.replace('+00:00', 'Z'))[0] == 400
        assert request(base, monitoring + '&file=synthetic')[0] == 400
        assert request(base, '/api/v1/observability/health')[1]['verified']
        assert request(base, '/api/v1/observability/unknown')[0] == 404
        assert request(base, '/api/v1/observability/predictions/' + '0' * 64)[0] == 404
        with store.connect() as db:
            db.execute('DROP TRIGGER no_record_update')
            db.execute("UPDATE records SET payload='{}' WHERE identity=?", (p,))
        code, error = request(base, '/api/v1/observability/predictions/' + p)
        assert code == 409 and 'integrity' in error['error']
        assert str(tmp_path) not in json.dumps(error)


def test_reads_do_not_append_labels_or_start_jobs(tmp_path, store):
    p, _ = issue(store)
    store.append_label(label(p))
    before = store.verify()
    with server_at(tmp_path, store.root) as base:
        path = '/api/v1/observability/predictions/' + p.identity
        code, pending = request(base, path + '?as_of=2026-06-01T00:00:00Z')
        assert code == 200 and pending['label_state'] == 'PENDING'
        code, available = request(base, path + '?as_of=2026-06-04T00:00:00Z')
        assert code == 200 and available['label_state'] == 'AVAILABLE'
    assert store.verify() == before
    assert not (tmp_path / 'data' / 'jobs.sqlite').exists()


def test_frozen_registry_and_experiment_history_routes_are_causal(tmp_path, store, dataset):
    from scripts.trading_lab.research.proposals import propose
    proposal = propose(dataset, limit=1)[0]
    h, e = proposal['hypothesis'], proposal['prepared']
    store.register(h, recorded_at=AT)
    store.start_trial(h, e, trial_id='synthetic-history', recorded_at=AT)
    store.observe_trial('synthetic-history', state='ABANDONED', outcome='ABANDONED',
                        evidence={'reason': 'synthetic-test'}, recorded_at=RECORDED)
    with server_at(tmp_path, store.root) as base:
        code, result = request(base, '/api/v1/research/hypotheses/' + h.identity + '?as_of=2026-06-04T00:00:00Z')
        assert code == 200 and result['criteria_hash'] == h.criteria_hash
        assert [r['payload']['state'] for r in result['trial_history']] == ['PREPARED', 'ABANDONED']
        code, result = request(base, '/api/v1/research/experiments/' + e.identity + '?as_of=2026-06-01T00:00:00Z')
        assert code == 200 and len(result['trial_history']) == 1
        assert request(base, '/api/v1/research/hypotheses/' + h.identity + '?as_of=2026-05-01T00:00:00Z')[0] == 404
        assert request(base, '/api/v1/research/hypotheses?product=BTC-USD')[0] == 400
        assert request(base, '/api/v1/research/hypotheses?limit=')[0] == 400
        assert request(base, '/api/v1/research/hypotheses?limit=1&unexpected=')[0] == 400

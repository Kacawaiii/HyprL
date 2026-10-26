"""Only SEC is prolonged; HTTP tests use an injected synthetic transport."""
from hashlib import sha256
import json

import pytest

from scripts.radar.core import Client, RadarError, instant
from scripts.radar.official import OfficialAuthorization
from tests.radar.test_official_v3 import official
from tests.radar.test_radar import state
OLD = '2026-11-05T00:00:00Z'


def extension(path, *, schema, scope):
    assert scope
    return {'schema': schema, 'decided_at': '2026-10-26T05:45:00Z',
            'decided_by': 'synthetic operator', 'decision': 'Expiry only', 'extends': [{
                'file': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
                'not_after_from': OLD, 'not_after_to': '2026-12-31T00:00:00Z',
                'scope_keys': ['https://www.sec.gov']}]}


def test_sec_extension_does_not_extend_fed_or_telegram(official, state):
    _, path = official
    payload = json.loads(path.read_text())
    payload['not_after'] = OLD
    path.write_text(json.dumps(payload))
    original = OfficialAuthorization(path)
    record = path.parent / 'sec-user-agent-extension-2026-10-26.json'
    record.write_text(json.dumps(extension(path, schema='authorization-extension-v1', scope=True)))
    grant = OfficialAuthorization(path)
    store, _, clock, _ = state
    clock.at = instant('2026-11-06T12:00:00Z')
    calls = []
    client = Client(store, grant, send=lambda *args: (calls.append(args) or (200, {}, b'{}')))
    client.get('https://www.sec.gov/files/company_tickers.json', 'sec', 'sec:tickers')
    assert grant.identity == original.identity
    assert grant.payload['not_after'] == OLD
    receipt = store.rows('http')[-1]
    assert receipt['authorization_hash'] == original.identity
    assert receipt['authorization_extension_sha256'] == sha256(record.read_bytes()).hexdigest()
    for group, url in [('fed', 'https://www.federalreserve.gov/feeds/press_all.xml')]:
        with pytest.raises(RadarError):
            client.get(url, group, group + ':press')
    with pytest.raises(RadarError):
        grant.check(clock(), group='telegram')
    assert len(calls) == 1


@pytest.mark.parametrize('fault', [None, 'missing', 'wrong_hash', 'extra_field', 'wrong_scope', 'self_reference'])
def test_hunter_checks_sec_extension_before_transport(official, fault, tmp_path):
    from io import BytesIO
    from scripts.equity_hunter.fund import get
    from scripts.authorization_extensions import ExtensionError
    _, path = official
    payload = json.loads(path.read_text())
    payload['not_after'] = OLD
    path.write_text(json.dumps(payload))
    value = extension(path, schema='authorization-extension-v1', scope=True)
    record = path.parent / 'sec-user-agent-extension-2026-10-26.json'
    if fault == 'wrong_hash':
        value['extends'][0]['sha256'] = '0' * 64
    elif fault == 'extra_field':
        value['extends'][0]['user_agent'] = 'changed'
    elif fault == 'wrong_scope':
        value['extends'][0]['scope_keys'] = ['https://www.federalreserve.gov']
    elif fault == 'self_reference':
        value['extends'][0]['file'] = str(record)
    record.write_text(json.dumps(value))
    if fault == 'missing':
        record.unlink()
    calls = []
    def send(request, **kwargs):
        calls.append(request)
        response = BytesIO(b'{"synthetic": true}')
        response.headers = {}
        return response
    args = dict(authorization=path, clock=lambda: instant('2026-11-06T12:00:00Z'),
                send=send, pause=lambda _: None, evidence=tmp_path / 'receipt.json')
    if fault:
        with pytest.raises(ExtensionError):
            get('https://www.sec.gov/files/company_tickers.json', **args)
        assert calls == []
    else:
        assert get('https://www.sec.gov/files/company_tickers.json', **args) == {'synthetic': True}
        metadata = json.loads((tmp_path / 'receipt.json').read_text())
        assert metadata['authorization_extension_sha256'] == sha256(record.read_bytes()).hexdigest()
        assert len(calls) == 1

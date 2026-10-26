"""SEC extension checks at the forward transport seam, with synthetic bytes."""
from hashlib import sha256
import json

import pytest

from scripts.trading_lab.agent_v2_quality.contracts import Blocked, instant
from tests.agent_v2_quality.test_boundaries import client, forward


def record(path):
    return {'schema': 'authorization-extension-v1', 'decided_at': '2026-10-26T05:45:00Z',
            'decided_by': 'synthetic operator', 'decision': 'Expiry only', 'extends': [{
                'file': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
                'not_after_from': '2026-11-05T00:00:00Z', 'not_after_to': '2026-12-31T00:00:00Z',
                'scope_keys': ['https://www.sec.gov']}]}


@pytest.mark.parametrize('fault', [None, 'missing', 'wrong_hash', 'extra_field', 'self_reference', 'missing_source'])
def test_producer_sec_extension_is_bound_before_reservation(forward, tmp_path, fault):
    journal, clock = forward
    calls = []
    value = client(forward, tmp_path, lambda *args: (calls.append(args) or (200, b'{}')))
    path = value.sec_authorization
    payload = json.loads(path.read_text())
    payload.update(authorization='news-radar-sec-fed-telegram-v1', extends='news-radar-v1', operator_signed=True,
                   not_after='2026-11-05T00:00:00Z')
    path.write_text(json.dumps(payload))
    extension = record(path)
    target = path.parent / 'sec-user-agent-extension-2026-10-26.json'
    if fault == 'wrong_hash':
        extension['extends'][0]['sha256'] = '0' * 64
    elif fault == 'extra_field':
        extension['extends'][0]['methods'] = ['POST']
    elif fault == 'self_reference':
        extension['extends'][0]['file'] = str(target)
    target.write_text(json.dumps(extension))
    if fault == 'missing':
        target.unlink()
    elif fault == 'missing_source':
        path.unlink()
    clock.at = instant('2026-11-06T12:00:00Z')
    if fault:
        with pytest.raises(Blocked, match='SEC_USER_AGENT_AUTHORIZATION_EXPIRED'):
            value.get('https://data.sec.gov/submissions/CIK0000000001.json')
        assert calls == [] and journal.records('http_reservation') == []
    else:
        _, receipt = value.get('https://data.sec.gov/submissions/CIK0000000001.json')
        evidence = receipt['sec_authorization']
        assert evidence['authorization_extension_sha256'] == sha256(target.read_bytes()).hexdigest()
        assert journal.records('http_reservation')[0]['body']['sec_authorization'] == evidence

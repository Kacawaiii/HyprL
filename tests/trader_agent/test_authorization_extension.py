"""Operator extension records are synthetic, never captured grants."""
from hashlib import sha256
import json

import pytest

from scripts.trading_lab.trader_agent.config import Authorization, TraderError, instant

OLD = '2026-11-05T00:00:00Z'
NEW = '2026-12-31T00:00:00Z'
AFTER = instant('2026-11-06T12:00:00Z')


def extension(path, *, schema='trader-authorization-extension-v1', scope=False):
    entry = {'file': str(path), 'sha256': sha256(path.read_bytes()).hexdigest(),
             'not_after_from': OLD, 'not_after_to': NEW}
    if scope:
        entry['scope_keys'] = ['https://www.sec.gov']
    value = {'schema': schema, 'decided_at': '2026-10-26T05:45:00Z',
             'decided_by': 'synthetic operator', 'decision': 'Extend expiry only', 'extends': [entry]}
    if not scope:
        value['slots'] = ['am']
    return value


def write_extension(path, value):
    target = path.parent / 'trader-v1-extension-2026-10-26.json'
    target.write_text(json.dumps(value))
    return target


def test_trader_extension_preserves_grant_and_authorizes_am(grant, tmp_path):
    path = tmp_path / 'grant.json'
    payload = {**grant.payload, 'not_after': OLD}
    path.write_text(json.dumps(payload))
    original = Authorization.load(path)
    write_extension(path, extension(path))
    extended = Authorization.load(path)
    extended.check(AFTER)
    assert extended.identity == original.identity
    assert extended.payload == payload
    with pytest.raises(TraderError):
        extended.check_slot('pm', AFTER)


@pytest.mark.parametrize('fault', ['missing', 'wrong_hash', 'extra_field', 'pm', 'wrong_schema',
                                    'wrong_from', 'shorten', 'missing_reference', 'mutated_grant',
                                    'duplicate_key', 'self_reference', 'future_decision'])
def test_invalid_extensions_fail_closed_without_changing_original_window(grant, tmp_path, fault):
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps({**grant.payload, 'not_after': OLD}))
    value = extension(path)
    if fault == 'wrong_hash':
        value['extends'][0]['sha256'] = '0' * 64
    elif fault == 'extra_field':
        value['extends'][0]['budgets'] = {'max_llm_calls_per_day': 1000}
    elif fault == 'pm':
        value['slots'] = ['am', 'pm']
    elif fault == 'wrong_schema':
        value['schema'] = 'authorization-extension-v1'
    elif fault == 'wrong_from':
        value['extends'][0]['not_after_from'] = '2026-11-04T00:00:00Z'
    elif fault == 'shorten':
        value['extends'][0]['not_after_to'] = OLD
    elif fault == 'missing_reference':
        value['extends'][0]['file'] = str(tmp_path / 'absent.json')
    elif fault == 'future_decision':
        value['decided_at'] = NEW
    target = write_extension(path, value)
    if fault == 'missing':
        target.unlink()
    elif fault == 'mutated_grant':
        body = json.loads(path.read_text())
        body['budgets']['max_llm_calls_per_day'] = 5
        path.write_text(json.dumps(body))
    elif fault == 'duplicate_key':
        target.write_text(target.read_text().replace('"slots":', '"slots": ["pm"], "slots":'))
    elif fault == 'self_reference':
        value['extends'][0]['file'] = str(target)
        target.write_text(json.dumps(value))
    loaded = Authorization.load(path)
    loaded.check(instant('2026-11-04T12:00:00Z'))
    with pytest.raises(TraderError, match='AUTHORIZATION_EXPIRED_OR_NOT_STARTED'):
        loaded.check(AFTER)


def test_extension_is_rechecked_and_expires_at_exact_boundary(grant, tmp_path):
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps({**grant.payload, 'not_after': OLD}))
    record = write_extension(path, extension(path))
    loaded = Authorization.load(path)
    loaded.check(instant(OLD))
    metadata = loaded.check(AFTER)
    assert metadata['authorization_extension_sha256'] == sha256(record.read_bytes()).hexdigest()
    with pytest.raises(TraderError):
        loaded.check(instant(NEW))
    record.unlink()
    with pytest.raises(TraderError):
        loaded.check(AFTER)



def test_future_label_drill_crosses_original_expiry(grant, tmp_path, capsys):
    from scripts.trading_lab.trader_agent.cli import main
    from scripts.trading_lab.trader_agent.drill_summary import summarize
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps({**grant.payload, 'not_after': OLD}))
    write_extension(path, extension(path))
    runtime = tmp_path / 'runtime'
    common = ['--authorization', str(path), '--runtime', str(runtime), '--dry-run']
    assert main(['run', *common, '--defer-synthetic-labels', '--synthetic-time', '2026-10-27T12:00:00Z']) == 0
    capsys.readouterr()
    assert main(['label', *common, '--synthetic-time', '2026-11-06T21:30:00Z']) == 0
    result = json.loads(capsys.readouterr().out)
    assert summarize(result, 'matured')['verified']
    assert result['authorization_hash'] == Authorization.load(path).identity
    assert result['authorization_extension_sha256']


def test_paper_extension_keeps_parent_and_paper_bindings(grant, tmp_path):
    from scripts.trading_lab.trader_agent.alpaca_paper import PaperAuthorization, LIMITS, PAPER_URL
    parent = tmp_path / 'grant.json'
    parent.write_text(json.dumps({**grant.payload, 'not_after': OLD}))
    path = tmp_path / 'paper.json'
    path.write_text(json.dumps({'authorization': 'trader-alpaca-paper-v1', 'extends': 'trader-agent-v1',
        'mode': 'ALPACA_PAPER_ONLY', 'endpoint': PAPER_URL,
        'granted_at': '2026-01-01T00:00:00Z', 'not_after': OLD,
        'accounts': {name: {'orders': name != 'momentum', 'account_suffix': str(1000+i),
                           'credentials': str(tmp_path / (name + '.env'))}
                     for i, name in enumerate(('ia_actions', 'ia_crypto', 'momentum'))},
        'risk_limits': {**LIMITS, 'max_orders_per_account_per_day': 40,
            'kill_switch': "equity below 90% of the account's peak: flatten that account, halt its executor, raise an alert; resuming needs the operator",
            'trade_only': 'consensus views of a COMPLETE or DEGRADED run that the reviewer did not reject; ABSTAIN, MISSING, FAILED or tainted runs never trade'}}))
    original = PaperAuthorization.load(path, Authorization.load(parent))
    value = extension(parent)
    value['extends'] += extension(path)['extends']
    record = write_extension(parent, value)
    loaded = PaperAuthorization.load(path, Authorization.load(parent))
    receipt = loaded.check(AFTER)
    assert loaded.identity == original.identity
    assert receipt['parent_authorization_hash'] == original.parent.identity
    assert receipt['authorization_extension_sha256'] == receipt['parent_authorization_extension_sha256']
    with pytest.raises(TraderError):
        loaded.check(AFTER, slot='pm')
    value['extends'][1]['endpoint'] = 'changed'
    record.write_text(json.dumps(value))
    with pytest.raises(TraderError):
        loaded.check(AFTER)

    # The paper record alone never renews its independent data capability.
    from scripts.trading_lab.trader_agent.alpaca_data import DataAuthorization, DATA_URL
    value['extends'][1].pop('endpoint')
    record.write_text(json.dumps(value))
    data_path = tmp_path / 'data.json'
    data_path.write_text(json.dumps({'authorization': 'trader-alpaca-data-v1',
        'extends': 'trader-alpaca-paper-v1', 'granted_at': '2026-01-01T00:00:00Z', 'not_after': OLD,
        'data_endpoint': {'origin': DATA_URL, 'methods': ['GET'], 'credentials': 'ia_actions keys'}}))
    data_grant = DataAuthorization.load(data_path, loaded)
    with pytest.raises(TraderError, match='DATA_AUTHORIZATION_EXPIRED_OR_NOT_STARTED'):
        data_grant.check(AFTER)
    value['extends'] += extension(data_path)['extends']
    record.write_text(json.dumps(value))
    assert data_grant.check(AFTER)['authorization_extension_sha256']


def test_base_extension_never_implicitly_extends_crypto_amendment(grant, tmp_path):
    from dataclasses import replace
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps({**grant.payload, 'not_after': OLD}))
    write_extension(path, extension(path))
    loaded = replace(Authorization.load(path), amendment={'granted_at': '2026-01-01T00:00:00Z', 'not_after': OLD})
    with pytest.raises(TraderError, match='CRYPTO_AMENDMENT_EXPIRED_OR_NOT_STARTED'):
        loaded.check(AFTER)

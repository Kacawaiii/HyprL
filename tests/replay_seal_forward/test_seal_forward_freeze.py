import json

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_seal_forward import experiment as ex, forward, schemas, spec


def test_new_freeze_binds_method_models_code_schema_and_forward():
    frozen = json.loads(ex.FREEZE.read_text())
    assert frozen['spec_hash'] == spec.SPEC_HASH
    assert frozen['code_hash'] == spec.code_hash()
    assert frozen['test_days'] == ex.planned('test', 'ab') and len(frozen['test_days']) == 21
    assert frozen['input_seal_required'] is True
    assert spec.spec()['schema_sha256'] == sha256_canonical(schemas.SCHEMAS)
    for f in ex.FAMILIES:
        assert frozen['config_hashes'][f] == {g: spec.config_hash('test', g, f) for g in ('ab', 'cd')}
        assert frozen['forward_config_hashes'][f] == sha256_canonical({'spec_hash': spec.SPEC_HASH,
            'code_hash': spec.code_hash(), 'family': f, 'schema': forward.SCHEMA, 'forward': spec.spec()['forward']})
    previous = json.loads((spec.ROOT / 'docs/artifacts/replay_duel_test_freeze_v1.json').read_text())
    assert frozen['code_hash'] != previous['code_hash'] and frozen['spec_hash'] != previous['spec_hash']

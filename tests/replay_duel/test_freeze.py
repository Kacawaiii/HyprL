"""The active execution must match its own committed freeze; previous freezes stay rejected."""
import json

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_duel import experiment as ex, schemas, spec


def test_active_freeze_binds_exact_code_and_both_models():
    frozen = json.loads(ex.FREEZE.read_text())
    assert frozen['spec_hash'] == spec.SPEC_HASH
    assert frozen['code_hash'] == spec.code_hash()
    assert frozen['input_seal_required'] is True
    assert frozen['test_days'] == ex.planned('test', 'ab')
    assert len(frozen['test_days']) == 28
    for family in ex.FAMILIES:
        assert frozen['config_hashes'][family] == {g: spec.config_hash('test', g, family) for g in ('ab', 'cd')}
        assert frozen['config_payloads'][family] == {g: spec.config_payload('test', g, family) for g in ('ab', 'cd')}
    parent = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_test_freeze_v3.json').read_text())
    assert frozen['parent_freeze_hash'] == sha256_canonical(parent)
    assert frozen['code_hash'] != parent['code_hash']
    assert frozen['spec_hash'] != parent['spec_hash']
    assert spec.spec()['schema_sha256'] == sha256_canonical(schemas.SCHEMAS)

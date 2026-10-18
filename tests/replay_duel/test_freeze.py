"""Revision 5 is preserved byte-for-byte and rejected by the newer execution."""
import json
import pytest
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench.grant import ReplayError
from scripts.trading_lab.replay_duel import experiment as ex, spec


def test_previous_freeze_preserved_and_execution_rejected(tmp_path, monkeypatch):
    frozen = json.loads(ex.FREEZE.read_text())
    assert sha256_canonical(frozen) == '130bde95e0a336741416d28151859191eaadf38f9c8062451ee26352d06f2def'
    assert frozen['spec_hash'] == spec.SPEC_HASH
    assert frozen['code_hash'] != spec.code_hash()
    monkeypatch.setattr(ex.audit, 'evidence', lambda root: {'evidence_hash': frozen['probe_evidence_hash']})
    with pytest.raises(ReplayError, match='TEST_CONFIG_NOT_FROZEN'):
        ex.check_freeze(tmp_path)

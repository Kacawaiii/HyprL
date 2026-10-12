"""Frozen execution variant bindings; changes require a new registered revision."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError

ARTIFACT = Path(__file__).resolve().parents[3] / 'docs/artifacts/trader_alpaca_execution_spec_v4.json'
SPEC_HASH = '92b00b05a4971f64bdea39f723c95ecbd28f4468592a908fd16f79ccccaa5cc6'
V3_SPEC_HASH = '42c025197da23ca3f4686a710f73f3a31c242edac89ac1043cd45c9cb6430252'
V3_DECISION_HASH = '61bb7b1758bcb7e8d1e7cab2b2bda5c61afbf213dd7fd88d79a31d611a33d57b'
PREVIOUS_SPEC_HASH = 'cc18ef9d4cf18ba33e47b4d9c6a17fa3ad82842891613168873b6a12372ff88b'
OPERATOR_DECISION_HASH = '4e332a0b31b8bd28bd26dddc2a59fd2b19ab5b5ea2233c4b1464ff197ef05a1f'


def execution_spec():
    value = json.loads(ARTIFACT.read_text())
    recorded = value.pop('canonical_sha256')
    if recorded != SPEC_HASH or sha256_canonical(value) != SPEC_HASH:
        raise TraderError('PAPER_SPEC_HASH_MISMATCH')
    return value

PREREG_HASH = 'bcee03f1f9136c72bc2663a653b3ecd9e6f4d5454c4f7ce7957fc21c95271c65'

def execution_preregistration():
    value = json.loads(ARTIFACT.with_name('trader_agent_preregistration_v7.json').read_text())
    recorded = value.pop('canonical_sha256')
    if (recorded != PREREG_HASH or sha256_canonical(value) != PREREG_HASH
            or value['execution_spec_hash'] != SPEC_HASH
            or value['operator_decision_hash'] != OPERATOR_DECISION_HASH):
        raise TraderError('PAPER_PREREGISTRATION_HASH_MISMATCH')
    execution_spec()
    return PREREG_HASH

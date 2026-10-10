"""Frozen execution variant bindings; changes require a new registered revision."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError

ARTIFACT = Path(__file__).resolve().parents[3] / 'docs/artifacts/trader_alpaca_execution_spec_v3.json'
SPEC_HASH = '42c025197da23ca3f4686a710f73f3a31c242edac89ac1043cd45c9cb6430252'
PREVIOUS_SPEC_HASH = 'cc18ef9d4cf18ba33e47b4d9c6a17fa3ad82842891613168873b6a12372ff88b'
OPERATOR_DECISION_HASH = '61bb7b1758bcb7e8d1e7cab2b2bda5c61afbf213dd7fd88d79a31d611a33d57b'


def execution_spec():
    value = json.loads(ARTIFACT.read_text())
    recorded = value.pop('canonical_sha256')
    if recorded != SPEC_HASH or sha256_canonical(value) != SPEC_HASH:
        raise TraderError('PAPER_SPEC_HASH_MISMATCH')
    return value

PREREG_HASH = '7b134cdaf0cb1b6d750ada86a9a8cd365745ad4dab1f14053354cbaa863a9adc'

def execution_preregistration():
    value = json.loads(ARTIFACT.with_name('trader_agent_preregistration_v4.json').read_text())
    recorded = value.pop('canonical_sha256')
    if (recorded != PREREG_HASH or sha256_canonical(value) != PREREG_HASH
            or value['execution_spec_hash'] != SPEC_HASH
            or value['operator_decision_hash'] != OPERATOR_DECISION_HASH):
        raise TraderError('PAPER_PREREGISTRATION_HASH_MISMATCH')
    execution_spec()
    return PREREG_HASH

"""Frozen execution variant bindings; changes require a new registered revision."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError

ARTIFACT = Path(__file__).resolve().parents[3] / 'docs/artifacts/trader_alpaca_execution_spec_v2.json'
SPEC_HASH = 'cc18ef9d4cf18ba33e47b4d9c6a17fa3ad82842891613168873b6a12372ff88b'
PREVIOUS_SPEC_HASH = 'c07ae3243d0b130c4ba9972158a2081ac3ff7fa5f280879a003915b98af5b386'
OPERATOR_DECISION_HASH = '0caee3e1145b4cc401cccaface8ca7ecc928c6337c335ab23cc1695a3677e7a9'


def execution_spec():
    value = json.loads(ARTIFACT.read_text())
    recorded = value.pop('canonical_sha256')
    if recorded != SPEC_HASH or sha256_canonical(value) != SPEC_HASH:
        raise TraderError('PAPER_SPEC_HASH_MISMATCH')
    return value

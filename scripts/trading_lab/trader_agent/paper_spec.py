"""Frozen execution variant bindings; changes require a new registered revision."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import TraderError

ARTIFACT = Path(__file__).resolve().parents[3] / 'docs/artifacts/trader_alpaca_execution_spec_v1.json'
SPEC_HASH = 'c07ae3243d0b130c4ba9972158a2081ac3ff7fa5f280879a003915b98af5b386'


def execution_spec():
    value = json.loads(ARTIFACT.read_text())
    recorded = value.pop('canonical_sha256')
    if recorded != SPEC_HASH or sha256_canonical(value) != SPEC_HASH:
        raise TraderError('PAPER_SPEC_HASH_MISMATCH')
    return value

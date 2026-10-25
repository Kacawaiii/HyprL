"""Versioned canonical bindings; no migration from changed research rules."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

def load(name, expected):
    value = json.loads((ROOT / 'docs/artifacts' / name).read_text())
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest() != expected:
        raise ValueError('RESEARCH_RULE_HASH_MISMATCH')
    return value

SPEC_HASH = '5950f6eab4257386015d7810601864131d67898dd975c32df3b65555ef02dedd'
REGISTRY_HASH = '1b7cc3255f79114f6fa5fd27d52b6f3117cb9732a206fcccd95fdd495972c2a0'
SPEC = load('research_loop_spec_v1.json', SPEC_HASH)
REGISTRY = load('research_loop_hypotheses_v1.json', REGISTRY_HASH)

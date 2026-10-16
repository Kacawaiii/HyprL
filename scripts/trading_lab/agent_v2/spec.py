"""New immutable spec family; no v1 binding or store migration."""
from pathlib import Path
import hashlib

from .contracts import Blocked, digest, strict_json

ARTIFACTS = Path(__file__).resolve().parents[3] / 'docs/artifacts'
HERE = Path(__file__).resolve().parent
SPEC_HASH = 'e7a219a8690ee50ab4880e73cbbb0f0d82295a948c934036763cdb2f306ec5ea'
PREREG_HASH = '26581e54b5804a2d9a40e6d309a8b8b2aeb22b77fb80c682852c4f8e826fe73e'


def load_pinned(name, expected):
    value = strict_json((ARTIFACTS / name).read_text())
    if value.pop('canonical_sha256') != expected or digest(value) != expected:
        raise Blocked('SPEC_HASH_MISMATCH')
    return value


def spec():
    value = load_pinned('trader_shadow_v2_spec_v1.json', SPEC_HASH)
    for name, expected in value['prompt_hashes'].items():
        if hashlib.sha256((HERE / 'prompts' / name).read_bytes()).hexdigest() != expected:
            raise Blocked('PROMPT_HASH_MISMATCH')
    from .contracts import NEWS, SUPERVISOR, SNAPSHOT, QUOTES
    shapes = {'news': NEWS, 'supervisor': SUPERVISOR, 'snapshot': SNAPSHOT, 'quotes': QUOTES}
    if value['schema_hashes'] != {k: digest(v) for k, v in shapes.items()}:
        raise Blocked('SCHEMA_HASH_MISMATCH')
    return value


def configuration():
    value = load_pinned('trader_shadow_v2_preregistration_v1.json', PREREG_HASH)
    if value['spec_hash'] != SPEC_HASH:
        raise Blocked('PREREGISTRATION_MISMATCH')
    spec()
    return value

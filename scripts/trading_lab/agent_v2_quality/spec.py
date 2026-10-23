"""New revision and store binding; revision-one artifacts are never modified."""
from pathlib import Path
import hashlib

from .contracts import Blocked, EXTRACTION, JUDGMENT, SNAPSHOT, digest, strict_json

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parents[2] / 'docs/artifacts'
SPEC_HASH = '43a4e9b24fe953e083acd1c4647c89220b8e2196dcfa717924bd0423b786c086'
PREREG_HASH = '42f64b26b67f38365a42d36862b5c1bb0c9fbc4f34211e0c702b55082de05f54'


def pinned(name, expected):
    value = strict_json((ARTIFACTS / name).read_text())
    if value.pop('canonical_sha256') != expected or digest(value) != expected:
        raise Blocked('QUALITY_SPEC_HASH_MISMATCH')
    return value


def spec():
    value = pinned('trader_shadow_v2_spec_v3.json', SPEC_HASH)
    if value['schema_hashes'] != {'extraction': digest(EXTRACTION), 'judgment': digest(JUDGMENT),
                                  'snapshot': digest(SNAPSHOT)}:
        raise Blocked('QUALITY_SCHEMA_HASH_MISMATCH')
    for name, expected in value['prompt_hashes'].items():
        if hashlib.sha256((HERE / 'prompts' / name).read_bytes()).hexdigest() != expected:
            raise Blocked('QUALITY_PROMPT_HASH_MISMATCH')
    return value


def configuration():
    value = pinned('trader_shadow_v2_preregistration_v3.json', PREREG_HASH)
    if value['spec_hash'] != SPEC_HASH:
        raise Blocked('QUALITY_PREREGISTRATION_MISMATCH')
    spec()
    return value


def implementation_hash():
    # Pure risk/calendar/momentum helpers are reused, but their implementation is frozen too.
    folders = [HERE, HERE.parent / 'agent_v2']
    return digest({str(p.relative_to(HERE.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for folder in folders for p in sorted(folder.rglob('*'))
                   if p.suffix in ('.py', '.txt')})

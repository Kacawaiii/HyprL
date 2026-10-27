"""Revision 20 research execution; parent model stores are never implicitly migrated."""
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench.spec import ROOT, file_hash
from scripts.trading_lab.replay_bench.grant import ReplayError

HERE = Path(__file__).resolve().parent
SPEC_FILE = ROOT / 'docs/artifacts/replay_duel_spec_v16.json'
SPEC_HASH = '09f4437a3177e10d9b6df544a2847e6a7ab6285b27e33accd028055b69c4a5c8'
PARENT_SPEC_HASH = 'ea9635cf341a9feaf14a83013094e0a8e25b5c921126ded8a36d577e987f78c6'
STATE_DIR = 'duel-v16'
PROVENANCE = 'APPROXIMATE_POINT_IN_TIME'


def spec():
    value = json.loads(SPEC_FILE.read_text())
    digest = value.pop('canonical_sha256')
    if digest != SPEC_HASH or sha256_canonical(value) != digest:
        raise ReplayError('SPEC_HASH_MISMATCH')
    for name, digest in value['prompt_hashes'].items():
        if file_hash(HERE / 'prompts' / name) != digest:
            raise ReplayError('PROMPT_HASH_MISMATCH')
    from .schemas import SCHEMAS, ANON_ANALYST_SCHEMA
    if sha256_canonical(SCHEMAS) != value['schema_sha256']:
        raise ReplayError('SCHEMA_HASH_MISMATCH')
    if sha256_canonical(ANON_ANALYST_SCHEMA) != value['anonymous_schema_sha256']:
        raise ReplayError('ANONYMOUS_SCHEMA_HASH_MISMATCH')
    parent = json.loads((ROOT / 'docs/artifacts/replay_duel_spec_v15.json').read_text())
    if parent['canonical_sha256'] != value['supersedes']['canonical_sha256'] or parent['canonical_sha256'] != PARENT_SPEC_HASH:
        raise ReplayError('SPEC_LINEAGE_MISMATCH')
    return value


def code_hash():
    from scripts.trading_lab.replay_bench.run import code_hash as inherited_hash
    paths = sorted((ROOT / 'deploy/research').glob('*'))
    return sha256_canonical({'runtime': inherited_hash(), 'research_deployment': {
        str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.is_file()}})


def config_payload(period, group, family):
    if family not in spec()['models']['identities'] or group not in ('ab', 'cd', 'anon'):
        raise ReplayError('MODEL_OR_GROUP_NOT_ALLOWED')
    return {'spec_hash': SPEC_HASH, 'code_hash': code_hash(), 'period': period, 'group': group,
            'family': family, 'model': spec()['models']['identities'][family]}


def config_hash(period, group, family):
    return sha256_canonical(config_payload(period, group, family))

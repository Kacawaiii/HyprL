"""Explicit, hash-bound references to prior answers; no source record is changed."""
import hashlib
import json
from pathlib import Path

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench.grant import ReplayError
from . import spec


def admission():
    rule = spec.spec().get('continuation')
    if not rule:
        raise ReplayError('REPLAY_CONTINUATION_NOT_ADMITTED')
    value = json.loads((spec.ROOT / 'docs/artifacts' / rule['artifact']).read_text())
    if sha256_canonical(value) != rule['sha256']:
        raise ReplayError('CONTINUATION_HASH_MISMATCH')
    return value


def lookup(root, family, prompt, schema, tag):
    source_state = spec.spec()['input_refresh']['source_state_dir']
    if not (Path(root) / source_state / 'transcripts').exists():
        return None
    value = admission()
    source = Path(root) / value['source_state_dir'] / 'transcripts'
    wanted = hashlib.sha256(prompt.encode()).hexdigest()
    matches = []
    for identity, digest in value['transcripts'].items():
        path = source / (identity + '.json')
        if spec.file_hash(path) != digest:
            raise ReplayError('CONTINUATION_SOURCE_CHANGED')
        record = json.loads(path.read_text())
        if (record.get('family') == family and record.get('prompt_sha256') == wanted and
                record['tag'].partition(':')[2] == tag.partition(':')[2]):
            expected = sha256_canonical([value['source_spec_hash'], value['source_code_hash'], family,
                                         prompt, schema, record['tag']])
            if expected != identity or record['spec_hash'] != value['source_spec_hash']:
                raise ReplayError('CONTINUATION_CALL_BINDING_MISMATCH')
            matches.append((record, identity, digest))
    if len(matches) > 1:
        raise ReplayError('CONTINUATION_AMBIGUOUS')
    if not matches:
        return None
    record, identity, digest = matches[0]
    if record['state'] == 'INTENT':
        if (identity != value['zero_byte_retry_identity'] or record.get('raw') or record.get('err') or
                not value['source_process_ended']):
            raise ReplayError('CALL_OUTCOME_UNKNOWN', identity)
    elif record['state'] != 'RECEIVED':
        raise ReplayError('CONTINUATION_INVALID_STATE')
    return record, {'identity': identity, 'sha256': digest, 'spec_hash': value['source_spec_hash'],
                    'code_hash': value['source_code_hash'], 'admission_sha256': sha256_canonical(value)}


def verify_sources(root):
    value = admission()
    for identity, digest in value['transcripts'].items():
        file = Path(root) / value['source_state_dir'] / 'transcripts' / (identity + '.json')
        if not file.is_file() or spec.file_hash(file) != digest:
            raise ReplayError('CONTINUATION_SOURCE_CHANGED')

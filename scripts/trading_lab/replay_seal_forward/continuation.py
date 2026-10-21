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
    if not any(Path(root).glob('duel-v*/transcripts')):
        return None
    value = admission()
    wanted = hashlib.sha256(prompt.encode()).hexdigest()
    matches = []
    for table in value.get('sources', [value]):
        source = Path(root) / table['source_state_dir'] / 'transcripts'
        for identity, digest in table['transcripts'].items():
            path = source / (identity + '.json')
            if spec.file_hash(path) != digest:
                raise ReplayError('CONTINUATION_SOURCE_CHANGED')
            record = json.loads(path.read_text())
            if (record.get('family') == family and record.get('prompt_sha256') == wanted and
                    record['tag'].partition(':')[2] == tag.partition(':')[2]):
                expected = sha256_canonical([table['source_spec_hash'], table['source_code_hash'], family,
                                             prompt, schema, record['tag']])
                if expected != identity or record['spec_hash'] != table['source_spec_hash']:
                    raise ReplayError('CONTINUATION_CALL_BINDING_MISMATCH')
                matches.append((record, identity, digest, table))
    if len(matches) > 1:
        raise ReplayError('CONTINUATION_AMBIGUOUS')
    if not matches:
        return None
    record, identity, digest, table = matches[0]
    if record['state'] == 'INTENT':
        if (identity != value.get('zero_byte_retry_identity') or record.get('raw') or record.get('err') or
                not value['source_process_ended']):
            raise ReplayError('CALL_OUTCOME_UNKNOWN', identity)
    elif record['state'] != 'RECEIVED':
        raise ReplayError('CONTINUATION_INVALID_STATE')
    return record, {'identity': identity, 'sha256': digest, 'spec_hash': table['source_spec_hash'],
                    'code_hash': table['source_code_hash'], 'admission_sha256': sha256_canonical(value)}


def verify_sources(root):
    value = admission()
    for table in (value.get('sources', [value]) + value.get('preserved_intents', []) +
                  value.get('preserved_nonanswers', [])):
        for identity, digest in table['transcripts'].items():
            file = Path(root) / table['source_state_dir'] / 'transcripts' / (identity + '.json')
            if not file.is_file() or spec.file_hash(file) != digest:
                raise ReplayError('CONTINUATION_SOURCE_CHANGED')


def response_record(root, call):
    """Resolve a scoring response through its admitted original protocol, never a guessed path."""
    identity = call['transcript']
    if len(identity) != 64 or any(c not in '0123456789abcdef' for c in identity):
        raise ReplayError('CONTINUATION_CALL_BINDING_MISMATCH')
    original = call.get('original_protocol')
    if original is None:
        return json.loads((Path(root) / spec.STATE_DIR / 'transcripts' / (identity + '.json')).read_text())
    value = admission()
    matches = []
    for table in value.get('sources', [value]):
        digest = table['transcripts'].get(identity)
        expected = {'identity': identity, 'sha256': digest, 'spec_hash': table['source_spec_hash'],
                    'code_hash': table['source_code_hash'], 'admission_sha256': sha256_canonical(value)}
        if digest and original == expected:
            matches.append(Path(root) / table['source_state_dir'] / 'transcripts' / (identity + '.json'))
    if len(matches) != 1:
        raise ReplayError('CONTINUATION_CALL_BINDING_MISMATCH')
    file = matches[0]
    if not file.is_file() or spec.file_hash(file) != original['sha256']:
        raise ReplayError('CONTINUATION_SOURCE_CHANGED')
    return json.loads(file.read_text())

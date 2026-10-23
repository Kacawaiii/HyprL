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
    original = {'identity': identity, 'sha256': digest, 'spec_hash': table['source_spec_hash'],
                'code_hash': table['source_code_hash'], 'admission_sha256': sha256_canonical(value)}
    if record['state'] in ('INTENT', 'INTERRUPTED') and identity in table.get('transport_streams', {}):
        from .models import retryable, unknown_outcome
        path = Path(root) / table['source_state_dir'] / 'transcripts' / (identity + '.json')
        verify_streams(path, table['transport_streams'][identity])
        error = unknown_outcome(record, path, identity, original_protocol=original)
        if error.evidence['terminal_model_failure'] or not retryable(record, path):
            raise error
        return record, original
    if record['state'] == 'INTENT':
        if (identity != value.get('zero_byte_retry_identity') or record.get('raw') or record.get('err') or
                not value['source_process_ended']):
            raise ReplayError('CALL_OUTCOME_UNKNOWN', identity)
    elif record['state'] != 'RECEIVED':
        raise ReplayError('CONTINUATION_INVALID_STATE')
    return record, original


def verify_streams(path, streams):
    for suffix, expected in streams.items():
        if suffix not in ('.stdout', '.stderr'):
            raise ReplayError('CONTINUATION_CALL_BINDING_MISMATCH')
        file = path.with_suffix(suffix)
        if (not file.is_file() or file.stat().st_size != expected['bytes'] or
                spec.file_hash(file) != expected['sha256']):
            raise ReplayError('CONTINUATION_SOURCE_CHANGED')


def transport_failure_hash(root, call):
    """Verify the original envelope and partial streams; never decode partial output."""
    original = call.get('original_protocol')
    path = Path(root) / spec.STATE_DIR / 'transcripts' / (call['transcript'] + '.json')
    if original:
        value = admission()
        tables = [t for t in value['sources'] if t['transcripts'].get(call['transcript']) == original['sha256']
                  and t['source_spec_hash'] == original['spec_hash'] and t['source_code_hash'] == original['code_hash']]
        if len(tables) != 1 or original['admission_sha256'] != sha256_canonical(value):
            raise ReplayError('CONTINUATION_CALL_BINDING_MISMATCH')
        path = Path(root) / tables[0]['source_state_dir'] / 'transcripts' / (call['transcript'] + '.json')
    if not path.is_file() or spec.file_hash(path) != call['envelope_sha256']:
        raise ReplayError('CONTINUATION_SOURCE_CHANGED')
    verify_streams(path, call['streams'])
    return sha256_canonical({'envelope': call['envelope_sha256'], 'streams': call['streams']})


def test_start(root, family):
    """An explicitly admitted started test keeps its original marker and publication."""
    rule = admission().get('test_continuation')
    if not rule:
        return None
    source = Path(root) / rule['source_state_dir']
    for relative, digest in rule['files'].items():
        path = source / relative
        if source.resolve() not in path.resolve().parents or not path.is_file() or spec.file_hash(path) != digest:
            raise ReplayError('CONTINUATION_SOURCE_CHANGED')
    value = json.loads((source / ('test_started_' + family + '.json')).read_text())
    freeze = json.loads((spec.ROOT / 'docs/artifacts' / rule['freeze_artifact']).read_text())
    dev = json.loads((source / 'results/dev.json').read_text())
    if (value['spec_hash'] != freeze['spec_hash'] or value['freeze_hash'] != sha256_canonical(freeze) or
            value['dev_result_hash'] != sha256_canonical(dev) or dev['status'] != 'SCORED' or
            dev['spec_hash'] != freeze['spec_hash'] or dev['code_hash'] != freeze['code_hash']):
        raise ReplayError('CONTINUATION_TEST_START_MISMATCH')
    return value


def verify_sources(root):
    value = admission()
    for table in (value.get('sources', [value]) + value.get('preserved_intents', []) +
                  value.get('preserved_nonanswers', [])):
        for identity, digest in table['transcripts'].items():
            file = Path(root) / table['source_state_dir'] / 'transcripts' / (identity + '.json')
            if not file.is_file() or spec.file_hash(file) != digest:
                raise ReplayError('CONTINUATION_SOURCE_CHANGED')
            if identity in table.get('transport_streams', {}):
                verify_streams(file, table['transport_streams'][identity])
    if value.get('test_continuation'):
        for family in ('opus', 'gpt'):
            test_start(root, family)


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

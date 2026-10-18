"""Common plans, committed configurations, sealed offline inputs, once-only paired test."""
from copy import copy
import json
from pathlib import Path
import subprocess

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench.grant import ReplayError, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json, append_event
from scripts.trading_lab.replay_bench import run as parent
from . import audit, run, spec

FREEZE = spec.ROOT / 'docs/artifacts/replay_duel_test_freeze_v2.json'
INPUT_SEAL = spec.ROOT / 'docs/artifacts/replay_duel_input_seal_v2.json'
FAMILIES = ('opus', 'gpt')
GROUPS = ('ab', 'cd', 'anon')


def head():
    return subprocess.check_output(['git', '-C', str(spec.ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def committed():
    if subprocess.check_output(['git', '-C', str(spec.ROOT), 'status', '--porcelain'], text=True).strip():
        raise ReplayError('CODE_NOT_COMMITTED')
    for file in (spec.SPEC_FILE, FREEZE):
        if subprocess.run(['git', '-C', str(spec.ROOT), 'cat-file', '-e', 'HEAD:' + str(file.relative_to(spec.ROOT))],
                          capture_output=True).returncode:
            raise ReplayError('DECLARATION_NOT_COMMITTED')


def planned(period, group):
    if period not in ('dev', 'test') or group not in GROUPS or period == 'test' and group == 'anon':
        raise ReplayError('PERIOD_OR_GROUP_NOT_ALLOWED')
    days = parent.sessions(*spec.spec()['periods'][period])
    return days[::4] if group == 'anon' else days


def input_proof(root):
    from .inputs import declaration
    declaration(root)
    root = Path(root)
    file = root / spec.STATE_DIR / 'input-manifest.json'
    if not file.is_file():
        raise ReplayError('PER_DAY_INPUT_MANIFESTS_MISSING')
    value = json.loads(file.read_text())
    if value.get('schema') != 'replay-input-bundles-v2' or value.get('spec_hash') != spec.SPEC_HASH:
        raise ReplayError('INPUT_MANIFEST_SPEC_MISMATCH')
    if value.get('method_hash') != sha256_canonical(declaration(root)):
        raise ReplayError('INPUT_METHOD_HASH_MISMATCH')
    wanted = {p + '/' + g + '/' + d for p in ('dev', 'test') for g in ('ab', 'cd') for d in planned(p, g)}
    if set(value.get('snapshots', {})) != wanted:
        raise ReplayError('INPUT_MANIFEST_PLAN_INCOMPLETE')
    for key, digest in value['snapshots'].items():
        path = root / spec.STATE_DIR / 'snapshots' / (key + '.json.gz')
        if root.resolve() not in path.resolve().parents or not path.is_file() or spec.file_hash(path) != digest:
            raise ReplayError('INPUT_BUNDLE_HASH_MISMATCH')
    for cache in ('prediction_cache', 'outcome_cache'):
        for key, digest in value.get(cache, {}).items():
            if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
                raise ReplayError('INPUT_CACHE_KEY_INVALID')
            path = root / 'cache' / key[:2] / (key + '.json.gz')
            if root.resolve() not in path.resolve().parents or not path.is_file() or spec.file_hash(path) != digest:
                raise ReplayError('INPUT_CACHE_HASH_MISMATCH')
    return value


def build_inputs(root):
    import os
    import tempfile
    from .archive import Archive
    from .inputs import build, declaration
    root = Path(root)
    directory = root / spec.STATE_DIR
    if ((directory / 'input-manifest.json').exists() or INPUT_SEAL.exists() or
            list(directory.glob('test_started_*.json'))):
        raise ReplayError('INPUTS_ALREADY_BUILT_OR_TEST_STARTED')
    method_hash = sha256_canonical(declaration(root))
    index = root / 'archive_index.json'
    if not index.is_file():
        raise ReplayError('VENDOR_REQUEST_INDEX_MISSING')
    archive = Archive(root, index)
    # Fail before producing any partial seal if inactive assets are absent.
    archive.require_assets()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='input-build-', dir=directory) as stage:
        stage = Path(stage)
        snapshots = {}
        for period in ('dev', 'test'):
            for day in planned(period, 'ab'):
                bundles = build(archive, parent.decision_time(day), method_hash)
                for group, bundle in bundles.items():
                    key = period + '/' + group + '/' + day
                    file = stage / (key + '.json.gz')
                    parent.write_json(file, bundle)
                    snapshots[key] = spec.file_hash(file)
        value = {'schema': 'replay-input-bundles-v2', 'spec_hash': spec.SPEC_HASH,
                 'method_hash': method_hash, 'provenance': spec.PROVENANCE,
                 'residual_risks': spec.spec()['residual_risks'], 'snapshots': snapshots,
                 'archive_index_hash': archive.proof()['index_hash'],
                 'prediction_cache': archive.objects, 'outcome_cache': archive.objects}
        target = directory / 'snapshots'
        if target.exists():
            raise ReplayError('INPUT_SNAPSHOT_DIRECTORY_ALREADY_EXISTS')
        os.replace(stage, target)
        # The TemporaryDirectory cleanup tolerates its moved staging directory.
        atomic_json(directory / 'input-manifest.json', value)
    return {'status': 'BUILT', 'manifest_hash': sha256_canonical(value), 'snapshot_count': len(snapshots)}


def offline(root, transport, *, outcomes=False):
    proof = input_proof(root)
    other = copy(transport)
    other.offline = True
    other.allowed_cache = dict(proof['prediction_cache'])
    if outcomes:
        if not proof.get('outcome_cache'):
            raise ReplayError('OUTCOME_CACHE_NOT_SEALED')
        for key, digest in proof['outcome_cache'].items():
            if key in other.allowed_cache and other.allowed_cache[key] != digest:
                raise ReplayError('CACHE_BINDING_CONFLICT')
            other.allowed_cache[key] = digest
    return other


def frozen_payload(root):
    evidence = audit.evidence(root)
    from .forward import SCHEMA
    return {'schema': 'replay-duel-test-freeze-v2', 'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(),
            'parent_freeze_hash': sha256_canonical(json.loads((spec.ROOT / 'docs/artifacts/replay_duel_test_freeze_v1.json').read_text())),
            'probe_evidence_hash': evidence['evidence_hash'],
            'config_payloads': {f: {g: spec.config_payload('test', g, f) for g in ('ab', 'cd')} for f in FAMILIES},
            'config_hashes': {f: {g: spec.config_hash('test', g, f) for g in ('ab', 'cd')} for f in FAMILIES},
            'test_days': planned('test', 'ab'), 'input_seal_required': True,
            'base_rate_rule': spec.spec()['statistics']['base_rate'],
            'paired_test_rule': spec.spec()['statistics']['paired_test'],
            'forward_config_hashes': {f: sha256_canonical({'spec_hash': spec.SPEC_HASH,
                'code_hash': spec.code_hash(), 'family': f, 'schema': SCHEMA,
                'forward': spec.spec()['forward']}) for f in FAMILIES}}


def seal_payload(root):
    if not FREEZE.is_file() or json.loads(FREEZE.read_text()) != frozen_payload(root):
        raise ReplayError('TEST_CONFIG_NOT_FROZEN')
    return {'schema': 'replay-duel-input-seal-v2', 'spec_hash': spec.SPEC_HASH,
            'freeze_hash': sha256_canonical(json.loads(FREEZE.read_text())),
            'input_proof_hash': sha256_canonical(input_proof(root)),
            'provenance': spec.PROVENANCE, 'snapshot_count': len(input_proof(root)['snapshots']),
            'residual_risks': spec.spec()['residual_risks']}


def check_freeze(root, *, test=False):
    if not FREEZE.is_file() or json.loads(FREEZE.read_text()) != frozen_payload(root):
        raise ReplayError('TEST_CONFIG_NOT_FROZEN')
    committed()
    if not INPUT_SEAL.is_file() or json.loads(INPUT_SEAL.read_text()) != seal_payload(root):
        raise ReplayError('INPUTS_NOT_SEALED')
    if subprocess.run(['git', '-C', str(spec.ROOT), 'cat-file', '-e', 'HEAD:' + str(INPUT_SEAL.relative_to(spec.ROOT))], capture_output=True).returncode:
        raise ReplayError('INPUTS_NOT_COMMITTED')
    if test:
        if not INPUT_SEAL.is_file() or json.loads(INPUT_SEAL.read_text()) != seal_payload(root):
            raise ReplayError('TEST_INPUTS_NOT_SEALED')
        if subprocess.run(['git', '-C', str(spec.ROOT), 'cat-file', '-e', 'HEAD:' + str(INPUT_SEAL.relative_to(spec.ROOT))],
                          capture_output=True).returncode:
            raise ReplayError('TEST_INPUTS_NOT_COMMITTED')
        # No second protocol after a v3/v4 test has begun. Preserve every prior record.
        if any((Path(root) / version / 'test_started.json').exists() for version in ('v3', 'v4', 'duel-v1')):
            raise ReplayError('SUPERSEDED_TEST_ALREADY_STARTED')
        if list((Path(root) / 'duel-v1').glob('test_started_*.json')):
            raise ReplayError('SUPERSEDED_TEST_ALREADY_STARTED')
        legacy = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_trials_v1.json').read_text())
        if any(t['period'] == 'test' for t in legacy['trials']):
            raise ReplayError('LEGACY_TEST_ALREADY_STARTED')


def meta(root, period, group, family):
    return dict(spec.config_payload(period, group, family), config_hash=spec.config_hash(period, group, family),
                input_proof_hash=sha256_canonical(input_proof(root)))


def record_path(root, period, group, family, day):
    return Path(root) / spec.STATE_DIR / 'store' / family / period / group / (day + '.json.gz')


def complete(root, period, family):
    return all(record_path(root, period, g, family, d).is_file()
               for g in (GROUPS if period == 'dev' else ('ab', 'cd')) for d in planned(period, g))


def result_path(root, period):
    return Path(root) / spec.STATE_DIR / 'results' / (period + '.json')


def ready_test(root):
    check_freeze(root, test=True)
    path = result_path(root, 'dev')
    value = json.loads(path.read_text()) if path.is_file() else {}
    if (not all(complete(root, 'dev', f) for f in FAMILIES) or value.get('status') != 'SCORED' or
        value.get('code_hash') != spec.code_hash() or value.get('spec_hash') != spec.SPEC_HASH or
        value.get('input_proof_hash') != sha256_canonical(input_proof(root)) or
        set(value.get('models', {})) != set(FAMILIES)):
        raise ReplayError('BOTH_DEV_PERIODS_MUST_BE_SCORED')
    return sha256_canonical(value)


def claim_test(root, family):
    dev_hash = ready_test(root)
    path = Path(root) / spec.STATE_DIR / ('test_started_' + family + '.json')
    binding = {'freeze_hash': sha256_canonical(json.loads(FREEZE.read_text())), 'spec_hash': spec.SPEC_HASH,
               'family': family, 'dev_result_hash': dev_hash, 'input_seal_hash': sha256_canonical(seal_payload(root))}
    if path.exists():
        value = json.loads(path.read_text())
        if any(value.get(k) != v for k, v in binding.items()):
            raise ReplayError('TEST_ALREADY_STARTED_WITH_OTHER_CONFIG')
    else:
        atomic_json(path, dict(binding, started_at=iso(now()), commit=head()))


def status(root, budget, clock=now):
    evidence = audit.evidence(root)
    reasons = []
    try:
        input_proof(root)
    except ReplayError as error:
        reasons.append(error.code)
    plans = {f: {p: {g: {'days': len(planned(p, g)),
                        'completed_days': sum(record_path(root, p, g, f, d).is_file() for d in planned(p, g))}
                        for g in (GROUPS if p == 'dev' else ('ab', 'cd'))} for p in ('dev', 'test')} for f in FAMILIES}
    return {'status': 'BLOCKED' if reasons else 'READY', 'reasons': reasons, 'plans': plans,
            'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(), 'commit': head(),
            'memory_floor': evidence['floor'], 'probe_evidence_hash': evidence['evidence_hash'],
            'calls_used_today': {f: budget.used('model:' + f, clock()) for f in FAMILIES},
            'subscription_stops': {f: budget.model_stopped(f, clock()) for f in FAMILIES}}


def run_one(root, period, group, day, family, transport, client):
    check_freeze(root, test=period == 'test')
    transport = offline(root, transport)
    if period == 'test':
        claim_test(root, family)
    expected = meta(root, period, group, family)
    path = record_path(root, period, group, family, day)
    if path.exists():
        return run.checked(path, expected)
    if day not in planned(period, group):
        raise ReplayError('DAY_NOT_PLANNED')
    value = run.decide(root, period, group, day, family, transport, client, expected, path)
    append_event(Path(root) / spec.STATE_DIR / 'trials.jsonl', dict(event='DAY_COMPLETED', period=period,
        group=group, day=day, family=family, config_hash=expected['config_hash'], decision_hash=value['decision_hash']))
    return value


def night(root, budget, transport, client, *, expected_commit):
    from scripts.trading_lab.replay_bench.models import night_remaining
    from .score import score
    night_remaining(now())
    if head() != expected_commit:
        raise ReplayError('INSTALLED_SHA_MISMATCH')
    current = status(root, budget)
    if current['status'] == 'BLOCKED':
        return current
    unavailable = {}
    for period in ('dev', 'test'):
        for group in (GROUPS if period == 'dev' else ('ab', 'cd')):
            for day in planned(period, group):
                # Alternating first model; each pair sees the exact same sealed context.
                order = FAMILIES if planned(period, group).index(day) % 2 == 0 else FAMILIES[::-1]
                for family in order:
                    if family in unavailable:
                        continue
                    try:
                        run_one(root, period, group, day, family, transport, client)
                    except ReplayError as error:
                        if error.code not in ('MODEL_USAGE_LIMIT', 'BUDGET_EXHAUSTED'):
                            raise
                        unavailable[family] = error.code
        if not all(complete(root, period, f) for f in FAMILIES):
            return dict(status(root, budget), status='BLOCKED', subscription_stops=unavailable,
                        reasons=['PERIOD_DECISIONS_INCOMPLETE'])
        score(root, period, transport)
    return {'status': 'DONE', 'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(), 'commit': head()}

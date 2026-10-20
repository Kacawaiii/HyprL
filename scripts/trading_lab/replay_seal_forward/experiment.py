"""Common plans, committed configurations, sealed offline inputs, once-only paired test."""
from copy import copy
from datetime import timedelta
import json
from pathlib import Path
import subprocess

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench.grant import ReplayError, iso, now
from scripts.trading_lab.replay_bench.storage import atomic_json, append_event
from scripts.trading_lab.replay_bench import run as parent
from . import audit, run, spec

FREEZE = spec.ROOT / 'docs/artifacts/replay_duel_test_freeze_v8.json'
INPUT_SEAL = spec.ROOT / 'docs/artifacts/replay_duel_input_seal_v8.json'
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


def test_cut():
    """Per-model last test day from the committed freeze (probe4 rule), else the period end."""
    frozen = json.loads(FREEZE.read_text()) if FREEZE.is_file() else {}
    return frozen.get('test_end', {f: spec.spec()['periods']['test'][1] for f in FAMILIES})


def planned_for(period, group, family):
    days = planned(period, group)
    if period == 'test':
        end = test_cut()[family]
        days = [d for d in days if d <= end]
    return days


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
    from .archive import Archive, UnavailableArchive
    from .inputs import build, declaration, paired_label_exclusions
    root = Path(root)
    directory = root / spec.STATE_DIR
    if ((directory / 'input-manifest.json').exists() or INPUT_SEAL.exists() or
            list(directory.glob('test_started_*.json'))):
        raise ReplayError('INPUTS_ALREADY_BUILT_OR_TEST_STARTED')
    method_hash = sha256_canonical(declaration(root))
    excluded_labels = paired_label_exclusions(root)
    index = root / 'archive_index.json'
    archive = Archive(root, index) if index.is_file() else UnavailableArchive('VENDOR_REQUEST_INDEX_MISSING')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    with tempfile.TemporaryDirectory(prefix='input-build-', dir=directory) as stage:
        stage = Path(stage)
        snapshots = {}
        from collections import Counter, defaultdict
        counts, identities = Counter(), defaultdict(set)
        excluded_days = []
        for period in ('dev', 'test'):
            for day in planned(period, 'ab'):
                bundles = build(archive, parent.decision_time(day), method_hash)
                quality = bundles['ab']['manifest']['data_quality']
                counts.update(quality['counts_by_reason'])
                if quality['excluded_day']:
                    excluded_days.append(day)
                for row in quality['records']:
                    identities[row['reason']].add(row['scope'] + ':' + row['identity'])
                atomic_json(directory / 'build-progress.json', {'period': period, 'day': day,
                    'days_built': len(snapshots) // 2 + 1, 'data_quality_occurrences': dict(counts)})
                for group, bundle in bundles.items():
                    key = period + '/' + group + '/' + day
                    file = stage / (key + '.json.gz')
                    parent.write_json(file, bundle)
                    snapshots[key] = spec.file_hash(file)
        value = {'schema': 'replay-input-bundles-v2', 'spec_hash': spec.SPEC_HASH,
                 'method_hash': method_hash, 'provenance': spec.PROVENANCE,
                 'residual_risks': spec.spec()['residual_risks'], 'snapshots': snapshots,
                 'archive_index_hash': archive.proof().get('index_hash'), 'archive_state': archive.proof().get('state', 'AVAILABLE'),
                 'prediction_cache': archive.objects, 'outcome_cache': archive.objects}
        value['paired_missing_labels'] = excluded_labels
        value['data_quality'] = {'occurrences_by_reason': dict(counts),
            'unique_by_reason': {k: {'count': len(v), 'identities_sha256': sha256_canonical(sorted(v))}
                                 for k, v in sorted(identities.items())},
            'excluded_days': len(excluded_days), 'excluded_days_sha256': sha256_canonical(excluded_days)}
        target = directory / 'snapshots'
        if target.exists():
            raise ReplayError('INPUT_SNAPSHOT_DIRECTORY_ALREADY_EXISTS')
        os.replace(stage, target)
        # The TemporaryDirectory cleanup tolerates its moved staging directory.
        atomic_json(directory / 'input-manifest.json', value)
    return {'status': 'BUILT', 'manifest_hash': sha256_canonical(value), 'snapshot_count': len(snapshots),
            'data_quality': value['data_quality']}


def offline(root, transport, *, outcomes=False):
    proof = input_proof(root)
    other = copy(transport)
    other.offline = True
    other.allowed_cache = dict(proof['prediction_cache'])
    other.excluded_labels = {(r['group'], r['day'], r['asset'], r['h']) for r in proof.get('paired_missing_labels', [])}
    if outcomes:
        if 'outcome_cache' not in proof:
            raise ReplayError('OUTCOME_CACHE_NOT_SEALED')
        for key, digest in proof['outcome_cache'].items():
            if key in other.allowed_cache and other.allowed_cache[key] != digest:
                raise ReplayError('CACHE_BINDING_CONFLICT')
            other.allowed_cache[key] = digest
    return other


def frozen_payload(root):
    evidence = audit.evidence(root)
    from .forward import SCHEMA, forward_config_hash
    from .stats import PRIMARY
    rate = json.loads((Path(root) / spec.STATE_DIR / 'dev-label-base-rate.json').read_text())
    if rate.get('spec_hash') != spec.SPEC_HASH or rate.get('n', -1) < 0 or rate.get('panel') != PRIMARY['panel']:
        raise ReplayError('DEV_BASE_RATE_NOT_FROZEN')
    return {'schema': 'replay-duel-test-freeze-v8', 'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(),
            'parent_freeze_hash': sha256_canonical(json.loads((spec.ROOT / 'docs/artifacts/replay_duel_test_freeze_v2.json').read_text())),
            'probe_evidence_hash': evidence['evidence_hash'],
            'test_end': evidence['test_end'], 'evidence_class': {'replay': 'EXPLORATORY', 'forward': 'LIVE_FORWARD'},
            'primary_endpoint': PRIMARY, 'frozen_dev_base_rate': rate,
            'config_payloads': {f: {g: spec.config_payload('test', g, f) for g in ('ab', 'cd')} for f in FAMILIES},
            'config_hashes': {f: {g: spec.config_hash('test', g, f) for g in ('ab', 'cd')} for f in FAMILIES},
            'test_days': planned('test', 'ab'), 'input_seal_required': True,
            'base_rate_rule': spec.spec()['statistics']['base_rate'],
            'paired_test_rule': spec.spec()['statistics']['paired_test'],
            'forward_config_hashes': {f: forward_config_hash(f) for f in FAMILIES},
            'forward_first_session': spec.spec()['forward']['first_session']}


def seal_payload(root):
    if not FREEZE.is_file() or json.loads(FREEZE.read_text()) != frozen_payload(root):
        raise ReplayError('TEST_CONFIG_NOT_FROZEN')
    return {'schema': 'replay-duel-input-seal-v8', 'spec_hash': spec.SPEC_HASH,
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
        if any(list((Path(root) / v).glob('test_started_*.json')) for v in ('duel-v1', 'duel-v2', 'duel-v3', 'duel-v4', 'duel-v5', 'duel-v6', 'duel-v7')):
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
               for g in (GROUPS if period == 'dev' else ('ab', 'cd')) for d in planned_for(period, g, family))


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


def claim_test(root, family, ledger=None):
    dev_hash = ready_test(root)
    path = Path(root) / spec.STATE_DIR / ('test_started_' + family + '.json')
    binding = {'freeze_hash': sha256_canonical(json.loads(FREEZE.read_text())), 'spec_hash': spec.SPEC_HASH,
               'family': family, 'dev_result_hash': dev_hash, 'input_seal_hash': sha256_canonical(seal_payload(root))}
    if path.exists():
        value = json.loads(path.read_text())
        if any(value.get(k) != v for k, v in binding.items()):
            raise ReplayError('TEST_ALREADY_STARTED_WITH_OTHER_CONFIG')
    else:
        value = dict(binding, started_at=iso(now()), commit=head())
        atomic_json(path, value)
    # The test-start marker is public (hash only) before any test call.
    from . import commitment
    if commitment.published(root, 'replay-test-start', family) is None:
        if ledger is None:
            raise ReplayError('TEST_START_NOT_PUBLISHED')
        ledger.publish('replay-test-start', family, {'test_start_marker': sha256_canonical(value)})
    commitment.require(root, 'replay-test-start', family, {'test_start_marker': sha256_canonical(value)}, now())


def failure_summary(root):
    file = Path(root) / spec.STATE_DIR / 'failures.jsonl'
    rows = [json.loads(l) for l in file.read_text().splitlines()] if file.is_file() else []
    return {f: {'failed_attempts': sum(r['family'] == f for r in rows),
                'failed_days': sorted({r['day'] for r in rows if r['family'] == f}),
                'errors': [r for r in rows if r['family'] == f]} for f in FAMILIES}


def call_order(day):
    days = sorted(set(planned('dev', 'ab') + planned('test', 'ab')))
    return FAMILIES if days.index(day) % 2 == 0 else FAMILIES[::-1]


def status(root, budget, clock=now):
    evidence = audit.evidence(root)
    reasons = []
    try:
        input_proof(root)
    except ReplayError as error:
        reasons.append(error.code)
    plans = {f: {p: {g: {'days': len(planned_for(p, g, f)),
                        'completed_days': sum(record_path(root, p, g, f, d).is_file() for d in planned_for(p, g, f))}
                        for g in (GROUPS if p == 'dev' else ('ab', 'cd'))} for p in ('dev', 'test')} for f in FAMILIES}
    return {'status': 'BLOCKED' if reasons else 'READY', 'reasons': reasons, 'plans': plans,
            'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(), 'commit': head(),
            'memory_floor': evidence['floor'], 'probe_evidence_hash': evidence['evidence_hash'],
            'calls_used_today': {f: budget.used('model:' + f, clock()) for f in FAMILIES},
            'subscription_stops': {f: budget.model_stopped(f, clock()) for f in FAMILIES},
            'failures': failure_summary(root)}


def run_one(root, period, group, day, family, transport, client, ledger=None):
    check_freeze(root, test=period == 'test')
    transport = offline(root, transport)
    if period == 'test':
        claim_test(root, family, ledger)
    expected = meta(root, period, group, family)
    path = record_path(root, period, group, family, day)
    if path.exists():
        return run.checked(path, expected)
    if day not in planned_for(period, group, family):
        raise ReplayError('DAY_NOT_PLANNED')
    try:
        value = run.decide(root, period, group, day, family, transport, client, expected, path)
    except ReplayError as error:
        append_event(Path(root) / spec.STATE_DIR / 'failures.jsonl',
                     {'family': family, 'period': period, 'group': group, 'day': day, 'error': error.code, 'at': iso(now())})
        raise
    append_event(Path(root) / spec.STATE_DIR / 'trials.jsonl', dict(event='DAY_COMPLETED', period=period,
        group=group, day=day, family=family, config_hash=expected['config_hash'], decision_hash=value['decision_hash']))
    return value


def publish_pending(root, ledger, night):
    """Publish every stored response hash not yet public (hash-only ledger + Discord)."""
    from . import commitment
    from .score import record_key
    public = commitment.published_hashes(root, 'replay-night', now() + timedelta(days=1))
    pending = {}
    for family in FAMILIES:
        for period in ('dev', 'test'):
            for group in (GROUPS if period == 'dev' else ('ab', 'cd')):
                for day in planned_for(period, group, family):
                    path = record_path(root, period, group, family, day)
                    if path.is_file():
                        key = record_key(family, period, group, day)
                        value = parent.read_json(path)['decision_hash']
                        if public.get(key) != value:
                            pending[key] = value
    # Include partial, invalid and usage-limited responses too, even when the
    # day never produced a complete decision record.
    for file in sorted((Path(root) / spec.STATE_DIR / 'transcripts').glob('*.json')):
        value = json.loads(file.read_text())
        if value.get('state') != 'INTENT' and 'raw' in value:
            key = 'response:' + file.stem
            digest = sha256_canonical({k: value[k] for k in ('raw', 'err', 'returncode')})
            if public.get(key) != digest:
                pending[key] = digest
    if not pending:
        return None
    digest = sha256_canonical(sorted(pending.items()))
    return ledger.publish('replay-night', night + '-' + digest[:12], pending,
                          summary=f'{len(pending)} réponses replay, empreinte {digest}')


def night(root, budget, transport, client, *, expected_commit, ledger=None):
    from scripts.trading_lab.replay_bench.models import night_remaining
    from .score import score
    night_remaining(now())
    if head() != expected_commit:
        raise ReplayError('INSTALLED_SHA_MISMATCH')
    current = status(root, budget)
    if current['status'] == 'BLOCKED':
        return current
    if ledger is None:
        from . import commitment
        ledger = commitment.from_environment(root)
    tag = iso(now())[:10]
    unavailable, failures = {}, {}
    try:
        for period in ('dev', 'test'):
            for group in (GROUPS if period == 'dev' else ('ab', 'cd')):
                for index, day in enumerate(planned(period, group)):
                    # Alternating first model; each pair sees the exact same sealed context.
                    order = call_order(day)
                    for family in order:
                        if family in unavailable or day not in planned_for(period, group, family):
                            continue
                        try:
                            run_one(root, period, group, day, family, transport, client, ledger)
                        except ReplayError as error:
                            if error.code not in ('MODEL_USAGE_LIMIT', 'BUDGET_EXHAUSTED', 'API_CREDIT_EXHAUSTED'):
                                failures.setdefault(family, []).append({'day': day, 'group': group, 'error': error.code})
                                raise
                            unavailable[family] = error.code
            if not all(complete(root, period, f) for f in FAMILIES):
                publish_pending(root, ledger, tag)
                return dict(status(root, budget), status='BLOCKED', subscription_stops=unavailable,
                            failures=failure_summary(root), reasons=['PERIOD_DECISIONS_INCOMPLETE'])
            publish_pending(root, ledger, tag)
            score(root, period, transport)
    finally:
        try:
            publish_pending(root, ledger, tag)
        except ReplayError:
            pass
    return {'status': 'DONE', 'spec_hash': spec.SPEC_HASH, 'code_hash': spec.code_hash(), 'commit': head(),
            'failures': failure_summary(root)}

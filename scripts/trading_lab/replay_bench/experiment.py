"""Frozen plans, one-shot test ownership, private trial ledger and sequential night batches."""
from datetime import timedelta
from copy import copy
import json
from pathlib import Path
import subprocess

from scripts.trading_lab.sources.canonical import sha256_canonical
from . import audit, marketdata as md, models, run, spec
from .grant import ReplayError, instant, iso, now
from .storage import append_event, atomic_json, load_lines

FREEZE = spec.ROOT / 'docs/artifacts/replay_bench_test_freeze_v2.json'
DECLARATION = spec.ROOT / 'docs/artifacts/replay_bench_trials_v2.json'
GROUP_MODELS = {'ab': ('sonnet',), 'cd': ('sonnet',), 'anon': ('sonnet',), 'opus': ('sonnet', 'opus')}


def head():
    return subprocess.check_output(['git', '-C', str(spec.ROOT), 'rev-parse', 'HEAD'], text=True).strip()


def committed():
    dirty = subprocess.check_output(['git', '-C', str(spec.ROOT), 'status', '--porcelain'], text=True)
    if dirty.strip():
        raise ReplayError('CODE_NOT_COMMITTED')
    for path in (spec.SPEC_FILE, FREEZE, DECLARATION):
        tracked = subprocess.run(['git', '-C', str(spec.ROOT), 'cat-file', '-e',
                                  'HEAD:' + str(path.relative_to(spec.ROOT))], capture_output=True)
        if tracked.returncode:
            raise ReplayError('DECLARATION_NOT_COMMITTED')


def input_proof(root):
    path = Path(root) / 'input_proof.json'
    if not path.exists():
        raise ReplayError('POINT_IN_TIME_INPUT_PROOF_MISSING')
    value = json.loads(path.read_text())
    if (value.get('schema') != 'replay-input-proof-v1' or value.get('spec_hash') != spec.SPEC_HASH or
            value.get('unrevised_prices_proven') is not True or
            value.get('historical_universe_and_identity_proven') is not True or not value.get('evidence') or
            not value.get('prediction_cache')):
        raise ReplayError('POINT_IN_TIME_INPUT_UNPROVEN')
    for item in value['evidence']:
        file = (Path(root) / item['file']).resolve()
        if Path(root).resolve() not in file.parents or spec.file_hash(file) != item['sha256']:
            raise ReplayError('INPUT_EVIDENCE_HASH_MISMATCH')
    for key, digest in value['prediction_cache'].items():
        if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
            raise ReplayError('INPUT_CACHE_KEY_INVALID')
        file = Path(root) / 'cache' / key[:2] / (key + '.json.gz')
        if not file.exists() or spec.file_hash(file) != digest:
            raise ReplayError('INPUT_CACHE_HASH_MISMATCH')
    return sha256_canonical(value)


def planned(root, period, group, evidence=None):
    value = spec.spec()
    first, last = value['periods'][period]
    ev = evidence or audit.evidence(root)
    floor = max(ev['floor'][f] for f in GROUP_MODELS[group])
    start = max(first, floor)
    all_days = run.sessions(first, last)
    days = [d for d in all_days if d >= start]
    if group in ('anon', 'opus'):
        if period != 'dev':
            raise ReplayError('AUDIT_IS_DEV_ONLY')
        if group == 'anon':
            selected = all_days[::4]
        else:
            used = set(value['opus_comparison']['legacy_replay_dates'])
            for day in all_days[::3]:
                if len(used) < 15:
                    used.add(day)
            selected = sorted(used.intersection(all_days))
        days = [d for d in selected if d >= start]
    return days, start


def frozen_payload(root):
    ev = audit.evidence(root)
    return {'schema': 'replay-test-freeze-v2', 'spec_hash': spec.SPEC_HASH, 'code_hash': run.code_hash(),
            'declaration_hash': sha256_canonical(json.loads(DECLARATION.read_text())),
            'probe_evidence_hash': ev['evidence_hash'],
            'input_proof_hash': input_proof(root),
            'config_hashes': {g: run.config_hash('test', g) for g in ('ab', 'cd')},
            'test_days': {g: planned(root, 'test', g, ev)[0] for g in ('ab', 'cd')}}


def check_freeze(root):
    frozen = json.loads(FREEZE.read_text())
    expected = frozen_payload(root)
    if any(frozen.get(k) != v for k, v in expected.items()):
        raise ReplayError('TEST_CONFIG_NOT_FROZEN')
    committed()
    if not all(expected['test_days'].values()):
        raise ReplayError('NO_POST_CUTOFF_TEST_DATES')
    # Protect against old test calls too; never silently start a replacement configuration.
    legacy = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_trials_v1.json').read_text())
    if any(t['period'] == 'test' for t in legacy['trials']):
        raise ReplayError('LEGACY_TEST_ALREADY_STARTED')
    return frozen


def trial_meta(period, group):
    config = run.config_hash(period, group)
    return {'trial_id': f'{period}-{group}-{config[:12]}', 'config_hash': config, 'spec_hash': spec.SPEC_HASH,
            'period': period, 'group': group}


def trial_out(root, meta):
    return Path(root) / 'v3' / 'store' / meta['period'] / meta['config_hash']


def trial_complete(root, period, group):
    days, _ = planned(root, period, group)
    if not days:
        return False
    meta = trial_meta(period, group)
    out = trial_out(root, meta)
    return all((out / group / f'{day}.json.gz').exists() for day in days)


def ready_test(root):
    check_freeze(root)
    for group in GROUP_MODELS:
        if not trial_complete(root, 'dev', group):
            raise ReplayError('DEV_NOT_COMPLETE', group)
    scored = Path(root) / 'v3/results/dev.json'
    if (not scored.exists() or json.loads(scored.read_text()).get('code_hash') != run.code_hash() or
            json.loads(scored.read_text()).get('status') != 'SCORED'):
        raise ReplayError('DEV_NOT_SCORED')


def claim_test(root):
    freeze = check_freeze(root)
    path = Path(root) / 'v3/test_started.json'
    value = {'freeze_hash': sha256_canonical(freeze), 'spec_hash': spec.SPEC_HASH}
    if path.exists():
        old = json.loads(path.read_text())
        if any(old.get(k) != v for k, v in value.items()):
            raise ReplayError('TEST_ALREADY_STARTED_WITH_OTHER_CONFIG')
    else:
        atomic_json(path, dict(value, started_at=iso(now()), commit=head()))
        append_event(Path(root) / 'v3/trials.jsonl', dict(value, event='TEST_STARTED', at=iso(now())))


def status(root, budget, clock=now):
    ev = audit.evidence(root)
    reasons = []
    plans = {}
    for period in ('dev', 'test'):
        plans[period] = {}
        for group in (GROUP_MODELS if period == 'dev' else ('ab', 'cd')):
            days, start = planned(root, period, group, ev)
            plans[period][group] = {'days': len(days), 'effective_start': start}
            if not days:
                reasons.append(f'NO_POST_CUTOFF_DATES:{period}:{group}')
    if budget.stopped(clock()):
        reasons.append('MODEL_USAGE_LIMIT')
    try:
        input_proof(root)
    except ReplayError as error:
        reasons.append(error.code)
    return {'status': 'BLOCKED' if reasons else 'READY', 'reasons': reasons, 'plans': plans,
            'memory_floor': ev['floor'], 'probe_evidence_hash': ev['evidence_hash'],
            'calls_used_today': {f: budget.used('model:' + f, clock()) for f in models.MODELS},
            'spec_hash': spec.SPEC_HASH, 'commit': head(), 'provenance': spec.PROVENANCE}


def run_group(root, period, group, transport, claude):
    days, start = planned(root, period, group)
    if not days:
        raise ReplayError('NO_POST_CUTOFF_DATES', {'period': period, 'group': group, 'start': start})
    committed()
    input_proof(root)
    # Predictions read only the cache sealed by the input proof; no silently refetched vintage.
    transport = copy(transport)
    transport.offline = True
    transport.allowed_cache = json.loads((Path(root) / 'input_proof.json').read_text())['prediction_cache']
    if period == 'test':
        ready_test(root)
        claim_test(root)
    meta = trial_meta(period, group)
    out = trial_out(root, meta)
    ledger = Path(root) / 'v3/trials.jsonl'
    append_event(ledger, dict(meta, event='BATCH_STARTED', days_planned=len(days), commit=head(), at=iso(now())))
    pending = []
    for day in days:
        path = out / group / f'{day}.json.gz'
        if path.exists():
            run.checked_record(path, meta, out)
        else:
            pending.append(day)
    if not pending:
        return {'complete': True, 'group': group}
    if group in ('ab', 'anon', 'opus'):
        # Prediction data end at the period's last decision; labels fetched only by the scorer.
        universe = run.Universe(transport, end=iso(run.decision_time(days[-1])))
        fn = lambda day: run.decide_ab(day, universe, claude, out, meta,
                                       anonymized=group == 'anon', opus_sample=group == 'opus')
    else:
        keywords = spec.spec()['variants']['c_news_small_mid']['negative_keywords']
        fn = lambda day: run.decide_c(day, transport, claude, out, meta, keywords)
    results, stop = run.run_days(pending, fn, 1)
    complete = all((out / group / f'{day}.json.gz').exists() for day in days)
    append_event(ledger, dict(meta, event='BATCH_STOPPED' if stop else 'BATCH_FINISHED', stop=stop,
                              complete=complete, processed=list(results), at=iso(now())))
    if stop:
        raise ReplayError(stop)
    return {'complete': complete, 'group': group}


def night(root, budget, transport, claude, *, expected_commit):
    models.night_remaining(now())
    if head() != expected_commit:
        raise ReplayError('INSTALLED_SHA_MISMATCH')
    current = status(root, budget)
    if current['status'] == 'BLOCKED':
        return current
    committed()
    for period in ('dev', 'test'):
        for group in (GROUP_MODELS if period == 'dev' else ('ab', 'cd')):
            run_group(root, period, group, transport, claude)
        from .score import score
        score(root, period, transport, spec.spec())
    return {'status': 'DONE', 'commit': head(), 'spec_hash': spec.SPEC_HASH}

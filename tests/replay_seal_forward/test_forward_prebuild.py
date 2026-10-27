"""Offline prospective snapshots and dispatch deadlines through the research entry points."""
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench import run as parent
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from scripts.trading_lab.replay_seal_forward import experiment as ex, forward as fw, inputs, quality, spec
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.replay_seal_forward.test_inputs import fixture as archive_fixture
from tests.replay_seal_forward.test_forward import runtime, T  # shared fake subscription transport and real hash ledger


def live_inputs(root, monkeypatch, at):
    import shutil
    _, index = archive_fixture(root, at)
    target = root / 'live-archive' / at.date().isoformat() / 'archive_index.json'
    target.parent.mkdir(parents=True)
    target.write_bytes(index.read_bytes())
    shutil.copytree(root / 'cache', target.parent / 'cache')
    monkeypatch.setattr(inputs, 'declaration', lambda root: {'synthetic': True})
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c' * 64)
    return target


def grant():
    return SimpleNamespace(check=lambda: None, model_limit=lambda family: 400,
        payload={'granted_at': '2026-10-01T00:00:00Z', 'not_after': '2026-11-01T00:00:00Z'})


def pinned(monkeypatch):
    monkeypatch.setattr(ex, 'committed', lambda: None)
    monkeypatch.setattr(ex, 'head', lambda: 'sha')
    monkeypatch.setattr(fw, 'checked_freeze', lambda: {})


def test_prebuild_seals_earlier_cutoff_and_dispatch_reads_hash_cache(tmp_path, monkeypatch):
    cutoff = T - timedelta(minutes=20)
    index = live_inputs(tmp_path, monkeypatch, cutoff)
    pinned(monkeypatch)
    result = fw.prebuild(tmp_path, grant(), expected_commit='sha', clock=lambda: cutoff)
    assert result['status'] == 'READY'
    assert result['model_calls'] == result['requests'] == result['publications'] == 0
    assert not (fw.directory(tmp_path, '2026-10-20') / 'snapshot.json.gz').exists()
    assert (tmp_path / fw.PREBUILD_DIR / '2026-10-20' / (result['snapshot_hash'] + '.json.gz')).is_file()
    # Later collection cannot replace or broaden the sealed inputs; no archive read at dispatch.
    index.write_text('{}')
    monkeypatch.setattr(inputs, 'build', lambda *a: pytest.fail('dispatch rebuilt a fresh prebuild'))
    sealed = fw.snapshot(tmp_path, T)
    assert sealed['input_cutoff'] == '2026-10-20T12:45:00Z'
    assert sealed['decision_time'] == iso(T)
    assert all(c['decision_time'] == iso(cutoff) for c in sealed['contexts'].values())
    assert sealed['snapshot_hash'] == result['snapshot_hash']
    assert fw.snapshot(tmp_path, T + timedelta(seconds=1)) == sealed
    assert 'Scheduled decision/entry anchor: ' + iso(T) in fw.prompt(sealed)


def test_missing_and_stale_prebuild_fall_back_without_relaxing_cutoff(tmp_path, monkeypatch):
    live_inputs(tmp_path, monkeypatch, T)
    sealed = fw.snapshot(tmp_path, T + timedelta(seconds=10))
    assert sealed['input_cutoff'] == sealed['decision_time'] == iso(T)
    # Make a hash-valid but stale prebuild. It cannot become the session snapshot.
    root = tmp_path / 'stale'
    live_inputs(root, monkeypatch, T)
    value = dict(sealed, input_cutoff=iso(T - timedelta(minutes=31)),
                 built_at=iso(T - timedelta(minutes=31)), completed_at=iso(T - timedelta(minutes=30)))
    value['contexts'] = {g: dict(c, decision_time=value['input_cutoff']) for g, c in sealed['contexts'].items()}
    value.pop('snapshot_hash')
    value['snapshot_hash'] = sha256_canonical(value)
    base = root / fw.PREBUILD_DIR / '2026-10-20'
    parent.write_json(base / (value['snapshot_hash'] + '.json.gz'), value)
    (base / 'ready.json').write_text(json.dumps({'snapshot_hash': value['snapshot_hash']}))
    fallback = fw.snapshot(root, T)
    assert fallback['input_cutoff'] == iso(T) and fallback['snapshot_hash'] != value['snapshot_hash']


def test_prebuild_hash_tampering_blocks_instead_of_rebuilding(tmp_path, monkeypatch):
    cutoff = T - timedelta(minutes=20)
    live_inputs(tmp_path, monkeypatch, cutoff)
    pinned(monkeypatch)
    result = fw.prebuild(tmp_path, grant(), expected_commit='sha', clock=lambda: cutoff)
    path = tmp_path / fw.PREBUILD_DIR / '2026-10-20' / (result['snapshot_hash'] + '.json.gz')
    value = parent.read_json(path)
    value['contexts']['ab']['universe'] = []
    parent.write_json(path, value)
    with pytest.raises(ReplayError, match='FORWARD_SNAPSHOT_HASH_MISMATCH'):
        fw.snapshot(tmp_path, T)


@pytest.mark.parametrize('elapsed, expected_calls', [(300, 2), (301, 0)])
def test_fallback_dispatch_requires_the_full_timeout_and_alerts(tmp_path, monkeypatch, runtime, ledger, elapsed, expected_calls):
    client, calls = runtime
    original = fw.snapshot
    current = [T]
    def build(root, at):
        current[0] = T + timedelta(seconds=elapsed)
        return original(root, at)
    monkeypatch.setattr(fw, 'snapshot', build)
    client.clock = lambda: current[0]
    alerts = []
    monkeypatch.setattr(quality, 'alert', lambda code: alerts.append(code) or True)
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: current[0], ledger=ledger)
    assert len(calls) == expected_calls
    assert result['dispatch_remaining_seconds'] == 1200 - elapsed
    if expected_calls:
        assert result['status'] == 'DONE' and not alerts
    else:
        assert result['models'] == {f: 'INSUFFICIENT_CALL_WINDOW' for f in ex.FAMILIES}
        assert result['window_alert_confirmed'] and len(alerts) == 1
        assert not (tmp_path / 'transcripts').exists()


def test_forward_rechecks_window_after_reservation(tmp_path, monkeypatch, runtime):
    client, calls = runtime
    current = [T]
    client.clock = lambda: current[0]
    reserve = client.budget.reserve_model
    def delayed(*args, **kwargs):
        reserve(*args, **kwargs)
        current[0] = T + timedelta(seconds=301)
    monkeypatch.setattr(client.budget, 'reserve_model', delayed)
    with pytest.raises(ReplayError, match='INSUFFICIENT_CALL_WINDOW'):
        client.call('opus', 'synthetic prompt', fw.SCHEMA, tag='synthetic', timeout=900)
    assert calls == []


def test_drill_builds_and_reads_cache_without_calling_or_publishing(tmp_path, monkeypatch):
    live_inputs(tmp_path, monkeypatch, T)
    pinned(monkeypatch)
    monkeypatch.setattr(ex, 'check_freeze', lambda root: None)
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda family: 400,
        payload={'granted_at': '2026-10-01T00:00:00Z', 'not_after': '2026-11-01T00:00:00Z'})
    seconds = iter([0, 4, 5, 0, 301, 302])
    alerts = []
    monkeypatch.setattr(quality, 'alert', lambda code: alerts.append(code) or True)
    for ready in (True, False):
        result = fw.drill(tmp_path, '2026-10-21', grant, expected_commit='sha', clock=lambda: T.replace(hour=22),
                          monotonic=lambda: next(seconds))
        assert result['prebuild_rehearsal']['fallback_ready'] is ready
        assert result['status'] == ('READY' if ready else 'BLOCKED')
        assert result['model_calls'] == result['publications'] == result['requests'] == 0
    assert len(alerts) == 1
    assert not (tmp_path / fw.STATE_DIR).exists()
    assert not (tmp_path / 'commitments').exists()


def test_prebuild_waits_for_collector_and_uses_actual_start_cutoff(tmp_path, monkeypatch):
    from scripts.trading_lab.replay_bench.storage import owned
    cutoff = T - timedelta(minutes=20)
    live_inputs(tmp_path, monkeypatch, cutoff)
    pinned(monkeypatch)
    current = [cutoff]
    lock = owned(tmp_path / 'collector')
    lock.__enter__()
    def finish_poll(seconds):
        lock.__exit__(None, None, None)
        current[0] += timedelta(minutes=1)
    monkeypatch.setattr(fw.timer, 'sleep', finish_poll)
    result = fw.prebuild(tmp_path, grant(), expected_commit='sha', clock=lambda: current[0])
    assert result['input_cutoff'] == '2026-10-20T12:46:00Z'
    assert fw.snapshot(tmp_path, T)['input_cutoff'] == result['input_cutoff']


def test_prebuild_does_not_publish_cache_when_build_crosses_dispatch(tmp_path, monkeypatch):
    cutoff = T - timedelta(minutes=20)
    live_inputs(tmp_path, monkeypatch, cutoff)
    pinned(monkeypatch)
    current = iter([cutoff, cutoff, T])
    with pytest.raises(ReplayError, match='FORWARD_PREBUILD_COMPLETED_TOO_LATE'):
        fw.prebuild(tmp_path, grant(), expected_commit='sha', clock=lambda: next(current))
    assert not (tmp_path / fw.PREBUILD_DIR / '2026-10-20' / 'ready.json').exists()


def test_dangling_cache_pointer_falls_back_with_scheduled_cutoff(tmp_path, monkeypatch):
    live_inputs(tmp_path, monkeypatch, T)
    base = tmp_path / fw.PREBUILD_DIR / '2026-10-20'
    base.mkdir(parents=True)
    (base / 'ready.json').write_text(json.dumps({'snapshot_hash': 'a' * 64}))
    sealed = fw.snapshot(tmp_path, T)
    assert sealed['input_cutoff'] == sealed['decision_time'] == iso(T)


@pytest.mark.parametrize('group', ['ab', 'cd'])
def test_prebuild_blocks_and_alerts_for_any_shared_excluded_day(tmp_path, monkeypatch, group):
    cutoff = T - timedelta(minutes=20)
    live_inputs(tmp_path, monkeypatch, cutoff)
    pinned(monkeypatch)
    build = fw.build_snapshot
    def excluded(*args, **kwargs):
        value = build(*args, **kwargs)
        value['manifests'][group]['data_quality']['excluded_day'] = True
        value.pop('snapshot_hash')
        value['snapshot_hash'] = sha256_canonical(value)
        return value
    monkeypatch.setattr(fw, 'build_snapshot', excluded)
    alerts = []
    monkeypatch.setattr(quality, 'alert', lambda code: alerts.append(code) or True)
    result = fw.prebuild(tmp_path, grant(), expected_commit='sha', clock=lambda: cutoff)
    assert result['status'] == 'BLOCKED' and result['excluded_day'] is True
    assert result['models'] == {'opus': 'DATA_QUALITY_EXCLUDED', 'gpt': 'DATA_QUALITY_EXCLUDED'}
    assert result['alert_confirmed'] and len(alerts) == 1
    assert not (tmp_path / fw.PREBUILD_DIR / '2026-10-20' / 'ready.json').exists()
    assert not fw.ForwardBudget(tmp_path).path.exists()


@pytest.mark.parametrize('gate', ['quota', 'credit', 'reserved', 'cap', 'failure', 'saved'])
def test_prebuild_checks_durable_model_skips_without_calling(tmp_path, monkeypatch, gate):
    from scripts.trading_lab.replay_bench.storage import atomic_json
    cutoff = T - timedelta(minutes=20)
    pinned(monkeypatch)
    budget = fw.ForwardBudget(tmp_path)
    expected = {'quota': 'MODEL_USAGE_LIMIT', 'credit': 'API_CREDIT_EXHAUSTED',
                'reserved': 'FORWARD_CALL_ALREADY_RESERVED', 'cap': 'BUDGET_EXHAUSTED',
                'failure': 'FORWARD_FAILURE_ALREADY_RECORDED:MODEL_FAILED', 'saved': 'FORWARD_DECISION_ALREADY_SAVED'}[gate]
    if gate == 'credit':
        (tmp_path / fw.models.CREDIT_STOP).write_text('{}')
    elif gate in ('failure', 'saved'):
        atomic_json(fw.directory(tmp_path, '2026-10-20') / ('opus-failure.json' if gate == 'failure' else 'opus.json'),
                    {'error': 'MODEL_FAILED'})
    else:
        atomic_json(budget.path, {'2026-10-20': {
            {'quota': 'stopped:opus', 'reserved': 'forward:model:opus', 'cap': 'model:opus'}[gate]:
            'MODEL_USAGE_LIMIT' if gate == 'quota' else 400 if gate == 'cap' else 1}})
    pointer = tmp_path / fw.PREBUILD_DIR / '2026-10-20' / 'ready.json'
    atomic_json(pointer, {'snapshot_hash': 'a' * 64})
    before = budget.path.read_bytes() if budget.path.exists() else None
    monkeypatch.setattr(fw, 'build_snapshot', lambda *a, **k: pytest.fail('durable blocker must precede build'))
    monkeypatch.setattr(quality, 'alert', lambda code: True)
    result = fw.prebuild(tmp_path, grant(), expected_commit='sha', clock=lambda: cutoff)
    assert result['status'] == 'BLOCKED' and result['models']['opus'] == expected
    assert result['models']['gpt'] == 'READY' and result['alert_confirmed']
    assert not pointer.exists()
    assert (budget.path.read_bytes() if budget.path.exists() else None) == before


def test_drill_blocks_on_source_exclusion_and_alert_failure_stays_blocked(tmp_path, monkeypatch):
    live_inputs(tmp_path, monkeypatch, T)
    pinned(monkeypatch)
    monkeypatch.setattr(ex, 'check_freeze', lambda root: None)
    build = fw.build_snapshot
    def excluded(*args, **kwargs):
        value = build(*args, **kwargs)
        value['manifests']['ab']['data_quality']['excluded_day'] = True
        value.pop('snapshot_hash')
        value['snapshot_hash'] = sha256_canonical(value)
        return value
    monkeypatch.setattr(fw, 'build_snapshot', excluded)
    monkeypatch.setattr(quality, 'alert', lambda code: False)
    result = fw.drill(tmp_path, '2026-10-21', grant(), expected_commit='sha', clock=lambda: T.replace(hour=22))
    assert result['status'] == 'BLOCKED' and result['excluded_day'] is True
    assert result['alert_confirmed'] is False
    assert not fw.directory(tmp_path, '2026-10-21').exists()

from datetime import timedelta
import json
from pathlib import Path
import subprocess
from threading import Barrier
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_seal_forward import experiment as ex, models, quality, spec
from scripts.trading_lab.sources.canonical import sha256_canonical

T = instant('2026-10-20T12:00:00Z')


def client(tmp_path, *, window=models.continuous_remaining, runner=None):
    if runner is None:
        def runner(args, **kwargs):
            raw = json.dumps({'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': {}})
            return subprocess.CompletedProcess(args, 0, raw, '')
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda family: 400)
    return models.Models(grant, models.DuelBudget(tmp_path), tmp_path/spec.STATE_DIR/'transcripts',
                         runner=runner, clock=lambda: T, window=window)


def test_short_window_never_consumes_a_call_or_truncates_timeout(tmp_path):
    calls = []
    def runner(args, **kwargs):
        calls.append(kwargs['timeout'])
        return subprocess.CompletedProcess(args, 0, '{"modelUsage":{"claude-opus-5-5":{}},"structured_output":{}}', '')
    c = client(tmp_path, window=lambda at: 899, runner=runner)
    with pytest.raises(ReplayError, match='INSUFFICIENT_CALL_WINDOW'):
        c.call('opus', 'p', {}, tag='t')
    assert c.budget.used('model:opus', T) == 0 and not calls
    c.window = lambda at: 900
    c.call('opus', 'p', {}, tag='t')
    assert calls == [900]


@pytest.mark.parametrize('hour', [0, 6, 12, 23])
def test_continuous_replay_accepts_every_hour(hour):
    assert models.continuous_remaining(T.replace(hour=hour)) >= 900


def test_usage_reset_resumes_only_at_provider_clock_and_keeps_the_daily_count(tmp_path):
    budget = models.DuelBudget(tmp_path)
    reset = T + timedelta(hours=1)
    budget.reserve_model('gpt', 400, T)
    budget.stop_model('gpt', T, reset_at=reset)
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        budget.reserve_model('gpt', 400, reset-timedelta(microseconds=1))
    budget.reserve_model('gpt', 400, reset)
    assert budget.used('model:gpt', T) == 2
    assert not budget.model_stopped('gpt', reset)
    assert models.quota_reset({'at':T.isoformat(), 'raw':json.dumps({'error':{'resets_at':reset.isoformat()}})}) == reset
    later = T + timedelta(days=2)
    budget.stop_model('gpt', T, reset_at=later)
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        budget.reserve_model('gpt', 400, T + timedelta(days=1))


@pytest.mark.parametrize('stream', ['.stdout', '.stderr'])
def test_any_recorded_byte_forbids_retry(tmp_path, stream):
    path = tmp_path/'intent.json'
    record = {'owner_pid': 999999999, 'owner_start': 'synthetic', 'retry': 0}
    for suffix in ('.stdout', '.stderr'):
        path.with_suffix(suffix).write_bytes(b'')
    assert models.retryable(record, path)
    path.with_suffix(stream).write_bytes(b'x')
    assert not models.retryable(record, path)
    path.with_suffix(stream).write_bytes(b'')
    assert not models.retryable(dict(record, retry=1), path)
    assert not models.retryable(dict(record, owner_pid=None), path)


def test_live_owner_cannot_be_retried(tmp_path):
    import os
    path = tmp_path/'intent.json'
    for suffix in ('.stdout', '.stderr'):
        path.with_suffix(suffix).write_bytes(b'')
    assert not models.retryable({'owner_pid': os.getpid(), 'owner_start': models.process_identity(os.getpid())}, path)


def test_retry_preserves_first_attempt_and_second_timeout_is_binding(tmp_path):
    attempts = []
    def runner(args, **kwargs):
        attempts.append(args)
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    c = client(tmp_path, runner=runner)
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call('opus', 'p', {}, tag='t')
    path = next(c.transcripts.glob('*.json'))
    first = json.loads(path.read_text())
    first.update(owner_pid=999999999, owner_start='synthetic')
    path.write_text(json.dumps(first))
    for suffix in ('.stdout', '.stderr'):
        path.with_suffix(suffix).write_bytes(b'')
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call('opus', 'p', {}, tag='t')
    assert json.loads(path.with_suffix('.attempt-0.json').read_text()) == first
    second = json.loads(path.read_text())
    second.update(owner_pid=999999999, owner_start='synthetic')
    path.write_text(json.dumps(second))
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call('opus', 'p', {}, tag='t')
    assert len(attempts) == c.budget.used('model:opus', T) == 2


def test_systemic_gap_boundaries_distinguish_eligibility_filters():
    fixed = tuple('ABCDE')
    rejected = [{'symbol': 'A'}]
    assert quality.systemic_gaps(rejected, {'X': 'INVALID_BAR'}, fixed, 2)
    assert quality.systemic_gaps([], {'X': 'HISTORY_LT_200_SESSIONS'}, fixed, 1)
    with pytest.raises(ReplayError, match='PIPELINE_DEFECT'):
        quality.systemic_gaps(rejected + [{'symbol': 'B'}], {}, fixed, 0)
    with pytest.raises(ReplayError, match='PIPELINE_DEFECT'):
        quality.systemic_gaps([], {'X': 'INVALID_BAR'}, fixed, 1)


def test_workers_overlap_and_test_waits_for_both_dev_scores(tmp_path, monkeypatch):
    barrier = Barrier(2, timeout=5)
    calls, scores = [], []
    monkeypatch.setattr(ex, 'head', lambda: 'sha')
    monkeypatch.setattr(ex, 'status', lambda *a: {'status': 'READY'})
    monkeypatch.setattr(ex, 'runtime_report', lambda *a: None)
    monkeypatch.setattr(ex, 'planned_for', lambda *a: ['2026-07-01'])
    monkeypatch.setattr(ex, 'GROUPS', ('ab',))
    monkeypatch.setattr(ex, 'publish_pending', lambda *a: None)
    monkeypatch.setattr(ex, 'complete', lambda *a: True)
    monkeypatch.setattr(ex, 'claim_test', lambda *a: None)
    monkeypatch.setattr(ex, 'failure_summary', lambda *a: {})
    from scripts.trading_lab.replay_seal_forward import score
    from scripts.trading_lab.replay_seal_forward import continuation
    monkeypatch.setattr(continuation, 'verify_sources', lambda *a: None)
    monkeypatch.setattr(score, 'score', lambda root, period, transport: scores.append(period))
    def run_one(root, period, group, day, family, *args):
        barrier.wait()
        if period == 'test':
            assert scores == ['dev']
        calls.append((period, family))
    monkeypatch.setattr(ex, 'run_one', run_one)
    budget = SimpleNamespace(model_stopped=lambda *a: None)
    result = ex.night(tmp_path, budget, None, None, expected_commit='sha', ledger=object())
    assert result['status'] == 'DONE'
    # Test groups are ab and cd, each traversed independently by both families.
    assert len(calls) == 6 and scores == ['dev', 'test']


@pytest.mark.parametrize('state', ['RECEIVED', 'INTENT'])
def test_explicit_continuation_never_redispatches_a_received_answer(tmp_path, monkeypatch, state):
    import hashlib
    from scripts.trading_lab.replay_seal_forward import continuation
    prior = {'state': state, 'family': 'opus', 'at': '2026-10-20T05:00:00Z', 'tag': 'old:2026-07-07:analyst-try0',
             'spec_hash': 'a'*64, 'prompt_sha256': hashlib.sha256(b'p').hexdigest()}
    if state == 'RECEIVED':
        prior.update(raw='{"modelUsage":{"claude-opus-5-5":{}},"structured_output":{}}', err='', returncode=0)
    identity = sha256_canonical(['a'*64, 'b'*64, 'opus', 'p', {}, prior['tag']])
    source = tmp_path/'duel-v8/transcripts'/(identity+'.json')
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(prior))
    digest = spec.file_hash(source)
    admission = {'source_state_dir': 'duel-v8', 'source_spec_hash': 'a'*64, 'source_code_hash': 'b'*64,
                 'transcripts': {identity:digest}, 'zero_byte_retry_identity': identity, 'source_process_ended':True}
    monkeypatch.setattr(continuation, 'admission', lambda: admission)
    calls = []
    def runner(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"modelUsage":{"claude-opus-5-5":{}},"structured_output":{}}', '')
    c = client(tmp_path, runner=runner)
    _, metadata = c.call('opus', 'p', {}, tag='new:2026-07-07:analyst-try0')
    c.call('opus', 'p', {}, tag='new:2026-07-07:analyst-try0')
    assert len(calls) == (0 if state == 'RECEIVED' else 1)
    assert spec.file_hash(source) == digest
    if state == 'RECEIVED':
        assert metadata['original_protocol']['identity'] == identity
    else:
        saved = json.loads(next(c.transcripts.glob('*.json')).read_text())
        assert saved['retry'] == 1 and saved['source_attempt']['identity'] == identity

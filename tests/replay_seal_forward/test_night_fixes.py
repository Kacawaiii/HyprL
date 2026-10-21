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
    assert models.quota_reset({'at':T.isoformat(), 'raw':'resets 5:10pm (UTC)'}) == T.replace(hour=17,minute=10)
    later = T + timedelta(days=2)
    budget.stop_model('gpt', T, reset_at=later)
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        budget.reserve_model('gpt', 400, T + timedelta(days=1))


CODEX_NOTICE = ("You've hit your usage limit. Upgrade to Pro (https://chatgpt.com/explore/pro), visit "
                "https://chatgpt.com/codex/settings/usage to purchase more credits or try again at {}.")


@pytest.mark.parametrize('notice, at, expected', [
    # 2026-10-21 incident: GPT stopped at 02:26Z and must resume at 05:07Z, not at midnight.
    (CODEX_NOTICE.format('5:07 AM'), '2026-10-21T02:26:00Z', '2026-10-21T05:07:00Z'),
    ("You've hit your usage limit. Try again at 9:12 PM.", '2026-10-21T02:26:00Z', '2026-10-21T21:12:00Z'),
    ("You've hit your usage limit. Try again at 9:12 PM.", '2026-10-21T21:30:00Z', '2026-10-22T21:12:00Z'),
    ("You've hit your session limit · resets 10:30pm (UTC)", '2026-10-21T02:26:00Z', '2026-10-21T22:30:00Z'),
    ("You've hit your session limit · resets 10:30pm (UTC)", '2026-10-21T22:31:00Z', '2026-10-22T22:30:00Z'),
    ("You've hit your session limit · resets 10pm (UTC)", '2026-10-21T23:00:00Z', '2026-10-22T22:00:00Z'),
    # Each CLI adds the date when the reset is on another day.
    (CODEX_NOTICE.format('Oct 22nd, 2026 5:07 AM'), '2026-10-21T23:30:00Z', '2026-10-22T05:07:00Z'),
    ("You've hit your weekly limit · resets Oct 24, 10am (UTC)", '2026-10-21T02:26:00Z', '2026-10-24T10:00:00Z'),
    ("You've hit your weekly limit · resets Jan 1, 10am (UTC)", '2026-12-31T23:30:00Z', '2027-01-01T10:00:00Z'),
    # The printed clock is truncated to the minute (05:07:45 prints 5:07): an error inside that
    # minute, e.g. after resuming at the stored 05:07:00, resets at its end, never a day later.
    (CODEX_NOTICE.format('5:07 AM'), '2026-10-21T05:07:30Z', '2026-10-21T05:08:00Z'),
    ("You've hit your session limit · resets 10:30pm (UTC)", '2026-10-21T22:30:00Z', '2026-10-21T22:31:00Z'),
])
def test_usage_limit_notice_resumes_at_next_printed_utc_clock(notice, at, expected):
    # Codex prints its notice inside JSONL events; Claude in its result envelope or on stderr.
    raw = '\n'.join(json.dumps(e) for e in [{'type': 'thread.started'}, {'type': 'error', 'message': notice}])
    assert models.quota_reset({'at': at, 'received_at': at, 'raw': raw, 'err': ''}) == instant(expected)
    assert models.quota_reset({'at': at, 'raw': '', 'err': notice}) == instant(expected)


@pytest.mark.parametrize('notice', ["You've hit your usage limit. Try again later.", "You've hit your session limit",
                                    'Try again at noon.', 'try again at 5.', 'try again at 13:07 PM.',
                                    'resets 25:00 (UTC)', 'resets 10pm (Mars/Olympus)'])
def test_usage_limit_without_a_parseable_clock_keeps_the_next_utc_day_fallback(notice):
    assert models.quota_reset({'at': '2026-10-21T02:26:00Z', 'raw': '', 'err': notice}) == instant('2026-10-22T00:00:00Z')


def test_gpt_usage_limit_stops_gpt_only_until_its_printed_reset(tmp_path):
    def runner(args, **kwargs):
        raw = '\n'.join(json.dumps(e) for e in [{'type': 'thread.started'},
                        {'type': 'error', 'message': CODEX_NOTICE.format('5:07 AM')},
                        {'type': 'turn.failed', 'error': {'message': CODEX_NOTICE.format('5:07 AM')}}])
        return subprocess.CompletedProcess(args, 1, raw, '')
    c = client(tmp_path, runner=runner)
    c.clock = lambda: instant('2026-10-21T02:26:00Z')
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        c.call('gpt', 'p', {}, tag='t')
    assert json.loads(c.budget.path.read_text())['2026-10-21']['reset_at:gpt'] == '2026-10-21T05:07:00Z'
    assert c.budget.model_stopped('gpt', instant('2026-10-21T05:06:59Z'))
    assert not c.budget.model_stopped('gpt', instant('2026-10-21T05:07:00Z'))
    assert not c.budget.model_stopped('opus', instant('2026-10-21T02:26:00Z'))
    # The 13:05Z forward dispatch is no longer refused.
    from scripts.trading_lab.replay_seal_forward.forward import ForwardBudget
    ForwardBudget(tmp_path).reserve_model('gpt', 400, instant('2026-10-21T13:05:00Z'))


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


def test_quota_worker_resumes_while_other_worker_still_runs(tmp_path,monkeypatch):
    from threading import Event
    from scripts.trading_lab.replay_seal_forward import continuation,score
    resumed=Event(); attempts=[]
    monkeypatch.setattr(continuation,'verify_sources',lambda *a:None)
    monkeypatch.setattr(ex,'head',lambda:'sha')
    monkeypatch.setattr(ex,'status',lambda *a:{'status':'READY'})
    monkeypatch.setattr(ex,'runtime_report',lambda *a:None)
    monkeypatch.setattr(ex,'planned_for',lambda *a:['2026-07-01'])
    monkeypatch.setattr(ex,'GROUPS',('ab',))
    monkeypatch.setattr(ex,'publish_pending',lambda *a:None)
    monkeypatch.setattr(ex,'complete',lambda *a:True)
    monkeypatch.setattr(ex,'claim_test',lambda *a:None)
    monkeypatch.setattr(ex,'failure_summary',lambda *a:{})
    monkeypatch.setattr(score,'score',lambda *a:None)
    def run_one(root,period,group,day,family,*a):
        if period=='dev' and family=='opus':
            attempts.append(family)
            if len(attempts)==1:raise ReplayError('MODEL_USAGE_LIMIT')
            resumed.set()
        elif period=='dev':
            assert resumed.wait(5),'Quota worker waited for the other worker to finish'
    monkeypatch.setattr(ex,'run_one',run_one)
    budget=SimpleNamespace(model_stopped=lambda *a:None)
    result=ex.night(tmp_path,budget,None,None,expected_commit='sha',ledger=object())
    assert result['status']=='DONE' and len(attempts)==2


def opus_news(n_candidates=40, n_views=10):
    """An Opus-like news answer: every forecast, but views only where the model has an opinion."""
    ctx = {'schema': 'replay-news-context-v1', 'session': '2026-07-06', 'decision_time': '2026-07-06T12:00:00Z',
           'candidates': [{'symbol': f'S{i:02d}', 'filings': [], 'headlines': [{'news_id': i, 'title': f'S{i:02d} news',
                           'published_at': '2026-07-06T11:00:00Z', 'url': f'https://news.invalid/{i}'}]}
                          for i in range(n_candidates)]}
    answer = {'views': [dict(symbol=f'S{i:02d}', view='UP', p_outperform=.6, holding_sessions=5,
                             event_code='contract_product', news_ids=[i], reason='r', counter_thesis='c', falsifier='f')
                        for i in range(n_views)],
              'forecasts': [dict(asset=c['symbol'], horizon=h, p_outperform=.55)
                            for c in ctx['candidates'] for h in ('2d', '10d')]}
    return ctx, answer


def test_stored_partial_view_answer_revalidates_from_raw_without_a_model_call(tmp_path, monkeypatch):
    """A RECEIVED answer that failed only on view coverage is re-decoded from its stored bytes."""
    import hashlib
    from scripts.trading_lab.replay_seal_forward import continuation, run, schemas
    ctx, answer = opus_news()
    monkeypatch.setattr(run, 'shared_snapshot', lambda *a: {'context': ctx, 'snapshot_hash': 'h'*64})
    text = run.prompt('news', ctx)
    prior = {'state': 'RECEIVED', 'family': 'opus', 'at': '2026-10-20T05:00:00Z', 'tag': 'old:2026-07-06:news-try0',
             'spec_hash': 'a'*64, 'prompt_sha256': hashlib.sha256(text.encode()).hexdigest(), 'returncode': 0, 'err': '',
             'raw': json.dumps({'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': answer})}
    identity = sha256_canonical(['a'*64, 'b'*64, 'opus', text, schemas.NEWS_SCHEMA, prior['tag']])
    source = tmp_path/'duel-v10/transcripts'/(identity+'.json')
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(prior))
    monkeypatch.setattr(continuation, 'admission', lambda: {
        'source_state_dir': 'duel-v10', 'source_spec_hash': 'a'*64, 'source_code_hash': 'b'*64,
        'transcripts': {identity: spec.file_hash(source)}, 'zero_byte_retry_identity': None, 'source_process_ended': True})
    calls = []
    c = client(tmp_path, runner=lambda args, **kwargs: calls.append(args))
    record = run.decide(tmp_path, 'dev', 'cd', '2026-07-06', 'opus', None, c, {'config_hash': 'new'},
                        tmp_path/'record.json.gz')
    assert calls == [] and c.budget.used('model:opus', T) == 0
    assert record['errors']['news'] == [] and record['calls']['news']['original_protocol']['identity'] == identity
    assert record[schemas.VIEWS_FILLED] == 30
    symbols = [x['symbol'] for x in ctx['candidates']]
    for variant in run.CD:
        views = record['decisions'][variant]
        assert len(views) == 40 and record['forecasts'][variant] == answer['forecasts']
        assert [v['view'] for v in views[10:]] == ['ABSTAIN'] * 30
        # The scorer's exact decision coverage now holds.
        schemas.coverage([(v['symbol'], 'trade') for v in views], [(s, 'trade') for s in symbols])

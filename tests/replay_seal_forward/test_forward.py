from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from scripts.trading_lab.replay_bench import run as parent
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.replay_seal_forward import experiment as ex, forward as fw, models, spec
from scripts.trading_lab.replay_seal_forward import score as scoring
from scripts.trading_lab.sources.canonical import sha256_canonical

T = instant('2026-10-19T13:05:00Z')


def sealed():
    a = {'schema': 'replay-context-v1', 'universe': ['AAPL'], 'headlines': {'AAPL': {'items': []}},
         'prices': {}, 'decision_time': iso(T), 'provenance': 'LIVE_FORWARD'}
    c = {'schema': 'replay-news-context-v1', 'candidates': [], 'decision_time': iso(T), 'provenance': 'LIVE_FORWARD'}
    value = {'contexts': {'ab': a, 'cd': c}, 'decision_time': iso(T), 'provenance': 'LIVE_FORWARD'}
    return dict(value, snapshot_hash=sha256_canonical(value))


def output():
    return {'analyst': {'regime': ['synthetic'], 'views': [dict(asset='AAPL', horizon=h, view='ABSTAIN', p_outperform=.7,
        confidence_reason='synthetic', catalysts=[], priced_in_assessment='', second_order='', counter_thesis='',
        falsifier='', event_risk=[]) for h in ('1d', '5d')]}, 'news': {'views': [], 'forecasts': []}}


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c'*64)
    monkeypatch.setattr(ex, 'committed', lambda: None)
    monkeypatch.setattr(ex, 'head', lambda: 'sha')
    file = tmp_path / 'freeze.json'; file.write_text('{}')
    monkeypatch.setattr(ex, 'FREEZE', file)
    monkeypatch.setattr(ex, 'frozen_payload', lambda root: {})
    monkeypatch.setattr(fw, 'snapshot', lambda root, at: sealed())
    calls = []
    def runner(args, **kwargs):
        calls.append((args, kwargs['input']))
        value = output()
        if args[0] == 'claude':
            raw = json.dumps({'subtype': 'success', 'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': value})
        else:
            raw = '\n'.join(json.dumps(e) for e in [{'type': 'thread.started'},
                {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': json.dumps(value)}},
                {'type': 'turn.completed'}])
        return subprocess.CompletedProcess(args, 0, raw, '')
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda family: 150)
    client = models.Models(grant, fw.ForwardBudget(tmp_path), tmp_path / 'transcripts', runner=runner,
                           clock=lambda: T, window=fw.window)
    return client, calls


def test_exchange_window_dst_holidays_and_no_backfill():
    assert fw.decision_time('2026-10-19') == T
    assert fw.decision_time('2026-11-02') == instant('2026-11-02T14:05:00Z')
    assert fw.window(T) == 600
    for at in (T-timedelta(microseconds=1), T+timedelta(minutes=10)):
        with pytest.raises(ReplayError, match='OUTSIDE_FORWARD_WINDOW'):
            fw.window(at)
    with pytest.raises(ReplayError, match='NOT_FORWARD_SESSION'):
        fw.window(fw.decision_time('2026-11-26'))
    with pytest.raises(ReplayError, match='NOT_FORWARD_SESSION'):
        fw.window(fw.decision_time('2026-10-18'))


def test_one_call_each_identical_inputs_probability_on_abstention_and_resume(tmp_path, runtime):
    client, calls = runtime
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)
    assert result['status'] == 'DONE' and len(calls) == 2
    assert calls[0][1] == calls[1][1]
    assert 'LIVE_FORWARD' in calls[0][1]
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)
    assert len(calls) == 2
    for family in ex.FAMILIES:
        value = json.loads((fw.directory(tmp_path, '2026-10-19') / (family+'.json')).read_text())
        assert value['provenance'] == 'LIVE_FORWARD' and value['decision_time'] == iso(T)
        assert value['snapshot_hash'] == result['snapshot_hash']
        assert value['output']['analyst']['views'][0]['p_outperform'] == .7
    with pytest.raises(ReplayError, match='FORWARD_CALL_ALREADY_RESERVED'):
        client.budget.reserve_model('opus', 150, T)


def test_timeout_never_retries_and_other_family_completes(tmp_path, runtime):
    client, calls = runtime
    original = client.runner
    attempts = []
    def runner(args, **kwargs):
        attempts.append(args[0])
        if args[0] == 'claude':
            raise subprocess.TimeoutExpired(args, 1)
        return original(args, **kwargs)
    client.runner = runner
    result = fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)
    assert result['status'] == 'BLOCKED'
    assert result['models']['opus'] == 'CALL_OUTCOME_UNKNOWN'
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)
    assert attempts == ['claude', 'codex']


def test_tampered_saved_decision_refused(tmp_path, runtime):
    client, _ = runtime
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)
    file = fw.directory(tmp_path, '2026-10-19') / 'opus.json'
    value = json.loads(file.read_text()); value['output']['analyst']['views'][0]['p_outperform'] = .1
    atomic_json(file, value)
    with pytest.raises(ReplayError, match='FORWARD_DECISION_BINDING_MISMATCH'):
        fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)


def test_mature_labels_only_and_weekly_at_most_once(tmp_path, runtime, monkeypatch):
    client, calls = runtime
    fw.run(tmp_path, client, expected_commit='sha', clock=lambda: T)
    file = fw.directory(tmp_path, '2026-10-19') / 'snapshot.json.gz'
    parent.write_json(file, sealed())
    seen = []
    class Outcomes:
        def __init__(self, *args):
            pass
        def get(self, group, day, asset, h):
            seen.append((asset, h))
            return {'rel': .01, 'cost': .003}
    monkeypatch.setattr(scoring, 'Outcomes', Outcomes)
    values = fw.score(tmp_path, None, clock=lambda: T.replace(hour=21))
    assert len(values['rows']) == 2 and seen == [('AAPL', 1), ('AAPL', 1)]
    assert all(r['horizon'] == '1d' and r['brier'] == pytest.approx(.09) for r in values['rows'])
    friday = instant('2026-10-23T21:15:00Z')
    values['scored_at'] = iso(friday)
    atomic_json(tmp_path/'live-forward-v1/scores.json', values)
    deliveries = []
    def deliver(args, **kwargs):
        deliveries.append((args, kwargs['input']))
        return subprocess.CompletedProcess(args, 0, 'discord\n', '')
    fw.weekly(tmp_path, notify=Path('/synthetic/notify'), clock=lambda: friday, runner=deliver)
    assert deliveries[0][0] == ['/synthetic/notify', 'brouillons']
    assert 'n=1' in deliveries[0][1] and 'n=0' in deliveries[0][1]
    with pytest.raises(ReplayError, match='ALREADY_ATTEMPTED'):
        fw.weekly(tmp_path, notify=Path('/synthetic/notify'), clock=lambda: friday, runner=deliver)
    assert len(deliveries) == 1

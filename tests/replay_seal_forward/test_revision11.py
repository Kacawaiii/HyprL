"""Regression checks for immutable replies and saved subscription reset evidence."""
import json
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_seal_forward import continuation, experiment as ex, run, score, spec
from scripts.trading_lab.replay_seal_forward import forward, models
from scripts.trading_lab.sources.canonical import sha256_canonical


@pytest.mark.parametrize('state', ['duel-v8', 'duel-v9', 'duel-v10', spec.STATE_DIR])
def test_dev_scoring_checks_the_original_published_response(tmp_path, monkeypatch, ledger, state):
    day = '2026-07-06'
    identity = 'a' * 64
    response = {'state': 'RECEIVED', 'raw': 'synthetic answer', 'err': '', 'returncode': 0}
    file = tmp_path / state / 'transcripts' / (identity + '.json')
    file.parent.mkdir(parents=True)
    file.write_text(json.dumps(response))
    admission = {'sources': [{'source_state_dir': state, 'source_spec_hash': 'b' * 64,
                             'source_code_hash': 'c' * 64, 'transcripts': {identity: spec.file_hash(file)}}]}
    monkeypatch.setattr(continuation, 'admission', lambda: admission)
    call = {'transcript': identity, 'prompt_sha256': 'd' * 64}
    if state != spec.STATE_DIR:
        call['original_protocol'] = dict(identity=identity, sha256=spec.file_hash(file), spec_hash='b' * 64,
                                        code_hash='c' * 64, admission_sha256=sha256_canonical(admission))
    record = {'decision_hash': 'e' * 64, 'calls': {'analyst': call}, 'snapshot_hash': 'f' * 64,
              'shown_context_hash': '0' * 64, 'forecasts': {v: [] for v in run.AB},
              'decisions': {v: [] for v in run.AB}}
    monkeypatch.setattr(ex, 'input_proof', lambda _: {})
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **k: None)
    monkeypatch.setattr(ex, 'complete', lambda *a: True)
    monkeypatch.setattr(ex, 'GROUPS', ('ab',))
    monkeypatch.setattr(ex, 'planned_for', lambda *a: [day])
    monkeypatch.setattr(ex, 'meta', lambda *a: {})
    monkeypatch.setattr(run, 'checked', lambda *a: record)
    monkeypatch.setattr(ex, 'offline', lambda *a, **k: None)
    monkeypatch.setattr(run, 'shared_snapshot', lambda *a: {'snapshot_hash': 'f' * 64,
                                                         'manifest': {}, 'context': {'universe': []}})
    monkeypatch.setattr(score, 'frozen_primary_rate', lambda _: {'n': 0, 'successes': 0, 'p': .5})
    hashes = {score.record_key(f, 'dev', 'ab', day): 'e' * 64 for f in ex.FAMILIES}
    hashes['response:' + identity] = sha256_canonical({k: response[k] for k in ('raw', 'err', 'returncode')})
    ledger.clock = lambda: instant('2026-10-21T00:00:00Z')
    ledger.publish('replay-night', 'synthetic', hashes)
    result = score.score(tmp_path, 'dev', None, outcome_factory=lambda *a: SimpleNamespace(),
                         clock=lambda: instant('2026-10-23T00:00:00Z'))
    assert result['status'] == 'SCORED'
    if state != spec.STATE_DIR:
        file.write_text(json.dumps(dict(response, raw='altered answer')))
        with pytest.raises(ReplayError, match='CONTINUATION_SOURCE_CHANGED'):
            score.score(tmp_path, 'dev', None, outcome_factory=lambda *a: SimpleNamespace(),
                        clock=lambda: instant('2026-10-23T00:00:00Z'))


def test_saved_usage_stops_reparse_without_refunds_or_invented_evidence(tmp_path):
    budget = models.DuelBudget(tmp_path)
    original = {'2026-10-21': {'model:gpt': 46, 'model:opus': 6, 'forward:model:opus': 1,
                              'stopped:gpt': 'MODEL_USAGE_LIMIT', 'reset_at:gpt': '2026-10-22T00:00:00Z'},
                '2026-10-20': {'stopped:opus': 'MODEL_USAGE_LIMIT'}}
    budget.path.write_text(json.dumps(original))
    source = tmp_path / 'duel-v10/transcripts/stop.json'
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps({'family': 'gpt', 'state': 'USAGE_LIMIT', 'at': '2026-10-21T02:26:00Z',
                                  'raw': '', 'err': "You've hit your usage limit. Try again at 5:07 AM.",
                                  'returncode': 1}))
    digest = spec.file_hash(source)
    result = budget.reparse_stops([source])
    repaired = json.loads(budget.path.read_text())
    assert repaired == dict(original, **{'2026-10-21': dict(original['2026-10-21'],
                                                         **{'reset_at:gpt': '2026-10-21T05:07:00Z'})})
    assert result['unresolved'] == [{'day': '2026-10-20', 'family': 'opus'}]
    assert result['evidence'][0]['sha256'] == digest and spec.file_hash(source) == digest
    assert budget.reparse_stops([source]) == result
    # Forward shares exactly the repaired stop, without consuming a forward reservation.
    assert not forward.ForwardBudget(tmp_path).model_stopped('gpt', instant('2026-10-21T13:05:00Z'))
    assert not forward.ForwardBudget(tmp_path).model_stopped('gpt', forward.decision_time('2026-10-22'))


def test_tomorrow_forward_dry_run_checks_both_families_without_reserving(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, 'committed', lambda: None)
    monkeypatch.setattr(ex, 'head', lambda: 'pinned')
    monkeypatch.setattr(ex, 'check_freeze', lambda *a: None)
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda _: 400,
                            payload={'granted_at': '2026-10-01T00:00:00Z', 'not_after': '2026-11-01T00:00:00Z'})
    budget = forward.ForwardBudget(tmp_path)
    budget.stop_model('gpt', instant('2026-10-21T02:26:00Z'), reset_at=instant('2026-10-21T05:07:00Z'))
    original = budget.path.read_bytes()
    result = forward.dry_run(tmp_path, '2026-10-22', grant, expected_commit='pinned')
    assert result['status'] == 'READY' and result['models'] == {'opus': 'READY', 'gpt': 'READY'}
    assert result['decision_time'] == '2026-10-22T13:05:00Z'
    assert result['model_calls'] == result['publications'] == result['requests'] == 0
    assert budget.path.read_bytes() == original
    assert not (tmp_path / forward.STATE_DIR).exists()
    with pytest.raises(ReplayError, match='INSTALLED_SHA_MISMATCH'):
        forward.dry_run(tmp_path, '2026-10-22', grant, expected_commit='wrong')
    budget.stop_model('gpt', instant('2026-10-21T02:26:00Z'), reset_at=instant('2026-10-23T00:00:00Z'))
    result = forward.dry_run(tmp_path, '2026-10-22', grant, expected_commit='pinned')
    assert result['status'] == 'BLOCKED' and result['models']['gpt'] == 'MODEL_USAGE_LIMIT'


def test_referenced_responses_are_published_under_the_current_protocol(tmp_path, monkeypatch, ledger):
    identity, wrapper = 'a' * 64, 'f' * 64
    record = {'state': 'RECEIVED', 'raw': 'synthetic answer', 'err': '', 'returncode': 0}
    source = tmp_path / 'duel-v8/transcripts' / (identity + '.json')
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(record))
    table = dict(source_state_dir='duel-v8', source_spec_hash='b' * 64, source_code_hash='c' * 64,
                 transcripts={identity: spec.file_hash(source)})
    admission = {'sources': [table]}
    monkeypatch.setattr(continuation, 'admission', lambda: admission)
    reference = dict(identity=identity, sha256=spec.file_hash(source), spec_hash='b' * 64, code_hash='c' * 64,
                     admission_sha256=sha256_canonical(admission))
    path = tmp_path / spec.STATE_DIR / 'transcripts' / (wrapper + '.json')
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'state': 'REFERENCED', 'source': reference}))
    monkeypatch.setattr(ex, 'planned_for', lambda *a: [])
    receipt = ex.publish_pending(tmp_path, ledger, 'synthetic')
    assert receipt['line']['hashes'] == {'response:' + identity:
                                      sha256_canonical({k: record[k] for k in ('raw', 'err', 'returncode')})}
    assert source.read_text() == json.dumps(record)

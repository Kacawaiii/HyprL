"""The operator's uniform unresolved-call rule and the in-code UTC gate."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError, instant
from scripts.trading_lab.replay_seal_forward import continuation, experiment as ex, models, run, schemas, spec
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.replay_seal_forward.test_night_fixes import opus_news


@pytest.mark.parametrize('bytes_recorded, retry', [(True, 0), (False, 1)])
def test_unresolved_first_pair_is_terminal_without_reask(tmp_path, monkeypatch, bytes_recorded, retry):
    ctx, answer = opus_news(1, 1)
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **k: None)
    monkeypatch.setattr(ex, 'offline', lambda *a: None)
    monkeypatch.setattr(ex, 'meta', lambda *a: {'config_hash': 'synthetic'})
    monkeypatch.setattr(ex, 'planned_for', lambda *a: ['2026-09-28', '2026-09-29'])
    monkeypatch.setattr(run, 'shared_snapshot', lambda *a: {'context': ctx, 'snapshot_hash': 's'*64})
    monkeypatch.setattr(continuation, 'admission', lambda: {'sources': []})
    dispatches = []
    def runner(*a, **k):
        dispatches.append(1)
        return SimpleNamespace(stdout=json.dumps({'modelUsage': {'claude-opus-5-5': {}},
                                                 'structured_output': deepcopy(answer)}), stderr='', returncode=0)
    client = models.Models(SimpleNamespace(check=lambda: None, model_limit=lambda _: 400),
        models.DuelBudget(tmp_path), tmp_path/spec.STATE_DIR/'transcripts', runner=runner,
        clock=lambda: instant('2026-10-23T01:00:00Z'))
    text = run.prompt('news', ctx)
    tag = 'synthetic:2026-09-28:news-try0'
    identity = sha256_canonical([spec.SPEC_HASH, spec.code_hash(), 'opus', text, schemas.NEWS_SCHEMA, tag])
    path = client.transcripts/(identity+'.json');path.parent.mkdir(parents=True)
    record = dict(state='INTENT', family='opus', at='2026-10-23T00:59:00Z', tag=tag, retry=retry,
                  owner_pid=999999999, owner_start='synthetic', prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    path.write_text(json.dumps(record))
    path.with_suffix('.stdout').write_text('partial synthetic response' if bytes_recorded else '')
    path.with_suffix('.stderr').write_text('')
    before = spec.file_hash(path)
    value = ex.run_one(tmp_path, 'dev', 'cd', '2026-09-28', 'opus', None, client)
    assert value['status'] == 'MODEL_FAILURE' and not value['forecasts'] and not value['decisions']
    assert value['validation_codes'] == ['CALL_OUTCOME_UNKNOWN'] and dispatches == []
    assert continuation.transport_failure_hash(tmp_path, value['uncertain_calls']['try0'])
    journal = tmp_path/spec.STATE_DIR/'failures.jsonl';digest = spec.file_hash(journal)
    assert ex.run_one(tmp_path, 'dev', 'cd', '2026-09-28', 'opus', None, client) == value
    assert spec.file_hash(journal) == digest and spec.file_hash(path) == before
    ex.run_one(tmp_path, 'dev', 'cd', '2026-09-29', 'opus', None, client)
    assert len(dispatches) == 1
    path.with_suffix('.stdout').write_text('tampered')
    with pytest.raises(ReplayError, match='CONTINUATION_SOURCE_CHANGED'):
        continuation.transport_failure_hash(tmp_path, value['uncertain_calls']['try0'])


@pytest.mark.parametrize('clock', ['20:29:59', '06:55:00', '07:00:00', '12:00:00'])
def test_gate_refuses_even_a_custom_unbounded_window_before_reserving(tmp_path, clock):
    calls = []
    at = instant('2026-10-23T'+clock+'Z')
    client = models.Models(SimpleNamespace(check=lambda: None, model_limit=lambda _: 400),
        models.DuelBudget(tmp_path), tmp_path/spec.STATE_DIR/'transcripts', runner=lambda *a, **k: calls.append(1),
        clock=lambda: at, window=lambda _: float('inf'))
    with pytest.raises(ReplayError, match='INSUFFICIENT_CALL_WINDOW'):
        client.call('gpt', 'synthetic', {}, tag='synthetic')
    assert not calls and client.budget.used('model:gpt', at) == 0


def test_admitted_unknown_response_preserves_envelope_and_rejects_changed_stream(tmp_path, monkeypatch):
    text, schema, tag = 'synthetic', {}, 'old:2026-09-28:analyst-try0'
    record = dict(state='INTENT', family='gpt', spec_hash='a'*64, tag=tag,
                  at='2026-10-23T06:59:43Z', owner_pid=999999999, owner_start='synthetic', retry=0,
                  prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    identity = sha256_canonical(['a'*64, 'b'*64, 'gpt', text, schema, tag])
    path = tmp_path/'duel-v12/transcripts'/(identity+'.json');path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record));path.with_suffix('.stdout').write_text('partial')
    path.with_suffix('.stderr').write_text('')
    table = dict(source_state_dir='duel-v12', source_spec_hash='a'*64, source_code_hash='b'*64,
        transcripts={identity: spec.file_hash(path)}, transport_streams={identity: {
            s: dict(sha256=spec.file_hash(path.with_suffix(s)), bytes=path.with_suffix(s).stat().st_size)
            for s in ('.stdout', '.stderr')}})
    monkeypatch.setattr(continuation, 'admission', lambda: {'sources': [table]})
    with pytest.raises(ReplayError) as caught:
        continuation.lookup(tmp_path, 'gpt', text, schema, 'new:2026-09-28:analyst-try0')
    assert caught.value.code == 'CALL_OUTCOME_UNKNOWN'
    assert caught.value.evidence['terminal_model_failure'] is True
    call = caught.value.evidence['uncertain_call']
    assert call['original_protocol']['identity'] == identity
    assert continuation.transport_failure_hash(tmp_path, call)
    path.with_suffix('.stdout').write_text('changed')
    with pytest.raises(ReplayError, match='CONTINUATION_SOURCE_CHANGED'):
        continuation.lookup(tmp_path, 'gpt', text, schema, 'new:2026-09-28:analyst-try0')


def test_admitted_test_start_keeps_original_binding_and_rejects_mutation(tmp_path, monkeypatch):
    source = tmp_path/'duel-v12';(source/'results').mkdir(parents=True)
    dev = dict(status='SCORED', spec_hash='a'*64, code_hash='b'*64)
    freeze = dict(spec_hash='a'*64, code_hash='b'*64)
    art = tmp_path/'docs/artifacts';art.mkdir(parents=True)
    (art/'freeze.json').write_text(json.dumps(freeze))
    marker = dict(spec_hash='a'*64, freeze_hash=sha256_canonical(freeze), dev_result_hash=sha256_canonical(dev))
    (source/'results/dev.json').write_text(json.dumps(dev))
    path = source/'test_started_gpt.json';path.write_text(json.dumps(marker))
    rule = dict(source_state_dir='duel-v12', freeze_artifact='freeze.json', files={
        'test_started_gpt.json': spec.file_hash(path), 'results/dev.json': spec.file_hash(source/'results/dev.json')})
    monkeypatch.setattr(spec, 'ROOT', tmp_path)
    monkeypatch.setattr(continuation, 'admission', lambda: {'test_continuation': rule})
    assert continuation.test_start(tmp_path, 'gpt') == marker
    path.write_text(json.dumps(dict(marker, spec_hash='c'*64)))
    with pytest.raises(ReplayError, match='CONTINUATION_SOURCE_CHANGED'):
        continuation.test_start(tmp_path, 'gpt')


def test_active_owner_and_unknown_owner_cannot_be_resolved_as_failure(tmp_path, monkeypatch):
    record = dict(family='gpt', at='2026-10-23T01:00:00Z', prompt_sha256='a'*64,
                  owner_pid=123, owner_start='alive', retry=0)
    path = tmp_path/('b'*64+'.json');path.write_text(json.dumps(record))
    path.with_suffix('.stdout').write_text('partial')
    monkeypatch.setattr(models, 'process_identity', lambda _: 'alive')
    assert models.unknown_outcome(record, path, 'b'*64).evidence['terminal_model_failure'] is False
    record.pop('owner_start')
    assert models.unknown_outcome(record, path, 'b'*64).evidence['terminal_model_failure'] is False


def test_new_timeout_with_recorded_bytes_is_terminal(tmp_path):
    import subprocess
    def runner(*a, **k):
        raise subprocess.TimeoutExpired('synthetic', 900, output=b'partial synthetic response')
    client = models.Models(SimpleNamespace(check=lambda: None, model_limit=lambda _: 400),
        models.DuelBudget(tmp_path), tmp_path/spec.STATE_DIR/'transcripts', runner=runner,
        clock=lambda: instant('2026-10-23T01:00:00Z'))
    with pytest.raises(ReplayError) as caught:
        client.call('gpt', 'synthetic', {}, tag='synthetic')
    assert caught.value.evidence['terminal_model_failure'] is True
    assert client.budget.used('model:gpt', instant('2026-10-23T01:00:00Z')) == 1


def test_admitted_dead_zero_byte_intent_retains_its_one_retry(tmp_path, monkeypatch):
    text, tag = 'synthetic', 'old:2026-09-28:analyst-try0'
    record = dict(state='INTENT', family='gpt', spec_hash='a'*64, tag=tag,
        at='2026-10-23T01:00:00Z', owner_pid=999999999, owner_start='synthetic', retry=0,
        prompt_sha256=hashlib.sha256(text.encode()).hexdigest())
    identity = sha256_canonical(['a'*64, 'b'*64, 'gpt', text, {}, tag])
    path = tmp_path/'duel-v12/transcripts'/(identity+'.json');path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record))
    for s in ('.stdout', '.stderr'):path.with_suffix(s).write_text('')
    table = dict(source_state_dir='duel-v12', source_spec_hash='a'*64, source_code_hash='b'*64,
        transcripts={identity: spec.file_hash(path)}, transport_streams={identity: {
            s: dict(sha256=spec.file_hash(path.with_suffix(s)), bytes=0) for s in ('.stdout', '.stderr')}})
    monkeypatch.setattr(continuation, 'admission', lambda: {'sources': [table]})
    saved, original = continuation.lookup(tmp_path, 'gpt', text, {}, 'new:2026-09-28:analyst-try0')
    assert saved == record and original['identity'] == identity

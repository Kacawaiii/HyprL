"""Synthetic regressions for terminal model failures and coded-time validation."""
from copy import deepcopy
import json
import hashlib
from types import SimpleNamespace

import pytest

from scripts.trading_lab.replay_bench.grant import ReplayError
from scripts.trading_lab.replay_seal_forward import continuation, experiment as ex, forward, models, run, schemas, score, spec
from scripts.trading_lab.sources.canonical import sha256_canonical
from tests.replay_seal_forward.test_night_fixes import opus_news
from tests.replay_duel.test_duel import analyst_output, context


def test_failed_first_pair_is_terminal_and_next_day_runs(tmp_path, monkeypatch):
    ctx, answer = opus_news(2, 2)
    calls = []
    def call(family, text, schema, *, tag):
        calls.append(tag)
        output = deepcopy(answer)
        if '2026-07-13' in tag:
            # A valid ID for another candidate is still an invalid citation.
            output['views'][0]['news_ids'] = [1]
        return output, {'transcript': str(len(calls)).zfill(64)}
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **k: None)
    monkeypatch.setattr(ex, 'offline', lambda *a: None)
    monkeypatch.setattr(ex, 'meta', lambda *a: {'config_hash': 'synthetic'})
    monkeypatch.setattr(ex, 'GROUPS', ('cd',))
    monkeypatch.setattr(ex, 'planned_for', lambda *a: ['2026-07-13', '2026-07-14'])
    monkeypatch.setattr(run, 'shared_snapshot', lambda *a: {'context': ctx, 'snapshot_hash': 's' * 64})
    client = SimpleNamespace(call=call)
    failed = ex.run_one(tmp_path, 'dev', 'cd', '2026-07-13', 'gpt', None, client)
    assert failed['status'] == 'MODEL_FAILURE'
    assert failed['forecasts'] == {} and failed['decisions'] == {}
    assert failed['validation_codes'] == ['CITATION_NOT_IN_SNAPSHOT'] * 2
    assert ex.run_one(tmp_path, 'dev', 'cd', '2026-07-13', 'gpt', None, client) == failed
    assert len(calls) == 2
    ex.run_one(tmp_path, 'dev', 'cd', '2026-07-14', 'gpt', None, client)
    assert len(calls) == 3 and ex.complete(tmp_path, 'dev', 'gpt')
    failures = [json.loads(l) for l in (tmp_path / spec.STATE_DIR / 'failures.jsonl').read_text().splitlines()]
    assert [r['code'] for r in failures] == ['CITATION_NOT_IN_SNAPSHOT'] * 2
    assert [r['attempt'] for r in failures] == [0, 1]
    rates = ex.failure_summary(tmp_path, period='dev')['gpt']
    assert rates['model_failure_cells'] == 1 and rates['resolved_cells'] == 2
    assert rates['model_failure_rate'] == .5


def test_anonymized_headline_times_validate_without_iso_parse_or_date_leak():
    ctx = context('2026-07-01')
    ctx['session'] = 'DAY+0'
    ctx['decision_time'] = 'DAY+0 12:00:00 UTC'
    ctx['headlines']['AAPL']['items'] = [{'published_at': 'DAY+0 11:55:08 UTC',
                                        'url': 'https://news.invalid/1'}]
    output = analyst_output(ctx)
    # This all-abstain answer reproduced INVALID_OR_MISSING_INPUT on July 1.
    schemas.analyst(output, ctx)
    output['views'][0].update(view='UP', catalysts=[dict(ctx['headlines']['AAPL']['items'][0], fact='synthetic')])
    schema = schemas.analyst_schema(ctx)
    envelope = json.dumps({'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': output})
    decoded, _ = models.decode(envelope, '', 0, 'opus', schema)
    schemas.analyst(decoded, ctx)
    assert 'DAY' in run.prompt('analyst', ctx)
    decoded['views'][0]['catalysts'][0]['published_at'] = 'DAY+0 11:55:09 UTC'
    with pytest.raises(ReplayError, match='CITATION_NOT_IN_SNAPSHOT'):
        schemas.analyst(decoded, ctx)
    with pytest.raises(ReplayError, match='MODEL_SCHEMA_INVALID'):
        schemas.validate(schemas.ANALYST_SCHEMA, output)


def test_original_anonymous_abstention_is_reused_without_dispatch(tmp_path, monkeypatch):
    ctx = context('2026-07-01')
    ctx['decision_time'] = 'DAY+0 12:00:00 UTC'
    ctx['headlines']['AAPL']['items'] = [{'published_at': 'DAY+0 11:55:08 UTC', 'url': 'https://news.invalid/1'}]
    original_text = run.prompt('analyst', ctx, schema=schemas.ANALYST_SCHEMA)
    tag = 'old:2026-07-01:analyst-try0'
    raw = json.dumps({'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': analyst_output(ctx)})
    record = dict(family='opus', state='RECEIVED', tag=tag, at='2026-10-21T00:00:00Z',
                  raw=raw, err='', returncode=0, spec_hash='a' * 64,
                  prompt_sha256=hashlib.sha256(original_text.encode()).hexdigest())
    identity = sha256_canonical(['a' * 64, 'b' * 64, 'opus', original_text, schemas.ANALYST_SCHEMA, tag])
    source = tmp_path / 'duel-v11/transcripts' / (identity + '.json')
    source.parent.mkdir(parents=True)
    source.write_text(json.dumps(record))
    admission = {'sources': [dict(source_state_dir='duel-v11', source_spec_hash='a' * 64,
        source_code_hash='b' * 64, transcripts={identity: spec.file_hash(source)})]}
    monkeypatch.setattr(continuation, 'admission', lambda: admission)
    dispatches = []
    client = models.Models(SimpleNamespace(), models.DuelBudget(tmp_path), tmp_path / spec.STATE_DIR / 'transcripts',
                           runner=lambda *a, **k: dispatches.append(a))
    output, meta = client.call('opus', run.prompt('analyst', ctx), schemas.analyst_schema(ctx),
                              tag='new:2026-07-01:analyst-try0',
                              previous_request=(original_text, schemas.ANALYST_SCHEMA))
    schemas.analyst(output, ctx)
    assert not dispatches and meta['transcript'] == identity
    assert meta['schema_compatibility'] == 'ANONYMOUS_CODED_CLOCK_V1'
    assert json.loads(source.read_text()) == record


def test_decoder_validation_error_keeps_call_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(continuation, 'admission', lambda: {'sources': []})
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda _: 400)
    raw = json.dumps({'modelUsage': {'claude-opus-5-5': {}}, 'structured_output': {}})
    client = models.Models(grant, models.DuelBudget(tmp_path), tmp_path / spec.STATE_DIR / 'transcripts',
                           runner=lambda *a, **k: SimpleNamespace(stdout=raw, stderr='', returncode=0),
                           clock=lambda: models.instant('2026-10-23T01:00:00Z'))
    with pytest.raises(ReplayError) as caught:
        client.call('opus', 'text', schemas.NEWS_SCHEMA, tag='synthetic')
    assert caught.value.code == 'MODEL_SCHEMA_INVALID'
    assert len(caught.value.evidence['call']['transcript']) == 64
    assert caught.value.evidence['call']['prompt_sha256'] == hashlib.sha256(b'text').hexdigest()


@pytest.mark.parametrize('failed_family', ['both', 'gpt'])
def test_scorer_reports_terminal_failures_without_creating_forecasts(tmp_path, monkeypatch, ledger, failed_family):
    day = '2026-07-13'
    monkeypatch.setattr(ex, 'input_proof', lambda _: {})
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **k: None)
    monkeypatch.setattr(ex, 'GROUPS', ('cd',))
    monkeypatch.setattr(ex, 'planned', lambda *a: [day])
    monkeypatch.setattr(ex, 'planned_for', lambda *a: [day])
    monkeypatch.setattr(ex, 'meta', lambda *a: {})
    monkeypatch.setattr(ex, 'offline', lambda *a, **k: None)
    monkeypatch.setattr(score, 'frozen_primary_rate', lambda _: {'n': 0, 'successes': 0, 'p': .5})
    ctx, answer = opus_news(1, 1)
    answer['views'][0]['view'] = 'ABSTAIN'
    monkeypatch.setattr(run, 'shared_snapshot', lambda *a: {'snapshot_hash': 's' * 64, 'manifest': {}, 'context': ctx})
    from scripts.trading_lab.replay_bench import run as parent
    hashes = {}
    for family in ex.FAMILIES:
        # Both failures are terminal, with no model probability available to score.
        record = dict(status='MODEL_FAILURE', provenance=spec.PROVENANCE, calls={},
                      snapshot_hash='s' * 64, shown_context_hash=sha256_canonical(ctx),
                      forecasts={}, decisions={}, validation_codes=['CITATION_NOT_IN_SNAPSHOT'] * 2)
        if failed_family != 'both' and family != failed_family:
            record.update(status='VALID', forecasts={v: answer['forecasts'] for v in run.CD},
                          decisions={v: answer['views'] for v in run.CD})
        record['decision_hash'] = sha256_canonical(record)
        parent.write_json(ex.record_path(tmp_path, 'dev', 'cd', family, day), record)
        hashes[score.record_key(family, 'dev', 'cd', day)] = record['decision_hash']
    ledger.publish('replay-night', 'synthetic-failures', hashes)
    outcomes = SimpleNamespace(get=lambda *a: dict(rel=.01, cluster=day, available_at='2026-07-14T20:00:00Z'))
    result = score.score(tmp_path, 'dev', None, outcome_factory=lambda *a: outcomes)
    assert result['forecasts']['gpt'] == []
    assert len(result['forecasts']['opus']) == (0 if failed_family == 'both' else 4)
    assert result['failures']['gpt']['model_failure_rate'] == 1
    assert result['models']['gpt']['c_news_small_mid:2d']['model_failure']['model_failure_cells'] == 1
    if failed_family == 'gpt':
        assert result['failures']['opus']['model_failure_rate'] == 0
        assert result['counts']['opus']['cd:unpaired_model_failure'] == 1
    text = ex.result_path(tmp_path, 'dev').with_suffix('.md').read_text()
    assert '| Model | Failed cells / resolved cells | Rate |' in text
    assert '| gpt | 1 / 1 | 1.0 |' in text


def test_previous_forward_proof_checks_its_original_publication_protocol(tmp_path, monkeypatch, ledger):
    from datetime import timedelta
    from scripts.trading_lab.replay_bench.grant import iso
    day = '2026-10-22'
    at = forward.decision_time(day)
    previous = json.loads((spec.ROOT / 'docs/artifacts/replay_duel_test_freeze_v11.json').read_text())
    sealed = dict(spec_hash=previous['spec_hash'], code_hash=previous['code_hash'],
                  snapshot_hash='a' * 64, decision_time=iso(at))
    records = {}
    for family in ex.FAMILIES:
        record = dict(family=family, model=spec.spec()['models']['identities'][family],
                      snapshot_hash=sealed['snapshot_hash'], provenance=forward.PROVENANCE,
                      configuration_hash=previous['forward_config_hashes'][family],
                      decision_time=iso(at), stored_at=iso(at + timedelta(minutes=1)),
                      call={'called_at': iso(at)})
        record['decision_hash'] = sha256_canonical(record)
        records[family] = record
    ledger.clock = lambda: at + timedelta(minutes=2)
    # Simulate publication before this revision, under the original immutable spec.
    with monkeypatch.context() as old:
        old.setattr(spec, 'SPEC_HASH', previous['spec_hash'])
        ledger.publish('forward', day, dict(snapshot=sealed['snapshot_hash'],
                       **{f: r['decision_hash'] for f, r in records.items()}))
    for family, record in records.items():
        forward.check_record(tmp_path, day, family, record, sealed, {})

"""Synthetic-only contracts, checkpoint recovery, full abstention scoring and paired inference."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench import marketdata as md, snapshot as sn, run as parent
from scripts.trading_lab.replay_bench.grant import ReplayError, instant, iso
from scripts.trading_lab.replay_bench.storage import atomic_json
from scripts.trading_lab.replay_duel import audit, cli, experiment as ex, metrics, models, run, schemas, score, spec

AT = instant('2026-10-19T01:00:00Z')


@pytest.fixture(autouse=True)
def fast_code_hash(monkeypatch):
    monkeypatch.setattr(spec, 'code_hash', lambda: 'c' * 64)


def context(day, news=False):
    previous = md.previous_close(parent.decision_time(day))
    if news:
        return {'schema': 'replay-news-context-v1', 'session': day, 'decision_time': day + 'T12:00:00Z',
                'news_window_start': iso(previous), 'candidates': [{'symbol': 'SMLL',
                'prices': {'last_price_at': iso(previous)}, 'headlines': [{'news_id': 1, 'title': 'SMLL cuts guidance',
                'published_at': day + 'T11:00:00Z', 'url': 'https://news.invalid/1'}]}]}
    return {'schema': 'replay-context-v1', 'session': day, 'decision_time': day + 'T12:00:00Z', 'universe': ['AAPL'],
            'prices': {'AAPL': {'asset': 'AAPL', 'state': 'AVAILABLE', 'last_price_at': iso(previous),
                               'recent_closes': [100.0, 101.0], 'returns': {'20': .01}}},
            'headlines': {'AAPL': {'state': 'AVAILABLE', 'items': []}}, 'exclusions': [], 'limitations': []}


def analyst_output(ctx, p=.8):
    return {'regime': ['synthetic'], 'views': [dict(asset=a, horizon=h, view='ABSTAIN', p_outperform=p,
        confidence_reason='synthetic', catalysts=[], priced_in_assessment='', second_order='', counter_thesis='',
        falsifier='', event_risk=[]) for a in ctx['universe'] for h in ('1d', '5d')]}


def test_pinned_spec_and_model_only_config_difference():
    value = spec.spec()
    assert len(value['variants']) == 4
    assert value['models']['identities'] == {'opus': 'claude-opus-5-5', 'gpt': 'gpt-6.1-sol'}
    assert value['models']['tools'] == []
    for group in ('ab', 'cd'):
        a, b = spec.config_payload('test', group, 'opus'), spec.config_payload('test', group, 'gpt')
        assert {k: v for k, v in a.items() if k not in ('family', 'model')} == {k: v for k, v in b.items() if k not in ('family', 'model')}
        assert sha256_canonical(a) != sha256_canonical(b)
    parent_spec = json.loads((spec.ROOT / 'docs/artifacts/replay_bench_spec_v4.json').read_text())
    for key in ('periods', 'costs', 'information_snapshot'):
        assert value[key] == parent_spec[key]


@pytest.mark.parametrize('p', [0, .2, .5, .9, 1])
def test_abstention_keeps_probability_and_exact_coverage(p):
    ctx = context('2026-07-15')
    output = analyst_output(ctx, p)
    schemas.analyst(output, ctx)
    output['views'].pop()
    with pytest.raises(ReplayError, match='FORECAST_COVERAGE_INVALID'):
        schemas.analyst(output, ctx)


def test_direction_reviewer_and_news_coverage():
    ctx = context('2026-07-15')
    a = analyst_output(ctx)
    a['views'][0]['view'] = 'DOWN'
    with pytest.raises(ReplayError, match='DIRECTION_INVALID'):
        schemas.analyst(a, ctx)
    a = analyst_output(ctx)
    review = {'verdicts': [dict(asset=v['asset'], horizon=v['horizon'], verdict='REJECT', p_outperform=.7, note='')
                           for v in a['views']]}
    schemas.reviewer(review, a)
    review['verdicts'][0]['p_outperform'] = .9
    with pytest.raises(ReplayError, match='REVIEW_RAISED_PROBABILITY'):
        schemas.reviewer(review, a)
    ctx = context('2026-07-15', True)
    out = news_output(ctx)
    schemas.news(out, ctx)
    out['forecasts'][0]['asset'] = 'WRONG'
    with pytest.raises(ReplayError, match='FORECAST_COVERAGE_INVALID'):
        schemas.news(out, ctx)


def news_output(ctx):
    return {'views': [dict(symbol=c['symbol'], view='ABSTAIN', p_outperform=.9, holding_sessions=3,
                          event_code='guidance_cut', news_ids=[1], reason='', counter_thesis='', falsifier='')
                      for c in ctx['candidates']],
            'forecasts': [dict(asset=c['symbol'], horizon=h, p_outperform=.7)
                          for c in ctx['candidates'] for h in ('2d', '10d')]}


def test_commands_are_no_tool_subscription_only():
    args = models.argv('gpt', schemas.ANALYST_SCHEMA, Path('/synthetic/schema.json'))
    assert args[:3] == ['codex', '--no-daemon', 'exec'] and '--search' not in args
    assert 'web_search="disabled"' in args and 'forced_login_method="chatgpt"' in args
    assert args[args.index('--model') + 1] == 'gpt-6.1-sol'
    for feature in ('shell_tool', 'code_mode_host', 'apps', 'plugins', 'multi_agent', 'hooks'):
        assert args[args.index(feature) - 1] == '--disable'
    opus = models.argv('opus', {}, Path('ignored'))
    assert opus[opus.index('--tools') + 1] == ''
    assert opus[opus.index('--model') + 1] == 'claude-opus-5-5'


def fake_result(args, **kwargs):
    assert list(Path(kwargs['cwd']).iterdir()) == []
    assert 'OPENAI_API_KEY' not in kwargs['env']
    if args[0] == 'codex':
        events = [{'type': 'thread.started', 'thread_id': 'synthetic'}, {'type': 'turn.started'},
                  {'type': 'item.completed', 'item': {'type': 'agent_message', 'text': '{"ok":true}'}},
                  {'type': 'turn.completed', 'usage': {}}]
        raw = '\n'.join(json.dumps(e) for e in events)
    else:
        raw = json.dumps({'subtype': 'success', 'structured_output': {'ok': True},
                          'modelUsage': {'claude-opus-5-5': {}}})
    return subprocess.CompletedProcess(args, 0, raw, '')


def client(root, runner=fake_result, at=AT):
    grant = SimpleNamespace(check=lambda: None, model_limit=lambda family: 150)
    return models.Models(grant, models.DuelBudget(root), root / 'transcripts', runner=runner, clock=lambda: at)


@pytest.mark.parametrize('family', ['gpt', 'opus'])
def test_caps_checkpoint_next_night_and_unknown_intent(tmp_path, family):
    seen = []
    def runner(*a, **kw):
        seen.append(a)
        return fake_result(*a, **kw)
    c = client(tmp_path, runner)
    atomic_json(c.budget.path, {'2026-10-19': {'model:' + family: 59}})
    first = c.call(family, 'q', {}, tag='60')
    assert c.call(family, 'q', {}, tag='60') == first and len(seen) == 1
    with pytest.raises(ReplayError, match='BUDGET_EXHAUSTED'):
        c.call(family, 'q', {}, tag='61')
    c.clock = lambda: AT + timedelta(days=1)
    c.call(family, 'q', {}, tag='61')
    assert len(seen) == 2
    file = next((tmp_path / 'transcripts').glob('*.json'))
    row = json.loads(file.read_text())
    row['state'] = 'INTENT'
    atomic_json(file, row)
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call(family, 'q', {}, tag=row['tag'])


def test_atomic_caps_are_separate_and_honor_legacy_usage(tmp_path):
    b = models.DuelBudget(tmp_path)
    atomic_json(b.path, {'2026-10-19': {'model:opus': 59, 'model:gpt': 59, 'model:sonnet': 36}})
    def reserve(family):
        try:
            b.reserve_model(family, 150, AT)
            return family
        except ReplayError as error:
            assert error.code == 'BUDGET_EXHAUSTED'
    with ThreadPoolExecutor(8) as pool:
        values = list(pool.map(reserve, ['opus', 'gpt'] * 4))
    assert values.count('opus') == values.count('gpt') == 1
    assert b.used('model:opus', AT) == b.used('model:gpt', AT) == 60


@pytest.mark.parametrize('family', ['opus', 'gpt'])
def test_quota_stops_only_its_subscription_and_no_blind_timeout_retry(tmp_path, family):
    c = client(tmp_path, lambda *a, **kw: subprocess.CompletedProcess(a, 1, '', 'usage limit'))
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        c.call(family, 'q', {}, tag='x')
    with pytest.raises(ReplayError, match='MODEL_USAGE_LIMIT'):
        c.call(family, 'q', {}, tag='different')
    other = 'gpt' if family == 'opus' else 'opus'
    c.runner = fake_result
    c.call(other, 'q', {}, tag='other')
    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired(a, 2)
    c.runner = timeout
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call(other, 'q', {}, tag='timeout')
    c.runner = lambda *a, **kw: pytest.fail('retried an unknown outcome')
    with pytest.raises(ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call(other, 'q', {}, tag='timeout')


@pytest.mark.parametrize('hour,minute', [(0,29), (5,30), (23,59)])
def test_outside_night_does_not_dispatch(tmp_path, hour, minute):
    c = client(tmp_path, lambda *a, **kw: pytest.fail('outside window'), AT.replace(hour=hour, minute=minute))
    with pytest.raises(ReplayError, match='OUTSIDE_NIGHT_WINDOW'):
        c.call('gpt', 'q', {}, tag='x')
    assert not c.budget.path.exists()


@pytest.mark.parametrize('kind', ['web_search', 'mcp_tool_call', 'command_execution', 'file_change', 'function_call'])
def test_tool_use_is_rejected_even_when_result_also_says_quota(kind):
    events = [{'type': 'item.completed', 'item': {'type': kind}}, {'type': 'error', 'message': 'usage limit'}]
    with pytest.raises(ReplayError, match='TAINTED_RUN'):
        models.decode('\n'.join(json.dumps(e) for e in events), '', 1, 'gpt', {})


def test_identity_and_strict_envelope_validation():
    raw = json.dumps({'structured_output': {}, 'modelUsage': {'claude-sonnet-5-5': {}}})
    with pytest.raises(ReplayError, match='MODEL_IDENTITY_CHANGED'):
        models.decode(raw, '', 0, 'opus', {})
    raw = '\n'.join(json.dumps(e) for e in [{'type': 'thread.started', 'model': 'wrong'}, {'type': 'turn.completed'}])
    with pytest.raises(ReplayError, match='MODEL_IDENTITY_CHANGED'):
        models.decode(raw, '', 0, 'gpt', {})


def proof(root):
    file = root / 'evidence.json'
    atomic_json(file, {'source': 'synthetic point-in-time evidence'})
    cache = root / 'cache' / ('a' * 2) / ('a' * 64 + '.json.gz')
    parent.write_json(cache, {'body': {'news': [], 'next_page_token': None}, 'received_at': iso(AT), 'body_sha256': 'b' * 64})
    value = {'schema': 'replay-duel-input-proof-v1', 'spec_hash': spec.SPEC_HASH,
             'unrevised_prices_proven': True, 'historical_universe_and_identity_proven': True,
             'evidence': [{'file': file.name, 'sha256': spec.file_hash(file)}],
             'prediction_cache': {'a' * 64: spec.file_hash(cache)}, 'outcome_cache': {'a' * 64: spec.file_hash(cache)}}
    atomic_json(root / 'duel_input_proof.json', value)
    return value


def test_missing_or_changed_input_evidence_blocks_before_transport(tmp_path, monkeypatch):
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **kw: None)
    with pytest.raises(ReplayError, match='POINT_IN_TIME_INPUT_PROOF_MISSING'):
        ex.run_one(tmp_path, 'dev', 'ab', '2026-07-15', 'gpt', object(), object())
    proof(tmp_path)
    ex.input_proof(tmp_path)
    (tmp_path / 'evidence.json').write_text('{}')
    with pytest.raises(ReplayError, match='INPUT_EVIDENCE_HASH_MISMATCH'):
        ex.input_proof(tmp_path)


def test_scores_and_clipping_include_abstentions():
    rows = [dict(p=p, outperform=y, session='2026-07-15', cluster='tech') for p,y in [(1, True), (0, False), (.8, True)]]
    out = metrics.proper(rows, [.5] * 3)
    assert out['n_forecasts'] == 3 and out['brier'] == pytest.approx(.04 / 3)
    assert out['log_loss'] < .1 and out['brier_half'] == .25
    assert out['probability_ess_cluster_day_kish'] == 1
    zero = metrics.trading([], ['2026-07-15'], {}, ['ABSTAIN'])
    assert zero['abstention_rate'] == 1 and zero['n_scored_trades'] == 0 and zero['max_drawdown'] == 0


def test_paired_block_test_and_holm_do_not_count_stocks_as_independent():
    days = parent.sessions('2026-07-01', '2026-08-31')[:28]
    left = [dict(panel='v:1d', session=d, asset='S'+str(i), horizon='1d', p=.9, outperform=True) for d in days for i in range(50)]
    right = [dict(r, p=.5) for r in left]
    result = metrics.paired(left, right, days)['v:1d:brier']
    assert result['n_pairs'] == 1400 and result['n_blocks'] == 3
    assert result['p_two_sided'] == .25 and result['p_holm'] == .5
    assert result['mean_opus_minus_gpt'] < 0
    with pytest.raises(ValueError, match='coverage'):
        metrics.paired(left, right[:-1], days)
    identical = metrics.paired(left, left, days)['v:1d:brier']
    assert identical['p_two_sided'] == 1


def test_dev_base_rate_waits_for_outcome_exit_and_test_rate_is_frozen():
    rows = [dict(panel='v:5d', session=d, available_at=end, outperform=True) for d,end in
            [('2026-07-01','2026-07-08T20:00:00Z'), ('2026-07-02','2026-07-09T20:00:00Z'),
             ('2026-07-09','2026-07-16T20:00:00Z')]]
    assert score.base_probabilities(rows, 'dev', {}) == [.5, .5, 2/3]
    assert score.base_probabilities(rows, 'test', {'v:5d': {'p': .6}}) == [.6] * 3


class SyntheticClient:
    def __init__(self):
        self.calls = []
    def call(self, family, text, schema, *, tag):
        self.calls.append((family, text, schema, tag))
        ctx = json.loads(text.split('\nCONTEXT_DATA:\n')[1].split('\nANALYST_DATA:\n')[0])
        if 'verdicts' in schema['properties']:
            original = json.loads(text.split('\nANALYST_DATA:\n')[1])
            output = {'verdicts': [dict(asset=v['asset'], horizon=v['horizon'], verdict='REJECT', p_outperform=.7, note='')
                                   for v in original['views']]}
        elif 'forecasts' in schema['properties']:
            output = news_output(ctx)
        else:
            output = analyst_output(ctx)
        import hashlib
        return output, {'prompt_sha256': hashlib.sha256(text.encode()).hexdigest()}


class SyntheticOutcomes:
    def __init__(self, *a):
        pass
    def get(self, group, day, asset, h):
        following = md.sessions(day, (parent.decision_time(day) + timedelta(days=21)).date().isoformat())
        return {'rel': .02, 'cost': .003, 'vol': .01, 'cluster': 'tech', 'gross_path': {day: .02},
                'available_at': iso(following[h - 1].close_at)}


def integration_setup(root, monkeypatch):
    proof(root)
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **kw: None)
    monkeypatch.setattr(ex, 'planned', lambda period, group: ['2026-07-15', '2026-07-16'] if group != 'anon' else ['2026-07-15'])
    def snapshot(root, period, group, day, transport):
        return sn.seal(context(day, group == 'cd'), {'news': []}, parent.decision_time(day))
    monkeypatch.setattr(run, 'shared_snapshot', snapshot)
    monkeypatch.setattr(parent, 'apply_primary_filter', lambda *a: ([], []))
    c = SyntheticClient()
    for group in ex.GROUPS:
        for day in ex.planned('dev', group):
            for family in ex.FAMILIES:
                ex.run_one(root, 'dev', group, day, family, SimpleNamespace(), c)
    return c


def test_full_paired_pipeline_all_abstain_still_scores_every_probability(tmp_path, monkeypatch):
    c = integration_setup(tmp_path, monkeypatch)
    result = score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=SyntheticOutcomes)
    assert result['status'] == 'SCORED' and set(result['models']) == {'opus', 'gpt'}
    for family in ex.FAMILIES:
        named = result['models'][family]['b_v1_no_reviewer:1d']
        assert named['n_forecasts'] == 2 and named['n_scored_trades'] == 0 and named['abstention_rate'] == 1
        assert named['brier'] == pytest.approx(.04)
        for variant in run.CD:
            assert result['models'][family][variant + ':2d']['n_forecasts'] == 2
    assert result['paired_tests']['b_v1_no_reviewer:1d:brier']['p_two_sided'] == 1
    assert (ex.result_path(tmp_path, 'dev').with_suffix('.md')).is_file()
    count = len(c.calls)
    ex.run_one(tmp_path, 'dev', 'ab', '2026-07-15', 'gpt', SimpleNamespace(), c)
    assert len(c.calls) == count
    # A saved scored result must be reused without fetching outcomes a second time.
    again = score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=lambda *a: pytest.fail('rescored'))
    assert again == result


def test_non_abstaining_signed_pnl_includes_both_side_costs():
    outcome = {'rel': .02, 'cost': .003, 'vol': .01, 'cluster': 'tech',
               'gross_path': {'2026-07-15': .03, '2026-07-16': -.01}}
    result, pnl = score.signed_outcome(outcome, 'UP', 2)
    assert sum(pnl.values()) == pytest.approx(.017)
    assert result['R'] == pytest.approx(.017 / (.01 * 2**.5))
    result, pnl = score.signed_outcome(outcome, 'DOWN', 2)
    assert sum(pnl.values()) == pytest.approx(-.023) and not result['hit']


def test_crypto_is_relative_to_synchronized_spy_and_holds_close_on_weekend():
    class Anchors:
        def crypto(self, at):
            price = 100 if at.day == 17 else (105 if at.hour == 0 else 110)
            return {'SOL-USD': {'o': price}}
        def stock(self, day, at, symbols):
            return {'SPY': {'p': 100 if day == '2026-07-17' else 108}}
    spy_rows = [{'close_at': instant('2026-07-17T20:00:00Z'), 'close': 102}]
    rel, path = score.crypto_outcome('2026-07-17', 'SOL-USD', Anchors(), spy_rows, {})
    assert rel == pytest.approx(.08) and sum(path.values()) == pytest.approx(.08)
    assert score.spy_mark(instant('2026-07-18T13:31:00Z'), spy_rows, Anchors()) == 102


def test_report_never_claims_done_for_only_one_scored_model(tmp_path, monkeypatch):
    target = tmp_path / 'report.md'
    monkeypatch.setenv('REPLAY_DUEL_REPORT', str(target))
    path = ex.result_path(tmp_path, 'test')
    value = dict(status='SCORED', spec_hash=spec.SPEC_HASH, code_hash=spec.code_hash(), models={'opus': {}}, paired_tests={'x': {}})
    atomic_json(path, value)
    cli.progress_report(tmp_path, {'status': 'DONE'})
    assert target.read_text().startswith('STATUS: BLOCKED')
    value['models']['gpt'] = {}
    atomic_json(path, value)
    cli.progress_report(tmp_path, {'status': 'DONE'})
    assert target.read_text().startswith('STATUS: DONE')


def test_code_or_input_binding_cannot_reuse_decisions(tmp_path, monkeypatch):
    integration_setup(tmp_path, monkeypatch)
    p = ex.record_path(tmp_path, 'dev', 'ab', 'gpt', '2026-07-15')
    value = parent.read_json(p)
    value['decisions'][run.AB[1]][0]['p_outperform'] = .1
    parent.write_json(p, value)
    with pytest.raises(ReplayError, match='DECISION_HASH_MISMATCH'):
        score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=SyntheticOutcomes)


def test_mismatched_snapshot_or_prompt_refuses_pairing(tmp_path, monkeypatch):
    integration_setup(tmp_path, monkeypatch)
    p = ex.record_path(tmp_path, 'dev', 'ab', 'gpt', '2026-07-15')
    value = parent.read_json(p)
    value['calls']['analyst']['prompt_sha256'] = 'different'
    value['decision_hash'] = sha256_canonical({k: v for k, v in value.items() if k != 'decision_hash'})
    parent.write_json(p, value)
    with pytest.raises(ReplayError, match='PAIRED_PROMPT_MISMATCH'):
        score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=SyntheticOutcomes)


def test_test_claim_is_once_per_model_and_bound_to_dev_result(tmp_path, monkeypatch):
    file = tmp_path / 'freeze.json'
    atomic_json(file, {'synthetic': 'freeze'})
    monkeypatch.setattr(ex, 'FREEZE', file)
    monkeypatch.setattr(ex, 'ready_test', lambda root: 'dev-score-a')
    monkeypatch.setattr(ex, 'seal_payload', lambda root: {'synthetic': 'seal'})
    monkeypatch.setattr(ex, 'head', lambda: 'synthetic-commit')
    ex.claim_test(tmp_path, 'opus')
    ex.claim_test(tmp_path, 'gpt')
    marker = tmp_path / spec.STATE_DIR / 'test_started_gpt.json'
    original = marker.read_bytes()
    ex.claim_test(tmp_path, 'gpt')
    assert marker.read_bytes() == original
    monkeypatch.setattr(ex, 'ready_test', lambda root: 'other-dev-result')
    with pytest.raises(ReplayError, match='TEST_ALREADY_STARTED_WITH_OTHER_CONFIG'):
        ex.claim_test(tmp_path, 'gpt')


def test_test_needs_both_scored_dev_periods(tmp_path, monkeypatch):
    proof(tmp_path)
    monkeypatch.setattr(ex, 'check_freeze', lambda *a, **kw: None)
    monkeypatch.setattr(ex, 'complete', lambda root, period, family: family == 'opus')
    atomic_json(ex.result_path(tmp_path, 'dev'), dict(status='SCORED', spec_hash=spec.SPEC_HASH,
        code_hash=spec.code_hash(), input_proof_hash=sha256_canonical(ex.input_proof(tmp_path)), models={'opus': {}}))
    with pytest.raises(ReplayError, match='BOTH_DEV_PERIODS_MUST_BE_SCORED'):
        ex.ready_test(tmp_path)


def test_genuine_shared_snapshot_rejects_changed_input_proof(tmp_path, monkeypatch):
    proof(tmp_path)
    day = '2026-07-15'
    monkeypatch.setattr(parent, 'Universe', lambda *a, **kw: SimpleNamespace(bars={}, news=[]))
    monkeypatch.setattr(sn, 'context_a', lambda *a: (context(day), {'news': []}))
    first = run.shared_snapshot(tmp_path, 'dev', 'ab', day, object())
    assert run.shared_snapshot(tmp_path, 'dev', 'ab', day, object()) == first
    value = ex.input_proof(tmp_path)
    value['extra_evidence_note'] = 'another manifest'
    atomic_json(tmp_path / 'duel_input_proof.json', value)
    with pytest.raises(ReplayError, match='SNAPSHOT_HASH_MISMATCH'):
        run.shared_snapshot(tmp_path, 'dev', 'ab', day, object())


def test_empty_news_candidates_keep_all_declared_panels(tmp_path, monkeypatch):
    integration_setup(tmp_path, monkeypatch)
    for family in ex.FAMILIES:
        for day in ex.planned('dev', 'cd'):
            p = ex.record_path(tmp_path, 'dev', 'cd', family, day)
            r = parent.read_json(p)
            r['decisions'] = {v: [] for v in run.CD}
            r['forecasts'] = {v: [] for v in run.CD}
            r['decision_hash'] = sha256_canonical({k: v for k, v in r.items() if k != 'decision_hash'})
            parent.write_json(p, r)
    original = run.shared_snapshot
    def empty(root, period, group, day, transport):
        value = original(root, period, group, day, transport)
        if group == 'cd':
            value['context']['candidates'] = []
            value = sn.seal(value['context'], value['manifest'], parent.decision_time(day))
        return value
    monkeypatch.setattr(run, 'shared_snapshot', empty)
    for family in ex.FAMILIES:
        for day in ex.planned('dev', 'cd'):
            p = ex.record_path(tmp_path, 'dev', 'cd', family, day)
            r = parent.read_json(p)
            sealed = empty(tmp_path, 'dev', 'cd', day, object())
            r['snapshot_hash'] = sealed['snapshot_hash']
            r['shown_context_hash'] = sealed['context_hash']
            r['decision_hash'] = sha256_canonical({k: v for k, v in r.items() if k != 'decision_hash'})
            parent.write_json(p, r)
    value = score.score(tmp_path, 'dev', SimpleNamespace(), outcome_factory=SyntheticOutcomes)
    assert value['head_to_head']['c_news_small_mid:2d']['gpt']['n_forecasts'] == 0


def test_offline_transport_cannot_make_official_request(tmp_path):
    proof(tmp_path)
    from scripts.trading_lab.replay_bench.grant import Transport
    grant = SimpleNamespace(payload={'scope': {}}, check=lambda: None)
    t = Transport(grant, models.DuelBudget(tmp_path), tmp_path / 'cache',
                  getter=lambda *a: pytest.fail('network invoked'), offline=True)
    t = ex.offline(tmp_path, t)
    with pytest.raises(ReplayError, match='PREDICTION_CACHE_NOT_SEALED'):
        t.get(md.ALPACA, '/v1beta1/news', {'start': '2026-07-15T00:00:00Z'})
    with pytest.raises(ReplayError, match='PROTECTED_REQUEST'):
        md.crypto_bars(t, ['ETH-USD'], '2026-09-01T00:00:00Z', '2026-09-02T00:00:00Z')


def test_future_snapshot_item_is_refused():
    ctx = context('2026-07-15', True)
    ctx['candidates'][0]['headlines'][0]['published_at'] = '2026-07-15T12:00:00Z'
    with pytest.raises(ReplayError, match='FUTURE_NEWS'):
        sn.seal(ctx, {'news': []}, parent.decision_time('2026-07-15'))

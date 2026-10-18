"""Safety boundaries: no real model, network, timers or private inputs in these tests."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

import pytest

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.replay_bench import audit, experiment, grant as gr, labels, models, probe, run, score, spec
from scripts.trading_lab.replay_bench.storage import atomic_json, owned
from test_replay_bench import FakeGrant, daily_bars

AT = gr.instant('2026-10-19T01:00:00Z')


def successful(args, **kwargs):
    event = {'subtype': 'success', 'structured_output': {'ok': True},
             'modelUsage': {models.MODELS['opus']: {}}, 'permission_denials': []}
    return subprocess.CompletedProcess(args, 0, stdout=json.dumps(event), stderr='')


def client(tmp_path, runner=successful, at=AT):
    return models.Claude(FakeGrant(), gr.Budget(tmp_path), tmp_path / 'transcripts',
                         runner=runner, clock=lambda: at)


@pytest.mark.parametrize('hour,minute', [(0, 29), (5, 30), (23, 59)])
def test_no_calls_outside_window(tmp_path, hour, minute):
    seen = []
    c = client(tmp_path, lambda *a, **kw: seen.append(a), AT.replace(hour=hour, minute=minute))
    with pytest.raises(gr.ReplayError, match='OUTSIDE_NIGHT_WINDOW'):
        c.call('opus', 'q', {}, tag='x')
    assert not seen and gr.Budget(tmp_path).used('model:opus', AT) == 0


def test_call_timeout_cannot_cross_end_of_night(tmp_path):
    seen = []
    def runner(args, **kwargs):
        seen.append(kwargs['timeout'])
        return successful(args, **kwargs)
    c = client(tmp_path, runner, AT.replace(hour=5, minute=29, second=50))
    c.call('opus', 'q', {}, tag='x')
    assert seen == [9]


def test_shared_atomic_night_cap_includes_legacy_models(tmp_path):
    b = gr.Budget(tmp_path)
    atomic_json(b.path, {'2026-10-19': {'model:opus': 12, 'model:sonnet': 46}})
    def reserve(_):
        try:
            b.reserve_model('opus', 150, AT)
            return True
        except gr.ReplayError as error:
            assert error.code == 'BUDGET_EXHAUSTED'
            return False
    with ThreadPoolExecutor(8) as workers:
        assert sum(workers.map(reserve, range(8))) == 2
    assert b.used('model:opus', AT) == 14


def test_usage_limit_stops_before_next_day_is_submitted():
    seen = []
    def fn(day):
        seen.append(day)
        raise gr.ReplayError('MODEL_USAGE_LIMIT')
    result, stop = run.run_days(['a', 'b', 'c'], fn, 1)
    assert seen == ['a'] and stop == 'MODEL_USAGE_LIMIT' and list(result) == ['a']


def test_saved_response_is_not_called_again_and_unknown_intent_blocks(tmp_path):
    seen = []
    def runner(args, **kwargs):
        seen.append(args)
        return successful(args, **kwargs)
    c = client(tmp_path, runner)
    first = c.call('opus', 'q', {}, tag='x')
    assert c.call('opus', 'q', {}, tag='x') == first and len(seen) == 1
    path = next((tmp_path / 'transcripts').glob('*.json'))
    atomic_json(path, {'state': 'INTENT'})
    with pytest.raises(gr.ReplayError, match='CALL_OUTCOME_UNKNOWN'):
        c.call('opus', 'q', {}, tag='x')
    assert len(seen) == 1


def test_timeout_is_not_blindly_retried(tmp_path):
    seen = []
    def runner(args, **kwargs):
        seen.append(args)
        raise subprocess.TimeoutExpired(args, kwargs['timeout'])
    c = client(tmp_path, runner)
    for _ in range(2):
        with pytest.raises(gr.ReplayError, match='CALL_OUTCOME_UNKNOWN'):
            c.call('opus', 'q', {}, tag='x')
    assert len(seen) == 1


def test_structured_quota_response_also_stops(tmp_path):
    def runner(args, **kwargs):
        event = {'is_error': True, 'api_error_status': 429, 'structured_output': {'ok': True}}
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(event), stderr='')
    c = client(tmp_path, runner)
    with pytest.raises(gr.ReplayError, match='MODEL_USAGE_LIMIT'):
        c.call('opus', 'q', {}, tag='x')
    assert c.budget.stopped(AT) == 'MODEL_USAGE_LIMIT'


def test_alias_drift_invalidates_the_probe_binding(tmp_path):
    def runner(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps({
            'structured_output': {'ok': True}, 'modelUsage': {'changed-model': {}}}), stderr='')
    with pytest.raises(gr.ReplayError, match='MODEL_IDENTITY_CHANGED'):
        client(tmp_path, runner).call('opus', 'q', {}, tag='x')


def test_schema_is_validated_locally_not_just_sent_to_cli(tmp_path):
    with pytest.raises(gr.ReplayError, match='MODEL_SCHEMA_INVALID'):
        client(tmp_path).call('opus', 'q', {'type': 'object', 'required': ['views']}, tag='x')


def test_exact_probe_event_coverage_and_grades_are_required(tmp_path):
    event = {'id': 'event', 'month': '2026-10', 'answer_key_all_of_any': [['synthetic fact']]}
    rows = [{'id': 'event', 'family': family, 'model': models.MODELS[family], 'reported': [models.MODELS[family]],
             'month': '2026-10', 'knows': True, 'correct': True, 'answer': 'synthetic fact'}
            for family in ('opus', 'sonnet')]
    path = tmp_path / 'probe.jsonl'
    def save(values):
        path.write_text(''.join(json.dumps(r) + '\n' for r in values))
    save(rows)
    assert len(audit.checked_rows(path, [event], forced=False)) == 2
    save(rows + rows[:1])
    with pytest.raises(gr.ReplayError, match='PROBE_INCOMPLETE_OR_DUPLICATE'):
        audit.checked_rows(path, [event], forced=False)
    save([dict(r, answer='incorrect') for r in rows])
    with pytest.raises(gr.ReplayError, match='PROBE_GRADE_MISMATCH'):
        audit.checked_rows(path, [event], forced=False)


def test_original_forced_threshold_cannot_be_relaxed_after_results():
    rows = [{'family': 'sonnet', 'month': month, 'id': f'{month}{i}', 'knows': False, 'correct': i < 3}
            for month in ('2026-08', '2026-10') for i in range(6)]
    summary = probe.summarize(rows, json.loads((spec.ROOT / 'docs/artifacts/replay_bench_spec_v3.json').read_text())['probe'])
    assert summary['sonnet']['replay_floor'] == '2026-11-01'
    assert not hasattr(probe, 'forced_floor')


def test_no_eligible_dates_and_no_proof_prevent_dispatch(tmp_path, monkeypatch):
    ev = {'floor': {'sonnet': '2026-11-01', 'opus': '2026-11-01'}}
    monkeypatch.setattr(audit, 'evidence', lambda root: ev)
    called = []
    with pytest.raises(gr.ReplayError, match='NO_POST_CUTOFF_DATES'):
        experiment.run_group(tmp_path, 'test', 'ab', called, called)
    assert not called
    with pytest.raises(gr.ReplayError, match='POINT_IN_TIME_INPUT_PROOF_MISSING'):
        experiment.input_proof(tmp_path)


def test_opus_full_period_and_anonymized_subset_are_declared(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'evidence', lambda root: {'floor': {'opus': '2026-07-01'}})
    days, start = experiment.planned(tmp_path, 'dev', 'ab')
    assert start == '2026-07-01' and days == run.sessions('2026-07-01', '2026-08-31')
    assert experiment.planned(tmp_path, 'dev', 'cd')[0] == days
    assert experiment.planned(tmp_path, 'dev', 'anon')[0] == days[::4]
    assert experiment.planned(tmp_path, 'test', 'ab')[0] == run.sessions('2026-09-01', '2026-10-09')
    assert all(families == ('opus',) for families in experiment.GROUP_MODELS.values())
    with pytest.raises(gr.ReplayError, match='AUDIT_IS_DEV_ONLY'):
        experiment.planned(tmp_path, 'test', 'anon')
    with pytest.raises(gr.ReplayError, match='GROUP_NOT_ALLOWED'):
        experiment.planned(tmp_path, 'dev', 'opus')


def test_trial_rejects_old_spec_before_reading_snapshot(tmp_path):
    path = tmp_path / 'old.json.gz'
    run.write_json(path, {'provenance': spec.PROVENANCE, 'spec_hash': spec.SUPERSEDED[2]})
    with pytest.raises(gr.ReplayError, match='RECORD_SPEC_MISMATCH'):
        run.checked_record(path, {}, tmp_path)


def test_test_claim_is_resume_only_for_same_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'check_freeze', lambda root: {'config': 'a'})
    monkeypatch.setattr(experiment, 'head', lambda: 'synthetic-sha')
    experiment.claim_test(tmp_path)
    original = (tmp_path / 'v4/test_started.json').read_bytes()
    experiment.claim_test(tmp_path)
    assert (tmp_path / 'v4/test_started.json').read_bytes() == original
    monkeypatch.setattr(experiment, 'check_freeze', lambda root: {'config': 'b'})
    with pytest.raises(gr.ReplayError, match='TEST_ALREADY_STARTED_WITH_OTHER_CONFIG'):
        experiment.claim_test(tmp_path)


def test_missing_dev_score_blocks_first_test_call(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'check_freeze', lambda root: {})
    monkeypatch.setattr(experiment, 'trial_complete', lambda *a: True)
    with pytest.raises(gr.ReplayError, match='DEV_NOT_SCORED'):
        experiment.ready_test(tmp_path)


def test_same_symbol_unrelated_8k_does_not_qualify_as_primary():
    record = {'decision_time': '2026-07-15T12:00:00Z', 'views': {'c_news_small_mid': [
        {'symbol': 'SMLL', 'view': 'DOWN', 'news_ids': [1]}]}}
    class Sec:
        def eight_ks(self, symbol):
            return [{'accepted_at': '2026-07-14T21:00:00Z', 'accession': '0000000001-26-000001'}]
    ctx = {'candidates': [{'symbol': 'SMLL', 'headlines': [{'news_id': 1, 'url': 'https://example.invalid/unrelated'}]}]}
    assert run.apply_primary_filter(record, Sec(), ctx)[0] == []
    ctx['candidates'][0]['headlines'][0]['url'] = 'https://www.sec.gov/Archives/edgar/data/1/000000000126000001/release.htm'
    assert len(run.apply_primary_filter(record, Sec(), ctx)[0]) == 1


def test_fill_is_first_trade_inside_exact_one_second_window():
    class Transport:
        def get(self, origin, path, params):
            assert path == '/v2/stocks/trades'
            return {'trades': {'AAPL': [{'t': '2026-07-15T13:30:29.999Z', 'p': 1},
                                         {'t': '2026-07-15T13:30:30.100Z', 'p': 100},
                                         {'t': '2026-07-15T13:30:31Z', 'p': 999}]}}, {}
    from scripts.trading_lab.replay_bench import marketdata
    assert marketdata.stock_trades(Transport(), ['AAPL'], gr.instant('2026-07-15T13:30:30Z'))['AAPL']['p'] == 100


def test_daily_marks_sum_to_after_cost_outcome_and_keep_overnight_move():
    rows = [{'bar_open_at': gr.instant(d + 'T13:30:00Z'), 'open': o, 'close': c}
            for d, o, c in [('2026-07-15', 100, 101), ('2026-07-16', 104, 105)]]
    spy = [dict(r, open=100, close=100) for r in rows]
    last, path = labels.marked_path('2026-07-15', 'SMLL', rows, spy, 2, 1, .004)
    assert last == '2026-07-16'
    assert path['2026-07-15'] == pytest.approx(.008)
    assert path['2026-07-16'] == pytest.approx(.038)
    assert sum(path.values()) == pytest.approx(.046)
    pnl = score.allocate_pnl([('2026-07-15', {}, path)], 10)
    assert pnl['2026-07-16'] == pytest.approx(.0038)


def test_two_process_ownership_is_exclusive(tmp_path):
    with owned(tmp_path):
        with pytest.raises(gr.ReplayError, match='REPLAY_ALREADY_RUNNING'):
            with owned(tmp_path):
                pass


def test_score_outputs_labels_and_daily_marks_for_all_groups(tmp_path, monkeypatch):
    from scripts.trading_lab.replay_bench import snapshot as sn, marketdata as md
    from test_replay_bench import universe_bars, item
    day = '2026-07-15'
    monkeypatch.setattr(score, 'now', lambda: AT)
    T = run.decision_time(day)
    value = spec.spec()
    value['periods']['dev'] = [day, '2026-07-17']
    bars = universe_bars()
    bars.update({s: daily_bars(end='2026-07-17') for s in sn.STOCKS + sn.ETFS + sn.BENCHMARKS})
    bars['SMLL'] = daily_bars(end='2026-07-17', price=20, volume=2e6)
    universe = type('Universe', (), {'bars': bars, 'news': []})()
    monkeypatch.setattr(run, 'Universe', lambda *a, **kw: universe)
    monkeypatch.setattr(experiment, 'input_proof', lambda root: 'synthetic')
    monkeypatch.setattr(experiment, 'planned', lambda root, period, group: ([day], day))
    monkeypatch.setattr(md, 'stock_bars', lambda *a, **kw: (bars, []))
    monkeypatch.setattr(md, 'SecFilings', lambda t: type('Sec', (), {'eight_ks': lambda self, s: []})())
    class Anchors:
        def __init__(self, transport):
            pass
        def stock(self, day, anchor, symbols):
            exit_fill = anchor != '13:30:30'
            return {s: {'p': 100 if s == 'SPY' or not exit_fill else 102} for s in symbols}
        def crypto(self, at):
            return {}
    monkeypatch.setattr(lb := labels, 'MinuteAnchors', Anchors)
    for group in experiment.GROUP_MODELS:
        meta = experiment.trial_meta('dev', group)
        out = experiment.trial_out(tmp_path, meta)
        if group == 'cd':
            ctx, manifest = sn.context_c(T, md.previous_close(T),
                [item(1, '2026-07-15T10:00:00Z', ['SMLL'], 'Synthetic guidance cut')], bars, ['cut'])
            views = {'c_news_small_mid': [{'symbol': 'SMLL', 'view': 'UP', 'p_outperform': .6,
                                          'holding_sessions': 2, 'event_code': 'guidance_cut', 'news_ids': [1]}]}
        else:
            ctx, manifest = sn.context_a(T, bars, [])
            views = {name: [{'asset': 'XLE', 'horizon': '1d', 'view': 'UP', 'p_outperform': .6}]
                     for name in run.VARIANTS_AB}
        sealed = sn.seal(ctx, manifest, T)
        run.write_json(out / 'snapshots' / f"{sealed['snapshot_hash']}.json.gz", sealed)
        record = {**meta, 'provenance': spec.PROVENANCE, 'decision_time': gr.iso(T), 'session': day,
                  'snapshot_hash': sealed['snapshot_hash'], 'context_hash': sealed['context_hash'], 'views': views}
        record['decision_hash'] = sha256_canonical(record)
        run.write_json(out / group / f'{day}.json.gz', record)
    result = score.score(tmp_path, 'dev', object(), value)
    assert (tmp_path / 'v4/results/dev.json').exists()
    assert result['summary']['a_v1_as_live:1d']['n_views'] == 1
    # 2% relative move less 2 * (0.1% fee + 0.05% liquidity-tier slippage).
    assert result['summary']['a_v1_as_live:1d']['mean_net_bp'] == pytest.approx(170)
    assert result['summary']['c_news_small_mid']['daily_pnl']['2026-07-16'] != 0
    assert result['paired_audits']['audit_anonymized'] == {'n_pairs': 1, 'agree': 1}
    assert 'audit_opus_a' not in result['paired_audits']
    monkeypatch.setattr(run, 'Universe', lambda *a, **kw: pytest.fail('A scored period was fetched again'))
    assert score.score(tmp_path, 'dev', object(), value) == result
    assert all(r['snapshot_hash'] and 'daily_pnl' in r for r in result['labels'])


def test_transport_enforces_holdout_even_without_marketdata_wrapper(tmp_path):
    transport = gr.Transport(FakeGrant(), gr.Budget(tmp_path), tmp_path / 'cache',
                             getter=lambda *a: pytest.fail('Protected query reached the network'))
    with pytest.raises(gr.ReplayError, match='PROTECTED_REQUEST'):
        transport.get('https://data.alpaca.markets', '/v1beta3/crypto/us/bars', {
            'symbols': 'BTC/USD', 'start': '2026-09-01T00:00:00Z', 'end': '2026-10-01T00:00:00Z'})


def test_prediction_cache_is_sealed_and_never_refetched(tmp_path):
    transport = gr.Transport(FakeGrant(), gr.Budget(tmp_path), tmp_path / 'cache',
                             getter=lambda *a: pytest.fail('Sealed prediction attempted a fetch'))
    transport.allowed_cache = {}
    with pytest.raises(gr.ReplayError, match='PREDICTION_CACHE_NOT_SEALED'):
        transport.get('https://data.alpaca.markets', '/v1beta1/news', {'start': 'x'})


def test_primary_submissions_cik_keeps_the_real_string_shape():
    from scripts.trading_lab.replay_bench import marketdata
    class Transport:
        def get(self, origin, path, params=None):
            if path.endswith('company_tickers.json'):
                return {'0': {'ticker': 'SMLL', 'cik_str': 1}}, {}
            assert path == '/submissions/CIK0000000001.json'
            return {'cik': '0000000001', 'filings': {'recent': {'form': ['8-K'],
                'acceptanceDateTime': ['2026-07-14T16:05:00.000Z'],
                'accessionNumber': ['0000000001-26-000001']}}}, {}
    filings = marketdata.SecFilings(Transport()).eight_ks('SMLL')
    assert filings[0]['accepted_at'] == '2026-07-14T20:05:00Z'


def test_freeze_must_be_committed_before_test(tmp_path, monkeypatch):
    freeze = tmp_path / 'freeze.json'
    value = {'config_hashes': {'ab': 'a', 'cd': 'b'}, 'test_days': {'ab': ['d'], 'cd': ['d']}}
    atomic_json(freeze, value)
    monkeypatch.setattr(experiment, 'FREEZE', freeze)
    monkeypatch.setattr(experiment, 'frozen_payload', lambda root: value)
    def uncommitted():
        raise gr.ReplayError('CODE_NOT_COMMITTED')
    monkeypatch.setattr(experiment, 'committed', uncommitted)
    with pytest.raises(gr.ReplayError, match='CODE_NOT_COMMITTED'):
        experiment.check_freeze(tmp_path)


def test_superseded_opus_freeze_is_preserved_and_refused(tmp_path, monkeypatch):
    frozen = json.loads(experiment.FREEZE.read_text())
    declaration = json.loads(experiment.DECLARATION.read_text())
    # Revision 5 has its own committed freeze. Never rewrite this v4 artifact to fit new code.
    assert frozen['code_hash'] == '86dc3e5090617b44ca1344b026974179d40934203e3f51290b2ced5eac95d551'
    assert frozen['code_hash'] != run.code_hash()
    monkeypatch.setattr(audit, 'evidence', lambda root: {
        'floor': {'opus': '2026-07-01'}, 'evidence_hash': frozen['probe_evidence_hash']})
    with pytest.raises(gr.ReplayError, match='TEST_CONFIG_NOT_FROZEN'):
        experiment.check_freeze(tmp_path)
    assert frozen['spec_hash'] == spec.SPEC_HASH
    assert frozen['config_hashes'] == {g: run.config_hash('test', g) for g in ('ab', 'cd')}
    assert frozen['declaration_hash'] == sha256_canonical(declaration)
    assert len(declaration['variants']) == 4
    assert frozen['input_seal_required'] is True
    assert frozen['test_days'] == {g: run.sessions('2026-09-01', '2026-10-09') for g in ('ab', 'cd')}


def test_timer_install_refuses_missing_environment_before_any_write(tmp_path):
    import importlib.util
    path = spec.ROOT / 'deploy/research/install_replay_timer.py'
    module_spec = importlib.util.spec_from_file_location('replay_timer_install', path)
    installer = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(installer)
    with pytest.raises(SystemExit, match='private research environment file'):
        installer.main(['--environment', str(tmp_path / 'missing.env'), '--claude', 'unused'])

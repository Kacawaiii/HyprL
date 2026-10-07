from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.trading_lab.app_api.contracts import AppApiError
from scripts.trading_lab.app_api.trader import TraderViews
from scripts.trading_lab.platform.contracts import ModelContract, PredictionRecord
from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.trader_agent.config import Authorization, TraderError, instant, private_root, strict_json
from scripts.trading_lab.trader_agent.data import PublicData, build_context, calendar_session, label_window, protected
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.portfolio import portfolios, roundtrip_cost, weights
from scripts.trading_lab.trader_agent.runners import ModelRunner, command, parse_claude, parse_gpt, prompt, tainted
from scripts.trading_lab.trader_agent.schedule import render
from scripts.trading_lab.trader_agent.schemas import GPT_SCHEMAS, HERE, SCHEMAS, validate_analyst, validate_reviewer
from scripts.trading_lab.trader_agent.scoring import realize, rows, scorecard, weekly_report
from scripts.trading_lab.trader_agent.service import preregistration, skills, views_with_review
from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner


def test_native_shape_demo_pending_labels_and_append_only_realization(service, clock):
    result = service.run()
    assert result['status'] == 'COMPLETE'
    assert len(result['decision']['views']) == 50
    issued = list(rows(service.store, 'prediction'))
    assert len(issued) == 50 and service.store.records('label') == []
    first = issued[0]
    assert service.store.prediction_view(first['identity'], as_of=first['recorded_at'])['label_state'] == 'PENDING'
    evidence = service.store.records('inputs', parent_id=first['identity'])[0]['payload']
    contract = ModelContract.from_dict(evidence['provenance']['model_contract'])
    assert contract.identity == first['payload']['model_contract_hash']
    context = service.store.get(evidence['snapshot']['context_record_hash'])
    from scripts.trading_lab.sources.canonical import sha256_canonical
    assert sha256_canonical(context['payload']['context']) == evidence['snapshot']['context_hash']
    clock.at += timedelta(days=9, hours=10)
    result = realize(service.store, service.ledger, service.data, at=clock())
    assert result['labels_added'] == 50 and result['pending'] == 0
    assert service.store.get(first['identity']) == first
    assert len(service.store.records('execution')) == 50
    executed = service.store.records('execution')
    assert any(r['payload']['state']=='NO_FILL' for r in executed)
    assert any(r['payload']['state']=='FILLED' for r in executed)
    assert realize(service.store, service.ledger, service.data, at=clock())['labels_added'] == 0
    assert service.store.verify()['verified']
    card = scorecard(service.store, synthetic=True)
    assert card['scores']['consensus/equity_etf/5d/SPY_relative']['non_abstained'] == 3
    assert card['hypothesis_state'] == 'PENDING_MINIMUM_SAMPLE'
    assert card['multiple_testing']['model_configuration_count'] == 1
    assert scorecard(service.store)['scores'] == {}
    assert weekly_report(service.store, service.ledger.root, at=clock(), synthetic=True)['identity']


@pytest.mark.parametrize('code', ['SCHEMA_INVALID', 'TAINTED_RUN', 'SKIPPED_QUOTA', 'MODEL_TIMEOUT', 'BUDGET_EXHAUSTED'])
def test_failed_inference_is_not_published_and_retries_bounded(service, code):
    class Broken(SyntheticRunner):
        def once(self, role, text):
            self.ledger.reserve(role)
            raise TraderError(code)
    service.runner = Broken(service.ledger, clock=service.clock)
    result = service.run()
    assert result['status'] == ('SKIPPED_QUOTA' if code == 'SKIPPED_QUOTA' else 'FAILED')
    assert not service.store.records('prediction')
    assert service.ledger.counts()['analyst_claude'] == (2 if code in {'SCHEMA_INVALID', 'MODEL_TIMEOUT'} else 1)
    assert json.loads((service.ledger.root / 'alert.json').read_text())['code'] == code


def test_invalid_actual_json_output_fails_schema_without_repair(service):
    class Invalid(SyntheticRunner):
        def once(self, role, text):
            output = super().once(role, text)
            output['unexpected'] = 'invalid'
            return output
    service.runner = Invalid(service.ledger, clock=service.clock)
    assert service.run()['error'] == 'SCHEMA_INVALID'
    assert not service.store.records('prediction')


def test_independence_and_reviewer_receives_both(service):
    class Capture(SyntheticRunner):
        def __init__(self, *args, **kw):
            super().__init__(*args, **kw)
            self.prompts = {}
        def once(self, role, text):
            self.prompts[role] = text
            return super().once(role, text)
    service.runner = Capture(service.ledger, clock=service.clock)
    assert service.run()['status'] == 'COMPLETE'
    assert service.runner.prompts['analyst_claude'] == service.runner.prompts['analyst_gpt']
    assert 'INDEPENDENT_ANALYST_DATA' not in service.runner.prompts['analyst_gpt']
    assert 'INDEPENDENT_ANALYST_DATA' in service.runner.prompts['reviewer']


def test_holiday_and_pause_never_dispatch(service, clock):
    clock.at = instant('2026-12-25T12:00:00Z')
    assert service.run()['status'] == 'SKIPPED_HOLIDAY'
    assert service.ledger.counts() == {}
    (service.ledger.root / 'PAUSED').touch()
    assert service.run()['status'] == 'PAUSED'


def test_expiry_and_late_decision_fail_without_dispatch(service, clock):
    clock.at = instant('2027-02-01T00:00:00Z')
    assert service.run()['error'] == 'AUTHORIZATION_EXPIRED_OR_NOT_STARTED'
    clock.at = instant('2026-10-06T14:00:00Z')
    assert service.run()['error'] == 'MISSED_DECISION_DEADLINE'
    assert service.ledger.counts() == {}


def test_same_day_durable_restart_does_not_run_twice(service):
    assert service.run()['status'] == 'COMPLETE'
    before = len(service.store.records('prediction'))
    assert service.run()['error'] == 'BUDGET_EXHAUSTED'
    assert len(service.store.records('prediction')) == before


def test_missing_prices_fail_before_models(service):
    def missing(*args, **kw):
        raise TraderError('MISSING_PRICES')
    service.data.yahoo = missing
    service.data.crypto = missing
    assert service.run()['error'] == 'MISSING_PRICES'
    assert service.ledger.counts() == {'run': 1}


def test_protection_before_fetch_and_label_windows(grant, ledger, clock):
    class OfflineReal(SyntheticData):
        synthetic = False
        def crypto(self, *args, **kw):
            pytest.fail('protected crypto must never be requested')
    context, _ = build_context(grant, OfflineReal(ledger, clock), at=clock())
    assert context['prices']['BTC-USD']['state'] == 'PROTECTED'
    assert 'BTC-USD' not in context['universe']
    assert protected('AAPL', instant('2026-11-25T12:00:00Z'), instant('2026-12-02T22:00:00Z'))
    assert label_window('AAPL', '2026-10-09', '5d', []) == (instant('2026-10-09T13:30:00Z'), instant('2026-10-15T20:00:00Z'))
    assert calendar_session('2026-11-02').open_at == instant('2026-11-02T14:30:00Z')


def test_source_budget_exhaustion_is_terminal(service):
    service.ledger.grant.payload['data_sources']['yahoo_chart']['max_requests_per_day'] = 1
    assert service.run()['error'] == 'BUDGET_EXHAUSTED'
    assert not service.store.records('prediction')


def test_atomic_daily_and_global_budgets(ledger, grant, clock):
    def reserve(_):
        try:
            return ledger.reserve('analyst_claude')
        except TraderError:
            return None
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(reserve, range(10)))
    assert len([r for r in result if r]) == 2
    restarted = Ledger(ledger.root, grant, clock=clock)
    assert restarted.counts()['analyst_claude'] == 2
    restarted.reserve('analyst_gpt')
    restarted.reserve('analyst_gpt')
    restarted.reserve('reviewer')
    restarted.reserve('reviewer')
    grant.payload['external_models']['reviewer']['retries_per_day'] = 3
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        restarted.reserve('reviewer')


def test_changing_runtime_path_cannot_reset_shared_budget_bank(tmp_path, grant, clock):
    bank = tmp_path / 'shared-bank'
    first = Ledger(tmp_path/'runtime-a',grant,clock=clock,budget_root=bank)
    second = Ledger(tmp_path/'runtime-b',grant,clock=clock,budget_root=bank)
    first.reserve('run')
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        second.reserve('run')
    with first.owner():
        with pytest.raises(TraderError, match='OWNER_BUSY'):
            with second.owner():
                pass
    (bank/'PAUSED').touch()
    with pytest.raises(TraderError, match='PAUSED'):
        second.reserve('yahoo_chart')


def test_archive_rejection_is_preserved_as_integrity_error(tmp_path):
    from scripts.trading_lab.trader_agent.data import archived
    assert archived(tmp_path/'missing','fomc',instant('2026-10-06T12:00:00Z'))['state'] == 'INTEGRITY_ERROR'


def test_gdelt_spacing_persists_and_expiry_checked_after_wait(ledger, clock):
    ledger.reserve('gdelt_doc_api')
    with pytest.raises(TraderError, match='SOURCE_SPACING'):
        ledger.reserve('gdelt_doc_api')
    clock.at += timedelta(seconds=6)
    ledger.reserve('gdelt_doc_api')
    clock.at = instant(ledger.grant.payload['not_after'])
    with pytest.raises(TraderError, match='AUTHORIZATION_EXPIRED'):
        ledger.reserve('yahoo_chart')


@pytest.mark.parametrize('event', [{'type':'command_execution'}, {'type':'file_change'},
    {'type':'function_call','name':'functions.exec_command'}, {'type':'mcp_tool_call'},
    {'nested': {'name':'apply_patch'}}])
def test_any_command_or_file_event_taints_even_before_final(event):
    text = json.dumps({'type':'item.completed','item':event}) + '\n' + json.dumps(
        {'type':'item.completed','item':{'type':'agent_message','text':'{}'}})
    with pytest.raises(TraderError, match='TAINTED_RUN'):
        parse_gpt(text)


def test_jsonl_protocol_and_claude_structured_output():
    assert parse_gpt(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'{"views":[]}'}})) == {'views': []}
    assert parse_claude('{"type":"result","subtype":"success","structured_output":{"views":[]}}') == {'views': []}
    for text in ('bad', '{}', '{"is_error":true}'):
        with pytest.raises(TraderError):
            parse_claude(text)
    with pytest.raises(TraderError):
        parse_gpt('{"type":"turn.failed"}')
    with pytest.raises(TraderError):
        strict_json('{"a":1,"a":2}')


def test_cli_restrictions_schema_and_empty_directory(ledger, monkeypatch):
    args = command('analyst_gpt', ledger.grant)
    assert args[:3] == ['codex', '--search', 'exec']
    for flag in ('--sandbox', '--ephemeral', '--output-schema', '--json', '--ignore-user-config'):
        assert flag in args
    assert args[args.index('--sandbox') + 1] == 'read-only'
    assert '--disable' in args and 'shell_tool' in args
    for role in ('analyst_claude', 'reviewer'):
        cmd = command(role, ledger.grant)
        assert cmd[cmd.index('--tools') + 1] == 'WebSearch,WebFetch'
        assert cmd[cmd.index('--allowedTools') + 1] == 'WebSearch,WebFetch'
        assert not any('dangerously' in a for a in cmd)
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic-cli-version'))
    class Process:
        returncode = 0
        def __init__(self, args, **kw):
            assert list(Path(kw['cwd']).iterdir()) == []
            assert kw['start_new_session']
            self.out = kw['stdout']
        def communicate(self, *a, **kw):
            self.out.write(json.dumps({'type':'item.completed','item':{'type':'agent_message','text':'{}'}}) + '\n')
            self.out.flush()
    monkeypatch.setattr('subprocess.Popen', Process)
    assert ModelRunner(ledger, clock=ledger.clock).once('analyst_gpt', 'synthetic prompt') == {}


@pytest.mark.parametrize('raw,code', [('{"error":"rate limit"}', 'SKIPPED_QUOTA'), ('bad-json','MODEL_JSON_INVALID')])
def test_raw_cli_quota_and_bad_json(ledger, monkeypatch, raw, code):
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic'))
    class Process:
        returncode = 0
        def __init__(self, args, **kw):
            self.out = kw['stdout']
        def communicate(self, *a, **kw):
            self.out.write(raw)
            self.out.flush()
    monkeypatch.setattr('subprocess.Popen', Process)
    with pytest.raises(TraderError, match=code):
        ModelRunner(ledger, clock=ledger.clock).once('analyst_gpt', 'synthetic')


def test_successful_forecast_mentions_trade_quota_without_being_usage_quota(ledger, monkeypatch):
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic'))
    class Process:
        returncode = 0
        def __init__(self, args, **kw):
            self.out = kw['stdout']
        def communicate(self, *a, **kw):
            self.out.write(json.dumps({'type':'item.completed','item':{'type':'agent_message',
                'text':'{"regime":["Synthetic trade quota announcement, 429 units"],"views":[]}'}}))
            self.out.flush()
    monkeypatch.setattr('subprocess.Popen', Process)
    assert ModelRunner(ledger, clock=ledger.clock).once('analyst_gpt', 'synthetic')['views'] == []


def test_schemas_pinned_skills_and_preregistration():
    assert len(preregistration()) == 64
    _, hashes = skills()
    assert len(hashes) == 2
    for name, schema in SCHEMAS.items():
        assert json.loads((HERE / 'schemas' / (name + '.json')).read_text()) == schema


def test_future_sources_and_reviewer_cannot_raise_probability(service):
    context, _ = build_context(service.ledger.grant, service.data, at=service.clock())
    text, _ = skills()
    output = service.runner.once('analyst_claude', prompt(context, text['TRADER_SKILL.md']))
    output['views'][0]['catalysts'][0]['published_at'] = '2027-01-01T00:00:00Z'
    with pytest.raises(TraderError, match='FUTURE_SOURCE'):
        validate_analyst(output, context['universe'], service.clock())
    output['views'][0]['catalysts'][0]['published_at'] = '2026-10-01T00:00:00Z'
    analysts = {'analyst_claude': output, 'analyst_gpt': output}
    review = service.runner.once('reviewer', prompt(context, text['REVIEWER_SKILL.md'], analysts=analysts))
    review['verdicts'][0].update(verdict='DOWNGRADE', adjusted_p=.7)
    with pytest.raises(TraderError, match='REVIEW_RAISED'):
        validate_reviewer(review, analysts)
    review['verdicts'][0]['adjusted_p'] = .52
    validate_reviewer(review, analysts)
    combined = views_with_review(analysts, review)
    assert next(v for v in combined if v['analyst']=='consensus')['p_outperform'] == .52


def test_long_short_cap_hedge_gross_and_costs():
    views = [{'asset': 'A' + str(i), 'horizon':'1d','analyst':'consensus','p_outperform': .56 if i%2 else .4,
              'view':'UP' if i%2 else 'DOWN','verdict':'KEEP'} for i in range(20)]
    w = weights(views)
    assert sum(abs(v) for v in w.values()) <= 1.00000001
    assert max(abs(v) for v in w.values()) <= .1
    assert w['A0'] < 0 and w['A1'] > 0
    for variant in portfolios(views)['1d'].values():
        if isinstance(variant, dict):
            assert sum(abs(v) for v in variant.values()) <= 1.00000001
            assert max(abs(v) for v in variant.values()) <= .1
    assert roundtrip_cost('BTC-USD') == .002
    assert roundtrip_cost('AAPL', half_spread_bps=2.5) == .0015


def test_api_read_only_bounded_no_private_transcripts(service, clock):
    result = service.run()
    api = TraderViews(service.ledger.root)
    before = service.store.verify()
    today = api.dispatch('/api/v1/trader/today', {'date':['2026-10-06']})
    assert today['runs'][-1]['payload']['status'] == 'COMPLETE'
    page = api.dispatch('/api/v1/trader/ledger', {'limit':['3']})
    assert len(page['records']) == 3 and page['next_after']
    assert len(api.dispatch('/api/v1/trader/ledger', {'limit':['3'],'after':[str(page['next_after'])]})['records']) == 3
    assert api.dispatch('/api/v1/trader/scorecard', {})['scores'] == {}
    assert api.dispatch('/api/v1/trader/alerts', {})['alerts'] == []
    assert service.store.verify() == before
    with pytest.raises(AppApiError):
        api.dispatch('/api/v1/trader/ledger', {'limit':['201']})
    with pytest.raises(AppApiError):
        TraderViews().dispatch('/api/v1/trader/today', {})


def test_unit_schedule_has_no_catchup_and_no_inference_on_install(tmp_path):
    units = render(tmp_path, '/synthetic/python', '/synthetic/grant', tmp_path / 'runtime', path='/synthetic/bin')
    assert len(units) == 6
    assert '12:00:00 UTC' in units['hyprl-trader-run.timer']
    assert '21:30:00 UTC' in units['hyprl-trader-label.timer']
    assert 'Persistent=false' in units['hyprl-trader-run.timer']
    assert 'KillMode=control-group' in units['hyprl-trader-run.service']
    assert 'UMask=0077' in units['hyprl-trader-run.service']
    assert 'WorkingDirectory=' + str(tmp_path) + '\n' in units['hyprl-trader-run.service']
    assert 'broker' not in ''.join(units.values())


def test_runtime_cannot_be_inside_public_repository():
    with pytest.raises(TraderError, match='RUNTIME_MUST_BE_OUTSIDE_GIT'):
        private_root(HERE / 'runtime')


def test_public_transport_refuses_other_paths_before_count(ledger):
    with pytest.raises(TraderError, match='UNAUTHORIZED_PATH'):
        PublicData(ledger).fetch('yahoo_chart', '/account', {})
    assert ledger.counts() == {}


def test_subprocess_timeout_kills_and_reaps_process_group(ledger, monkeypatch):
    import subprocess
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: SimpleNamespace(stdout='synthetic'))
    killed, calls = [], []
    monkeypatch.setattr('os.killpg', lambda *a: killed.append(a))
    class Process:
        pid = 123456789
        returncode = -9
        def __init__(self, *args, **kw):
            pass
        def communicate(self, *args, **kw):
            calls.append(kw)
            if len(calls) == 1:
                raise subprocess.TimeoutExpired('synthetic', 60)
    monkeypatch.setattr('subprocess.Popen', Process)
    with pytest.raises(TraderError, match='MODEL_TIMEOUT'):
        ModelRunner(ledger, clock=ledger.clock).once('analyst_gpt', 'synthetic')
    assert killed[0][0] == Process.pid and len(calls) == 2


def test_no_network_request_on_expired_grant_or_redirect(ledger, clock, monkeypatch):
    import scripts.trading_lab.trader_agent.data as data_module
    def forbidden(*args, **kw):
        pytest.fail('expired authorization must not invoke transport')
    monkeypatch.setattr(data_module, 'build_opener', forbidden)
    clock.at = instant(ledger.grant.payload['not_after'])
    with pytest.raises(TraderError, match='AUTHORIZATION_EXPIRED'):
        PublicData(ledger, clock=clock).fetch('yahoo_chart', '/v8/finance/chart/AAPL', {})
    assert not ledger.counts()
    with pytest.raises(TraderError, match='REDIRECT_REFUSED'):
        data_module.NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid/')


def test_model_finishing_after_open_is_not_recorded(service, clock):
    class Late(SyntheticRunner):
        def once(self, role, text):
            output = super().once(role, text)
            if role == 'reviewer':
                clock.at = instant('2026-10-06T13:31:00Z')
            return output
    service.runner = Late(service.ledger, clock=clock)
    assert service.run()['error'] == 'MISSED_DECISION_DEADLINE'
    assert not service.store.records('prediction')


def test_rejected_original_views_remain_scoreable(service, clock):
    class Reject(SyntheticRunner):
        def once(self, role, text):
            output = super().once(role, text)
            if role == 'reviewer':
                output['verdicts'][0].update(verdict='REJECT', reason_code='generic')
            return output
    service.runner = Reject(service.ledger, clock=clock)
    result = service.run()
    assert result['status'] == 'COMPLETE'
    original = next(v for v in result['decision']['views'] if v['analyst']=='analyst_claude')
    assert original['verdict'] == 'REJECT' and original['view'] == 'UP'
    consensus = next(v for v in result['decision']['views'] if v['analyst']=='consensus')
    assert consensus['view'] == 'ABSTAIN'
    clock.at += timedelta(days=9, hours=10)
    realize(service.store, service.ledger, service.data, at=clock())
    scores = scorecard(service.store, synthetic=True)['scores']
    assert scores['reviewer_rejected/equity_etf/1d/SPY_relative']['non_abstained'] == 1
    assert scores['reviewer_rejected/equity_etf/1d/SPY_relative']['false_positive_rate'] is not None


def test_first_timer_guard_and_health_expectations(service, clock):
    from scripts.trading_lab.trader_agent.cli import health
    service.data.synthetic = False
    clock.at = instant('2026-10-05T12:00:00Z')
    assert service.run()['status'] == 'NOT_STARTED'
    assert not service.ledger.counts()
    assert health(service.ledger)['state'] == 'HEALTHY'
    clock.at = instant('2026-10-06T14:00:00Z')
    assert health(service.ledger)['state'] == 'MISSING_DAILY_RUN'
    clock.at = instant('2026-10-06T22:10:00Z')
    assert health(service.ledger)['state'] == 'MISSING_LABEL_JOB'
    (service.ledger.budget_root/'PAUSED').touch()
    assert health(service.ledger)['state'] == 'PAUSED'


def test_pending_cohort_is_not_partial_realized_portfolio(service, clock):
    service.run()
    card = scorecard(service.store, synthetic=True)
    assert all(c['state'] == 'PENDING' for c in card['portfolio_cohorts'])
    clock.at = instant('2026-10-06T21:30:00Z')
    realize(service.store, service.ledger, service.data, at=clock())
    card = scorecard(service.store, synthetic=True)
    assert next(c for c in card['portfolio_cohorts'] if c['horizon']=='5d')['state'] == 'PENDING'


def test_minimum_sample_and_registered_weekly_bootstrap():
    from scripts.trading_lab.trader_agent.scoring import primary_test
    assert primary_test([])['state'] == 'PENDING_MINIMUM_SAMPLE'
    start = instant('2026-01-01T00:00:00Z')
    samples = [{'session':(start + timedelta(days=i)).date().isoformat(), 'view':'UP', 'p':.6,
                'y':1, 'climatology':.5} for i in range(60) for _ in range(25)]
    result = primary_test(samples, draws=100)
    assert result['state'] == 'SUPPORTED' and result['hit_lower_95'] == 1
    assert result['brier_difference_upper_95'] < 0


def test_http_endpoints_refuse_writes(service):
    import threading
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen
    from scripts.trading_lab.app_api.server import make_server
    service.run()
    server = make_server('tests/fixtures/crypto', port=0, trader_root=service.ledger.root)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = 'http://127.0.0.1:' + str(server.server_address[1])
    try:
        for leaf in ('today', 'ledger', 'scorecard', 'alerts'):
            with urlopen(base + '/api/v1/trader/' + leaf, timeout=5) as response:
                assert response.status == 200
        with pytest.raises(HTTPError) as error:
            urlopen(Request(base + '/api/v1/trader/today', data=b'{}', method='POST'), timeout=5)
        assert error.value.code == 405
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_native_coinbase_forming_candle_is_excluded_before_adapter(ledger, clock):
    data = PublicData(ledger, clock=clock)
    complete = instant('2026-10-05T00:00:00Z')
    forming = instant('2026-10-06T00:00:00Z')
    payload = [[int(d.timestamp()),99,103,100,101,20] for d in (complete, forming)]
    data.fetch = lambda *a, **kw: (payload, {'received_at':'2026-10-06T12:00:00Z'})
    bars, _ = data.crypto('BTC-USD', complete, clock())
    assert len(bars) == 1 and bars[0]['bar_open_at'] == complete


def test_daily_context_is_stored_once_with_prediction_references(service):
    service.run()
    contexts = [r for r in rows(service.store,'replay-summary') if r['payload'].get('schema')=='trader-context-evidence-v1']
    assert len(contexts) == 1
    inputs = list(rows(service.store,'inputs'))
    assert all(r['payload']['snapshot']['context_record_hash'] == contexts[0]['identity'] for r in inputs)


def test_crash_between_label_and_execution_leaves_neither_and_resume_completes(service, clock, monkeypatch):
    service.run()
    clock.at += timedelta(days=9, hours=10)
    original = ResearchStore._append

    def crash_on_execution(self, db, kind, *args, **kwargs):
        if kind == 'execution':
            raise RuntimeError('simulated crash after the label insert')
        return original(self, db, kind, *args, **kwargs)
    monkeypatch.setattr(ResearchStore, '_append', crash_on_execution)
    with pytest.raises(RuntimeError):
        realize(service.store, service.ledger, service.data, at=clock())
    # One transaction: the label of the crashed prediction was rolled back with its execution.
    assert service.store.records('label') == [] and service.store.records('execution') == []
    monkeypatch.setattr(ResearchStore, '_append', original)
    result = realize(service.store, service.ledger, service.data, at=clock())
    assert result['labels_added'] == 50 and result['executions_repaired'] == 0
    assert len(service.store.records('execution')) == len(service.store.records('label')) == 50


def test_a_labelled_prediction_missing_its_execution_is_repaired_once(service, clock, monkeypatch):
    service.run()
    clock.at += timedelta(days=9, hours=10)
    # The old code: label written, then a crash before the execution (separate transactions).
    monkeypatch.setattr(ResearchStore, 'append_label_and_execution', lambda self, label, execution: self.append_label(label))
    realize(service.store, service.ledger, service.data, at=clock())
    assert len(service.store.records('label')) == 50 and service.store.records('execution') == []
    monkeypatch.undo()
    repaired = realize(service.store, service.ledger, service.data, at=clock())
    assert repaired['executions_repaired'] == 50 and repaired['labels_added'] == 0
    executions = service.store.records('execution')
    assert len(executions) == 50 and {r['payload']['state'] for r in executions} == {'FILLED', 'NO_FILL'}
    assert realize(service.store, service.ledger, service.data, at=clock())['executions_repaired'] == 0
    assert service.store.verify()['verified']


def test_false_positive_rate_is_fp_over_actual_negatives():
    from scripts.trading_lab.trader_agent.scoring import metrics

    def sample(p, y):
        return {'view': 'UP' if p > .5 else 'DOWN', 'p': p, 'y': y, 'climatology': .5, 'return': .01 if y else -.01,
                'pnl': 0., 'session': '2026-10-06'}
    card = metrics([sample(.6, 1), sample(.6, 1), sample(.6, 0), sample(.4, 0)], 4)
    assert card['hit_rate'] == .75 and card['error_rate'] == .25
    assert card['false_positive_rate'] == .5          # 1 FP among 2 actual negatives (the old formula said 0.25)
    assert card['false_discovery_rate'] == 1 / 3
    assert card['confusion'] == {'tp': 2, 'fp': 1, 'tn': 1, 'fn': 0}
    no_negatives = metrics([sample(.6, 1), sample(.4, 1)], 2)
    assert no_negatives['false_positive_rate'] is None and no_negatives['confusion']['fn'] == 1


def test_confusion_counts_sum_to_non_abstained_at_the_threshold():
    from scripts.trading_lab.trader_agent.scoring import metrics
    sample = {'view': 'DOWN', 'p': .5, 'y': 1, 'climatology': .5, 'return': .01, 'pnl': 0., 'session': '2026-10-06'}
    card = metrics([sample, dict(sample, y=0)], 2)
    assert sum(card['confusion'].values()) == card['non_abstained'] == 2


def test_catchup_after_a_modelless_failure_enters_at_the_close_and_is_scored_apart(service, clock):
    from scripts.trading_lab.trader_agent.service import CATCHUP_VARIANT
    # A reservation alone is not proof of a failed daily run.
    clock.at = instant('2026-10-06T15:00:00Z')
    assert service.run(catchup=True)['error'] == 'CATCHUP_NOT_ALLOWED'
    service.ledger.reserve('run')                       # the morning run consumed, no model call (the GDELT crash)
    failed_daily_run(service, clock)
    result = service.run(catchup=True)
    assert result['status'] == 'COMPLETE' and result['run_id'].endswith(':catchup')
    assert {v['asset'] for v in result['decision']['views']} == {'AAPL', 'MSFT', 'XLK'}   # equities only
    issued = list(rows(service.store, 'prediction'))
    definition = issued[0]['payload']['signal']['label_definition']
    assert definition['variant'] == CATCHUP_VARIANT and definition['entry_price'] == 'close'
    assert definition['entry_at'].startswith('2026-10-06T20:00:00')                    # today's close (EDT)
    assert service.run(catchup=True)['error'] in {'BUDGET_EXHAUSTED', 'CATCHUP_NOT_ALLOWED'}   # once per day
    clock.at += timedelta(days=9)
    labelled = realize(service.store, service.ledger, service.data, at=clock())
    assert labelled['labels_added'] == len(issued) and labelled['pending'] == 0
    scores = scorecard(service.store, synthetic=True)['scores']
    assert any(k.startswith(CATCHUP_VARIANT + ':consensus/') for k in scores)
    assert not any(k.startswith('consensus/') for k in scores)                           # never pooled with primary


def test_catchup_refused_too_close_to_the_close(service, clock):
    service.ledger.reserve('run')
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T19:00:00Z')                                          # less than 80 min to the close
    assert service.run(catchup=True)['error'] == 'MISSED_DECISION_DEADLINE'


def failed_daily_run(service, clock, status='FAILED'):
    run_id = 'trader:' + clock().date().isoformat() + ':synthetic'
    service.summary(run_id, status, clock(), error='MODEL_FAILED')


def test_catchup_after_a_failed_run_with_model_calls_uses_remaining_budgets(service, clock):
    service.ledger.reserve('run')
    service.ledger.reserve('analyst_claude')
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T15:00:00Z')
    result = service.run(catchup=True)
    assert result['status'] == 'COMPLETE'
    assert result['budget_counts']['analyst_claude'] == 2
    assert result['budget_counts']['analyst_gpt'] == 1
    assert result['budget_counts']['reviewer'] == 1


@pytest.mark.parametrize('missing', ['analyst_claude', 'analyst_gpt'])
def test_catchup_exhausted_analyst_is_missing_without_dispatch_and_scored_apart(service, clock, missing):
    from scripts.trading_lab.trader_agent.service import CATCHUP_VARIANT
    present = 'analyst_claude' if missing == 'analyst_gpt' else 'analyst_gpt'
    service.ledger.reserve('run')
    service.ledger.reserve(present)
    service.ledger.reserve(missing)
    service.ledger.reserve(missing)
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T15:00:00Z')
    calls, original = [], service.runner.once
    def capture(role, text):
        calls.append(role)
        if role == 'reviewer':
            analysts = json.loads(text.split('\nINDEPENDENT_ANALYST_DATA:\n')[1])
            assert set(analysts) == {present}
        return original(role, text)
    service.runner.once = capture
    result = service.run(catchup=True)
    assert calls == [present, 'reviewer']
    assert result['status'] == 'DEGRADED' and result['degraded'] == {missing: 'BUDGET_EXHAUSTED'}
    views = result['decision']['views']
    absent = [v for v in views if v['verdict'] == 'MISSING']
    assert len(absent) == 12 and {v['analyst'] for v in absent} == {missing, 'reviewer_' + missing.split('_')[-1]}
    assert all(v['error'] == 'BUDGET_EXHAUSTED' and v['raw_view'] is None for v in absent)
    assert all(v['view'] == 'ABSTAIN' for v in views if v['analyst'] == 'consensus')
    assert {v['asset'] for v in views} == {'AAPL', 'MSFT', 'XLK'}
    counts = result['budget_counts']
    assert counts[missing] == counts[present] == 2 and counts['reviewer'] == 1
    assert sum(counts[r] for r in ('analyst_claude', 'analyst_gpt', 'reviewer')) == 5
    predictions = service.store.records('prediction')
    assert len(predictions) == 18
    assert all(p['payload']['signal']['label_definition']['variant'] == CATCHUP_VARIANT for p in predictions)
    before = service.ledger.counts()
    assert service.run(catchup=True)['error'] == 'BUDGET_EXHAUSTED'
    assert service.ledger.counts() == before
    clock.at += timedelta(days=9)
    assert realize(service.store, service.ledger, service.data, at=clock())['labels_added'] == 18
    scores = scorecard(service.store, synthetic=True)['scores']
    assert scores and all(k.startswith(CATCHUP_VARIANT + ':') for k in scores)
    assert not any(missing in k or ('reviewer_' + missing.split('_')[-1]) in k for k in scores)


@pytest.mark.parametrize('status', [None, 'RUNNING', 'COMPLETE', 'DEGRADED', 'SKIPPED_QUOTA'])
def test_catchup_requires_a_failed_daily_run_and_refuses_success_even_after_later_failure(service, clock, status):
    service.ledger.reserve('run')
    if status:
        failed_daily_run(service, clock, status)
    if status in {'COMPLETE', 'DEGRADED'}:
        clock.at += timedelta(seconds=1)
        failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T15:00:00Z')
    before = service.ledger.counts()
    assert service.run(catchup=True)['error'] == 'CATCHUP_NOT_ALLOWED'
    assert service.ledger.counts() == before


@pytest.mark.parametrize('used', [
    ('analyst_claude', 'analyst_claude', 'analyst_gpt', 'analyst_gpt'),
    ('reviewer', 'reviewer'),
    ('analyst_claude', 'reviewer', 'reviewer'),
    ('analyst_claude', 'analyst_gpt', 'analyst_gpt', 'reviewer'),
])
def test_catchup_refuses_unavailable_analysts_reviewer_or_aggregate_budget(service, clock, used):
    # Last case leaves only one aggregate call, but needs the remaining analyst plus reviewer (two).
    if used == ('analyst_claude', 'analyst_gpt', 'analyst_gpt', 'reviewer'):
        service.ledger.grant.payload['budgets']['max_llm_calls_per_day'] = 5
    service.ledger.reserve('run')
    for role in used:
        service.ledger.reserve(role)
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T15:00:00Z')
    before = service.ledger.counts()
    assert service.run(catchup=True)['error'] == 'BUDGET_EXHAUSTED'
    assert service.ledger.counts() == before
    assert not service.store.records('prediction')


def test_catchup_analyst_that_exhausts_its_last_retry_can_still_be_missing(service, clock):
    service.ledger.reserve('run')
    service.ledger.reserve('analyst_gpt')
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T15:00:00Z')
    service.runner = FailOne(service.ledger, 'analyst_gpt', 'SCHEMA_INVALID', clock=clock)
    result = service.run(catchup=True)
    assert result['status'] == 'DEGRADED' and result['degraded'] == {'analyst_gpt': 'BUDGET_EXHAUSTED'}
    assert result['budget_counts']['analyst_gpt'] == 2
    assert result['budget_counts']['reviewer'] == 1


@pytest.mark.parametrize('reason', ['previous_day', 'other_population', 'catchup_only'])
def test_catchup_failure_must_belong_to_todays_primary_run(service, clock, reason):
    service.ledger.reserve('run')
    run_id = {'previous_day': 'trader:2026-10-05:synthetic',
              'other_population': 'trader:2026-10-06:real',
              'catchup_only': 'trader:2026-10-06:synthetic:catchup'}[reason]
    service.summary(run_id, 'FAILED', clock(), error='MODEL_FAILED')
    clock.at = instant('2026-10-06T15:00:00Z')
    assert service.run(catchup=True)['error'] == 'CATCHUP_NOT_ALLOWED'
    assert service.ledger.counts() == {'run': 1}


@pytest.mark.parametrize('reason', ['paused', 'expired', 'reviewer_failure'])
def test_catchup_budget_exception_never_overrides_pause_expiry_or_reviewer_failure(service, clock, reason):
    service.ledger.reserve('run')
    service.ledger.reserve('analyst_gpt')
    service.ledger.reserve('analyst_gpt')
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T15:00:00Z')
    if reason == 'paused':
        (service.ledger.root / 'PAUSED').touch()
    elif reason == 'expired':
        service.ledger.grant.payload['not_after'] = '2026-10-06T15:00:00Z'
    else:
        service.runner = FailOne(service.ledger, 'reviewer', 'MODEL_FAILED', clock=clock)
    result = service.run(catchup=True)
    assert result['status'] == ('PAUSED' if reason == 'paused' else 'FAILED')
    assert not service.store.records('prediction')
    assert service.ledger.counts()['analyst_gpt'] == 2


@pytest.mark.parametrize('stage', ['context', 'analyst', 'reviewer'])
def test_catchup_decision_must_finish_before_close_minus_80_minutes(service, clock, stage):
    service.ledger.reserve('run')
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T18:30:00Z')
    if stage == 'context':
        original = service.data.headlines
        def slow_context(*args):
            output = original(*args)
            clock.at = instant('2026-10-06T18:40:00Z')
            return output
        service.data.headlines = slow_context
    else:
        original = service.runner.once
        def slow_model(role, text):
            output = original(role, text)
            if role == ('reviewer' if stage == 'reviewer' else 'analyst_claude'):
                clock.at = instant('2026-10-06T18:40:00Z')
            return output
        service.runner.once = slow_model
    result = service.run(catchup=True)
    assert result['status'] == 'FAILED'
    assert result['error'] in {'MISSED_DECISION_DEADLINE', 'MISSED_DECISION_DEADLINE_OR_PAUSED'}
    assert not service.store.records('prediction')


def test_catchup_refused_at_exact_close_minus_80_minutes(service, clock):
    service.ledger.reserve('run')
    failed_daily_run(service, clock)
    clock.at = instant('2026-10-06T18:40:00Z')
    assert service.run(catchup=True)['error'] == 'MISSED_DECISION_DEADLINE'
    assert service.ledger.counts() == {'run': 1}


def test_close_entry_window_is_strictly_after_the_decision_and_equities_only():
    from scripts.trading_lab.trader_agent.data import label_window
    entry, exit_1d = label_window('AAPL', '2026-10-06', '1d', ['BTC-USD'], 'close')
    assert entry.isoformat().startswith('2026-10-06T20:00:00') and exit_1d.isoformat().startswith('2026-10-07T20:00:00')
    with pytest.raises(TraderError, match='UNSUPPORTED_ENTRY'):
        label_window('BTC-USD', '2026-10-06', '1d', ['BTC-USD'], 'close')


def test_gpt_schema_passes_openai_strict_mode_and_local_validator_keeps_uri_checks():
    from scripts.trading_lab.trader_agent.schemas import for_openai_strict, openai_strict_problems
    for name, schema in GPT_SCHEMAS.items():
        assert openai_strict_problems(schema) == [], name
        assert json.loads((HERE / 'schemas' / (name + '.gpt.json')).read_text()) == schema
    url = GPT_SCHEMAS['analyst']['properties']['views']['items']['properties']['catalysts']['items']['properties']['url']
    assert url == {'type': 'string'}   # the incident: 'format: uri' was rejected with invalid_json_schema
    assert openai_strict_problems(SCHEMAS['analyst'])   # the full schema is NOT strict-compatible: it must stay local
    assert for_openai_strict(SCHEMAS['analyst']) == GPT_SCHEMAS['analyst']
    assert openai_strict_problems({'type': 'object', 'properties': {'a': {'type': 'string', 'format': 'uri'}},
                                   'required': [], 'additionalProperties': True})


def test_codex_receives_the_strict_schema_and_claude_the_full_one(ledger):
    gpt = command('analyst_gpt', ledger.grant)
    assert gpt[gpt.index('--output-schema') + 1].endswith('analyst.gpt.json')
    assert json.loads(Path(gpt[gpt.index('--output-schema') + 1]).read_text()) == GPT_SCHEMAS['analyst']
    claude = command('analyst_claude', ledger.grant)
    assert json.loads(claude[claude.index('--json-schema') + 1]) == SCHEMAS['analyst']


def test_local_validator_still_rejects_non_https_catalyst_even_though_gpt_schema_has_no_uri_format(service):
    context, _ = build_context(service.ledger.grant, service.data, at=service.clock())
    text, _ = skills()
    output = service.runner.once('analyst_claude', prompt(context, text['TRADER_SKILL.md']))
    output['views'][0]['catalysts'][0]['url'] = 'http://example.invalid/x'
    with pytest.raises(TraderError, match='UNSAFE_SOURCE_URL'):
        validate_analyst(output, context['universe'], service.clock())


class FailOne(SyntheticRunner):
    def __init__(self, ledger, role, code, **kw):
        super().__init__(ledger, **kw)
        self.failing, self.code = role, code

    def once(self, role, text):
        if role == self.failing:
            self.ledger.reserve(role)
            raise TraderError(self.code)
        return super().once(role, text)


@pytest.mark.parametrize('failing', ['analyst_gpt', 'analyst_claude'])
@pytest.mark.parametrize('code', ['MODEL_FAILED', 'TAINTED_RUN', 'SCHEMA_INVALID', 'SKIPPED_QUOTA'])
def test_one_failed_analyst_degrades_the_run_instead_of_losing_the_day(service, failing, code):
    service.runner = FailOne(service.ledger, failing, code, clock=service.clock)
    result = service.run()
    assert result['status'] == 'DEGRADED' and result['degraded'] == {failing: code}
    views = result['decision']['views']
    other = 'analyst_claude' if failing == 'analyst_gpt' else 'analyst_gpt'
    suffix = lambda role: 'reviewer_' + role.split('_')[-1]
    missing = [v for v in views if v['verdict'] == 'MISSING']
    assert {v['analyst'] for v in missing} == {failing, suffix(failing)}
    assert all(v['error'] == code and v['view'] == 'ABSTAIN' and v['raw_view'] is None and v['review'] is None for v in missing)
    assert len(missing) == 2 * len([v for v in views if v['analyst'] == other])
    consensus = [v for v in views if v['analyst'] == 'consensus']
    assert consensus and all(v['view'] == 'ABSTAIN' and v['p_outperform'] == .5 and v['verdict'] == 'ABSTAIN' for v in consensus)
    assert {v['analyst'] for v in views if v['verdict'] == 'KEEP'} == {other, suffix(other)}
    assert len(consensus) == len([v for v in views if v['analyst'] == other])
    predicted = {r['payload']['model_id'] for r in service.store.records('prediction')}
    assert predicted == {'trader:' + other, 'trader:' + suffix(other), 'trader:consensus'}
    assert service.runner.metadata[other]['model'] and result['decision']['models'][failing]['cli_version'] == 'NOT_RUN'
    assert service.ledger.counts()['reviewer'] == 1
    assert scorecard(service.store, synthetic=True)['scores']


def test_both_analysts_failing_is_still_a_failed_run_and_the_reviewer_is_never_called(service):
    class Broken(SyntheticRunner):
        def once(self, role, text):
            self.ledger.reserve(role)
            raise TraderError('MODEL_FAILED')
    service.runner = Broken(service.ledger, clock=service.clock)
    result = service.run()
    assert result['status'] == 'FAILED' and result['error'] == 'MODEL_FAILED'
    assert 'reviewer' not in service.ledger.counts() and not service.store.records('prediction')


@pytest.mark.parametrize('code', ['BUDGET_EXHAUSTED', 'PAUSED', 'MISSED_DECISION_DEADLINE', 'AUTHORIZATION_EXPIRED_OR_NOT_STARTED'])
def test_budget_pause_deadline_and_authorization_errors_never_degrade(service, code):
    service.runner = FailOne(service.ledger, 'analyst_gpt', code, clock=service.clock)
    assert service.run()['status'] == 'FAILED'
    assert not service.store.records('prediction')


def test_reviewer_failure_after_a_degraded_analyst_is_a_failed_run(service):
    service.runner = FailOne(service.ledger, 'analyst_gpt', 'MODEL_FAILED', clock=service.clock)
    original = service.runner.once
    service.runner.once = lambda role, text: (_ for _ in ()).throw(TraderError('MODEL_FAILED')) if role == 'reviewer' else original(role, text)
    assert service.run()['status'] == 'FAILED'


def test_degraded_run_counts_as_the_daily_run_for_health_and_the_preregistration_registers_it(service):
    from scripts.trading_lab.trader_agent.service import preregistration
    artifact = json.loads((Path(__file__).resolve().parents[2] / 'docs/artifacts/trader_agent_preregistration_v1.json').read_text())
    assert artifact['revision'] == 4 and 'DEGRADED' in artifact['degraded_continuation']['run_status']
    from scripts.trading_lab.trader_agent.service import CATCHUP_VARIANT, VARIANTS
    assert artifact['variants'] == VARIANTS and CATCHUP_VARIANT in artifact['variants']
    assert artifact['catchup']['failed_daily_run_required'] is True
    assert preregistration() == artifact['canonical_sha256']

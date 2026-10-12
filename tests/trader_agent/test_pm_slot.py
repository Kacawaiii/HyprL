"""Second-slot behavior through durable budgets, decisions and the broker boundary."""
from dataclasses import replace
import pytest

from scripts.trading_lab.trader_agent.config import TraderError, instant
from scripts.trading_lab.trader_agent.ledger import Ledger


def pm_grant(grant):
    return replace(grant, pm_amendment={'effective_from': '2026-10-13T16:00:00Z',
        'granted_at': '2026-10-12T19:20:00Z', 'not_after': grant.payload['not_after']},
        pm_identity='a' * 64)


def test_slot_budgets_survive_restart_and_never_borrow(tmp_path, grant, clock, monkeypatch):
    clock.at = instant('2026-10-13T16:00:00Z')
    grant = pm_grant(grant)
    am = Ledger(tmp_path / 'am', grant, clock=clock, budget_root=tmp_path / 'bank')
    pm = Ledger(tmp_path / 'pm', grant, clock=clock, budget_root=tmp_path / 'bank', slot='pm')
    for bank in (am, pm):
        bank.reserve('run')
        bank.reserve('analyst_claude')
        bank.reserve('analyst_claude')
        with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
            bank.reserve('analyst_claude')
    restarted = Ledger(tmp_path / 'restart', grant, clock=clock, budget_root=tmp_path / 'bank', slot='pm')
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        restarted.reserve('run')
    assert am.counts()['analyst_claude'] == pm.counts()['analyst_claude'] == 2
    from scripts.trading_lab.trader_agent.config import Authorization
    monkeypatch.setattr(Authorization, 'check_news', lambda self, at: self.check(at))
    pm.arm = 'news'
    pm.reserve('analyst_claude')
    assert pm.counts()['news_analyst_claude'] == 1
    assert am.counts().get('news_analyst_claude', 0) == 0


def test_pm_run_freezes_slot_start_and_labels_follow_full_next_session(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    clock.at = instant('2026-10-13T16:00:03Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    result = service.run()
    assert result['status'] == 'COMPLETE'
    assert result['run_id'] == 'trader:2026-10-13:synthetic:pm'
    context = service.store.records('replay-summary', object_id=result['run_id'] + ':context')[0]['payload']['context']
    assert context['decision_time'] == '2026-10-13T16:00:00Z'
    assert result['decision']['slot'] == 'pm'
    predictions = service.store.records('prediction')
    equity = next(r['payload'] for r in predictions if r['payload']['product'] == 'AAPL' and r['payload']['signal']['label_definition']['horizon'] == '1d')
    assert equity['signal']['label_definition']['entry_at'] == '2026-10-13T17:30:00Z'
    assert equity['signal']['label_definition']['exit_at'] == '2026-10-14T20:00:00Z'


@pytest.mark.parametrize('day,entry,exit_at', [
    ('2026-10-16', '2026-10-16T17:30:00Z', '2026-10-19T20:00:00Z'),
    ('2026-11-27', '2026-11-27T17:30:00Z', '2026-11-30T21:00:00Z'),
    ('2026-11-25', '2026-11-25T17:30:00Z', '2026-11-27T18:00:00Z'),
])
def test_pm_horizon_skips_weekends_holidays_and_uses_half_day_calendar(day, entry, exit_at):
    from scripts.trading_lab.trader_agent.slots import label_window
    assert label_window('AAPL', day, '1d', []) == (instant(entry), instant(exit_at))


def test_pm_failed_primary_recovers_once_in_its_own_budget(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.recovery import recover
    clock.at = instant('2026-10-13T16:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    ledger.reserve('run')
    service.summary('trader:2026-10-13:synthetic:pm', 'FAILED', clock(), error='SOURCE_UNAVAILABLE')
    clock.at = instant('2026-10-13T16:10:00Z')
    result = recover(service)
    assert result['status'] == 'COMPLETE'
    assert result['decision']['slot'] == 'pm'
    assert ledger.counts()['catchup_run'] == 1
    assert recover(service)['state'] in {'ALREADY_ATTEMPTED', 'NO_RECOVERY_NEEDED'}
    clock.at = instant('2026-10-13T17:30:00Z')
    assert service.run(catchup=True)['status'] == 'FAILED'


from tests.trader_agent.test_alpaca_paper import environment, issue


def enable_pm(executor):
    from scripts.trading_lab.trader_agent import slots
    executor.grant = executor.ledger.grant = replace(executor.grant, parent=pm_grant(executor.grant.parent))
    executor.ledger.append('pm_registration', None, **slots.registration_target(executor.grant))


@pytest.mark.parametrize('direction', ['UP', 'DOWN'])
@pytest.mark.parametrize('filled', [True, False])
def test_pm_open_am_lot_or_pending_intent_blocks_both_directions(environment, direction, filled):
    executor, broker, _, clock = environment
    enable_pm(executor)
    clock.at = instant('2026-10-13T13:30:30Z')
    if not filled:
        clock.at = instant('2026-10-13T16:04:00Z')
    issue(executor, horizons=('1d',), slot='am' if filled else 'pm')
    executor.run('execute', accounts=['ia_actions'])
    if filled:
        broker.fill('ia_actions', broker.posts()[0][3]['client_order_id'])
    clock.at = instant('2026-10-13T16:05:00Z')
    issue(executor, horizons=('1d',), direction=direction, probability=.8 if direction == 'UP' else .2, slot='pm')
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 1
    assert any(r['payload']['signal']['slot'] == 'pm' for r in executor.research.records('prediction'))


def test_pm_entry_is_day_market_one_lot_and_cannot_exit_on_entry_day(environment):
    executor, broker, _, clock = environment
    enable_pm(executor)
    clock.at = instant('2026-10-13T16:05:00Z')
    issue(executor, slot='pm')
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 1
    order = broker.posts()[0][3]
    assert order['type'] == 'market' and order['time_in_force'] == 'day'
    broker.fill('ia_actions', order['client_order_id'])
    executor.run('audit', accounts=['ia_actions'])
    live = executor.ledger.projection('ia_actions')[2]
    assert len(live) == 1
    assert next(iter(live.values()))['exit_at'] == '2026-10-14T20:00:00Z'
    clock.at = instant('2026-10-13T19:55:00Z')
    executor.run('exit', accounts=['ia_actions'])
    assert len(broker.posts()) == 1
    clock.at = instant('2026-10-14T19:55:00Z')
    executor.run('exit', accounts=['ia_actions'])
    assert len(broker.posts()) == 2


@pytest.mark.parametrize('at,allowed', [
    ('2026-10-13T16:00:00Z', True), ('2026-10-13T19:44:59Z', True),
    ('2026-10-13T19:45:00Z', False), ('2026-10-13T20:00:00Z', False),
    ('2026-11-27T17:44:59Z', True), ('2026-11-27T17:45:00Z', False),
    ('2026-10-17T16:00:00Z', False), ('2026-12-25T16:00:00Z', False),
])
def test_pm_market_window_refuses_closed_and_last_fifteen_minutes(at, allowed):
    from scripts.trading_lab.trader_agent.alpaca_paper import order_terms
    from scripts.trading_lab.trader_agent.data import calendar_session
    at = instant(at)
    if allowed:
        assert order_terms('ia_actions', 'entry', at, calendar_session(at.date().isoformat()), pm=True)['time_in_force'] == 'day'
    else:
        with pytest.raises(TraderError, match='PAPER_PM_ENTRY_WINDOW'):
            order_terms('ia_actions', 'entry', at, calendar_session(at.date().isoformat()), pm=True)


def test_labels_and_scorecard_separate_am_pm_and_combined(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.scoring import realize, scorecard
    clock.at = instant('2026-10-13T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock)
    data = SyntheticData(ledger, clock)
    service = TraderService(ledger, data, SyntheticRunner(ledger, clock=clock), clock=clock)
    assert service.run()['status'] == 'COMPLETE'
    ledger.slot = 'pm'
    clock.at = instant('2026-10-13T16:00:00Z')
    assert service.run()['status'] == 'COMPLETE'
    clock.at = instant('2026-10-21T21:30:00Z')
    assert realize(service.store, ledger, data, at=clock())['pending'] == 0
    card = scorecard(service.store, synthetic=True)
    assert card['by_slot']['am']['issued_predictions'] == 50
    assert card['by_slot']['pm']['issued_predictions'] == 50
    assert card['combined']['issued_predictions'] == 100
    assert card['combined']['realized_predictions'] == 100
    assert card['by_slot']['pm']['scores']['consensus/equity_etf/1d/raw']['realized'] == 3


from test_execution_v2 import amended
from test_weekend_crypto import weekend_grant


@pytest.mark.parametrize('day', ['2026-10-17', '2026-12-25'])
def test_pm_closed_day_crypto_only_and_recovery(weekend_grant, tmp_path, clock, day):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.recovery import recover
    clock.at = instant(day + 'T16:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(weekend_grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    ledger.reserve('run')
    service.summary('trader:' + day + ':synthetic:pm', 'FAILED', clock(), error='SOURCE_UNAVAILABLE')
    clock.at = instant(day + 'T16:10:00Z')
    result = recover(service)
    assert result['status'] == 'COMPLETE'
    assert {v['asset'] for v in result['decision']['views']} == set(weekend_grant.payload['universe']['crypto'])
    assert ledger.counts().get('yahoo_chart', 0) == 0
    assert all(r['payload']['signal']['label_definition']['variant'] == 'pm_weekend_crypto_recovery_v1'
               for r in service.store.records('prediction'))


@pytest.mark.parametrize('source', ['yahoo_chart', 'coinbase_exchange_public', 'gdelt_doc_api'])
def test_source_budgets_are_independent_and_global_spacing_remains(tmp_path, grant, clock, source):
    from datetime import timedelta
    clock.at = instant('2026-10-13T16:00:00Z')
    grant = pm_grant(grant)
    am = Ledger(tmp_path / 'am', grant, clock=clock, budget_root=tmp_path / 'bank')
    pm = Ledger(tmp_path / 'pm', grant, clock=clock, budget_root=tmp_path / 'bank', slot='pm')
    limit = grant.payload['data_sources'][source]['max_requests_per_day']
    for _ in range(limit):
        am.reserve(source)
        clock.at += timedelta(seconds=6)
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        am.reserve(source)
    pm.reserve(source)
    assert pm.counts()[source] == 1
    assert am.counts()[source] == limit
    if source == 'gdelt_doc_api':
        with pytest.raises(TraderError, match='SOURCE_SPACING'):
            pm.reserve(source)


def test_pm_units_have_preflight_retry_recovery_and_existing_exit_windows(tmp_path):
    from scripts.trading_lab.trader_agent.schedule import render
    from scripts.trading_lab.trader_agent.rehearsal_schedule import render as monitoring
    units = render(tmp_path, '/python', '/grant', tmp_path, path='/bin', paper_authorization='/paper', include_pm=True)
    units.update(monitoring(tmp_path, '/python', '/grant', tmp_path, '/paper', '/data', path='/bin', include_pm=True))
    assert '16:00:00 UTC' in units['hyprl-trader-run-pm.timer']
    assert '15:15:00 UTC' in units['hyprl-trader-rehearsal-pm.timer']
    assert '15:45:00 UTC' in units['hyprl-trader-rehearsal-pm-retry.timer']
    assert '16..23:05/10:00 UTC' in units['hyprl-trader-recover-pm.timer']
    assert '"--slot" "pm"' in units['hyprl-trader-run-pm.service']
    assert '12:55:00 America/New_York' in units['hyprl-trader-paper-exit-ia-actions.timer']
    assert '*:30:00 UTC' in units['hyprl-trader-paper-exit-ia-crypto.timer']


def test_pm_preflight_reuses_successful_probes_and_has_independent_attempts(tmp_path, clock):
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState, rehearse
    clock.at = instant('2026-10-13T15:15:00Z')
    state = MonitorState(tmp_path / 'monitor')
    state.save('2026-10-13', 'primary', {'state': 'GREEN'})
    result = rehearse(state, [('executor', lambda: {'methods': ['GET']}), ('model', lambda: {'cached': True})],
                      clock=clock, reports=tmp_path, run_slot='pm', notify=lambda *args: 'DELIVERED')
    assert result['state'] == 'GREEN' and result['slot'] == 'pm-primary'
    assert state.get('2026-10-13', 'primary') == {'state': 'GREEN'}


def test_pm_registration_is_required_and_old_am_binding_is_preserved(environment):
    from scripts.trading_lab.trader_agent import slots
    executor, broker, _, clock = environment
    bindings = executor.ledger.events(event='binding')
    executor.grant = executor.ledger.grant = replace(executor.grant, parent=pm_grant(executor.grant.parent))
    clock.at = instant('2026-10-13T16:05:00Z')
    issue(executor, slot='pm')
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'BLOCKED'
    assert not broker.posts()
    executor.ledger.append('pm_registration', None, **slots.registration_target(executor.grant))
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert executor.ledger.events(event='binding') == bindings


def test_pm_budget_exhaustion_keeps_pending_without_am_borrow(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.recovery import recover
    clock.at = instant('2026-10-13T16:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    ledger.reserve('run')
    for _ in range(2):
        ledger.reserve('reviewer')
    service.summary('trader:2026-10-13:synthetic:pm', 'FAILED', clock(), error='MODEL_FAILED')
    clock.at = instant('2026-10-13T16:10:00Z')
    assert recover(service)['state'] == 'BUDGET_EXHAUSTED'
    assert not service.store.records('prediction')
    ledger.slot = 'am'
    assert ledger.counts() == {}


def test_missing_pm_primary_can_start_late_but_keeps_sixteen_cutoff(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.recovery import recover
    clock.at = instant('2026-10-13T16:05:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    result = recover(service)
    assert result['status'] == 'COMPLETE'
    context = service.store.records('replay-summary', object_id=result['run_id'] + ':context')[0]['payload']['context']
    assert context['decision_time'] == '2026-10-13T16:00:00Z'


def test_pm_post_audit_catches_missing_pm_even_when_am_complete(environment):
    from scripts.trading_lab.trader_agent.post_audit import inspect_audit
    executor, _, _, clock = environment
    enable_pm(executor)
    clock.at = instant('2026-10-13T13:30:30Z')
    issue(executor, direction='ABSTAIN', probability=.5)
    clock.at = instant('2026-10-13T17:40:00Z')
    result = inspect_audit(executor, require_daily_run=True, require_pm_run=True)
    assert 'MISSING_PM_RUN' in result['errors']
    assert 'MISSING_DAILY_RUN' not in result['errors']


def test_pm_cockpit_export_marks_run_slot_and_slot_scorecards(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.scoring import scorecard
    from scripts.radar.cockpit_export import load_trader, build_home
    clock.at = instant('2026-10-13T16:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    service.run()
    trader = load_trader(service.store, scorecard(service.store, synthetic=True), include_synthetic=True)
    home = build_home({'events': [], 'sources': []}, trader)
    assert home['trader']['slot'] == 'pm'
    assert home['trader']['by_slot']['pm']['issued_predictions'] == 50
    assert home['trader']['combined']['issued_predictions'] == 50


def test_pm_news_arm_uses_pm_base_and_separate_budget(tmp_path, grant, clock, monkeypatch):
    from scripts.trading_lab.trader_agent.config import Authorization
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.news_spec import SPEC_HASH
    from scripts.trading_lab.sources.canonical import sha256_canonical
    monkeypatch.setattr(Authorization, 'check_news', lambda self, at: self.check(at))
    grant = replace(pm_grant(grant), news_decision={'schema': 'synthetic-decision'})
    clock.at = instant('2026-10-13T16:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', grant, clock=clock, slot='pm')
    def news(cutoff):
        context = {'decision_time': '2026-10-13T16:00:00Z', 'spec_hash': SPEC_HASH, 'events': [], 'veille': {'status': 'UNAVAILABLE'}}
        return context, sha256_canonical(context)
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock, news_builder=news)
    base = service.run()
    result = service.run_news()
    assert result['status'] == 'COMPLETE'
    assert result['base_run_id'] == base['run_id']
    assert ledger.counts()['analyst_claude'] == ledger.counts()['news_analyst_claude'] == 1
    news_predictions = [r['payload'] for r in service.store.records('prediction') if r['payload']['signal'].get('arm') == 'news']
    assert news_predictions and all(p['signal']['slot'] == 'pm' for p in news_predictions)
    assert all(p['signal']['label_definition']['variant'] == 'news_pm_slot_v1' for p in news_predictions)


def test_pm_recovery_after_original_anchor_stays_prospective_and_same_day(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    from scripts.trading_lab.trader_agent.recovery import recover
    clock.at = instant('2026-10-13T16:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    ledger.reserve('run')
    service.summary('trader:2026-10-13:synthetic:pm', 'FAILED', clock(), error='SOURCE_UNAVAILABLE')
    clock.at = instant('2026-10-13T18:05:00Z')
    result = recover(service)
    assert result['status'] == 'COMPLETE'
    assert result['decision']['pm_label_anchor'] == '2026-10-13T19:30:00Z'
    definition = next(r['payload']['signal']['label_definition'] for r in service.store.records('prediction')
                      if r['payload']['product'] == 'AAPL' and r['payload']['signal']['label_definition']['horizon'] == '1d')
    assert definition['entry_at'] == '2026-10-13T19:30:00Z'
    assert definition['exit_at'] == '2026-10-14T20:00:00Z'
    assert definition['variant'] == 'pm_slot_recovery_v1'

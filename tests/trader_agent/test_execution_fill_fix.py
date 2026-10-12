"""Execution behavior at the mocked broker boundary; no real dispatch."""
from datetime import timedelta

from scripts.trading_lab.trader_agent.config import instant
from scripts.trading_lab.trader_agent.data import calendar_session
from tests.trader_agent.test_alpaca_paper import environment, issue


def test_equity_waits_for_open_then_submits_day_once(environment):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-12T12:00:00Z')
    issue(executor, horizons=('1d',))
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert not broker.posts()
    assert not executor.ledger.events(event='intent')
    clock.at = instant('2026-10-12T13:30:30Z')
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert broker.posts()[0][3]['time_in_force'] == 'day'
    executor.run('execute', accounts=['ia_actions'])
    assert len(broker.posts()) == 1


def test_expired_entry_alerts_once_and_never_creates_phantom_lot(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:30:30Z')
    executor.run('execute', accounts=['ia_actions'])
    key = broker.posts()[0][3]['client_order_id']
    broker.orders[('ia_actions', key)]['status'] = 'expired'
    executor.run('audit', accounts=['ia_actions'])
    executor.run('audit', accounts=['ia_actions'])
    assert not any(float(l['qty']) for l in executor.ledger.projection('ia_actions')[2].values())
    alerts = [e for e in executor.ledger.events('ia_actions', 'alert') if e.get('client_id') == key]
    assert len(alerts) == 1 and alerts[0]['code'] == 'PAPER_ORDER_EXPIRED'
    assert executor.ledger.events('ia_actions', 'order_audit')[-1]['orders'][0]['disposition'] == 'ALERTED'
    assert len(broker.posts()) == 1


def test_expired_day_exit_retries_residual_at_next_open_once(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:30:30Z')
    executor.run('execute', accounts=['ia_actions'])
    broker.fill('ia_actions', broker.posts()[0][3]['client_order_id'])
    clock.at = instant('2026-10-12T19:55:00Z')
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'COMPLETE'
    exit_order = broker.posts()[-1][3]
    assert exit_order['time_in_force'] == 'day' and exit_order['qty'] == '36'
    broker.fill('ia_actions', exit_order['client_order_id'], .5)
    broker.orders[('ia_actions', exit_order['client_order_id'])]['status'] = 'expired'
    clock.at = instant('2026-10-13T13:30:30Z')
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'COMPLETE'
    retry = broker.posts()[-1][3]
    assert retry['qty'] == '18' and retry['client_order_id'] != exit_order['client_order_id']
    executor.run('exit', accounts=['ia_actions'])
    assert len(broker.posts()) == 3
    broker.fill('ia_actions', retry['client_order_id'])
    executor.run('audit', accounts=['ia_actions'])
    assert all(float(l['qty']) == 0 for l in executor.ledger.projection('ia_actions')[2].values())
    assert not executor.ledger.halted('ia_actions')


def test_next_open_cancels_pending_exit_before_retry(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:30:30Z')
    executor.run('execute', accounts=['ia_actions'])
    broker.fill('ia_actions', broker.posts()[0][3]['client_order_id'])
    clock.at = instant('2026-10-12T19:55:00Z')
    executor.run('exit', accounts=['ia_actions'])
    old = broker.posts()[-1][3]['client_order_id']
    clock.at = instant('2026-10-13T13:30:30Z')
    executor.run('exit', accounts=['ia_actions'])
    assert broker.orders[('ia_actions', old)]['status'] == 'canceled'
    assert len(broker.posts()) == 3
    assert broker.posts()[-1][3]['qty'] == '36'


def test_partial_entry_is_canceled_before_horizon_exit_and_no_late_fill(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:30:30Z')
    executor.run('execute', accounts=['ia_actions'])
    entry = broker.posts()[0][3]['client_order_id']
    broker.fill('ia_actions', entry, .5)
    clock.at = instant('2026-10-12T19:55:00Z')
    executor.run('exit', accounts=['ia_actions'])
    assert broker.orders[('ia_actions', entry)]['status'] == 'canceled'
    assert broker.posts()[-1][3]['qty'] == '18'


import pytest
from scripts.trading_lab.trader_agent import alpaca_paper as paper
from scripts.trading_lab.trader_agent.schedule import render
from scripts.trading_lab.trader_agent.config import TraderError


@pytest.mark.parametrize('day,entry,exit_at', [
    ('2026-10-12', '13:30:30', '19:55:00'),
    ('2026-11-02', '14:30:30', '20:55:00'),
    ('2026-11-27', '14:30:30', '17:55:00'),
])
def test_day_windows_follow_dst_and_half_days(day, entry, exit_at):
    session = calendar_session(day)
    assert paper.order_terms('ia_actions', 'entry', instant(day + 'T' + entry + 'Z'), session)['time_in_force'] == 'day'
    assert paper.order_terms('ia_actions', 'exit', instant(day + 'T' + exit_at + 'Z'), session)['time_in_force'] == 'day'
    with pytest.raises(TraderError, match='PAPER_ENTRY_WINDOW'):
        paper.order_terms('ia_actions', 'entry', session.open_at, session)
    with pytest.raises(TraderError, match='PAPER_EXIT_WINDOW'):
        paper.order_terms('ia_actions', 'exit', session.close_at, session)
    assert calendar_session('2026-12-25') is None


def test_user_timers_separate_open_entries_sweep_and_daily_audit(tmp_path):
    units = render(tmp_path, '/python', '/grant', tmp_path, path='/bin', paper_authorization='/paper')
    assert '09:30:30 America/New_York' in units['hyprl-trader-paper-open.timer']
    service = units['hyprl-trader-paper-open.service']
    assert service.index('paper-exit') < service.index('paper-execute')
    assert 'ia_actions' in service
    assert '15:55:00 America/New_York' in units['hyprl-trader-paper-exit-ia-actions.timer']
    assert '12:55:00 America/New_York' in units['hyprl-trader-paper-exit-ia-actions.timer']
    assert 'paper-audit' in units['hyprl-trader-paper-audit.service']
    assert 'paper-audit' in units['hyprl-trader-health.service']
    assert '*:30:00 UTC' in units['hyprl-trader-paper-exit-ia-crypto.timer']
    assert '--paper-account" "ia_crypto' in units['hyprl-trader-run.service']


def test_real_fill_gap_is_separate_from_open_close_label(environment):
    from scripts.trading_lab.platform.contracts import LabelRecord
    from scripts.trading_lab.trader_agent.config import iso
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:30:30Z')
    executor.run('execute', accounts=['ia_actions'])
    entry = broker.posts()[0][3]['client_order_id']
    broker.fill('ia_actions', entry)
    broker.orders[('ia_actions', entry)]['filled_avg_price'] = '101'
    clock.at = instant('2026-10-12T19:55:00Z')
    executor.run('exit', accounts=['ia_actions'])
    exit_key = broker.posts()[-1][3]['client_order_id']
    broker.fill('ia_actions', exit_key)
    broker.orders[('ia_actions', exit_key)]['filled_avg_price'] = '109'
    executor.run('audit', accounts=['ia_actions'])
    lot = next(iter(executor.ledger.projection('ia_actions')[2].values()))
    clock.at = instant('2026-10-12T21:30:00Z')
    prediction = next(r['payload'] for r in executor.research.records('prediction') if r['identity'] == lot['prediction_hash'])
    label = LabelRecord(label_id='synthetic-label', prediction_id=prediction['prediction_id'], prediction_hash=lot['prediction_hash'],
        product='AAPL', horizon_seconds=100, target='synthetic', realized_at=lot['exit_at'],
        available_at=iso(clock()), recorded_at=iso(clock()), value={'raw_prices': {'AAPL': [100, 110]}, 'raw_return': .1},
        provenance={'method': 'synthetic'}, version='1')
    executor.research.append_label(label)
    before = executor.research.verify()
    result = executor.report()
    gap = result['execution_gap_scores'][0]
    assert gap['entry_gap_bps'] == '100.00'
    assert float(gap['exit_gap_bps']) == pytest.approx(90.9090909)
    assert gap['entry_price'] == '101' and gap['exit_price'] == '109'
    assert executor.research.verify() == before


def test_expiry_exposes_operator_alert_file(environment):
    import json
    executor, broker, _, _ = environment
    issue(executor, horizons=('1d',))
    executor.run('execute', accounts=['ia_actions'])
    key = broker.posts()[0][3]['client_order_id']
    broker.orders[('ia_actions', key)]['status'] = 'expired'
    executor.run('audit', accounts=['ia_actions'])
    alert = json.loads((executor.ledger.root / 'paper-alert-ia_actions.json').read_text())
    assert alert['code'] == 'PAPER_ORDER_EXPIRED'


def test_sweep_restart_after_acceptance_does_not_duplicate_retry(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    executor.run('execute', accounts=['ia_actions'])
    broker.fill('ia_actions', broker.posts()[0][3]['client_order_id'])
    clock.at = instant('2026-10-13T13:30:30Z')  # no close invocation at all
    broker.crash = True
    with pytest.raises(SystemExit):
        executor.run('exit', accounts=['ia_actions'])
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 2


def test_missing_entry_window_alerts_without_late_order(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    clock.at = instant('2026-10-12T13:35:00Z')
    executor.run('execute', accounts=['ia_actions'])
    executor.run('audit', accounts=['ia_actions'])
    assert not broker.posts()
    assert any(e['code'] == 'PAPER_ELIGIBLE_ENTRY_NOT_DISPATCHED' for e in executor.ledger.events('ia_actions', 'alert'))


def test_audit_is_get_only_and_classifies_unfilled_orders(environment):
    executor, broker, _, _ = environment
    issue(executor, horizons=('1d',))
    executor.run('execute', accounts=['ia_actions'])
    before = len(broker.calls)
    result = executor.run('audit', accounts=['ia_actions'])
    assert result['accounts'][0]['audit']['orders'][0]['disposition'] == 'ALERTED'
    assert all(c[1] == 'GET' for c in broker.calls[before:])

"""Closed native paper lots, modeled costs, persistent safety stop and planner isolation."""
from dataclasses import replace
from datetime import timedelta

import pytest

from scripts.trading_lab.trader_agent import alpaca_paper as paper
from scripts.trading_lab.trader_agent.config import Authorization, TraderError, iso
from scripts.trading_lab.trader_agent.downgrade_guard import evaluate, realized_trades
from scripts.trading_lab.trader_agent.execution_tiers import HALF_DOWNGRADE
from scripts.trading_lab.trader_agent.news_spec import registration_target
from test_alpaca_paper import environment, issue


def seed(ledger, clock, index, pnl, *, partial=False, filled=True):
    identity = f'synthetic-lot-{index}'
    lot = {'lot_id': identity, 'variant': HALF_DOWNGRADE, 'qty': '2' if partial else '1',
           'symbol': 'AAPL', 'reference_price': '100', 'run_id': 'synthetic-run', 'horizon': '1d',
           'prediction_hash': 'a' * 64, 'entry_at': iso(clock()), 'exit_at': iso(clock()), 'session': clock().date().isoformat()}
    for purpose in ('entry', 'exit'):
        client = purpose + identity
        ledger.append('intent', 'ia_actions', client_id=client, purpose=purpose, run_id='synthetic-run',
                      spec_hash=paper.SPEC_HASH, lots=[lot], order={'qty': lot['qty'], 'symbol': 'AAPL'})
        quantity = '1' if filled else '0'
        ledger.append('order', 'ia_actions', client_id=client, status='partially_filled' if partial and purpose == 'exit' else 'filled',
                      filled_qty=quantity, filled_avg_price='100', filled_at=iso(clock()))
        ledger.append('trade_outcome', 'ia_actions', client_id=client, lot_id=identity, purpose=purpose,
                      cumulative_lot_qty=quantity, broker_status='partially_filled' if partial and purpose == 'exit' else 'filled',
                      state='FILLED' if purpose == 'entry' and filled else 'NOT_TRADED' if purpose == 'entry' else 'CLOSED',
                      entry_price='100', lot_pnl_before_broker_fees=str(pnl))
    clock.at += timedelta(seconds=1)


@pytest.fixture
def guarded(environment, monkeypatch):
    executor, broker, grant, clock = environment
    monkeypatch.setattr(Authorization, 'check_news', lambda self, at: self.check(at))
    grant = replace(grant, parent=replace(grant.parent, news_decision={'schema': 'synthetic-operator-decision'}))
    executor.grant = executor.ledger.grant = grant
    executor.ledger.append('news_guard_registration', None, **registration_target(grant))
    return executor, broker, grant, clock


@pytest.mark.parametrize('n,positive,state', [(19, 0, 'TOO_EARLY'), (20, 8, 'HALTED'), (20, 9, 'ACTIVE'), (20, 20, 'ACTIVE')])
def test_twenty_realized_trades_strict_forty_five_percent_threshold(guarded, n, positive, state):
    executor, broker, _, clock = guarded
    for i in range(n):
        seed(executor.ledger, clock, i, 1 if i < positive else -1)
    result = evaluate(executor.ledger)
    assert result['state'] == state
    assert result['realized'] == n
    assert result['positive_rate_after_costs'] == positive / n
    assert len(executor.ledger.events('ia_actions', 'tier_halt')) == int(state == 'HALTED')
    assert not executor.ledger.halted('ia_actions')
    assert not broker.calls


def test_small_gross_profit_is_negative_after_costs_and_unfilled_partial_are_excluded(guarded):
    executor, _, _, clock = guarded
    for i in range(20):
        seed(executor.ledger, clock, i, .10)
    seed(executor.ledger, clock, 'partial', 10, partial=True)
    seed(executor.ledger, clock, 'unfilled', 10, filled=False)
    trades = realized_trades(executor.ledger)
    assert len(trades) == 20
    assert {t['pnl_after_costs'] for t in trades} == {'-0.0500'}
    assert evaluate(executor.ledger)['state'] == 'HALTED'


def test_repeated_exit_outcome_dedupes_and_halt_survives_restart_and_improved_rate(guarded):
    executor, _, grant, clock = guarded
    for i in range(20):
        seed(executor.ledger, clock, i, 1 if i < 8 else -1)
    last = executor.ledger.events('ia_actions', 'trade_outcome')[-1]
    executor.ledger.append('trade_outcome', 'ia_actions', **{k: v for k, v in last.items()
                          if k not in {'schema', 'event_id', 'event', 'at', 'account', 'account_suffix'}})
    assert len(realized_trades(executor.ledger)) == 20
    assert evaluate(executor.ledger)['state'] == 'HALTED'
    assert len(executor.ledger.events('ia_actions', 'alert')) == 1
    for i in range(20, 40):
        seed(executor.ledger, clock, i, 1)
    reopened = paper.PaperLedger(executor.ledger.root, grant, clock=clock)
    result = evaluate(reopened)
    assert result['state'] == 'HALTED' and result['positive_rate_after_costs'] == .7
    assert len(reopened.events('ia_actions', 'tier_halt')) == 1
    assert len(reopened.events('ia_actions', 'alert')) == 1


@pytest.mark.parametrize('invalid_hash', ['0' * 64, 'e677c1874c2d549e80bc8dcdf48f8302593407b0560f2c29ca803c5d66d89a0a'])
def test_guard_registration_is_required_no_implicit_rebinding(environment, monkeypatch, invalid_hash):
    executor, _, grant, clock = environment
    monkeypatch.setattr(Authorization, 'check_news', lambda self, at: self.check(at))
    grant = replace(grant, parent=replace(grant.parent, news_decision={'schema': 'synthetic-operator-decision'}))
    executor.ledger.grant = grant
    before = executor.ledger.events(event='binding')
    with pytest.raises(TraderError, match='PAPER_NEWS_GUARD_REGISTRATION_REQUIRED'):
        evaluate(executor.ledger)
    assert executor.ledger.events(event='binding') == before
    executor.ledger.append('news_guard_registration', None, **(registration_target(grant) | {'news_spec_hash': invalid_hash}))
    with pytest.raises(TraderError, match='PAPER_NEWS_GUARD_REGISTRATION_REQUIRED'):
        evaluate(executor.ledger)


def test_guard_blocks_downgrade_entries_but_keeps_full_size_and_scoring(guarded):
    executor, broker, _, clock = guarded
    for i in range(20):
        seed(executor.ledger, clock, i, -1)
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_direction='ABSTAIN')
    assert executor.entries('ia_actions', paper.D(100000), []) == []
    assert executor.research.records('prediction')
    clock.at += timedelta(seconds=1)
    issue(executor, horizons=('1d',), verdict='KEEP', peer_direction='ABSTAIN')
    assert [(l['qty'], l['variant']) for l in executor.entries('ia_actions', paper.D(100000), [])] == [('36', 'alpaca_open_entry_v2')]
    assert not broker.calls


def test_halted_tier_cancels_accepted_unfilled_entries(guarded):
    executor, broker, _, clock = guarded
    clock.at -= timedelta(days=1)
    for i in range(19):
        seed(executor.ledger, clock, i, -1)
    clock.at += timedelta(days=1)
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_direction='ABSTAIN')
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 1
    seed(executor.ledger, clock, 19, -1)
    result = executor.run('execute', accounts=['ia_actions'])
    assert result['state'] == 'COMPLETE'
    assert result['accounts'][0]['downgrade_guard']['state'] == 'HALTED'
    assert any(c[1] == 'DELETE' for c in broker.calls)
    assert len(broker.posts()) == 1


def test_halted_tier_preserves_scheduled_exits_and_outcomes(guarded):
    executor, broker, _, clock = guarded
    clock.at -= timedelta(days=1)
    for i in range(19):
        seed(executor.ledger, clock, i, -1)
    clock.at += timedelta(days=1)
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_direction='ABSTAIN')
    executor.run('execute', accounts=['ia_actions'])
    key = next(iter(broker.orders))[1]
    broker.fill('ia_actions', key)
    executor.run('status', accounts=['ia_actions'])
    seed(executor.ledger, clock, 19, -1)
    clock.at = clock.at.replace(hour=19, minute=55, second=0)
    result = executor.run('exit', accounts=['ia_actions'])
    assert result['state'] == 'COMPLETE'
    assert result['accounts'][0]['downgrade_guard']['state'] == 'HALTED'
    assert broker.posts()[-1][3]['time_in_force'] == 'day'
    exit_key = list(broker.orders)[-1][1]
    broker.fill('ia_actions', exit_key)
    assert executor.run('status', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert evaluate(executor.ledger)['realized'] == 21


def test_halted_tier_marks_crashed_unsubmitted_intent_not_traded(guarded):
    executor, broker, _, clock = guarded
    clock.at -= timedelta(days=1)
    for i in range(19):
        seed(executor.ledger, clock, i, -1)
    clock.at += timedelta(days=1)
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_direction='ABSTAIN')
    plan = executor.plans('ia_actions', executor.entries('ia_actions', paper.D(100000), []), 'entry')[0]
    executor.ledger.reserve('ia_actions', plan)  # crash before broker submission
    seed(executor.ledger, clock, 19, -1)
    result = executor.run('execute', accounts=['ia_actions'])
    assert result['state'] == 'COMPLETE'
    assert not broker.posts()
    assert executor.ledger.projection('ia_actions')[1][plan['client_id']]['status'] == 'not_submitted'
    assert executor.ledger.events('ia_actions', 'trade_outcome')[-1]['state'] == 'NOT_TRADED'


def test_removing_decision_file_cannot_disable_registered_guard_or_resume_halt(guarded):
    executor, _, grant, clock = guarded
    for i in range(20):
        seed(executor.ledger, clock, i, -1)
    assert evaluate(executor.ledger)['state'] == 'HALTED'
    executor.ledger.grant = replace(grant, parent=replace(grant.parent, news_decision=None))
    assert evaluate(executor.ledger)['state'] == 'HALTED'
    assert len(executor.ledger.events('ia_actions', 'tier_halt')) == 1

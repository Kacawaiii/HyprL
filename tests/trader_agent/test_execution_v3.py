"""Operator v3 tiers through the paper planner and immutable ledger boundary."""
from dataclasses import replace
from datetime import timedelta
import json

import pytest

from scripts.trading_lab.trader_agent import alpaca_paper as paper, weekend
from scripts.trading_lab.trader_agent.config import instant
from scripts.trading_lab.trader_agent.execution_tiers import HALF_CRYPTO, HALF_DOWNGRADE, execution_scores
from scripts.trading_lab.trader_agent.paper_spec import execution_preregistration
from scripts.trading_lab.trader_agent.execution_replay import replay
from test_alpaca_paper import environment, issue
from test_execution_v2 import amended
from test_weekend_crypto import weekend_grant


@pytest.mark.parametrize('probability,verdict,quantity,tier', [
    (.80, 'KEEP', '36', 'alpaca_open_entry_v2'),
    (.80, 'DOWNGRADE', '18', HALF_DOWNGRADE),
    (.53, 'KEEP', '3', 'alpaca_open_entry_v2'),
    (.53, 'DOWNGRADE', '1', HALF_DOWNGRADE),
    (.51, 'DOWNGRADE', None, None),
])
def test_action_tiers_floor_v2_whole_shares_then_halve(environment, probability, verdict, quantity, tier):
    executor, broker, _, _ = environment
    issue(executor, horizons=('1d',), probability=probability, verdict=verdict, peer_direction='ABSTAIN')
    lots = executor.entries('ia_actions', paper.D(100000), [])
    assert [(l['qty'], l['variant']) for l in lots] == ([(quantity, tier)] if quantity else [])
    assert not broker.calls


def test_keep_precedes_agreeing_downgrade_without_double_order(environment):
    executor, _, _, _ = environment
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_verdict='KEEP')
    lots = executor.entries('ia_actions', paper.D(100000), [])
    assert [(l['qty'], l['variant']) for l in lots] == [('36', 'alpaca_open_entry_v2')]


@pytest.mark.parametrize('peer,verdict,peer_verdict,expected', [
    ('UP', 'KEEP', 'KEEP', [('113.63636363', 'alpaca_open_entry_v1')]),
    ('ABSTAIN', 'KEEP', 'KEEP', [('56.81818181', HALF_CRYPTO)]),
    ('ABSTAIN', 'KEEP', 'MISSING', [('56.81818181', HALF_CRYPTO)]),
    ('ABSTAIN', 'DOWNGRADE', 'KEEP', []),
    ('UP', 'KEEP', 'DOWNGRADE', []),
    ('DOWN', 'KEEP', 'REJECT', []),
])
def test_crypto_full_single_half_downgrade_and_conflict(environment, peer, verdict, peer_verdict, expected):
    executor, _, grant, _ = environment
    grant.parent.payload['universe']['crypto'] = ['SOL-USD']
    issue(executor, assets=('SOL-USD',), horizons=('1d',), verdict=verdict, peer_direction=peer, peer_verdict=peer_verdict)
    assert [(l['qty'], l['variant']) for l in executor.entries('ia_crypto', paper.D(100000), [])] == expected


def test_crypto_down_single_stays_flat_and_threshold_is_unchanged(environment):
    executor, _, grant, _ = environment
    grant.parent.payload['universe']['crypto'] = ['SOL-USD']
    issue(executor, assets=('SOL-USD',), horizons=('1d',), direction='DOWN', probability=.2, peer_direction='ABSTAIN')
    assert executor.entries('ia_crypto', paper.D(100000), []) == []
    executor.clock.at += timedelta(seconds=1)
    issue(executor, assets=('SOL-USD',), horizons=('1d',), probability=.54, peer_direction='ABSTAIN')
    assert executor.entries('ia_crypto', paper.D(100000), []) == []


def test_weekend_single_half_is_separate_population_and_idempotent(environment):
    executor, broker, grant, clock = environment
    clock.at = instant('2026-10-11T12:10:00Z')
    grant.parent.payload['universe']['crypto'] = ['SOL-USD', 'AVAX-USD', 'LINK-USD', 'DOGE-USD', 'LTC-USD']
    executor.grant = executor.ledger.grant = replace(grant, parent=replace(grant.parent,
        amendment_identity='synthetic-amendment', weekend_decision={'decided_at': '2026-10-09T22:10:00Z'}))
    executor.ledger.append('weekend_registration', None, **weekend.registration_target(executor.grant))
    issue(executor, assets=('SOL-USD',), horizons=('1d',), variant=weekend.VARIANT,
          artifact_hash=weekend.PREREG_HASH, peer_direction='ABSTAIN')
    assert executor.run('execute')['state'] == 'COMPLETE'
    assert broker.posts()[0][3]['qty'] == '56.81818181'
    assert executor.run('execute')['state'] == 'COMPLETE'
    assert len(broker.posts()) == 1
    score = execution_scores(executor.ledger)
    assert list(score) == ['ia_crypto/weekend_crypto_v1/crypto_single_kept_half_v1/1d']
    assert next(iter(score.values()))['filled'] == 1
    clock.at = instant('2026-10-12T13:30:00Z')
    assert executor.run('exit', accounts=['ia_crypto'])['state'] == 'COMPLETE'
    score = execution_scores(executor.ledger)
    assert next(iter(score.values()))['open_lots'] == 0
    assert paper.decimal(next(iter(score.values()))['realized_lot_pnl_before_broker_fees']) == 0


def test_v3_execution_registration_precedes_first_new_run(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.ledger import Ledger
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    clock.at = instant('2026-10-11T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    result = service.run()
    assert result['decision']['preregistration_hash'] == weekend.PREREG_HASH
    assert result['decision']['execution_preregistration_hash'] == execution_preregistration()
    assert {HALF_CRYPTO, HALF_DOWNGRADE} <= set(result['multiple_testing_variants'])


def test_action_half_after_hours_keeps_limit_quote_and_time_rules(environment, tmp_path):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-12T20:10:00Z')
    quotes = tmp_path / 'quotes.json'
    quotes.write_text(json.dumps({'quotes': {'AAPL': {'bid': 99.99, 'ask': 100.01, 'at': paper.iso(clock())}}}))
    executor.quotes = paper.PrivateQuotes(quotes)
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_direction='ABSTAIN', variant='alpaca_after_hours_v1')
    result = executor.run('execute', dry_run=True, accounts=['ia_actions'])
    assert result['accounts'][0]['planned_orders'] == [{'symbol': 'AAPL', 'qty': '18', 'side': 'buy',
        'type': 'limit', 'time_in_force': 'day', 'extended_hours': True, 'limit_price': '100.01'}]
    assert not broker.posts()


def test_offline_replay_uses_recorded_inputs_and_leaves_both_journals_unchanged(environment):
    executor, broker, grant, clock = environment
    issue(executor, horizons=('1d', '5d'), probability=.53, verdict='DOWNGRADE', peer_direction='ABSTAIN')
    research_before, paper_before = executor.research.verify(), executor.ledger.store.verify()
    result = replay(executor.research, grant, clock().date().isoformat())
    actions = result['accounts'][0]
    assert actions['orders'] == [{'symbol': 'AAPL', 'qty': '1', 'side': 'buy', 'type': 'market', 'time_in_force': 'opg'}]
    assert [(l['horizon'], l['qty'], l['variant']) for l in actions['lots']] == [('1d', '1', HALF_DOWNGRADE)]
    assert result['network_requests'] == 0 and result['no_order']
    assert executor.research.verify() == research_before and executor.ledger.store.verify() == paper_before
    assert not broker.calls


def test_scored_pnl_keeps_full_and_half_tiers_apart_without_double_counting(environment):
    executor, broker, _, clock = environment
    issue(executor, horizons=('1d',))
    executor.run('execute', accounts=['ia_actions'])
    first = next(iter(broker.orders))[1]
    broker.fill('ia_actions', first)
    clock.at += timedelta(seconds=1)
    issue(executor, horizons=('1d',), verdict='DOWNGRADE', peer_direction='ABSTAIN')
    executor.run('execute', accounts=['ia_actions'])
    second = list(broker.orders)[-1][1]
    broker.fill('ia_actions', second)
    clock.at = instant('2026-10-12T19:40:00Z')
    executor.run('exit', accounts=['ia_actions'])
    exit_key = list(broker.orders)[-1][1]
    broker.fill('ia_actions', exit_key)
    broker.orders[('ia_actions', exit_key)]['filled_avg_price'] = '110'
    executor.run('status', accounts=['ia_actions'])
    scores = execution_scores(executor.ledger)
    assert scores['ia_actions/weekday/alpaca_open_entry_v2/1d']['realized_lot_pnl_before_broker_fees'] == '360'
    assert scores['ia_actions/weekday/half_size_downgrade_v1/1d']['realized_lot_pnl_before_broker_fees'] == '180'
    assert all(s['open_lots'] == 0 for s in scores.values())
    executor.run('status', accounts=['ia_actions'])
    assert execution_scores(executor.ledger) == scores


def test_rebind_retains_immutable_v1_v2_chain_and_weekend_registration(environment, tmp_path):
    from scripts.trading_lab.research.store import ResearchStore
    from scripts.trading_lab.sources.canonical import sha256_canonical
    from scripts.trading_lab.trader_agent.paper_spec import PREVIOUS_SPEC_HASH
    from pathlib import Path
    executor, broker, grant, clock = environment
    root = tmp_path / 'historical-paper'
    store = ResearchStore(root / 'paper-evidence')
    first = {'schema': 'alpaca-paper-event-v1', 'event': 'binding', 'event_id': 'synthetic-v1',
             'at': paper.iso(clock()), 'account': None, 'grant_hash': grant.identity,
             'spec_hash': 'c07ae3243d0b130c4ba9972158a2081ac3ff7fa5f280879a003915b98af5b386'}
    second = {**first, 'event_id': 'synthetic-v2', 'spec_hash': PREVIOUS_SPEC_HASH,
              'previous_binding_hash': sha256_canonical(first), 'previous_spec_hash': first['spec_hash'],
              'previous_grant_hash': grant.identity,
              'operator_decision_hash': '0caee3e1145b4cc401cccaface8ca7ecc928c6337c335ab23cc1695a3677e7a9'}
    registration = {**first, 'event': 'weekend_registration', 'event_id': 'synthetic-weekend',
                    **weekend.registration_target(grant)}
    for event in (first, second, registration):
        store.append('replay-summary', event, object_id=event['event_id'], recorded_at=event['at'])
    # Synthetic decision with the real approved shape, no broker acquisition.
    decision = {'schema': 'synthetic-operator-decision', 'decided_at': paper.iso(clock())}
    decision_path = tmp_path / 'decision.json'
    decision_path.write_text(json.dumps(decision))
    from unittest.mock import patch
    with patch.object(paper, 'OPERATOR_DECISION_HASH', sha256_canonical(decision)):
        factory = lambda g, name, **kw: paper.PaperClient(g, name, transport=broker.transport(name), **kw)
        rebound = paper.PaperLedger.rebind(root, grant, decision_path, clock=clock, client_factory=factory)
        assert rebound.events(event='binding')[:2] == [first, second]
        assert rebound.events(event='weekend_registration') == [registration]
        assert rebound.events(event='binding')[-1]['preregistration_hash'] == execution_preregistration()
        assert rebound.store.verify()['verified']
        reopened = paper.PaperLedger(root, grant, clock=clock)
        assert reopened.events(event='binding') == rebound.events(event='binding')
        assert all(c[1] == 'GET' and c[0] != 'momentum' for c in broker.calls)

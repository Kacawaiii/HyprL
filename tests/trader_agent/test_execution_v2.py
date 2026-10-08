"""Execution v2 approval boundaries; synthetic authority and broker only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.trader_agent import alpaca_paper as paper, config
from scripts.trading_lab.trader_agent.config import Authorization, TraderError, instant, iso
from scripts.trading_lab.trader_agent.data import PublicData
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.paper_spec import PREVIOUS_SPEC_HASH
from scripts.trading_lab.trader_agent.service import TraderService, views_with_review
from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
from test_alpaca_paper import environment, issue


@pytest.fixture
def amended(tmp_path, grant, monkeypatch):
    base = tmp_path / 'synthetic-grant.json'
    body = json.loads((Path(__file__).resolve().parents[2] / 'docs/artifacts/trader_crypto_universe_amendment_v1.json').read_text())
    body.pop('canonical_sha256')
    body['base_grant_hash'] = grant.identity
    body['canonical_sha256'] = sha256_canonical(body)
    monkeypatch.setattr(config, 'AMENDMENT_HASH', body['canonical_sha256'])
    path = tmp_path / config.AMENDMENT_NAME
    path.write_text(json.dumps(body))
    return Authorization.load(base), base, path


def test_amendment_keeps_original_grant_and_budget_history_and_enters_analyst_context(amended, tmp_path, clock):
    grant, base, _ = amended
    assert json.loads(base.read_text())['universe']['crypto'] == ['BTC-USD', 'ETH-USD']
    clock.at = instant('2026-10-09T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', grant, clock=clock)
    ledger.reserve('coinbase_exchange_public')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    result = service.run()
    assert result['status'] == 'COMPLETE'
    assert {v['asset'] for v in result['decision']['views']} >= set(config.AMENDED_CRYPTO)
    assert not {'BTC-USD', 'ETH-USD'} & {v['asset'] for v in result['decision']['views']}
    assert ledger.counts()['coinbase_exchange_public'] == 6
    assert grant.payload['data_sources']['coinbase_exchange_public']['max_requests_per_day'] == 40
    assert grant.base_identity != grant.identity


def test_amendment_fails_closed_on_tamper_or_wrong_parent(amended):
    _, base, path = amended
    body = json.loads(path.read_text())
    body['crypto'].append('BTC-USD')
    path.write_text(json.dumps(body))
    with pytest.raises(TraderError, match='CRYPTO_AMENDMENT_HASH_MISMATCH'):
        Authorization.load(base)


def test_amended_source_cannot_fetch_protected_old_products(amended, tmp_path, clock):
    grant, _, _ = amended
    clock.at = instant('2026-10-09T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', grant, clock=clock)
    with pytest.raises(TraderError, match='UNAUTHORIZED_PATH'):
        PublicData(ledger).fetch('coinbase_exchange_public', '/products/BTC-USD/candles', {})
    assert ledger.counts() == {}


def test_consensus_still_abstains_and_both_analyst_variants_remain_recorded():
    analysts = {'analyst_claude': {'views': [{'asset': 'XOM', 'horizon': '1d', 'view': 'UP', 'p_outperform': .60}]},
                'analyst_gpt': {'views': [{'asset': 'XOM', 'horizon': '1d', 'view': 'ABSTAIN', 'p_outperform': .50}]}}
    reviewer = {'verdicts': [{'analyst': a, 'asset': 'XOM', 'horizon': '1d', 'verdict': 'KEEP'} for a in analysts]}
    views = views_with_review(analysts, reviewer)
    assert next(v for v in views if v['analyst'] == 'consensus')['view'] == 'ABSTAIN'
    assert {v['analyst'] for v in views} == {'analyst_claude', 'analyst_gpt', 'reviewer_claude', 'reviewer_gpt', 'consensus'}


@pytest.mark.parametrize('risk, expected', [('pending', 'PAPER_REBIND_PENDING_INTENTS'),
    ('filled', 'PAPER_REBIND_OPEN_LOTS'), ('position', 'PAPER_REBIND_BROKER_NOT_EMPTY'),
    ('order', 'PAPER_REBIND_BROKER_NOT_EMPTY'), ('decision', 'PAPER_REBIND_OPERATOR_DECISION_REFUSED')])
def test_rebinding_refuses_uncovered_risk_or_unapproved_decision(environment, tmp_path, monkeypatch, risk, expected):
    executor, broker, grant, clock = environment
    root = tmp_path / 'old-paper'
    store = ResearchStore(root / 'paper-evidence')
    event = {'schema': 'alpaca-paper-event-v1', 'event_id': 'synthetic-binding', 'event': 'binding',
             'at': iso(clock()), 'account': None, 'account_suffix': None,
             'grant_hash': grant.identity, 'spec_hash': PREVIOUS_SPEC_HASH}
    store.append('replay-summary', event, object_id=event['event_id'], recorded_at=event['at'])
    if risk in {'pending', 'filled'}:
        issue(executor)
        executor.run('execute', accounts=['ia_actions'])
        if risk == 'filled':
            for name, key in broker.orders:
                broker.fill(name, key)
            executor.run('status')
        for event in executor.ledger.events('ia_actions'):
            if event['event'] in {'intent', 'order'}:
                store.append('replay-summary', event, object_id=event['event_id'], recorded_at=event['at'])
    elif risk == 'position':
        broker.positions[('ia_actions', 'AAPL')] = paper.D(1)
    elif risk == 'order':
        broker.orders[('ia_actions', 'unrecognized')] = {'status': 'accepted'}
    decision = {'schema': 'synthetic-operator-decision', 'decided_at': iso(clock()), 'constraints': 'no open lots'}
    path = tmp_path / 'operator.json'
    path.write_text(json.dumps(decision))
    monkeypatch.setattr(paper, 'OPERATOR_DECISION_HASH', '0' * 64 if risk == 'decision' else sha256_canonical(decision))
    factory = lambda g, name, **kw: paper.PaperClient(g, name, transport=broker.transport(name), **kw)
    before = len(broker.calls)
    with pytest.raises(TraderError, match=expected):
        paper.PaperLedger.rebind(root, grant, path, clock=clock, client_factory=factory)
    assert all(c[1] == 'GET' for c in broker.calls[before:])
    assert len(store.records('replay-summary', object_id='synthetic-binding')) == 1
    assert not any(r['payload'].get('previous_spec_hash') for r in store.records('replay-summary'))


def test_replay_requires_dry_run_and_never_submits(environment):
    executor, broker, _, clock = environment
    issue(executor, peer_direction='ABSTAIN')
    with pytest.raises(TraderError, match='PAPER_REPLAY_REQUIRES_DRY_RUN'):
        executor.run('execute', replay_at=iso(clock()))
    result = executor.run('execute', dry_run=True, accounts=['ia_actions'], replay_at=iso(clock()))
    assert result['accounts'][0]['planned_orders']
    assert all(c[1] == 'GET' for c in broker.calls)
    assert not executor.ledger.events(event='intent')
    assert not executor.replay_prior_registration


def test_claude_book_is_get_only_and_absent_from_ai_baselines(environment):
    executor, broker, _, _ = environment
    result = executor.report()
    assert result['claude_book']['mode'] == 'GET_ONLY'
    assert result['claude_book']['baseline'] is False
    assert not {'momentum', 'claude_book'} & result['versus']['complet'].keys()
    assert not executor.ledger.events('momentum', 'benchmark')
    assert all(c[1] == 'GET' for c in broker.calls)


@pytest.mark.parametrize('native', ['SOLUSD', 'AVAXUSD', 'LINKUSD', 'DOGEUSD', 'LTCUSD'])
def test_new_crypto_native_broker_symbols_normalize(native):
    assert paper.broker_symbol(native) == native[:-3] + '/USD'

@pytest.mark.parametrize('bad', [
    [[1791504000, 99, 103, 100, 101]],
    [[1791504000, 99, 103, True, 101, 20]],
    [[1791504000, 102, 103, 100, 101, 20]],
    [[1791504000, 99, 103, 100, 101, -1]],
    [[1791504000, 99, 103, 100, 101, 20]] * 2,
    [[1791504000.5, 99, 103, 100, 101, 20]],
    [[1791504000, 99, 103, 100, float('nan'), 20]],
])
def test_native_candle_validation_refuses_bad_shapes(bad):
    from scripts.trading_lab.trader_agent.coinbase import daily_closes
    with pytest.raises(TraderError, match='CRYPTO_SHAPE_INVALID'):
        daily_closes(bad, instant('2026-10-10T12:00:00Z'))


def test_native_candles_preserve_completed_close_and_exact_minute_open():
    from scripts.trading_lab.trader_agent.coinbase import daily_closes, candles
    before = instant('2026-10-09T12:00:00Z')
    prior, forming = int(instant('2026-10-08T00:00:00Z').timestamp()), int(instant('2026-10-09T00:00:00Z').timestamp())
    assert daily_closes([[forming, 99, 103, 100, 102, 20], [prior, 99, 103, 100, 101, 20]], before) == [
        {'bar_open_at': instant('2026-10-08T00:00:00Z'), 'close': 101}]
    anchor = instant('2026-10-08T13:30:00Z')
    assert candles([[int(anchor.timestamp()), 99, 103, 100, 101, 20]])[0]['open'] == 100

@pytest.mark.parametrize('verdict', ['DOWNGRADE', 'REJECT'])
def test_issued_conflict_vetoes_even_if_opposing_view_was_not_kept(environment, verdict):
    executor, broker, _, _ = environment
    issue(executor, peer_direction='DOWN', peer_verdict=verdict)
    result = executor.run('execute', dry_run=True, accounts=['ia_actions'])
    assert result['accounts'][0]['planned_orders'] == []
    assert not broker.posts()


def test_old_registration_is_replay_only(environment):
    executor, broker, _, clock = environment
    issue(executor, peer_direction='ABSTAIN', artifact_hash='b' * 64)
    assert executor.run('execute', accounts=['ia_actions'])['accounts'][0]['planned_orders'] == []
    result = executor.run('execute', dry_run=True, accounts=['ia_actions'], replay_at=iso(clock()))
    assert result['accounts'][0]['planned_orders']
    assert not broker.posts()

def test_new_crypto_labels_costs_and_scores_stay_outside_equity_primary(amended, tmp_path, clock):
    from datetime import timedelta
    from scripts.trading_lab.trader_agent.scoring import realize, scorecard
    from scripts.trading_lab.trader_agent.portfolio import portfolios, roundtrip_cost
    grant, _, _ = amended
    clock.at = instant('2026-10-09T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', grant, clock=clock)
    data = SyntheticData(ledger, clock)
    service = TraderService(ledger, data, SyntheticRunner(ledger, clock=clock), clock=clock)
    result = service.run()
    assert result['status'] == 'COMPLETE'
    assert roundtrip_cost('SOL-USD', half_spread_bps=2.5) == .002
    views = [{'asset': 'SOL-USD', 'horizon': '1d', 'analyst': 'consensus', 'view': 'UP', 'p_outperform': .60, 'verdict': 'KEEP'}]
    assert portfolios(views)['1d']['spy_hedged'] == {'SOL-USD': .1}
    clock.at += timedelta(days=9, hours=10)
    assert realize(service.store, ledger, data, at=clock())['state'] == 'COMPLETE'
    card = scorecard(service.store, synthetic=True)
    assert card['scores']['consensus/crypto/5d/raw']['realized'] == 5
    assert card['scores']['consensus/equity_etf/5d/SPY_relative']['realized'] == 3

def test_v2_accepts_native_nine_decimal_crypto_partial_fills(environment):
    executor, broker, grant, _ = environment
    grant.parent.payload['universe']['crypto'] = ['SOL-USD']
    broker.crypto_fill = False
    issue(executor, assets=('SOL-USD',), horizons=('1d',))
    assert executor.run('execute', accounts=['ia_crypto'])['state'] == 'COMPLETE'
    order = next(iter(broker.orders.values()))
    order.update(status='partially_filled', filled_qty='0.000000001', filled_avg_price='100', filled_at=iso(executor.clock()))
    broker.positions[('ia_crypto', 'SOL/USD')] = paper.D('0.000000001')
    assert executor.run('status', accounts=['ia_crypto'])['state'] == 'COMPLETE'
    assert next(iter(executor.ledger.projection('ia_crypto')[2].values()))['qty'] == '1E-9'

def test_every_directional_keep_equity_view_is_sized_under_literal_union(environment):
    executor, broker, _, _ = environment
    issue(executor, horizons=('1d',), probability=.53, peer_direction='ABSTAIN')
    result = executor.run('execute', dry_run=True, accounts=['ia_actions'])
    assert result['accounts'][0]['planned_orders'] == [
        {'symbol': 'AAPL', 'qty': '3', 'side': 'buy', 'type': 'market', 'time_in_force': 'opg'}]
    assert not broker.posts()


def test_crypto_consensus_threshold_is_retained(environment):
    executor, broker, _, _ = environment
    issue(executor, assets=('BTC-USD',), probability=.54)
    assert executor.run('execute', accounts=['ia_crypto'])['accounts'][0]['planned_orders'] == []
    assert not broker.posts()

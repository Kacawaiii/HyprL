"""Native broker shapes, synthetic credentials, injected HTTP only."""
from copy import deepcopy
from datetime import timedelta
import json
from urllib.parse import parse_qs, urlparse

import pytest

from scripts.trading_lab.platform.contracts import PredictionRecord
from scripts.trading_lab.research.contracts import PredictionEvidence
from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.trader_agent import alpaca_paper as paper
from scripts.trading_lab.trader_agent.config import TraderError, instant, iso
from scripts.trading_lab.trader_agent.data import calendar_session, label_window
from scripts.trading_lab.trader_agent.paper_spec import SPEC_HASH, execution_spec
from scripts.trading_lab.trader_agent.schedule import render


class MockBroker:
    def __init__(self, grant, clock):
        self.clock, self.calls, self.orders, self.positions = clock, [], {}, {}
        self.accounts = {name: {'account_number': 'SYNTHETIC-' + cfg['account_suffix'], 'equity': '100000',
                               'last_equity': '100000', 'status': 'ACTIVE', 'shorting_enabled': True, 'trading_blocked': False}
                         for name, cfg in grant.payload['accounts'].items()}
        self.crash = False

    def fill(self, name, key, fraction=1):
        order = self.orders[(name, key)]
        old = paper.decimal(order['filled_qty'])
        filled = paper.decimal(order['qty']) * paper.decimal(fraction)
        order.update(status='filled' if fraction == 1 else 'partially_filled', filled_qty=str(filled),
                     filled_avg_price='100', filled_at=iso(self.clock()))
        asset = order['symbol']
        self.positions[(name, asset)] = self.positions.get((name, asset), paper.D(0)) + (filled - old) * (1 if order['side'] == 'buy' else -1)

    def transport(self, name):
        def dispatch(method, url, headers, body):
            assert url.startswith(paper.PAPER_URL + '/v2/')
            self.calls.append((name, method, url, deepcopy(body)))
            parsed = urlparse(url)
            path, query = parsed.path, parse_qs(parsed.query)
            if path == '/v2/account':
                return deepcopy(self.accounts[name])
            if path == '/v2/positions':
                return [{'symbol': s, 'qty': str(q), 'current_price': '100', 'market_value': str(q * 100)}
                        for (a, s), q in self.positions.items() if a == name and q]
            if path.startswith('/v2/assets/'):
                return {'tradable': True, 'shortable': True}
            if path == '/v2/orders:by_client_order_id':
                result = self.orders.get((name, query['client_order_id'][0]))
                if result is None:
                    raise TraderError('PAPER_HTTP_404')
                return deepcopy(result)
            if method == 'GET' and path == '/v2/orders':
                return [deepcopy(o) for (a, _), o in self.orders.items() if a == name and o['status'] not in paper.TERMINAL]
            if method == 'DELETE':
                for (a, _), order in self.orders.items():
                    if a == name and path.endswith(order['id']):
                        order['status'] = 'canceled'
                return None
            assert method == 'POST' and path == '/v2/orders'
            key = body['client_order_id']
            assert (name, key) not in self.orders, 'duplicate order after crash'
            result = {**body, 'id': 'synthetic-broker-' + key, 'status': 'accepted', 'filled_qty': '0',
                      'filled_avg_price': None, 'filled_at': None}
            self.orders[(name, key)] = result
            if body['time_in_force'] == 'gtc':
                self.fill(name, key)
            if self.crash:
                self.crash = False
                raise SystemExit('synthetic crash after broker acceptance')
            return deepcopy(result)
        return dispatch

    def posts(self, name=None):
        return [c for c in self.calls if c[1] == 'POST' and (name is None or c[0] == name)]


@pytest.fixture
def environment(tmp_path, grant, clock, monkeypatch):
    clock.at = instant('2026-10-08T12:00:00Z')
    # No holdout prices are read. Synthetic broker tests model an authorized, unprotected universe.
    monkeypatch.setattr(paper, 'protected', lambda *a: None)
    accounts = {}
    for name, suffix in (('ia_actions', 'SYN1'), ('ia_crypto', 'SYN2'), ('momentum', 'SYN3')):
        path = tmp_path / (name + '.env')
        path.write_text('APCA_API_KEY_ID=synthetic-key\nAPCA_API_SECRET_KEY=synthetic-secret\nAPCA_API_BASE_URL=' + paper.PAPER_URL + '\n')
        accounts[name] = {'credentials': str(path), 'account_suffix': suffix, 'orders': name != 'momentum'}
    payload = {'authorization': 'trader-alpaca-paper-v1', 'extends': 'trader-agent-v1', 'mode': 'ALPACA_PAPER_ONLY',
               'endpoint': paper.PAPER_URL, 'granted_at': '2026-01-01T00:00:00Z', 'not_after': '2027-01-01T00:00:00Z',
               'accounts': accounts, 'risk_limits': {**deepcopy(paper.LIMITS), 'max_orders_per_account_per_day': 40,
                'kill_switch': "equity below 90% of the account's peak: flatten that account, halt its executor, raise an alert; resuming needs the operator",
                'trade_only': 'consensus views of a COMPLETE or DEGRADED run that the reviewer did not reject; ABSTAIN, MISSING, FAILED or tainted runs never trade'}}
    path = tmp_path / 'paper-grant.json'
    path.write_text(json.dumps(payload))
    authorization = paper.PaperAuthorization.load(path, grant)
    ledger = paper.PaperLedger(tmp_path / 'paper-private', authorization, clock=clock)
    research = ResearchStore(tmp_path / 'research')
    broker = MockBroker(authorization, clock)
    factory = lambda g, name, **kw: paper.PaperClient(g, name, transport=broker.transport(name), **kw)
    executor = paper.PaperExecutor(ledger, research, clock=clock, client_factory=factory)
    return executor, broker, authorization, clock


def issue(executor, *, assets=('AAPL',), horizons=('1d', '5d'), direction='UP', probability=.80,
          status='COMPLETE', verdict='KEEP', tainted=False, reviewer=True, variant=None):
    at = executor.clock()
    day, run_id = at.date().isoformat(), 'synthetic-test:' + iso(at)
    views = []
    for asset in assets:
        for horizon in horizons:
            view = {'asset': asset, 'horizon': horizon, 'analyst': 'consensus', 'view': direction,
                    'p_outperform': probability, 'verdict': verdict}
            views.append(view)
            entry, exit_at = label_window(asset, day, horizon, ['BTC-USD', 'ETH-USD'], 'close' if variant else 'open')
            definition = {'entry_at': iso(entry), 'exit_at': iso(exit_at), 'horizon': horizon}
            if variant:
                definition['variant'] = variant
            identity = sha256_canonical({'run': run_id, 'asset': asset, 'horizon': horizon})
            p = PredictionRecord(prediction_id=identity, model_id='trader:consensus', model_contract_hash='a' * 64,
                artifact_hash='b' * 64, product=asset, decision_at=iso(at), horizon_seconds=100,
                snapshot_hash=sha256_canonical({'price': {'recent_closes': [100]}}), features_hash=sha256_canonical([]), event_ids=(),
                outputs={'return': None, 'target_price': None, 'class': direction, 'probabilities': {'outperform': probability},
                         'quantiles': None, 'scenarios': None},
                signal={'run_id': run_id, 'session': day, 'view': view, 'label_definition': definition}, risk={'tainted': tainted}, synthetic=False)
            evidence = PredictionEvidence(prediction_id=identity, prediction_hash=p.identity, recorded_at=iso(at),
                features=(), snapshot={'price': {'recent_closes': [100]}}, baselines={}, input_quality={'state': 'AVAILABLE'},
                split='SYNTHETIC_HTTP_TEST', provenance={'method': 'synthetic'})
            executor.research.issue(p, evidence)
    executor.research.append('replay-summary', {'schema': 'trader-run-v1', 'run_id': run_id, 'at': iso(at),
        'status': status, 'synthetic': False, 'tainted': tainted,
        'decision': {'session': day, 'models': {'reviewer': {'cli_version': 'synthetic'}} if reviewer else {}, 'views': views}}, recorded_at=iso(at))


def test_live_url_refused_before_even_get(environment):
    executor, broker, grant, _ = environment
    path = grant.payload['accounts']['ia_actions']['credentials']
    from pathlib import Path
    Path(path).write_text('APCA_API_KEY_ID=synthetic\nAPCA_API_SECRET_KEY=synthetic\nAPCA_API_BASE_URL=https://api.alpaca.markets\n')
    assert executor.run('status')['state'] == 'BLOCKED'
    assert not any(c[0] == 'ia_actions' for c in broker.calls)


@pytest.mark.parametrize('method', ['POST', 'DELETE', 'PATCH', 'PUT'])
def test_momentum_mutation_refused_even_after_account_verification(environment, method):
    _, broker, grant, clock = environment
    client = paper.PaperClient(grant, 'momentum', clock=clock, transport=broker.transport('momentum'))
    client.start()
    before = len(broker.calls)
    with pytest.raises(TraderError, match='PAPER_MOMENTUM_READ_ONLY'):
        client.request(method, '/v2/orders')
    assert len(broker.calls) == before


def test_account_suffix_mismatch_refuses_and_alerts(environment):
    executor, broker, _, _ = environment
    broker.accounts['ia_actions']['account_number'] = 'SYNTHETIC-WRONG'
    issue(executor)
    assert executor.run('execute')['accounts'][0]['reason'] == 'PAPER_ACCOUNT_SUFFIX_MISMATCH'
    assert not broker.posts('ia_actions')
    assert executor.ledger.events('ia_actions', 'alert')[-1]['code'] == 'PAPER_ACCOUNT_SUFFIX_MISMATCH'


@pytest.mark.parametrize('options', [dict(direction='ABSTAIN', verdict='ABSTAIN'), dict(verdict='MISSING'),
    dict(verdict='REJECT'), dict(tainted=True), dict(status='FAILED'), dict(reviewer=False),
    dict(variant='catchup_close_entry_v1'), dict(probability=.54)])
def test_ineligible_views_produce_no_order(environment, options):
    executor, broker, _, _ = environment
    issue(executor, **options)
    result = executor.run('execute')
    assert result['state'] == 'COMPLETE'
    assert not broker.posts() and not executor.ledger.events(event='intent')


def test_degraded_with_review_and_valid_consensus_trades(environment):
    executor, broker, _, _ = environment
    issue(executor, status='DEGRADED')
    assert executor.run('execute')['state'] == 'COMPLETE'
    assert len(broker.posts('ia_actions')) == 1
    order = broker.posts()[0][3]
    assert order['time_in_force'] == 'opg' and order['type'] == 'market' and 'extended_hours' not in order
    assert len(executor.ledger.events('ia_actions', 'intent')[0]['lots']) == 2
    executor.run('execute')
    assert len(broker.posts()) == 1


def test_dry_run_only_gets_and_reserves_nothing(environment):
    executor, broker, _, _ = environment
    issue(executor)
    result = executor.run('execute', dry_run=True)
    assert result['accounts'][0]['planned_orders'][0]['qty'] == '43'
    assert all(c[1] == 'GET' for c in broker.calls)
    assert not executor.ledger.events(event='intent')
    assert 'synthetic-key' not in json.dumps(list(executor.ledger.events()))


def test_crash_after_acceptance_recovers_without_duplicate_post(environment):
    executor, broker, _, _ = environment
    issue(executor, assets=('BTC-USD',), horizons=('1d',))
    broker.crash = True
    with pytest.raises(SystemExit):
        executor.run('execute')
    assert len(broker.posts()) == 1
    assert not executor.ledger.events('ia_crypto', 'order')
    assert executor.run('execute')['state'] == 'COMPLETE'
    assert len(broker.posts()) == 1
    assert executor.ledger.events('ia_crypto', 'reconciliation')[-1]['matched']
    assert executor.ledger.projection('ia_crypto')[2]


def test_reconciliation_discrepancy_halts_without_correction(environment):
    executor, broker, _, _ = environment
    broker.positions[('ia_actions', 'AAPL')] = paper.D(1)
    issue(executor)
    result = executor.run('execute')
    assert result['accounts'][0]['reason'] == 'PAPER_RECONCILIATION_MISMATCH'
    assert executor.ledger.halted('ia_actions') and not broker.posts()
    broker.positions.clear()
    assert executor.run('execute')['accounts'][0]['halted']
    assert not broker.posts()


def test_crypto_long_only_and_caps(environment):
    executor, broker, _, clock = environment
    issue(executor, assets=('BTC-USD', 'ETH-USD'), direction='DOWN', probability=.20)
    assert executor.run('execute')['state'] == 'COMPLETE' and not broker.posts()
    clock.at += timedelta(seconds=1)
    issue(executor, assets=('BTC-USD', 'ETH-USD'))
    executor.run('execute')
    assert len(broker.posts()) == 2
    assert all(c[3]['side'] == 'buy' and c[3]['time_in_force'] == 'gtc' for c in broker.posts())
    _, _, lots = executor.ledger.projection('ia_crypto')
    weights = {}
    for lot in lots.values():
        weights[lot['symbol']] = weights.get(lot['symbol'], 0) + float(lot['qty']) * 100 / 100000
    assert all(w <= .25 for w in weights.values()) and sum(weights.values()) <= .50


def test_equity_gross_short_and_per_name_caps_include_pending_orders(environment):
    executor, broker, grant, clock = environment
    grant.parent.payload['universe']['stocks'] = ['SYN' + str(i) for i in range(30)]
    issue(executor, assets=tuple(grant.assets('ia_actions')), direction='DOWN', probability=.20)
    executor.run('execute')
    orders = broker.posts('ia_actions')
    assert len(orders) == 20  # reserve the other 20 orders for today's 1d exits
    weights = [float(c[3]['qty']) * 100 / 100000 for c in orders]
    assert sum(weights) <= .30 and all(w <= .08 for w in weights)
    clock.at += timedelta(seconds=1)
    issue(executor, assets=tuple(grant.assets('ia_actions')), direction='DOWN', probability=.20)
    executor.run('execute')
    assert len(broker.posts()) == len(orders)  # batch would exceed the 40/day cap


def test_partial_fill_projection_and_due_cls_delta(environment):
    executor, broker, _, clock = environment
    issue(executor)
    executor.run('execute')
    intent = executor.ledger.events('ia_actions', 'intent')[0]
    broker.fill('ia_actions', intent['client_id'], paper.D(21) / 43)
    status = executor.run('status')
    assert not status['accounts'][0]['halted']
    assert sum(paper.decimal(l['qty']) for l in executor.ledger.projection('ia_actions')[2].values()) == paper.D('21')
    assert all(paper.decimal(l['qty']) == paper.decimal(l['qty']).quantize(paper.D(1)) for l in executor.ledger.projection('ia_actions')[2].values())
    broker.fill('ia_actions', intent['client_id'])
    clock.at = instant('2026-10-08T19:40:00Z')
    result = executor.run('exit', accounts=['ia_actions'])
    assert result['state'] == 'COMPLETE'
    exit_order = broker.posts()[-1][3]
    assert exit_order['qty'] == '36' and exit_order['side'] == 'sell' and exit_order['time_in_force'] == 'cls'
    executor.run('exit', accounts=['ia_actions'])
    assert len(broker.posts()) == 2
    broker.fill('ia_actions', exit_order['client_order_id'])
    executor.run('status')
    remaining = [l for l in executor.ledger.projection('ia_actions')[2].values() if paper.decimal(l['qty'])]
    assert len(remaining) == 1 and remaining[0]['horizon'] == '5d'


def test_kill_switch_flattens_then_stays_halted(environment):
    executor, broker, _, _ = environment
    issue(executor, assets=('BTC-USD',))
    executor.run('execute')
    broker.accounts['ia_crypto']['equity'] = '89999'
    result = executor.run('execute')
    assert result['accounts'][1]['halted']
    assert broker.posts()[-1][3]['side'] == 'sell'
    assert all(q == 0 for (a, _), q in broker.positions.items() if a == 'ia_crypto')
    count = len(broker.posts())
    assert executor.run('execute')['accounts'][1]['halted']
    assert len(broker.posts()) == count


def test_no_mutation_today_even_kill_switch(environment):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-07T21:00:00Z')
    broker.accounts['ia_crypto']['equity'] = '80000'
    assert executor.run('execute')['state'] == 'NOT_STARTED'
    client = executor.client_factory(executor.grant, 'ia_crypto', clock=clock)
    client.start()
    with pytest.raises(TraderError, match='PAPER_ORDERS_NOT_STARTED'):
        client.request('POST', '/v2/orders', body={})
    assert not broker.posts()


@pytest.mark.parametrize('day,open_at,cls_at,cutoff', [
    ('2026-10-08', '13:30', '19:40', '23:30'), ('2026-11-02', '14:30', '20:40', '00:30')])
def test_auction_and_after_hours_dst(day, open_at, cls_at, cutoff):
    session = calendar_session(day)
    assert iso(session.open_at)[11:16] == open_at
    at = instant(day + 'T' + cls_at + ':00Z')
    assert paper.order_terms('ia_actions', 'exit', at, session)['time_in_force'] == 'cls'
    assert iso(paper.after_hours_cutoff(session))[11:16] == cutoff
    with pytest.raises(TraderError, match='PAPER_CLS_CUTOFF'):
        paper.order_terms('ia_actions', 'exit', session.close_at - timedelta(minutes=10), session)


def test_early_close_cls_uses_calendar():
    session = calendar_session('2026-11-27')
    assert iso(session.close_at) == '2026-11-27T18:00:00Z'
    assert paper.order_terms('ia_actions', 'exit', instant('2026-11-27T17:40:00Z'), session)['time_in_force'] == 'cls'


@pytest.mark.parametrize('side', ['buy', 'sell'])
def test_after_hours_limit_spread_and_cutoff(side):
    session = calendar_session('2026-10-08')
    at = instant('2026-10-08T22:00:00Z')
    terms = paper.order_terms('ia_actions', 'entry', at, session, after_hours=True, bid='99.95', ask='100.05', side=side)
    assert terms['type'] == 'limit' and terms['extended_hours'] is True and terms['time_in_force'] == 'day'
    with pytest.raises(TraderError, match='PAPER_SPREAD_CAP'):
        paper.order_terms('ia_actions', 'entry', at, session, after_hours=True, bid='99', ask='101')
    with pytest.raises(TraderError, match='PAPER_AFTER_HOURS_CUTOFF'):
        paper.order_terms('ia_actions', 'entry', paper.after_hours_cutoff(session), session, after_hours=True, bid='100', ask='100.1')


def test_after_hours_requires_fresh_private_quote_and_no_other_http_host(environment, tmp_path):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-08T20:10:00Z')
    issue(executor, variant='alpaca_after_hours_v1')
    assert executor.run('execute')['accounts'][0]['reason'] == 'PAPER_QUOTE_UNAVAILABLE'
    path = tmp_path / 'synthetic-quotes.json'
    path.write_text(json.dumps({'quotes': {'AAPL': {'bid': 99.95, 'ask': 100.05, 'at': iso(clock())}}}))
    executor.quotes = paper.PrivateQuotes(path)
    assert executor.run('execute')['state'] == 'COMPLETE'
    assert broker.posts()[0][3]['extended_hours'] is True
    assert all(urlparse(c[2]).netloc == 'paper-api.alpaca.markets' for c in broker.calls)
    clock.at += timedelta(seconds=61)
    with pytest.raises(TraderError, match='PAPER_QUOTE_STALE'):
        executor.quotes.get('AAPL', clock())


def test_report_complet_sums_two_accounts_and_momentum_is_get_only(environment):
    executor, broker, _, _ = environment
    broker.accounts['ia_actions']['equity'] = '101000'
    broker.accounts['ia_crypto']['equity'] = '99000'
    result = executor.report()
    assert result['complet']['equity'] == '200000' and result['complet']['pnl'] == '0'
    assert result['momentum']['mode'] == 'GET_ONLY'
    assert all(c[1] == 'GET' for c in broker.calls if c[0] == 'momentum')
    assert not broker.posts()
    assert executor.ledger.store.verify()['verified']


def test_schedule_chains_execute_report_and_uses_dst_and_crypto_anchor(tmp_path):
    units = render(tmp_path, '/synthetic/python', '/synthetic/grant', tmp_path / 'runtime',
                   path='/synthetic/bin', paper_authorization='/synthetic/paper-grant')
    assert 'ExecStartPost=' in units['hyprl-trader-run.service'] and 'paper-execute' in units['hyprl-trader-run.service']
    assert 'paper-report' in units['hyprl-trader-label.service']
    assert '15:40:00 America/New_York' in units['hyprl-trader-paper-exit-ia-actions.timer']
    assert '12:40:00 America/New_York' in units['hyprl-trader-paper-exit-ia-actions.timer']
    assert '*:30:00 UTC' in units['hyprl-trader-paper-exit-ia-crypto.timer']
    assert all('Persistent=false' in v for k, v in units.items() if k.endswith('.timer'))


def test_registered_execution_spec_and_private_store_binding(environment):
    executor, _, grant, clock = environment
    assert execution_spec()['revision'] == 1 and len(SPEC_HASH) == 64
    assert executor.ledger.events(event='binding')[0]['spec_hash'] == SPEC_HASH
    grant = paper.PaperAuthorization({**grant.payload}, 'e' * 64, grant.parent)
    with pytest.raises(TraderError, match='PAPER_LEDGER_BINDING_MISMATCH'):
        paper.PaperLedger(executor.ledger.root, grant, clock=clock)


def test_close_order_budget_reserved_before_entries(environment):
    executor, broker, grant, clock = environment
    grant.parent.payload['universe']['stocks'] = ['SYN' + str(i) for i in range(25)]
    issue(executor, assets=tuple(grant.assets('ia_actions')))
    result = executor.run('execute', accounts=['ia_actions'])
    assert result['accounts'][0]['orders_reserved_for_close'] == 20
    assert len(result['accounts'][0]['budget_skipped_symbols']) == 6
    for (name, key) in list(broker.orders):
        broker.fill(name, key)
    clock.at = instant('2026-10-08T19:40:00Z')
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 40


def test_stale_unsubmitted_intent_never_reschedules_next_open(environment):
    executor, broker, _, clock = environment
    issue(executor)
    broker.crash = True
    with pytest.raises(SystemExit):
        executor.run('execute', accounts=['ia_actions'])
    # Model an unacknowledged dispatch that never reached the broker.
    broker.orders.clear()
    clock.at = instant('2026-10-08T13:29:00Z')
    result = executor.run('execute', accounts=['ia_actions'])
    assert result['accounts'][0]['reason'] == 'PAPER_OPG_CUTOFF'
    assert len(broker.posts()) == 1
    assert executor.ledger.events('ia_actions', 'order')[-1]['status'] == 'not_submitted'


def test_after_hours_local_session_continues_across_utc_midnight(environment, tmp_path):
    executor, broker, _, clock = environment
    clock.at = instant('2026-11-02T23:59:50Z')
    issue(executor, variant='alpaca_after_hours_v1')
    clock.at = instant('2026-11-03T00:10:00Z')
    path = tmp_path / 'quotes.json'
    path.write_text(json.dumps({'quotes': {'AAPL': {'bid': '99.95', 'ask': '100.05', 'at': iso(clock())}}}))
    executor.quotes = paper.PrivateQuotes(path)
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert broker.posts()[0][3]['extended_hours'] is True


def test_kill_uses_recorded_peak_and_operator_resume_is_required(environment):
    executor, broker, _, _ = environment
    broker.accounts['ia_crypto']['equity'] = '120000'
    executor.run('status')
    broker.accounts['ia_crypto']['equity'] = '108000'
    assert not executor.run('status')['accounts'][1]['halted']
    broker.accounts['ia_crypto']['equity'] = '107999'
    assert executor.run('status')['accounts'][1]['halted']
    broker.accounts['ia_crypto']['equity'] = '120000'
    assert executor.run('execute')['accounts'][1]['halted']


def test_after_hours_catchup_records_distinct_predictions_and_does_not_shadow_label(service, clock):
    from scripts.trading_lab.trader_agent.scoring import realize
    original = service.data.yahoo
    service.data.yahoo = lambda *a, **kw: (_ for _ in ()).throw(TraderError('MISSING_PRICES'))
    assert service.run()['status'] == 'FAILED'
    service.data.yahoo = original
    clock.at = instant('2026-10-06T20:10:00Z')
    result = service.run(catchup=True, after_hours=True)
    assert result['status'] == 'COMPLETE'
    predictions = service.store.records('prediction')
    assert predictions and all(p['payload']['signal']['label_definition']['variant'] == 'alpaca_after_hours_v1' for p in predictions)
    clock.at += timedelta(days=9)
    labels = realize(service.store, service.ledger, service.data, at=clock())
    assert labels['labels_added'] == 0 and labels['pending'] == len(predictions)
    assert service.ledger.budget_day is None


def test_after_hours_budget_cannot_refund_at_utc_midnight(ledger, clock):
    ledger.budget_day = '2026-11-02'
    clock.at = instant('2026-11-02T23:59:00Z')
    ledger.reserve('analyst_claude')
    ledger.reserve('analyst_claude')
    clock.at = instant('2026-11-03T00:10:00Z')
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        ledger.reserve('analyst_claude')
    assert ledger.counts()['analyst_claude'] == 2


def test_reject_retains_order_and_not_traded_outcome(environment):
    executor, broker, _, _ = environment
    issue(executor)
    executor.run('execute', accounts=['ia_actions'])
    key = broker.posts()[0][3]['client_order_id']
    broker.orders[('ia_actions', key)]['status'] = 'rejected'
    assert executor.run('status')['state'] == 'COMPLETE'
    assert executor.ledger.events('ia_actions', 'order')[-1]['status'] == 'rejected'
    assert all(e['state'] == 'NOT_TRADED' for e in executor.ledger.events('ia_actions', 'trade_outcome')[-2:])
    executor.run('execute', accounts=['ia_actions'])
    assert len(broker.posts()) == 1


def test_kill_cancels_pending_entry_and_reconciles_before_flatten(environment):
    executor, broker, _, _ = environment
    issue(executor)
    executor.run('execute', accounts=['ia_actions'])
    key = broker.posts()[0][3]['client_order_id']
    broker.fill('ia_actions', key, paper.D(21) / 43)
    broker.accounts['ia_actions']['equity'] = '89999'
    result = executor.run('execute', accounts=['ia_actions'])
    assert result['accounts'][0]['halted']
    assert any(c[1] == 'DELETE' for c in broker.calls)
    assert len(broker.posts()) == 2
    assert broker.posts()[-1][3]['qty'] == '21' and broker.posts()[-1][3]['side'] == 'sell'


def test_pause_and_expiry_block_transport_mutations(environment):
    executor, broker, _, clock = environment
    issue(executor)
    marker = executor.ledger.root / 'PAPER_PAUSED'
    marker.touch()
    assert executor.run('execute')['state'] == 'PAUSED'
    assert not broker.posts()
    marker.unlink()
    clock.at = instant('2027-01-01T00:00:00Z')
    with pytest.raises(TraderError, match='PAPER_AUTHORIZATION_EXPIRED_OR_NOT_STARTED'):
        executor.run('execute')
    assert not broker.posts()


def test_after_hours_unfilled_is_canceled_at_cutoff_and_scored_not_traded(environment, tmp_path):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-08T20:10:00Z')
    issue(executor, variant='alpaca_after_hours_v1')
    path = tmp_path / 'quotes.json'
    path.write_text(json.dumps({'quotes': {'AAPL': {'bid': '99.95', 'ask': '100.05', 'at': iso(clock())}}}))
    executor.quotes = paper.PrivateQuotes(path)
    executor.run('execute', accounts=['ia_actions'])
    clock.at = instant('2026-10-08T23:30:00Z')
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert any(c[1] == 'DELETE' for c in broker.calls)
    assert len(broker.posts()) == 1
    assert all(e['state'] == 'NOT_TRADED' for e in executor.ledger.events('ia_actions', 'trade_outcome')[-2:])
    assert all(paper.decimal(l['qty']) == 0 for l in executor.ledger.projection('ia_actions')[2].values())


def test_public_benchmarks_use_first_execution_open_and_protected_btc_remains_pending(environment, tmp_path, monkeypatch):
    from scripts.trading_lab.trader_agent import paper_reporting
    executor, _, _, clock = environment
    executor.run('status')
    clock.at = instant('2026-10-08T21:30:00Z')
    monkeypatch.setattr(paper_reporting, 'protected', lambda asset, *a: asset == 'BTC-USD')
    class Quotes:
        def yahoo(self, asset, start, end):
            assert start.date().isoformat() == '2026-10-07'
            return [{'bar_open_at': calendar_session('2026-10-08').open_at, 'open': 100, 'close': 105}], {'digest': 'a' * 64}
        def crypto(self, *a, **kw):
            pytest.fail('protected BTC must not be fetched')
    values = paper_reporting.benchmarks(executor.ledger, tmp_path / 'runtime', executor.grant.parent, data=Quotes())
    assert values['SPY']['return_since_paper_start'] == '0.05'
    assert values['SPY']['base_at'] == '2026-10-08T13:30:00Z'
    assert values['BTC']['state'] == 'PENDING' and values['BTC']['reason'] == 'PROTECTED_BENCHMARK'
    report = executor.report(values)
    assert report['versus']['complet']['SPY'] == '-0.05' and report['versus']['complet']['BTC'] is None


def test_no_entry_when_label_exit_is_after_grant_expiry(environment):
    executor, broker, _, clock = environment
    executor.grant.payload['not_after'] = '2026-10-09T00:00:00Z'
    issue(executor, assets=('AAPL', 'BTC-USD'))
    result = executor.run('execute')
    assert result['state'] == 'COMPLETE'
    assert len(broker.posts()) == 1
    assert broker.posts()[0][0] == 'ia_actions' and broker.posts()[0][3]['qty'] == '36'
    assert all(l['horizon'] == '1d' for l in executor.ledger.events('ia_actions', 'intent')[0]['lots'])

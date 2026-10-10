"""Closed-day prospective population; synthetic inputs and broker only."""
import json

import pytest

from scripts.trading_lab.trader_agent.config import instant
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.service import TraderService
from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
from test_execution_v2 import amended
from test_alpaca_paper import environment, issue


@pytest.fixture
def weekend_grant(amended):
    from dataclasses import replace
    grant, _, _ = amended
    return replace(grant, weekend_decision={'decided_at': '2026-10-09T22:10:00Z'},
                   amendment={**grant.amendment, 'not_after': '2027-01-01T00:00:00Z'})


@pytest.mark.parametrize('day', ['2026-10-10', '2026-10-11', '2026-12-25'])
def test_closed_day_issues_only_amended_crypto_and_crypto_context(weekend_grant, tmp_path, clock, day):
    # Christmas is inside this synthetic grant's window; production expires Nov 5.
    clock.at = instant(day + 'T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    result = service.run()
    assert result['status'] == 'COMPLETE'
    assert {v['asset'] for v in result['decision']['views']} == set(weekend_grant.payload['universe']['crypto'])
    context = service.store.records('replay-summary', object_id=result['run_id'] + ':context')[0]['payload']['context']
    assert set(context['prices']) == set(context['universe']) == set(weekend_grant.payload['universe']['crypto'])
    assert set(context['headlines']) == set(context['universe']) | {'macro'}
    assert 'edgar' not in context['archives']
    assert not ledger.counts().get('yahoo_chart')
    definitions = [r['payload']['signal']['label_definition'] for r in service.store.records('prediction')]
    assert all(d['variant'] == 'weekend_crypto_v1' for d in definitions)
    assert {d['entry_at'] for d in definitions} == {day + 'T13:30:00Z'}


def test_weekend_paper_population_requires_registration_then_trades_crypto_only(environment):
    from dataclasses import replace
    from scripts.trading_lab.trader_agent import weekend
    executor, broker, grant, clock = environment
    clock.at = instant('2026-10-11T12:10:00Z')
    # Synthetic authority has no protection; preserve native broker shapes.
    grant.parent.payload['universe']['crypto'] = ['SOL-USD', 'AVAX-USD', 'LINK-USD', 'DOGE-USD', 'LTC-USD']
    executor.grant = executor.ledger.grant = replace(grant, parent=replace(grant.parent,
        amendment_identity='synthetic-amendment', weekend_decision={'decided_at': '2026-10-09T22:10:00Z'}))
    issue(executor, assets=('SOL-USD',), variant=weekend.VARIANT, artifact_hash=weekend.PREREG_HASH)
    assert executor.run('execute')['state'] == 'BLOCKED'
    assert not broker.posts()
    executor.ledger.append('weekend_registration', None, **weekend.registration_target(executor.grant))
    result = executor.run('execute')
    assert result['state'] == 'COMPLETE'
    assert {c[0] for c in broker.calls} == {'ia_crypto'}
    assert len(broker.posts()) == 1
    order = broker.posts()[0][3]
    assert order['symbol'] == 'SOL/USD' and order['type'] == 'market' and order['time_in_force'] == 'gtc'
    assert {l['variant'] for l in executor.ledger.events('ia_crypto', 'intent')[0]['lots']} == {weekend.VARIANT}
    executor.run('execute')
    assert len(broker.posts()) == 1
    clock.at = instant('2026-10-12T13:30:00Z')
    assert executor.run('exit')['state'] == 'COMPLETE'
    assert broker.posts()[-1][3]['side'] == 'sell'
    assert len(broker.posts()) == 2
    clock.at = instant('2026-10-16T13:30:00Z')
    assert executor.run('exit', accounts=['ia_crypto'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 3 and broker.posts()[-1][3]['side'] == 'sell'
    assert executor.run('exit', accounts=['ia_crypto'])['accounts'][0]['open_lots'] == 0
    assert len(broker.posts()) == 3


def test_closed_day_recovery_alerts_explicit_skip_and_reserves_nothing(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.recovery import recover
    clock.at = instant('2026-10-10T14:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    assert recover(service)['state'] == 'WEEKEND_CRYPTO_SKIPPED'
    assert json.loads((ledger.root / 'alert.json').read_text())['code'] == 'RECOVERY_WEEKEND_CRYPTO_SKIPPED'
    assert ledger.counts() == {}
    before = (ledger.root / 'alerts.jsonl').read_bytes()
    assert recover(service)['state'] == 'WEEKEND_CRYPTO_SKIPPED'
    assert (ledger.root / 'alerts.jsonl').read_bytes() == before


def test_daily_timers_cover_closed_days_and_report_hook_skips_equity(weekend_grant, tmp_path):
    from scripts.trading_lab.trader_agent.schedule import render
    units = render(tmp_path, '/synthetic/python', '/synthetic/grant', tmp_path, path='/synthetic/bin',
                   paper_authorization='/synthetic/paper')
    assert 'OnCalendar=*-*-* 12:00:00 UTC' in units['hyprl-trader-run.timer']
    assert 'OnCalendar=*-*-* 21:30:00 UTC' in units['hyprl-trader-label.timer']
    assert 'OnCalendar=*-*-* 08..19:00/10:00 America/New_York' in units['hyprl-trader-recover.timer']


def test_weekend_health_requires_run_and_labels(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.cli import health
    clock.at = instant('2026-10-10T14:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    from scripts.trading_lab.research.store import ResearchStore
    ResearchStore(ledger.root / 'evidence').close()
    assert health(ledger)['state'] == 'MISSING_DAILY_RUN'
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    service.summary('trader:2026-10-10:real', 'COMPLETE', clock(), synthetic=False)
    assert health(ledger)['state'] == 'HEALTHY'
    clock.at = instant('2026-10-10T22:00:00Z')
    assert health(ledger)['state'] == 'MISSING_LABEL_JOB'


def test_weekend_labels_observe_exact_24_and_120_hour_anchors_and_score_apart(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.scoring import realize, scorecard
    from scripts.trading_lab.app_api.trader import TraderViews
    clock.at = instant('2026-10-10T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    data = SyntheticData(ledger, clock)
    service = TraderService(ledger, data, SyntheticRunner(ledger, clock=clock), clock=clock)
    assert service.run()['status'] == 'COMPLETE'
    definitions = [r['payload']['signal']['label_definition'] for r in service.store.records('prediction')]
    assert {d['exit_at'] for d in definitions if d['horizon'] == '1d'} == {'2026-10-11T13:30:00Z'}
    assert {d['exit_at'] for d in definitions if d['horizon'] == '5d'} == {'2026-10-15T13:30:00Z'}
    clock.at = instant('2026-10-11T13:30:00Z')
    assert realize(service.store, ledger, data, at=clock())['labels_added'] == 0
    clock.at = instant('2026-10-11T13:31:00Z')
    assert realize(service.store, ledger, data, at=clock())['labels_added'] == 25
    assert ledger.counts()['coinbase_exchange_public'] == 10
    clock.at = instant('2026-10-15T13:31:00Z')
    assert realize(service.store, ledger, data, at=clock())['labels_added'] == 25
    card = scorecard(service.store, synthetic=True)
    assert card['scores']['weekend_crypto_v1:consensus/crypto/5d/raw']['realized'] == 5
    assert all(k.startswith('weekend_crypto_v1:') and '/crypto/' in k for k in card['scores'])
    assert card['hypothesis_state'] == 'PENDING_MINIMUM_SAMPLE'
    api = TraderViews(ledger.root)
    assert set(api.dispatch('/api/v1/trader/context', {'date': ['2026-10-10']})['context']['prices']) == set(weekend_grant.payload['universe']['crypto'])
    assert set(api.dispatch('/api/v1/trader/series', {})['series']) == set(weekend_grant.payload['universe']['crypto'])
    assert {r['payload']['product'] for r in api.dispatch('/api/v1/trader/ledger', {})['records']} == set(weekend_grant.payload['universe']['crypto'])
    assert service.store.verify()['verified']


def test_weekend_budgets_share_utc_day_bank_across_runtimes(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.config import TraderError
    clock.at = instant('2026-10-10T12:00:00Z')
    first = Ledger(tmp_path / 'first', weekend_grant, clock=clock, budget_root=tmp_path / 'bank')
    second = Ledger(tmp_path / 'second', weekend_grant, clock=clock, budget_root=tmp_path / 'bank')
    first.reserve('analyst_claude')
    service = TraderService(second, SyntheticData(second, clock), SyntheticRunner(second, clock=clock), clock=clock)
    assert service.run()['status'] == 'COMPLETE'
    assert first.counts() == second.counts()
    assert first.counts()['analyst_claude'] == 2
    assert first.counts()['run'] == 1
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        first.reserve('analyst_claude')
    duplicate = TraderService(first, SyntheticData(first, clock), SyntheticRunner(first, clock=clock), clock=clock)
    assert duplicate.run()['error'] == 'BUDGET_EXHAUSTED'
    clock.at = instant('2026-10-11T12:00:00Z')
    assert duplicate.run()['status'] == 'COMPLETE'
    assert first.counts()['run'] == 1 and first.counts()['analyst_claude'] == 1


@pytest.mark.parametrize('at,expected', [('2026-10-10T11:59:59Z', 'NOT_STARTED'),
    ('2026-10-10T13:30:00Z', 'FAILED'), ('2026-11-07T12:00:00Z', 'FAILED')])
def test_weekend_start_deadline_and_grant_expiry_fail_without_dispatch(weekend_grant, tmp_path, clock, at, expected):
    from dataclasses import replace
    grant = replace(weekend_grant, amendment={**weekend_grant.amendment, 'not_after': '2026-11-05T00:00:00Z'})
    clock.at = instant(at)
    ledger = Ledger(tmp_path / 'runtime', grant, clock=clock)
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    assert service.run()['status'] == expected
    assert ledger.counts() == {}


def test_weekend_decision_loading_preserves_identity_and_refuses_tamper(amended, monkeypatch):
    from scripts.trading_lab.trader_agent import config
    from scripts.trading_lab.sources.canonical import sha256_canonical
    grant, path, _ = amended
    decision = {'schema': 'synthetic-operator-decision', 'decided_at': '2026-10-09T22:10:00Z'}
    monkeypatch.setattr(config, 'WEEKEND_DECISION_HASH', sha256_canonical(decision))
    target = path.parent / config.WEEKEND_DECISION_NAME
    target.write_text(json.dumps(decision))
    loaded = config.Authorization.load(path)
    assert loaded.identity == grant.identity and loaded.payload == grant.payload
    loaded.check_weekend(instant('2026-10-10T12:00:00Z'))
    target.write_text(json.dumps({**decision, 'extra': 'tampered'}))
    with pytest.raises(config.TraderError, match='WEEKEND_OPERATOR_DECISION_HASH_MISMATCH'):
        config.Authorization.load(path)


def test_weekday_run_with_weekend_authority_keeps_original_population(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.service import PREREG_HASH
    clock.at = instant('2026-10-09T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    result = service.run()
    assert result['status'] == 'COMPLETE'
    assert result['decision']['preregistration_hash'] == PREREG_HASH
    assert {v['asset'] for v in result['decision']['views']} == set(weekend_grant.universe)
    assert all('variant' not in r['payload']['signal']['label_definition'] for r in service.store.records('prediction'))


def test_closed_day_equity_exit_never_opens_account(environment):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-10T23:30:00Z')
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'SKIPPED_CLOSED_EQUITY_MARKET'
    assert not broker.calls and not executor.ledger.events('ia_actions')


def test_explicit_weekend_registration_is_additive_idempotent_and_offline(environment, tmp_path, monkeypatch):
    from dataclasses import replace
    from scripts.trading_lab.trader_agent import config
    from scripts.trading_lab.sources.canonical import sha256_canonical
    executor, broker, grant, clock = environment
    clock.at = instant('2026-10-10T12:00:00Z')
    decision = {'schema': 'synthetic-operator-decision', 'decided_at': '2026-10-09T22:10:00Z'}
    monkeypatch.setattr(config, 'WEEKEND_DECISION_HASH', sha256_canonical(decision))
    grant.parent.payload['universe']['crypto'] = list(config.AMENDED_CRYPTO)
    executor.grant = executor.ledger.grant = replace(grant, parent=replace(grant.parent,
        amendment_identity='synthetic-amendment', weekend_decision=decision))
    path = tmp_path / 'synthetic-decision.json'
    path.write_text(json.dumps(decision))
    binding = executor.ledger.events(event='binding')
    registered = executor.ledger.register_weekend(path)
    assert executor.ledger.register_weekend(path) == registered
    assert executor.ledger.events(event='binding') == binding
    assert len(executor.ledger.events(event='weekend_registration')) == 1
    executor.ledger.check_weekend_registration()
    assert not broker.calls
    path.write_text('{}')
    with pytest.raises(config.TraderError, match='WEEKEND_OPERATOR_DECISION_HASH_MISMATCH'):
        executor.ledger.register_weekend(path)


def test_weekend_retry_uses_existing_role_budget(weekend_grant, tmp_path, clock):
    from scripts.trading_lab.trader_agent.config import TraderError
    class RetryRunner(SyntheticRunner):
        failed = False
        def once(self, role, text):
            if role == 'analyst_gpt' and not self.failed:
                self.failed = True
                self.ledger.reserve(role)
                raise TraderError('MODEL_FAILED')
            return super().once(role, text)
    clock.at = instant('2026-10-10T12:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', weekend_grant, clock=clock)
    runner = RetryRunner(ledger, clock=clock)
    service = TraderService(ledger, SyntheticData(ledger, clock), runner, clock=clock)
    assert service.run()['status'] == 'COMPLETE'
    assert ledger.counts()['analyst_gpt'] == 2
    assert sum(ledger.counts()[role] for role in ('analyst_claude', 'analyst_gpt', 'reviewer')) == 4


@pytest.mark.parametrize('risk', ['kill', 'pause', 'mismatch'])
def test_weekend_executor_retains_safety_guards(environment, risk):
    executor, broker, _, clock = environment
    clock.at = instant('2026-10-10T12:00:00Z')
    if risk == 'kill':
        broker.accounts['ia_crypto']['equity'] = '89000'
    elif risk == 'pause':
        executor.pause_paths[0].touch()
    else:
        broker.positions[('ia_crypto', 'SOL/USD')] = 1
    result = executor.run('execute')
    assert result['state'] == ('PAUSED' if risk == 'pause' else 'BLOCKED' if risk == 'mismatch' else 'COMPLETE')
    if risk != 'pause':
        assert executor.ledger.halted('ia_crypto')
    assert not broker.posts() and not any(c[0] == 'ia_actions' for c in broker.calls)

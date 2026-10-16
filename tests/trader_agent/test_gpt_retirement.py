"""Offline forward-only retirement through run, budget, planner and scorecard interfaces."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from scripts.trading_lab.sources.canonical import sha256_canonical
from scripts.trading_lab.trader_agent import retirement_spec as retirement
from scripts.trading_lab.trader_agent.config import TraderError, instant
from scripts.trading_lab.trader_agent.ledger import Ledger
from scripts.trading_lab.trader_agent.preflight import preflight
from scripts.trading_lab.trader_agent.runners import ModelRunner
from scripts.trading_lab.trader_agent.scoring import realize, scorecard
from scripts.trading_lab.trader_agent.synthetic import SyntheticRunner
from test_alpaca_paper import environment
from test_news_ab import news_service, radar
from test_pm_slot import pm_grant


@pytest.mark.parametrize('slot', ['am', 'pm'])
def test_base_and_news_only_call_claude_and_preserve_frozen_labels(news_service, clock, slot):
    service = news_service
    clock.at = instant('2026-10-19T' + ('16' if slot == 'pm' else '12') + ':00:00Z')
    service.ledger.slot = slot
    service.ledger.grant = pm_grant(service.ledger.grant)
    calls = []
    class Capture(SyntheticRunner):
        def once(self, role, text):
            calls.append((self.ledger.arm, role))
            assert role in {'analyst_claude', 'reviewer'}
            assert 'two independent analysts' not in text
            return super().once(role, text)
    service.runner = Capture(service.ledger, clock=clock)
    base = service.run()
    assert base['status'] == 'COMPLETE', base
    before = service.store.records('prediction')
    news = service.run_news()
    assert news['status'] == 'COMPLETE', news
    assert calls == [('base', 'analyst_claude'), ('base', 'reviewer'), ('news', 'analyst_claude'), ('news', 'reviewer')]
    for run in (base, news):
        assert run['decision']['preregistration_hash'] == retirement.preregistration()
        assert run['decision']['models']['analyst_gpt']['model'] == 'RETIRED'
        assert {v['analyst'] for v in run['decision']['views']} == {'analyst_claude', 'reviewer_claude'}
    assert service.store.records('prediction')[:len(before)] == before
    equity = next(r for r in before if r['payload']['product'] == 'AAPL' and r['payload']['signal']['label_definition']['horizon'] == '1d')
    window = equity['payload']['signal']['label_definition']
    assert window['entry_at'] == ('2026-10-19T17:30:00Z' if slot == 'pm' else '2026-10-19T13:30:00Z')
    assert window['exit_at'] == ('2026-10-20T20:00:00Z' if slot == 'pm' else '2026-10-19T20:00:00Z')
    clock.at += timedelta(days=9)
    assert realize(service.store, service.ledger, service.data, at=clock())['state'] == 'COMPLETE'
    card = scorecard(service.store, synthetic=True, include_samples=True)
    assert card['_primary_samples'] == []
    assert all(key.startswith(retirement.POPULATION + ':') for key in card['scores'])
    assert card['role_lifecycle']['analyst_gpt']['state'] == 'RETIRED'
    assert card['news_ab']['complete_news_runs'] == 1
    assert all(c['role'] not in retirement.RETIRED for c in card['news_ab']['comparisons'])
    assert service.store.verify()['verified']


@pytest.mark.parametrize('slot', ['am', 'pm'])
@pytest.mark.parametrize('arm', ['base', 'news'])
def test_retired_budgets_cannot_be_borrowed_across_restart(tmp_path, grant, clock, monkeypatch, slot, arm):
    from scripts.trading_lab.trader_agent.config import Authorization
    clock.at = instant('2026-10-19T16:00:00Z')
    grant = pm_grant(grant)
    monkeypatch.setattr(Authorization, 'check_news', lambda self, at: self.check(at))
    ledger = Ledger(tmp_path / 'one', grant, clock=clock, budget_root=tmp_path / 'bank', slot=slot)
    ledger.arm = arm
    for kind in ('analyst_gpt', 'reviewer_gpt', 'gpt_preflight'):
        with pytest.raises(TraderError, match='GPT_RETIRED'):
            ledger.reserve(kind)
    for role in ('analyst_claude', 'reviewer'):
        ledger.reserve(role)
        ledger.reserve(role)
    other = Ledger(tmp_path / 'two', grant, clock=clock, budget_root=tmp_path / 'bank', slot=slot)
    other.arm = arm
    for role in ('analyst_claude', 'reviewer'):
        with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
            other.reserve(role)
    assert sum(other.counts().values()) == 4
    assert not any('gpt' in key for key in other.counts())


def test_retirement_blocks_before_even_cli_version_or_probe_launch(ledger, clock, monkeypatch, tmp_path):
    from scripts.trading_lab.trader_agent.rehearsal import LiveChecks, MonitorState
    clock.at = instant('2026-10-19T10:00:00Z')
    def forbidden(*a, **kw):
        pytest.fail('A retired GPT boundary launched a subprocess')
    monkeypatch.setattr('subprocess.run', forbidden)
    monkeypatch.setattr('subprocess.Popen', forbidden)
    runner = ModelRunner(ledger, clock=clock)
    with pytest.raises(TraderError, match='GPT_RETIRED'):
        runner.infer('analyst_gpt', 'Synthetic')
    with pytest.raises(TraderError, match='GPT_RETIRED'):
        runner.once('analyst_gpt', 'Synthetic', preflight=True)
    assert preflight(ledger, runner)['state'] == 'RETIRED'
    assert ledger.counts() == {}
    checks = LiveChecks(SimpleNamespace(), MonitorState(tmp_path / 'monitor'), clock=clock)
    assert 'analyst_gpt' not in {name for name, _ in checks.steps()}


@pytest.mark.parametrize('verdict,quantity,tier', [('KEEP', '28', 'alpaca_open_entry_v2'),
                                                ('DOWNGRADE', '14', 'half_size_downgrade_v1'),
                                                ('REJECT', None, None)])
def test_new_execution_requires_explicit_registration_and_keeps_action_sizes(environment, tmp_path, monkeypatch, verdict, quantity, tier):
    from scripts.trading_lab.trader_agent import alpaca_paper as paper
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData
    executor, broker, grant, clock = environment
    clock.at = instant('2026-10-19T12:00:00Z')
    parent = replace(grant.parent, retirement_decision={'schema': 'synthetic-operator-decision'})
    ledger = Ledger(tmp_path / 'decisions', parent, clock=clock)
    class Data(SyntheticData):
        synthetic = False
    class Runner(SyntheticRunner):
        def once(self, role, text):
            output = super().once(role, text)
            if role == 'reviewer':
                for review in output['verdicts']:
                    review.update(verdict=verdict, adjusted_p=.6 if verdict == 'DOWNGRADE' else None)
            else:
                for view in output['views']:
                    view['p_outperform'] = .8
            return output
    service = TraderService(ledger, Data(ledger, clock), Runner(ledger, clock=clock), clock=clock)
    assert service.run()['status'] == 'COMPLETE'
    executor.research = service.store
    clock.at = instant('2026-10-19T13:30:30Z')
    before = executor.ledger.events()
    with pytest.raises(TraderError, match='PAPER_GPT_RETIREMENT_REGISTRATION_REQUIRED'):
        executor.entries('ia_actions', paper.D('100000'), [])
    decision = {'schema': 'synthetic-operator-decision', 'decided_at': '2026-10-15T22:40:00Z'}
    path = tmp_path / 'decision.json'
    path.write_text(json.dumps(decision))
    with pytest.raises(TraderError, match='GPT_RETIREMENT_OPERATOR_DECISION_REFUSED'):
        executor.ledger.register_gpt_retirement(path)
    # Operator authorization is the sole synthetic boundary; all registration and planner checks are real.
    monkeypatch.setattr(paper, 'sha256_canonical', lambda value: retirement.DECISION_HASH if value == decision else sha256_canonical(value))
    executor.ledger.register_gpt_retirement(path)
    assert executor.ledger.events()[:len(before)] == before
    assert executor.ledger.events(event='binding') == [before[0]]
    count = len(executor.ledger.events())
    executor.ledger.register_gpt_retirement(path)
    assert len(executor.ledger.events()) == count
    lots = executor.entries('ia_actions', paper.D('100000'), [])
    aapl = [lot for lot in lots if lot['asset'] == 'AAPL' and lot['horizon'] == '1d']
    assert [(lot['qty'], lot['variant']) for lot in aapl] == ([(quantity, tier)] if quantity else [])
    assert all(plan['spec_hash'] == retirement.SPEC_HASH for plan in executor.plans('ia_actions', lots, 'entry'))
    from scripts.trading_lab.trader_agent.execution_replay import replay
    chain = executor.research.verify()
    result = replay(executor.research, grant, '2026-10-19')
    assert result['network_requests'] == 0 and result['no_order']
    assert executor.research.verify() == chain
    assert not broker.calls


def test_crypto_cannot_resurrect_consensus_or_full_size():
    from scripts.trading_lab.trader_agent.execution_tiers import selections
    from decimal import Decimal
    view = {'asset': 'SOL-USD', 'horizon': '1d', 'analyst': 'analyst_claude',
            'view': 'UP', 'p_outperform': .8, 'verdict': 'KEEP'}
    assert selections([view], 'ia_crypto', retired=True)[('SOL-USD', '1d')][1:] == (Decimal('.5'), 'crypto_single_kept_half_v1')
    with pytest.raises(TraderError, match='PAPER_RETIRED_VIEW_REFUSED'):
        selections([view, {**view, 'analyst': 'consensus'}], 'ia_crypto', retired=True)


def test_history_is_kept_and_new_claude_scores_do_not_enter_consensus_sample(service, clock):
    assert service.run()['status'] == 'COMPLETE'
    historical = service.store.records('prediction')
    prior = scorecard(service.store, synthetic=True)
    clock.at = instant('2026-10-19T12:00:00Z')
    assert service.run()['status'] == 'COMPLETE'
    assert service.store.records('prediction')[:len(historical)] == historical
    card = scorecard(service.store, synthetic=True, include_samples=True)
    for key, value in prior['scores'].items():
        assert card['scores'][key] == value
    assert card['scores']['analyst_gpt/equity_etf/1d/SPY_relative']['issued'] == 3
    assert card['scores']['gpt_retired_v1:analyst_claude/equity_etf/1d/SPY_relative']['issued'] == 3
    assert card['_primary_samples'] == []


def test_failed_claude_never_falls_back_to_retired_gpt(service, clock):
    clock.at = instant('2026-10-19T12:00:00Z')
    class Failure(SyntheticRunner):
        def once(self, role, text):
            assert role == 'analyst_claude'
            self.ledger.reserve(role)
            raise TraderError('MODEL_FAILED')
    service.runner = Failure(service.ledger, clock=clock)
    assert service.run()['status'] == 'FAILED'
    assert service.ledger.counts()['analyst_claude'] == 2
    assert not service.store.records('prediction')


def test_registration_preserves_existing_open_lots_and_their_original_spec(environment, tmp_path, monkeypatch):
    from scripts.trading_lab.trader_agent import alpaca_paper as paper
    from test_alpaca_paper import issue
    executor, broker, _, clock = environment
    issue(executor, horizons=('5d',))
    executor.run('execute', accounts=['ia_actions'])
    client_id = next(iter(broker.orders))[1]
    broker.fill('ia_actions', client_id)
    executor.run('status', accounts=['ia_actions'])
    before = executor.ledger.events()
    projection = executor.ledger.projection('ia_actions')
    clock.at = instant('2026-10-19T10:00:00Z')
    decision = {'schema': 'synthetic-operator-decision', 'decided_at': '2026-10-15T22:40:00Z'}
    path = tmp_path / 'decision.json'
    path.write_text(json.dumps(decision))
    monkeypatch.setattr(paper, 'sha256_canonical', lambda value: retirement.DECISION_HASH if value == decision else sha256_canonical(value))
    executor.ledger.register_gpt_retirement(path)
    assert executor.ledger.events()[:len(before)] == before
    assert executor.ledger.projection('ia_actions') == projection
    assert len(executor.ledger.events()) == len(before) + 1
    assert all(intent['spec_hash'] == paper.SPEC_HASH for intent in projection[0].values())


def test_wrong_retirement_hash_does_not_migrate_a_paper_store(environment):
    executor, _, _, clock = environment
    clock.at = instant('2026-10-19T10:00:00Z')
    target = retirement.registration_target(executor.ledger)
    executor.ledger.append('gpt_retirement_registration', None, **{**target, 'execution_spec_hash': 'a' * 64})
    with pytest.raises(TraderError, match='PAPER_GPT_RETIREMENT_REGISTRATION_REQUIRED'):
        executor.ledger.check_gpt_retirement_registration()

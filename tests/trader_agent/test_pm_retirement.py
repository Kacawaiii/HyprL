"""Forward PM retirement preserves AM monitoring and outstanding PM obligations."""
import json
from contextlib import closing
from datetime import timedelta

import pytest

from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.trader_agent.cli import health
from scripts.trading_lab.trader_agent.config import Authorization, TraderError, instant
from scripts.trading_lab.trader_agent.ledger import Ledger
from tests.trader_agent.test_pm_slot import enable_pm, pm_grant
from tests.trader_agent.test_alpaca_paper import environment, issue


@pytest.mark.parametrize('at,pm_status,expected', [
    ('2026-10-25T23:59:59Z', None, 'MISSING_PM_RUN'),
    ('2026-10-25T23:59:59Z', 'FAILED', 'FAILED_PM_RUN'),
    ('2026-10-26T00:00:00Z', None, 'HEALTHY'),
    ('2026-10-26T18:00:00Z', 'FAILED', 'HEALTHY'),
])
def test_health_stops_expecting_pm_at_retirement(tmp_path, grant, clock, at, pm_status, expected):
    clock.at = instant(at)
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock)
    with closing(ResearchStore(ledger.root / 'evidence')) as store:
        for slot, status in [('am', 'COMPLETE'), ('pm', pm_status)]:
            if status:
                store.append('replay-summary', {'schema': 'trader-run-v1', 'slot': slot,
                    'at': at, 'synthetic': False, 'status': status}, recorded_at=at)
    (ledger.root / 'last-label.json').write_text(json.dumps({'at': at, 'state': 'COMPLETE'}))
    result = health(ledger)
    assert result['state'] == expected
    assert result['slots']['am'] == 'HEALTHY'
    assert result['slots']['pm'] == ('RETIRED' if at >= '2026-10-26' else expected)


@pytest.mark.parametrize('am_status,label,expected', [
    (None, True, 'MISSING_DAILY_RUN'),
    ('FAILED', True, 'FAILED_DAILY_RUN'),
    ('SKIPPED_QUOTA', True, 'FAILED_DAILY_RUN'),
    ('DEGRADED', True, 'HEALTHY'),
    ('COMPLETE', False, 'MISSING_LABEL_JOB'),
])
def test_retirement_keeps_am_and_label_health_gates(tmp_path, grant, clock, am_status, label, expected):
    clock.at = instant('2026-10-26T22:05:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock)
    with closing(ResearchStore(ledger.root / 'evidence')) as store:
        # Even an unexpected successful PM run cannot satisfy the AM check.
        for slot, status in [('am', am_status), ('pm', 'COMPLETE')]:
            if status:
                store.append('replay-summary', {'schema': 'trader-run-v1', 'slot': slot,
                    'at': '2026-10-26T18:00:00Z', 'synthetic': False, 'status': status})
    if label:
        (ledger.root / 'last-label.json').write_text('{"at":"2026-10-26T21:30:00Z"}')
    assert health(ledger)['state'] == expected
    clock.at = instant('2027-02-02T18:00:00Z')
    assert health(ledger)['state'] == 'EXPIRED_OR_UNAVAILABLE'


@pytest.mark.parametrize('kind', ['run', 'catchup_run', 'analyst_claude', 'reviewer',
    'yahoo_chart', 'coinbase_exchange_public', 'gdelt_doc_api'])
@pytest.mark.parametrize('arm,preflight', [('base', False), ('news', False), ('base', True)])
def test_no_pm_dispatch_or_budget_charge_after_retirement(tmp_path, grant, clock, kind, arm, preflight):
    clock.at = instant('2026-10-26T16:00:00Z')
    grant = pm_grant(grant)
    ledger = Ledger(tmp_path / 'runtime', grant, clock=clock, slot='pm', preflight=preflight)
    ledger.arm = arm
    with pytest.raises(TraderError, match='PM_SLOT_RETIRED'):
        ledger.reserve(kind)
    assert ledger.counts() == {}
    am = Ledger(tmp_path / 'runtime', grant, clock=clock)
    am.reserve('run')
    with pytest.raises(TraderError, match='BUDGET_EXHAUSTED'):
        am.reserve('run')
    assert am.counts()['run'] == 1


def test_decision_and_new_spec_are_bound_without_rebinding_old_pm(tmp_path, grant):
    from scripts.trading_lab.trader_agent import pm_retirement_spec, slots
    assert pm_retirement_spec.spec()['lifecycle'] == {'am': 'ACTIVE', 'pm': 'RETIRED'}
    assert slots.spec()['schema'] == 'trader-pm-slot-spec-v2'
    assert slots.preregistration() == slots.PREREG_HASH
    path = tmp_path / 'grant.json'
    path.write_text(json.dumps(grant.payload))
    decision = tmp_path / pm_retirement_spec.DECISION_NAME
    decision.write_text((pm_retirement_spec.ROOT / decision.name).read_text())
    assert Authorization.load(path).identity == grant.identity
    decision.write_text('{}')
    with pytest.raises(TraderError, match='PM_RETIREMENT_OPERATOR_DECISION_HASH_MISMATCH'):
        Authorization.load(path)


@pytest.mark.parametrize('horizon,exit_day', [('1d', '2026-10-26'), ('5d', '2026-10-30')])
def test_retirement_preserves_old_pm_exits_checkpoints_and_label_obligations(environment, horizon, exit_day):
    from scripts.trading_lab.trader_agent.post_audit import checkpoints, inspect_audit
    from scripts.trading_lab.trader_agent.alpaca_paper import decimal
    executor, broker, _, clock = environment
    enable_pm(executor)
    clock.at = instant('2026-10-23T16:05:00Z')
    issue(executor, slot='pm', horizons=(horizon,))
    assert executor.run('execute', accounts=['ia_actions'])['state'] == 'COMPLETE'
    broker.fill('ia_actions', broker.posts()[0][3]['client_order_id'])
    executor.run('audit', accounts=['ia_actions'])
    bindings = executor.ledger.events(event='binding')
    clock.at = instant(exit_day + 'T19:55:00Z')
    assert executor.run('exit', accounts=['ia_actions'])['state'] == 'COMPLETE'
    assert len(broker.posts()) == 2
    assert broker.posts()[-1][3]['side'] == 'sell'
    broker.fill('ia_actions', broker.posts()[-1][3]['client_order_id'])
    executor.run('audit', accounts=['ia_actions'])
    assert executor.ledger.events(event='binding') == bindings
    live = executor.ledger.projection('ia_actions')[2]
    assert live and all(decimal(lot['qty']) == 0 for lot in live.values())
    clock.at += timedelta(hours=2)
    due = checkpoints(executor.grant.parent, executor.research, executor.ledger.events(), clock())
    assert not any(key.startswith('slot-entry:pm:') for key in due)
    assert any(key.startswith('pm-plan:') for key in due)
    assert any(key.startswith('planned-exit:') for key in due)
    result = inspect_audit(executor, label_at=clock(), require_daily_run=True,
                           require_pm_run=True, require_label_job=True)
    assert 'MISSING_PM_RUN' not in result['errors']
    assert {'MISSING_DAILY_RUN', 'LABELS_MISSING', 'MISSING_OR_FAILED_LABEL_JOB'} <= set(result['errors'])
    assert len(broker.posts()) == 2


@pytest.mark.parametrize('slot,orders', [('am', 1), ('pm', 0)])
def test_retirement_still_admits_am_entries_and_ignores_retired_pm_plans(environment, slot, orders):
    executor, broker, _, clock = environment
    enable_pm(executor)
    clock.at = instant('2026-10-26T13:30:30Z' if slot == 'am' else '2026-10-26T16:05:00Z')
    issue(executor, slot=slot)
    result = executor.run('execute', accounts=['ia_actions'])
    assert result['state'] == 'COMPLETE'
    assert len(broker.posts()) == orders
    if slot == 'pm':
        assert executor.eligible_predictions('ia_actions') == []


def test_midnight_audit_keeps_prior_pm_evidence_without_requiring_entry_permission(environment):
    from scripts.trading_lab.trader_agent.post_audit import inspect_audit
    executor, broker, _, clock = environment
    enable_pm(executor)
    clock.at = instant('2026-10-25T16:05:00Z')
    issue(executor, slot='pm', assets=('BTC-USD',), horizons=('1d',))
    clock.at = instant('2026-10-26T00:01:30Z')
    # The NY date is still Oct 25, so the prior PM summary is considered by the
    # legacy cross-midnight lookup. Its forecasts must not become new entries.
    result = inspect_audit(executor)
    assert result['state'] == 'GREEN', result['errors']
    assert not broker.posts()
    assert executor.research.records('prediction')


def test_post_audit_does_not_restart_pm_retry_but_keeps_am_retry(tmp_path, clock, monkeypatch):
    from scripts.trading_lab.trader_agent import post_audit
    from scripts.trading_lab.trader_agent.rehearsal import MonitorState
    clock.at = instant('2026-10-26T18:11:30Z')
    monkeypatch.setattr(post_audit, 'now', clock)
    monkeypatch.setenv('HOME', str(tmp_path))
    monitor = tmp_path / 'monitor'
    state = MonitorState(monitor)
    for prefix in ('', 'pm-'):
        state.save('2026-10-26', prefix + 'primary',
                   {'state': 'BLOCKED', 'retry_at': '2026-10-26T17:00:00Z'})
    commands = []
    monkeypatch.setattr(post_audit.subprocess, 'run', lambda argv, **kw: commands.append(argv))
    # Missing synthetic auth fails before any broker request; retry scheduling still runs.
    assert post_audit.main(['--authorization', str(tmp_path / 'missing'), '--runtime', str(tmp_path),
        '--paper-authorization', '/synthetic-paper', '--data-authorization', '/synthetic-data',
        '--monitor-runtime', str(monitor), '--reports', str(tmp_path / 'reports'),
        '--telegram-env', str(tmp_path / 'missing-notifications')]) == 1
    assert commands == [['systemctl', '--user', 'start', '--no-block', 'hyprl-trader-rehearsal-retry.service']]


@pytest.mark.parametrize('extra', [[], ['--retry'], ['--failure-hook']])
def test_retired_pm_rehearsal_is_a_noop_before_monitor_or_notification(tmp_path, clock, monkeypatch, extra):
    from scripts.trading_lab.trader_agent import rehearsal
    clock.at = instant('2026-10-26T15:15:00Z')
    monkeypatch.setattr(rehearsal, 'now', clock)
    monitor = tmp_path / 'monitor'
    assert rehearsal.main(['--authorization', '/synthetic', '--runtime', '/synthetic',
        '--paper-authorization', '/synthetic', '--data-authorization', '/synthetic',
        '--monitor-runtime', str(monitor), '--slot', 'pm', *extra]) == 0
    assert not monitor.exists()


@pytest.mark.parametrize('action', ['run', 'news', 'recover', 'catchup', 'catchup-after-hours', 'paper-execute'])
def test_retired_pm_cli_does_no_work(tmp_path, grant, clock, monkeypatch, capsys, action):
    from scripts.trading_lab.trader_agent import cli
    clock.at = instant('2026-10-26T16:00:00Z')
    monkeypatch.setattr(cli, 'now', clock)
    monkeypatch.setenv('HOME', str(tmp_path))
    path = tmp_path / 'synthetic-grant.json'
    runtime = tmp_path / 'runtime'
    path.write_text(json.dumps(grant.payload))
    assert cli.main([action, '--authorization', str(path), '--runtime', str(runtime), '--slot', 'pm']) == 0
    assert json.loads(capsys.readouterr().out) == {'state': 'RETIRED', 'slot': 'pm'}
    assert not runtime.exists()


def test_retired_pm_recovery_uses_no_calls_or_alerts(tmp_path, grant, clock):
    from scripts.trading_lab.trader_agent.recovery import recover
    from scripts.trading_lab.trader_agent.service import TraderService
    from scripts.trading_lab.trader_agent.synthetic import SyntheticData, SyntheticRunner
    clock.at = instant('2026-10-26T18:00:00Z')
    ledger = Ledger(tmp_path / 'runtime', pm_grant(grant), clock=clock, slot='pm')
    service = TraderService(ledger, SyntheticData(ledger, clock), SyntheticRunner(ledger, clock=clock), clock=clock)
    assert recover(service)['state'] == 'RETIRED'
    assert ledger.counts() == {}
    assert service.store.records('prediction') == []
    assert not (ledger.root / 'alerts.jsonl').exists()


def test_audit_retains_am_and_crypto_windows_after_pm_retirement():
    from scripts.trading_lab.trader_agent.post_audit import execution_window
    assert execution_window(instant('2026-10-23T16:00:00Z'))
    assert not execution_window(instant('2026-10-26T16:00:00Z'))
    assert execution_window(instant('2026-10-26T12:00:00Z'))
    assert execution_window(instant('2026-10-26T16:30:00Z'))

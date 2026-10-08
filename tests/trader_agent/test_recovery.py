from datetime import timedelta
import json

import pytest

from scripts.trading_lab.trader_agent.config import instant
from scripts.trading_lab.trader_agent.recovery import recover
from scripts.trading_lab.trader_agent.data import calendar_session


def fail_primary(service):
    service.ledger.reserve('run')
    day = service.clock().date().isoformat()
    run_id = 'trader:' + day + (':synthetic' if service.data.synthetic else ':real')
    service.summary(run_id, 'FAILED', service.clock(), error='SYNTHETIC_FAILURE')


def alert_code(service):
    return json.loads((service.ledger.root / 'alert.json').read_text())['code']


def test_failed_primary_recovers_once_and_alerts_every_outcome(service, clock):
    fail_primary(service)
    clock.at = instant('2026-10-06T14:00:00Z')
    result = recover(service)
    assert result['status'] == 'COMPLETE' and result['mode'] == 'close'
    assert alert_code(service) == 'RECOVERY_COMPLETE'
    counts = service.ledger.counts()
    assert counts['catchup_run'] == 1 and counts['run'] == 1
    assert sum(counts.get(r, 0) for r in ('analyst_claude', 'analyst_gpt', 'reviewer')) == 3
    assert recover(service)['state'] == 'ALREADY_ATTEMPTED'
    assert alert_code(service) == 'RECOVERY_ALREADY_ATTEMPTED'
    assert service.ledger.counts() == counts


def test_late_failure_waits_until_close_then_uses_after_hours(service, clock):
    fail_primary(service)
    clock.at = instant('2026-10-06T19:00:00Z')
    assert recover(service)['state'] == 'WAITING_AFTER_HOURS'
    assert not service.ledger.counts().get('catchup_run')
    clock.at = instant('2026-10-06T20:00:00Z')
    result = recover(service)
    assert result['status'] == 'COMPLETE' and result['mode'] == 'after_hours'
    assert all(r['payload']['signal']['label_definition']['variant'] == 'alpaca_after_hours_v1'
               for r in service.store.records('prediction'))


@pytest.mark.parametrize('at,expected', [('2026-10-10T14:00:00Z', 'SKIPPED_HOLIDAY'),
                                      ('2026-12-25T14:00:00Z', 'SKIPPED_HOLIDAY'),
                                      ('2026-10-06T22:10:00Z', 'MISSED_CUTOFF')])
def test_calendar_and_exact_latest_start_cutoff(service, clock, at, expected):
    clock.at = instant(at)
    if expected == 'MISSED_CUTOFF':
        fail_primary(service)
    assert recover(service)['state'] == expected
    assert alert_code(service) == 'RECOVERY_' + expected
    assert not service.ledger.counts().get('catchup_run')


def test_half_day_uses_early_close(service, clock):
    clock.at = instant('2026-11-27T16:41:00Z')
    fail_primary(service)
    assert calendar_session('2026-11-27').close_at == instant('2026-11-27T18:00:00Z')
    assert recover(service)['state'] == 'WAITING_AFTER_HOURS'
    clock.at = instant('2026-11-27T18:00:00Z')
    assert recover(service)['mode'] == 'after_hours'


def test_successful_primary_never_recovers_even_after_later_failed_attempt(service, clock):
    assert service.run()['status'] == 'COMPLETE'
    service.summary('trader:2026-10-06:synthetic', 'FAILED', clock(), error='SYNTHETIC_LATER_FAILURE')
    counts = service.ledger.counts()
    assert recover(service)['state'] == 'NO_RECOVERY_NEEDED'
    assert service.ledger.counts() == counts


def test_pause_and_exhausted_reviewer_do_not_launch_recovery(service, clock):
    fail_primary(service)
    marker = service.ledger.root / 'PAUSED'
    marker.touch()
    assert recover(service)['state'] == 'PAUSED'
    marker.unlink()
    service.ledger.reserve('reviewer')
    service.ledger.reserve('reviewer')
    assert recover(service)['state'] == 'BUDGET_EXHAUSTED'
    assert not service.ledger.counts().get('catchup_run')


def test_recorded_primary_decision_without_success_summary_never_recovers(service, monkeypatch):
    from scripts.trading_lab.trader_agent.config import TraderError
    original = service.record
    def fail_after_recording(decision, context):
        original(decision, context)
        raise TraderError('SYNTHETIC_CRASH_AFTER_DECISION')
    monkeypatch.setattr(service, 'record', fail_after_recording)
    assert service.run()['status'] == 'FAILED'
    assert recover(service)['state'] == 'DECISION_ALREADY_RECORDED'
    assert not service.ledger.counts().get('catchup_run')


def test_successful_recovery_resolves_current_health_and_keeps_failure_history(service, clock):
    from scripts.trading_lab.trader_agent.cli import health
    service.data.synthetic = False
    clock.at = instant('2026-10-09T12:00:00Z')
    fail_primary(service)
    clock.at = instant('2026-10-09T14:00:00Z')
    assert health(service.ledger)['state'] == 'FAILED_DAILY_RUN'
    assert recover(service)['status'] == 'COMPLETE'
    assert health(service.ledger)['state'] == 'HEALTHY'
    assert any(r['payload'].get('status') == 'FAILED' for r in service.store.records('replay-summary'))


def test_busy_global_owner_does_not_reserve_or_launch_a_second_run(service):
    fail_primary(service)
    with service.ledger.owner():
        assert recover(service)['state'] == 'OWNER_BUSY'
    assert not service.ledger.counts().get('catchup_run')
    assert alert_code(service) == 'RECOVERY_OWNER_BUSY'

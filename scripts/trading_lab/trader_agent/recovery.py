"""Automatic same-session recovery using the existing durable catch-up budget."""
from datetime import timedelta
from zoneinfo import ZoneInfo

from .config import TraderError, iso
from .data import calendar_session
from .alpaca_paper import after_hours_cutoff
from .scoring import rows

NY = ZoneInfo('America/New_York')


def recover(service):
    ledger, at = service.ledger, service.clock()
    if ledger.slot == 'pm':
        return recover_pm(service)
    day = at.astimezone(NY).date().isoformat()
    utc_closed = calendar_session(at.date().isoformat()) is None
    if utc_closed and ledger.grant.weekend_decision:
        day = at.date().isoformat()
    base = {'schema': 'trader-recovery-v1', 'at': iso(at), 'session': day}

    def outcome(state, **extra):
        ledger.alert_state('recovery', 'RECOVERY_' + state)
        return {**base, 'state': state, **extra}

    previous_day = ledger.budget_day
    ledger.budget_day = day
    try:
        ledger.grant.check(at)
        if ledger.paused:
            return outcome('PAUSED')
        if utc_closed and ledger.grant.weekend_decision:
            ledger.grant.check_weekend(at)
            return outcome('WEEKEND_CRYPTO_SKIPPED')
        session = calendar_session(day)
        if not session:
            return outcome('SKIPPED_HOLIDAY')
        if ledger.counts().get('catchup_run'):
            return outcome('ALREADY_ATTEMPTED')
        run_id = 'trader:' + day + (':synthetic' if service.data.synthetic else ':real')
        runs = [r['payload'] for r in rows(service.store, 'replay-summary')
                if r['payload'].get('schema') == 'trader-run-v1' and r['payload'].get('run_id') == run_id]
        if any(r.get('decision') or r['status'] in {'COMPLETE', 'DEGRADED'} for r in runs):
            return outcome('NO_RECOVERY_NEEDED')
        if any(r['payload'].get('signal', {}).get('run_id') == run_id for r in rows(service.store, 'prediction')):
            return outcome('DECISION_ALREADY_RECORDED')
        if not any(r['status'] == 'FAILED' for r in runs):
            return outcome('NO_FAILED_PRIMARY')
        service.catchup_missing(run_id)
        if at < session.close_at - timedelta(minutes=80):
            mode = 'close'
        elif at < session.close_at:
            return outcome('WAITING_AFTER_HOURS')
        elif at < after_hours_cutoff(session) - timedelta(minutes=80):
            mode = 'after_hours'
        else:
            return outcome('MISSED_CUTOFF')
        # Service acquires the global owner and rechecks eligibility, deadlines,
        # pause and all budgets before reserving or dispatching anything.
        result = service.run(catchup=True, after_hours=mode == 'after_hours')
        ledger.alert_state('recovery', 'RECOVERY_' + result['status'], force=True)
        return {**base, **result, 'mode': mode}
    except TraderError as error:
        return outcome(error.code)
    except Exception:
        ledger.alert('RECOVERY_UNEXPECTED_FAILURE')
        raise
    finally:
        ledger.budget_day = previous_day


def recover_pm(service):
    """Same-day pm retry; never switch slots, anchors or replenish reservations."""
    from . import slots
    ledger, at = service.ledger, service.clock()
    day = at.date().isoformat()
    base = {'schema': 'trader-recovery-v1', 'at': iso(at), 'session': day, 'slot': 'pm'}
    try:
        ledger.grant.check_slot('pm', at)
        if ledger.paused:
            return {**base, 'state': 'PAUSED'}
        if at < slots.start_at(day, 'pm'):
            return {**base, 'state': 'NOT_STARTED'}
        crypto_only = calendar_session(day) is None
        if crypto_only:
            ledger.grant.check_weekend(at)
        if at >= slots.recovery_deadline(day, crypto_only):
            return {**base, 'state': 'MISSED_CUTOFF'}
        if ledger.counts().get('catchup_run'):
            return {**base, 'state': 'ALREADY_ATTEMPTED'}
        run = slots.run_id(day, service.data.synthetic, 'pm')
        attempts = [r['payload'] for r in rows(service.store, 'replay-summary')
                    if r['payload'].get('run_id') == run and r['payload'].get('schema') == 'trader-run-v1']
        if any(r.get('decision') or r['status'] in {'COMPLETE', 'DEGRADED'} for r in attempts):
            return {**base, 'state': 'NO_RECOVERY_NEEDED'}
        if not attempts and not ledger.counts().get('run') and at >= slots.start_at(day, 'pm') + timedelta(minutes=5):
            # A service lost before reserving a run has no journal marker. The
            # original primary may still start, under the same durable owner.
            if at < slots.deadline(day, crypto_only):
                return {**base, **service.run(), 'mode': 'pm'}
            with ledger.owner():
                ledger.reserve('run')
                service.summary(run, 'FAILED', at, error='PM_PRIMARY_MISSED')
            attempts = [{'status': 'FAILED'}]
        if not any(r['status'] == 'FAILED' for r in attempts):
            if any(r['status'] == 'RUNNING' for r in attempts):
                with ledger.owner():
                    if any(r['payload'].get('signal', {}).get('run_id') == run for r in rows(service.store, 'prediction')):
                        return {**base, 'state': 'DECISION_ALREADY_RECORDED'}
                    service.summary(run, 'FAILED', at, error='PM_PRIMARY_INTERRUPTED')
            else:
                return {**base, 'state': 'NO_FAILED_PRIMARY'}
        service.catchup_missing(run)
        result = service.run(catchup=True)
        ledger.alert_state('recovery:pm', 'RECOVERY_' + result['status'], force=True)
        return {**base, **result, 'mode': 'pm'}
    except TraderError as error:
        ledger.alert_state('recovery:pm', 'RECOVERY_' + error.code)
        return {**base, 'state': error.code}

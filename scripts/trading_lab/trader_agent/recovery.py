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

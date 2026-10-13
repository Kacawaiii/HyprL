"""Read-only order/lot/label checkpoints, ten minutes after planned and actual execution."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import re
import subprocess

from scripts.trading_lab.research.store import ResearchStore
from scripts.trading_lab.sources.canonical import sha256_canonical

from .config import Authorization, TraderError, instant, iso, now, strict_json
from .data import calendar_session
from .scoring import rows
from .rehearsal import (MonitorState, SnapshotPaperLedger, arguments, error_code, get_only_client,
                        population, readonly_executor, telegram)


def checkpoints(grant, research, events, at, last_label=None):
    """Scheduled checkpoints catch a completely missing run/order as well as existing intents."""
    day = at.date().isoformat()
    session = calendar_session(day)
    result = {}
    if population(grant, at) != 'closed':
        entry = session.open_at + timedelta(seconds=30) if session else at.replace(hour=13, minute=30, second=0, microsecond=0)
        result['session-entry:' + day] = entry + timedelta(minutes=10)
        result['label-deadline:' + day] = at.replace(hour=21, minute=40, second=0, microsecond=0)
    if grant.pm_amendment and at >= instant(grant.pm_amendment['effective_from']):
        from .slots import deadline
        result['slot-entry:pm:' + day] = deadline(day, session is None) + timedelta(minutes=10)
    for row in rows(research, 'prediction'):
        p = row['payload']
        if p.get('synthetic', True):
            continue
        definition = p['signal']['label_definition']
        if p['signal'].get('slot') == 'pm':
            result['pm-plan:' + p['signal']['run_id']] = instant(p['decision_at']) + timedelta(minutes=10)
        exit_at = instant(definition['exit_at'])
        equity = p['product'] not in grant.payload['universe']['crypto']
        due = exit_at + timedelta(minutes=5 if equity else 10)
        result['planned-exit:' + iso(due)] = due
    for event in events:
        if event.get('event') == 'intent':
            result['intent:' + event['client_id']] = instant(event['at']) + timedelta(minutes=10)
    if last_label:
        result['after-label:' + sha256_canonical(last_label)] = instant(last_label['at'])
    # Historical checkpoints are audited at adoption, not silently discarded after a restart.
    return {key: due for key, due in result.items() if due <= at}


def inspect_audit(executor, *, label_at=None, require_daily_run=False, require_label_job=False, last_label=None, require_pm_run=False):
    from .alpaca_paper import decimal
    errors, accounts, issues = [], [], []
    def problem(code, identity):
        errors.append(code)
        issues.append({'code': code, 'identity': sha256_canonical([code, identity])})
    executor.grant.check(executor.dispatch_clock())
    executor.research.verify()
    for name in ('ia_actions', 'ia_crypto'):
        try:
            _, account, positions = executor.start(name)
            intents, observed, lots = executor.ledger.projection(name)
            canceled = {e['client_id'] for e in executor.ledger.events(name, 'cancel') if e.get('reason')}
            dispositions = {}
            for key, intent in intents.items():
                order = observed.get(key, {})
                status = order.get('status')
                if instant(intent['at']) + timedelta(minutes=10) > executor.clock():
                    continue
                if status == 'filled' and decimal(order.get('filled_qty', 0)) == decimal(intent['order']['qty']):
                    disposition = 'FILLED'
                elif status == 'internal' or decimal(intent['order']['qty']) == 0:
                    disposition = 'SETTLED_INTERNAL_RULE'
                elif status in {'canceled', 'not_submitted'} and (key in canceled or order.get('reason')):
                    disposition = 'CANCELED_BY_RULE'
                else:
                    disposition = 'ALERTED'
                    problem('PAPER_ORDER_' + status.upper() if status in {'expired', 'rejected'} else 'PAPER_ORDER_UNRESOLVED',
                            [name, key])
                dispositions[disposition] = dispositions.get(disposition, 0) + 1
            for lot in lots.values():
                grace = timedelta(minutes=5 if name == 'ia_actions' else 10)
                if decimal(lot['qty']) and instant(lot['exit_at']) + grace <= executor.clock():
                    problem('PAPER_LOT_PAST_HORIZON', [name, lot['lot_id']])
            # Reconstruct the executor's actual selection at the planned entry clock.
            # This catches eligible plans never reserved without guessing from all analyst views.
            original = executor.clock
            try:
                candidates = executor.eligible_predictions(name)
                for row, _, _, _ in candidates:
                    p = row['payload']
                    definition = p['signal']['label_definition']
                    if name == 'ia_actions':
                        session = calendar_session(p['signal']['session'])
                        after_hours = definition.get('variant') == 'alpaca_after_hours_v1'
                        dispatch_at = instant(p['decision_at']) + timedelta(seconds=1) if after_hours or p['signal'].get('slot') == 'pm' else session.open_at + timedelta(seconds=30)
                    else:
                        dispatch_at = instant(p['decision_at']) + timedelta(seconds=1)
                    if dispatch_at + timedelta(minutes=10) > original():
                        continue
                    executor.clock = lambda dispatch_at=dispatch_at: dispatch_at
                    planned = executor.entries(name, decimal(account['equity']), positions)
                    if planned:
                        problem('PAPER_ELIGIBLE_ENTRY_NOT_DISPATCHED', [name, row['identity']])
                        break
            finally:
                executor.clock = original
            accounts.append({'account': name, 'suffix': executor.grant.payload['accounts'][name]['account_suffix'],
                             'reconciled': True, 'open_lots': sum(bool(decimal(l['qty'])) for l in lots.values()),
                             'orders': dispositions})
        except Exception as error:
            problem(error_code(error), [name, executor.clock().date().isoformat()])
            accounts.append({'account': name, 'reconciled': False})
    at = executor.clock()
    day = at.date().isoformat()
    if require_daily_run:
        runs = [r['payload'] for r in rows(executor.research, 'replay-summary')
                if r['payload'].get('schema') == 'trader-run-v1' and not r['payload'].get('synthetic', True)
                and r['payload'].get('slot', 'am') == 'am'
                and r['payload']['at'][:10] == day and r['payload'].get('status') in {'COMPLETE', 'DEGRADED'}]
        if not runs:
            problem('MISSING_DAILY_RUN', day)
    if require_pm_run:
        runs = [r['payload'] for r in rows(executor.research, 'replay-summary')
                if r['payload'].get('schema') == 'trader-run-v1' and not r['payload'].get('synthetic', True)
                and r['payload'].get('slot') == 'pm' and r['payload']['at'][:10] == day
                and r['payload'].get('status') in {'COMPLETE', 'DEGRADED'}]
        if not runs:
            problem('MISSING_PM_RUN', day)
    if require_label_job and (not last_label or last_label.get('at', '')[:10] != day or last_label.get('state') != 'COMPLETE'):
        problem('MISSING_OR_FAILED_LABEL_JOB', day)
    if label_at and last_label and last_label.get('state') != 'COMPLETE':
        problem('LABEL_JOB_FAILED', last_label['at'])
    if label_at:
        labeled = {r['payload']['prediction_hash'] for r in rows(executor.research, 'label')}
        missing = [r for r in rows(executor.research, 'prediction') if not r['payload'].get('synthetic', True)
                   and instant(r['payload']['signal']['label_definition']['exit_at']) <= label_at and r['identity'] not in labeled]
        if missing:
            for row in missing:
                problem('LABELS_MISSING', row['identity'])
    else:
        missing = []
    return {'schema': 'trader-post-audit-v1', 'at': iso(at), 'state': 'BLOCKED' if errors else 'GREEN',
            'errors': sorted(set(errors)), 'accounts': accounts, 'labels_missing': len(missing),
            'issues': issues,
            'broker_methods': ['GET'], 'production_writes': 0}


def execution_window(at, research=None):
    """Leave the two-minute service budget clear of entry/exit and their retry minutes."""
    anchors = [at.replace(hour=hour, minute=0, second=0, microsecond=0) for hour in (12, 16)]
    anchors.append(at.replace(minute=30, second=0, microsecond=0))  # hourly crypto exit check
    session = calendar_session(at.date().isoformat())
    if session:
        anchors.extend((session.open_at, session.close_at - timedelta(minutes=5)))
    if research:
        for row in rows(research, 'prediction'):
            if not row['payload'].get('synthetic', True):
                definition = row['payload']['signal']['label_definition']
                anchors.extend(instant(definition[k]) for k in ('entry_at', 'exit_at'))
    return any(anchor - timedelta(minutes=2) <= at < anchor + timedelta(minutes=3) for anchor in anchors)


def audit(args, state, *, clock=now, force=False, notify=None, client_factory=get_only_client):
    at = clock()
    if execution_window(at):
        return {'state': 'PAPER_EXECUTION_WINDOW'}
    key = 'audit-infrastructure:' + at.date().isoformat()
    try:
        grant = Authorization.load(args.authorization)
        grant.check(at)
        from .alpaca_paper import PaperAuthorization
        paper_grant = PaperAuthorization.load(args.paper_authorization, grant)
        research = ResearchStore(Path(args.runtime) / 'evidence', read_only=True)
        paper = SnapshotPaperLedger(args.paper_runtime, paper_grant, clock)
        try:
            if execution_window(at, research):
                return {'state': 'PAPER_EXECUTION_WINDOW'}
            with paper.owner():
                path = Path(args.runtime) / 'last-label.json'
                last_label = strict_json(path.read_text()) if path.exists() else None
                due = checkpoints(grant, research, paper.events(), at, last_label)
                with state.connect() as db:
                    done = {r[0] for r in db.execute('SELECT id FROM checkpoints')}
                due = {k: v for k, v in due.items() if k not in done}
                if force:
                    due['forced:' + iso(at)] = at
                if not due:
                    return {'state': 'NO_CHECKPOINT_DUE'}
                label_check = any(k.startswith(('after-label:', 'label-deadline:', 'forced:')) for k in due)
                executor = readonly_executor(paper, research, clock=clock, client_factory=client_factory)
                result = inspect_audit(executor,
                    require_daily_run=any(k.startswith('session-entry:') for k in due),
                    require_pm_run=any(k.startswith('slot-entry:pm:') for k in due),
                    require_label_job=any(k.startswith('label-deadline:') for k in due),
                    label_at=instant(last_label['at']) if label_check and last_label else None, last_label=last_label)
                result['checkpoints'] = sorted(due)
                if not paper.unchanged():
                    return {'state': 'PAPER_EXECUTION_IN_PROGRESS'}
                with state.connect() as db:
                    for checkpoint in due:
                        db.execute('INSERT OR IGNORE INTO checkpoints VALUES(?,?)', (checkpoint, json.dumps(result)))
        finally:
            research.close()
            paper.store.close()
    except Exception as error:
        if isinstance(error, TraderError) and error.code == 'PAPER_OWNER_BUSY':
            return {'state': 'PAPER_EXECUTION_IN_PROGRESS'}
        result = {'schema': 'trader-post-audit-v1', 'at': iso(at), 'state': 'BLOCKED', 'errors': [error_code(error)]}
    apply_acknowledgements(state, result)
    if result['state'] == 'BLOCKED':
        issues = result.get('issues', [{'code': code, 'identity': sha256_canonical([key, code])}
                                       for code in result['errors']])
        result['alert'] = alert_issues(state, issues, notify)
    root = Path(args.reports)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = root / ('trader-post-audit-' + at.strftime('%Y-%m-%dT%H%M%S') + '.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    return result


def acknowledge_issues(state, issues, *, reason, at, unlabelable=False, not_activated=False, decision_sha256=None):
    """Explicit operator disposition of exact historical identities, never a code-wide mute."""
    if not re.fullmatch(r'[A-Z][A-Z0-9_]{2,159}', reason):
        raise TraderError('AUDIT_ACKNOWLEDGEMENT_REFUSED')
    if (not_activated and (unlabelable or not isinstance(decision_sha256, str)
                           or not re.fullmatch(r'[a-f0-9]{64}', decision_sha256))
            or not not_activated and decision_sha256 is not None):
        raise TraderError('AUDIT_ACKNOWLEDGEMENT_REFUSED')
    allowed = {'MISSING_PM_RUN'} if not_activated else {'LABELS_MISSING', 'PAPER_ORDER_EXPIRED'}
    for issue in issues:
        if (issue['code'] not in allowed
                or not re.fullmatch(r'[a-f0-9]{64}', issue['identity'])
                or (unlabelable and issue['code'] != 'LABELS_MISSING')):
            raise TraderError('AUDIT_ACKNOWLEDGEMENT_REFUSED')
    with state.connect() as db:
        for issue in issues:
            inserted = db.execute('INSERT OR IGNORE INTO acknowledgements VALUES(?,?,?,?,?)',
                       (issue['identity'], issue['code'], reason,
                        'NOT_ACTIVATED' if not_activated else 'UNLABELABLE' if unlabelable else 'ACKNOWLEDGED', iso(at))).rowcount
            if inserted and not_activated:
                db.execute('INSERT INTO acknowledgement_decisions VALUES(?,?)', (issue['identity'], decision_sha256))


def apply_acknowledgements(state, result):
    if 'issues' not in result:
        return
    active, known = [], []
    with state.connect() as db:
        for issue in result['issues']:
            row = db.execute('SELECT reason,disposition,at FROM acknowledgements WHERE identity=? AND code=?',
                             (issue['identity'], issue['code'])).fetchone()
            if row:
                disposition = {**issue, 'reason': row[0], 'disposition': row[1], 'at': row[2]}
                decision = db.execute('SELECT sha256 FROM acknowledgement_decisions WHERE identity=?',
                                      (issue['identity'],)).fetchone()
                if decision:
                    disposition['decision_sha256'] = decision[0]
                known.append(disposition)
            else:
                active.append(issue)
    result['observed_errors'] = result['errors']
    result['acknowledged_issues'] = known
    result['labels_unlabelable'] = sum(i['disposition'] == 'UNLABELABLE' for i in known)
    result['issues'] = active
    result['errors'] = sorted({i['code'] for i in active})
    result['state'] = 'BLOCKED' if active else 'GREEN'


def alert_issues(state, issues, notify):
    """New checkpoints must not repeatedly notify an already reported expired intent.

    Reserve individual issue identities, then send one bounded batch. A new order
    with the same error still alerts; hundreds of missing labels do not spam.
    """
    fresh = {}
    with state.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        for issue in issues:
            if db.execute('SELECT 1 FROM acknowledgements WHERE identity=? AND code=?',
                          (issue['identity'], issue['code'])).fetchone():
                continue
            key = 'audit-issue:' + issue['identity']
            if db.execute('INSERT OR IGNORE INTO notifications VALUES(?,?)', (key, 'RESERVED')).rowcount:
                fresh[key] = issue['code']
    if not fresh:
        return {'delivery': 'ALREADY_NOTIFIED', 'new_issues': 0}
    delivery = state.alert('audit-batch:' + sha256_canonical(sorted(fresh)),
        'Trader post-run audit failed: ' + ', '.join(sorted(set(fresh.values()))) +
        ' (' + str(len(fresh)) + ' new issues)', notify)
    with state.connect() as db:
        for key in fresh:
            db.execute('UPDATE notifications SET state=? WHERE id=?', (delivery, key))
    return {'delivery': delivery, 'new_issues': len(fresh)}


def main(argv=None):
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--acknowledge-issue', action='append', default=[], metavar='CODE:SHA256')
    parser.add_argument('--acknowledge-reason')
    parser.add_argument('--unlabelable', action='store_true')
    parser.add_argument('--not-activated', action='store_true')
    parser.add_argument('--decision-sha256')
    args = parser.parse_args(argv)
    state = MonitorState(args.monitor_runtime)
    if args.acknowledge_issue:
        if not args.acknowledge_reason or args.force:
            parser.error('acknowledgement requires a reason and cannot force an audit')
        try:
            issues = [dict(zip(('code', 'identity'), value.split(':', 1))) for value in args.acknowledge_issue]
            with state.owner():
                acknowledge_issues(state, issues, reason=args.acknowledge_reason, at=now(), unlabelable=args.unlabelable,
                                   not_activated=args.not_activated, decision_sha256=args.decision_sha256)
        except (KeyError, TraderError):
            parser.error('invalid historical acknowledgement')
        print(json.dumps({'state': 'NOT_ACTIVATED' if args.not_activated else 'ACKNOWLEDGED',
                          'issues': len(issues), 'production_writes': 0}))
        return 0
    if args.acknowledge_reason or args.unlabelable or args.not_activated or args.decision_sha256:
        parser.error('disposition requires an exact issue identity')
    request_retries = []
    try:
        with state.owner():
            at = now()
            for prefix, unit in (('', 'hyprl-trader-rehearsal-retry.service'),
                                 ('pm-', 'hyprl-trader-rehearsal-pm-retry.service')):
                first = state.get(at.date().isoformat(), prefix + 'primary')
                if (first and first['state'] == 'BLOCKED' and first.get('retry_at')
                        and instant(first['retry_at']) <= at and not state.get(at.date().isoformat(), prefix + 'retry')):
                    request_retries.append(unit)
            result = audit(args, state, force=args.force, notify=telegram(args.telegram_env))
    except TraderError as error:
        if error.code != 'REHEARSAL_OWNER_BUSY':
            raise
        result = {'state': 'REHEARSAL_IN_PROGRESS'}
    for unit in request_retries:
        subprocess.run(['systemctl', '--user', 'start', '--no-block', unit],
                       check=True, timeout=15, capture_output=True)
    print(json.dumps(result, indent=2))
    return 1 if result['state'] == 'BLOCKED' else 0


if __name__ == '__main__':
    raise SystemExit(main())

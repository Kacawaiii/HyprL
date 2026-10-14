"""Validate a drill command's top-level result, never a nested scoring state."""
import argparse
import json
from pathlib import Path


def first_label_at(day):
    """Equities mature at today's close; a crypto-only session needs 24 hours."""
    from datetime import timedelta
    from .config import instant, iso
    from .data import calendar_session
    at = instant(day + 'T21:30:00Z')
    return iso(at if calendar_session(day) else at + timedelta(days=1))


def summarize(result, kind):
    state = result.get('status', result.get('state', result.get('run_status')))
    evidence = {}
    if kind == 'run':
        decision = result.get('decision', {})
        views = decision.get('views', [])
        evidence = {'views': len(views), 'synthetic': result.get('synthetic')}
        valid = (state in {'COMPLETE', 'DEGRADED'} and isinstance(views, list)
                 and evidence['views'] > 0 and evidence['synthetic'] is True)
    elif kind in {'label', 'matured'}:
        evidence = {k: result.get(k) for k in ('labels_added', 'pending')}
        valid = state == 'COMPLETE' and type(evidence['labels_added']) is int and evidence['labels_added'] > 0
        if kind == 'matured':
            valid = valid and evidence['pending'] == 0
    elif kind == 'paper':
        evidence = {k: result.get(k) for k in ('production_writes', 'broker_methods', 'snapshot_unchanged')}
        safe = (evidence['production_writes'] == 0 and evidence['broker_methods'] == ['GET']
                and evidence['snapshot_unchanged'] is True)
        valid = state == 'COMPLETE' and safe
        if state == 'SKIPPED_CLOSED_EQUITY_MARKET':
            from .data import calendar_session
            day = result.get('at', '')[:10]
            closed = bool(day) and calendar_session(day) is None
            evidence['calendar_closed'] = closed
            valid = safe and closed
    else:
        valid = state == 'NO_RECOVERY_NEEDED'
    return {'state': state or 'INVALID_RESULT', 'verified': valid, 'evidence': evidence}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=('run', 'label', 'matured', 'paper', 'recover'))
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    try:
        summary = summarize(json.loads(args.output.read_text()), args.kind)
    except (ValueError, TypeError, AttributeError):
        summary = {'state': 'INVALID_RESULT', 'verified': False, 'evidence': {}}
    print(summary['state'] + ' ' + json.dumps(summary['evidence'], sort_keys=True))
    return 0 if summary['verified'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

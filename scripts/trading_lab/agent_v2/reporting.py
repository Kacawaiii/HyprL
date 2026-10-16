"""Recompute accounting and layer counts from the ledger; publish only bounded safe views."""
from collections import Counter
from datetime import timedelta
from pathlib import Path
import os

from .contracts import Blocked, canonical, digest, instant, iso, utc
from .spec import PREREG_HASH, SPEC_HASH


def summary(ledger, *, clock=utc):
    tip = ledger.verify()
    state = ledger.state()
    marks = state['marks']
    last = marks[-1] if marks else None
    layers = {}
    for layer in ('core', 'news'):
        trades = [p for p in state['closed'] if p['layer'] == layer]
        wins = sum(p['pnl'] > 0 for p in trades)
        units = {(p['filled_at'][:10], p['cluster']) for p in trades}
        layers[layer] = {'closed': len(trades), 'wins': wins, 'pnl': sum(p['pnl'] for p in trades),
                         'expectancy_r': sum(p['r'] for p in trades) / len(trades) if trades else None,
                         'cluster_days': len(units), 'evidence': 'UNPROVEN'}
    codes = Counter(p['reason'] for p in state['vetoes'] if p['by'] == 'supervisor')
    last_age = (clock() - instant(last['at'])).total_seconds() if last else None
    return {'schema': 'agent-v2-summary-v1', 'mode': ledger.binding['mode'], 'asof': iso(clock()),
            'spec_hash': SPEC_HASH, 'preregistration_hash': PREREG_HASH, 'code_sha': ledger.binding['code_sha'],
            'ledger_hash': tip, 'events': len(ledger.records()), 'last_mark': last,
            'state': 'NO_QUOTES' if last is None else 'STALE' if not 0 <= last_age <= 600 else 'SHADOW',
            'positions': [{k: p[k] for k in ('symbol', 'layer', 'playbook', 'qty', 'entry', 'stop', 'initial_risk', 'cluster', 'filled_at')}
                          for p in state['positions'].values()],
            'pending': len(state['pending']), 'layers': layers,
            'supervisor_vetoes': {'count': sum(codes.values()), 'codes': dict(codes), 'counterfactual': 'UNMEASURED'},
            'risk_vetoes': dict(Counter(p['reason'] for p in state['vetoes'] if p['by'] == 'risk')),
            'last_blocked': state['blocked'][-1] if state['blocked'] else None,
            'promotion': 'OPERATOR_ONLY_AFTER_REPLAY', 'broker_orders': 0}


def reconcile(ledger, *, clock=utc):
    """Identity checks plus explicit freshness; a stale or missing close is never PASS."""
    with ledger.owned():
        state = ledger.state()
        initial = ledger.binding['config']['initial_equity']
        expected = initial
        for row in ledger.records():
            body = row['body']
            if row['kind'] == 'entry_fill':
                expected -= body['qty'] * body['entry'] + body['entry_fee']
            elif row['kind'] == 'exit_fill':
                expected += body['qty'] * body['exit'] - body['exit_fee']
        mismatch = abs(expected - state['cash']) > .000001
        view = summary(ledger, clock=clock)
        at = clock()
        from .inputs import session
        try:
            _, opened, closed = session(at)
            # At night, require a mark near the actual close, including early-close days.
            mark_at = instant(state['marks'][-1]['at']) if state['marks'] else None
            valuation_ok = mark_at is not None and (0 <= (at - mark_at).total_seconds() <= 600
                if at <= closed else closed - timedelta(minutes=10) <= mark_at <= closed)
        except Blocked:
            valuation_ok = False
        status = 'BLOCKED' if mismatch or not valuation_ok else 'PASS'
        report = {'schema': 'agent-v2-reconciliation-v1', 'at': iso(clock()), 'status': status,
                  'reason': 'CASH_MISMATCH' if mismatch else 'RECONCILED' if valuation_ok else 'CLOSE_MARK_MISSING_OR_STALE', 'cash': state['cash'],
                  'expected_cash': expected, 'ledger_hash_before_report': view['ledger_hash'],
                  'positions': len(state['positions']), 'pending': len(state['pending']), 'layers': view['layers']}
        ledger.append([('reconciliation', report)], clock())
        return report


def write_report(root, report):
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    target = root / ('reconciliation-' + report['at'][:10] + '.json')
    temporary = target.with_suffix('.tmp')
    with temporary.open('w') as stream:
        os.chmod(temporary, 0o600)
        stream.write(canonical(report) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(target)
    return digest(report)

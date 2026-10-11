"""Latched execution-only safety stop; closed lots, not forecast labels or orders."""
from decimal import Decimal as D

from scripts.trading_lab.sources.canonical import sha256_canonical
from .config import TraderError, instant
from .execution_tiers import HALF_DOWNGRADE
from .news_spec import registration_target, spec

CODE = 'PAPER_DOWNGRADE_TIER_HALTED'


def realized_trades(ledger):
    from .alpaca_paper import TERMINAL
    intents, observations, live = ledger.projection('ia_actions')
    latest = {(e['client_id'], e['lot_id']): e for e in ledger.events('ia_actions', 'trade_outcome')}
    trades = []
    for intent in intents.values():
        if intent['purpose'] != 'entry':
            continue
        entry = observations.get(intent['client_id'], {})
        if entry.get('status') not in TERMINAL:
            continue
        for lot in intent['lots']:
            if lot['variant'] != HALF_DOWNGRADE or D(live[lot['lot_id']]['qty']) != 0:
                continue
            filled = latest.get((intent['client_id'], lot['lot_id']), {})
            quantity = abs(D(filled.get('cumulative_lot_qty', '0')))
            if not quantity or filled.get('state') != 'FILLED':
                continue
            exits = [e for (_, identity), e in latest.items()
                     if identity == lot['lot_id'] and e['purpose'] != 'entry'
                     and D(e.get('cumulative_lot_qty', '0')) != 0]
            if not exits or any(e.get('broker_status') not in TERMINAL for e in exits):
                continue
            if sum((abs(D(e['cumulative_lot_qty'])) for e in exits), D(0)) != quantity:
                raise TraderError('PAPER_GUARD_QUANTITY_MISMATCH')
            gross = sum((D(e['lot_pnl_before_broker_fees']) for e in exits), D(0))
            cost = quantity * D(filled['entry_price']) * D(spec()['downgrade_breaker']['cost_roundtrip'])
            trades.append({'lot_id': lot['lot_id'], 'realized_at': max(e['at'] for e in exits),
                           'pnl_after_costs': str(gross - cost), 'modeled_cost': str(cost)})
    return sorted(trades, key=lambda t: (instant(t['realized_at']), t['lot_id']))


def evaluate(ledger, *, persist=True):
    """Registered additive contract, no automatic rebind/resume or account-level halt."""
    parent = ledger.grant.parent
    registrations = ledger.events(event='news_guard_registration')
    halts = ledger.events('ia_actions', 'tier_halt')
    if (not registrations and not halts and
            (parent.news_decision is None or ledger.clock() < instant(spec()['effective_from']))):
        return {'state': 'NOT_ACTIVE', 'realized': 0, 'positive_rate_after_costs': None}
    parent.check(ledger.clock())
    if parent.news_decision is not None:
        parent.check_news(ledger.clock())
    # A journaled safety contract survives removal of its authorization file.
    # Removing a file cannot resume a latched tier or disable its evaluation.
    target = registration_target(ledger.grant)
    if not registrations or any(registrations[-1].get(k) != v for k, v in target.items()):
        raise TraderError('PAPER_NEWS_GUARD_REGISTRATION_REQUIRED')
    trades = realized_trades(ledger)
    rule = spec()['downgrade_breaker']
    positive, crossing = 0, None
    for n, trade in enumerate(trades, 1):
        positive += D(trade['pnl_after_costs']) > 0
        if n >= rule['minimum_realized'] and D(positive) / n < D(rule['positive_rate_below']) and crossing is None:
            crossing = {'realized': n, 'positive': positive, 'trade_hash': sha256_canonical(trades[:n])}
    halted = any(h.get('tier') == HALF_DOWNGRADE for h in halts)
    if crossing and not halted and persist:
        ledger.append('tier_halt', 'ia_actions', tier=HALF_DOWNGRADE, code=CODE, **target, **crossing)
        ledger.append('alert', 'ia_actions', code=CODE, tier=HALF_DOWNGRADE, **crossing)
        # The supervisor already exposes this private alert filename.
        import json
        from .config import iso
        path = ledger.root / 'paper-alert-ia_actions.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps({'at': iso(ledger.clock()), 'code': CODE, 'tier': HALF_DOWNGRADE,
                                    'account': 'ia_actions', **crossing}))
        temp.replace(path)
    return {'state': 'HALTED' if halted or crossing else 'TOO_EARLY' if len(trades) < rule['minimum_realized'] else 'ACTIVE',
            'tier': HALF_DOWNGRADE, 'realized': len(trades),
            'positive_rate_after_costs': sum(D(t['pnl_after_costs']) > 0 for t in trades) / len(trades) if trades else None,
            'minimum_realized': rule['minimum_realized'], 'stop_below': rule['positive_rate_below'],
            'cost_basis': rule['cost_basis'], 'resume': 'OPERATOR_REQUIRED'}

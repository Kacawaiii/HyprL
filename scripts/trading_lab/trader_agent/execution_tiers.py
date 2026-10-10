"""Execution-only selection; original research views are never rewritten."""
from decimal import Decimal

from .config import TraderError

HALF_DOWNGRADE = 'half_size_downgrade_v1'
HALF_CRYPTO = 'crypto_single_kept_half_v1'
ROLES = {'analyst_claude', 'analyst_gpt'}


def selections(views, account):
    groups, consensus = {}, {}
    for view in views:
        key = (view['asset'], view['horizon'])
        if view['analyst'] in ROLES:
            groups.setdefault(key, []).append(view)
        elif view['analyst'] == 'consensus':
            if key in consensus:
                raise TraderError('PAPER_ANALYST_BINDING_MISMATCH')
            consensus[key] = view
    result = {}
    for key, pair in groups.items():
        if len(pair) > 2 or len({v['analyst'] for v in pair}) != len(pair):
            raise TraderError('PAPER_ANALYST_BINDING_MISMATCH')
        if {v['view'] for v in pair} >= {'UP', 'DOWN'}:
            continue
        directional = [v for v in pair if v['view'] in {'UP', 'DOWN'}]
        if account == 'ia_actions':
            for verdict, multiplier, tier in [('KEEP', '1', None), ('DOWNGRADE', '.5', HALF_DOWNGRADE)]:
                eligible = [v for v in directional if v['verdict'] == verdict]
                if eligible:
                    selected = min(eligible, key=lambda v: (abs(Decimal(str(v['p_outperform'])) - Decimal('.5')), v['analyst']))
                    result[key] = (selected, Decimal(multiplier), tier)
                    break
        else:
            full = consensus.get(key)
            if (len(directional) == 2 and all(v['verdict'] == 'KEEP' for v in directional)
                    and full and full['view'] == directional[0]['view'] and full['verdict'] == 'KEEP'):
                result[key] = (full, Decimal(1), None)
            elif len(directional) == 1 and directional[0]['verdict'] == 'KEEP':
                result[key] = (directional[0], Decimal('.5'), HALF_CRYPTO)
    return result


def execution_scores(ledger):
    """Cumulative outcome scores; partial exit observations are not double counted."""
    scores = {}
    for account in ('ia_actions', 'ia_crypto'):
        intents, observations, lots = ledger.projection(account)
        latest = {(e['client_id'], e['lot_id']): e for e in ledger.events(account, 'trade_outcome')}
        for intent in intents.values():
            if intent['purpose'] != 'entry':
                continue
            for lot in intent['lots']:
                population = lot.get('population', 'weekend_crypto_v1' if lot['variant'] == 'weekend_crypto_v1'
                                     else 'after_hours' if lot['variant'].startswith('alpaca_after_hours') else 'weekday')
                key = '/'.join((account, population, lot['variant'], lot['horizon']))
                score = scores.setdefault(key, {'lots': 0, 'filled': 0, 'not_traded': 0, 'pending': 0,
                                                'open_lots': 0, 'realized_lot_pnl_before_broker_fees': '0'})
                score['lots'] += 1
                entry = latest.get((intent['client_id'], lot['lot_id']), {})
                state = entry.get('state', 'PENDING')
                score['filled' if state == 'FILLED' else 'not_traded' if state == 'NOT_TRADED' else 'pending'] += 1
                score['open_lots'] += int(Decimal(lots[lot['lot_id']]['qty']) != 0)
                pnl = sum((Decimal(e.get('lot_pnl_before_broker_fees', '0')) for (_, identity), e in latest.items()
                           if identity == lot['lot_id'] and e['purpose'] != 'entry'), Decimal(0))
                score['realized_lot_pnl_before_broker_fees'] = str(Decimal(score['realized_lot_pnl_before_broker_fees']) + pnl)
    return scores

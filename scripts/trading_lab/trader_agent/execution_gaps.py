"""Actual execution gaps against immutable, causally available research labels."""
from decimal import Decimal as D

from .config import instant
from .scoring import rows


def execution_gaps(ledger, research, at):
    labels = {r['payload']['prediction_hash']: r['payload'] for r in rows(research, 'label')
              if instant(r['recorded_at']) <= at and instant(r['payload']['available_at']) <= at}
    latest = {(e['client_id'], e['lot_id']): e for e in ledger.events(event='trade_outcome')}
    result = []
    for account in ('ia_actions',):
        intents, _, _ = ledger.projection(account)
        for intent in intents.values():
            if intent['purpose'] != 'entry':
                continue
            for lot in intent['lots']:
                if lot.get('population') == 'after_hours':
                    continue
                entry = latest.get((intent['client_id'], lot['lot_id']), {})
                qty = D(entry.get('cumulative_lot_qty', '0'))
                if not qty:
                    continue
                score = {'lot_id': lot['lot_id'], 'variant': lot['variant'], 'horizon': lot['horizon'],
                         'entry_price': entry['entry_price'], 'entry_at': entry['entry_at'],
                         'state': 'PENDING_LABEL', 'entry_gap_bps': None, 'exit_gap_bps': None}
                prices = labels.get(lot['prediction_hash'], {}).get('value', {}).get('raw_prices', {}).get(lot['asset'])
                if prices and len(prices) == 2 and all(D(str(p)) > 0 for p in prices):
                    op, close = map(lambda p: D(str(p)), prices)
                    sign = D(1) if qty > 0 else D(-1)
                    score.update(state='ENTRY_SCORED', label_open=str(op), label_close=str(close),
                                 entry_gap_bps=str((D(entry['entry_price']) / op - 1) * sign * 10000))
                    exits = [e for (_, identity), e in latest.items() if identity == lot['lot_id']
                             and e['purpose'] != 'entry' and e.get('broker_fill')]
                    exited = sum((abs(D(e['cumulative_lot_qty'])) for e in exits), D(0))
                    if exited:
                        price = sum((abs(D(e['cumulative_lot_qty'])) * D(e['exit_price']) for e in exits), D(0)) / exited
                        score.update(state='CLOSED_SCORED' if exited == abs(qty) else 'PARTIAL_EXIT_SCORED',
                                     exit_price=str(price), exit_gap_bps=str((1 - price / close) * sign * 10000),
                                     signed_return_gap_bps=str((price / D(entry['entry_price']) - close / op) * sign * 10000))
                result.append(score)
    return result

"""Sizing and hard vetoes. A model can never relax these rules."""
import math
from datetime import timedelta

from .inputs import event_block


def equity(state, quotes):
    return state['cash'] + sum(p['qty'] * quotes[s]['bid'] for s, p in state['positions'].items())


def governor(state, nav, at):
    marks = state['marks']
    initial = state.get('initial_equity', 100000)
    peak = max([initial, nav] + [m['equity'] for m in marks])
    if nav <= peak * .92:
        return 0, 'PEAK_DRAWDOWN_HALT'
    monday = (at.date() - timedelta(days=at.weekday())).isoformat()
    month = at.date().replace(day=1).isoformat()
    def start(day):
        prior = [m['equity'] for m in marks if m['at'][:10] < day]
        return prior[-1] if prior else initial
    if nav <= start(month) * .94:
        return .25, 'MONTH_QUARTER_REVIEW'
    if nav <= start(monday) * .97:
        return .5, 'WEEK_HALF'
    return 1, 'NORMAL'


def exposure(state, quotes, settings):
    risks, notional = {}, 0
    for symbol, pos in state['positions'].items():
        price = quotes[symbol]['bid']
        # Never release initial stop risk merely because the position has a paper loss.
        risk = (max(0, max(price, pos['entry']) - pos['stop']) * pos['qty'] + pos['entry_fee']
                + pos['stop'] * pos['qty'] * (settings['slippage_per_side'] + settings['fee_per_side']))
        risks[pos['cluster']] = risks.get(pos['cluster'], 0) + risk
        notional += pos['qty'] * quotes[symbol]['ask']
    for p in state['pending'].values():
        risks[p['cluster']] = risks.get(p['cluster'], 0) + p['initial_risk']
        notional += p['qty'] * p['entry']
    return sum(risks.values()), risks, notional


def size(candidate, state, quotes, snapshot, news, settings, at):
    symbol, cluster = candidate['symbol'], candidate['cluster']
    if symbol in state['positions'] or any(p['symbol'] == symbol for p in state['pending'].values()):
        return None, 'ALREADY_HELD'
    if candidate['side'] != 'BUY':
        return None, 'SHORT_FORBIDDEN'
    if any(s not in quotes for s in state['positions']) or symbol not in quotes:
        return None, 'QUOTE_MISSING'
    if not cluster or cluster not in settings['clusters']:
        return None, 'CLUSTER_UNKNOWN'
    if event_block(snapshot, symbol, at, 7 if candidate['playbook'] == 'hunter_runner' else candidate['holding_days']):
        return None, 'EVENT_RISK'
    if any(e['symbol'] == symbol and e['primary'] and e['timely'] and e['sentiment'] == 'negative' for e in news):
        return None, 'NEGATIVE_PRIMARY_NEWS'
    quote = quotes[symbol]
    if (quote['ask'] - quote['bid']) / quote['ask'] > settings['max_spread_fraction']:
        return None, 'SPREAD'
    entry = quote['ask'] * (1 + settings['slippage_per_side'])
    stop = candidate['stop']
    if not 0 < stop < entry or candidate['atr'] <= 0 or entry - quote['bid'] > .3 * candidate['atr']:
        return None, 'STOP_OR_ENTRY_INVALID'
    if candidate['playbook'] == 'hunter_runner' and (entry > snapshot_close(snapshot, symbol) * 1.1
                                                   or sum(p['playbook'] == 'hunter_runner' for p in
                                                          [*state['positions'].values(), *state['pending'].values()]) >= 8):
        return None, 'HUNTER_CAP'
    if candidate['layer'] == 'news' and not 2 <= candidate['holding_days'] <= 10:
        return None, 'NEWS_HORIZON'
    nav = equity(state, quotes)
    if nav <= 0:
        return None, 'INSOLVENT'
    factor, reason = governor(state, nav, at)
    if not factor:
        return None, reason
    heat, clusters, gross = exposure(state, quotes, settings)
    unit_risk = entry - stop + entry * settings['fee_per_side'] + stop * (settings['slippage_per_side'] + settings['fee_per_side'])
    risk_budget = min(nav * candidate['risk_fraction'] * factor,
                      nav * settings['max_heat'] - heat,
                      nav * settings['max_cluster'] - clusters.get(cluster, 0))
    qty = math.floor(min(max(0, risk_budget) / unit_risk,
                         nav * settings['max_name_notional'] / entry,
                         max(0, nav * settings['max_gross'] - gross) / entry,
                         max(0, state['cash'] - sum(p['qty'] * p['entry'] * (1 + settings['fee_per_side'])
                                                    for p in state['pending'].values())) / (entry * (1 + settings['fee_per_side'])),
                         quote['ask_size']))
    if candidate['playbook'] == 'equity_momentum':
        # Book core: 30% sleeve, 0.1% of sleeve per ATR, 15% of sleeve per name.
        sleeve_used = sum(p['qty'] * p['entry'] for p in [*state['positions'].values(), *state['pending'].values()]
                          if p['playbook'] == 'equity_momentum')
        qty = min(qty, math.floor(nav * .3 * .001 * factor / candidate['atr']),
                  math.floor(nav * .3 * .15 / entry), math.floor(max(0, nav * .3 - sleeve_used) / entry))
    if qty < 1:
        return None, 'RISK_OR_CAP'
    return {**candidate, 'entry': entry, 'qty': qty, 'initial_risk': qty * unit_risk,
            'risk_budget': risk_budget, 'governor': reason}, None


def snapshot_close(snapshot, symbol):
    return next(r['bars'][-1]['c'] for r in snapshot['instruments'] if r['symbol'] == symbol)

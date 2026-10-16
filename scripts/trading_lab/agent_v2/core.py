"""Pure Clenow ranking (same formula as the book), two-layer hunter and ETF trend."""
import math
from .contracts import digest
from .inputs import fresh


def momentum_score(closes):
    y = [math.log(c) for c in closes]
    n = len(y)
    xm, ym = (n - 1) / 2, sum(y) / n
    slope = sum((i - xm) * (v - ym) for i, v in enumerate(y)) / sum((i - xm) ** 2 for i in range(n))
    total = sum((v - ym) ** 2 for v in y)
    residual = sum((v - ym - slope * (i - xm)) ** 2 for i, v in enumerate(y))
    return (math.exp(slope) ** 250 - 1) * (1 - residual / total if total > 0 else 0)


def features(bars, window=20):
    if len(bars) < max(100, window + 1):
        return None
    c = [b['c'] for b in bars]
    tr = [max(bars[i]['h'] - bars[i]['l'], abs(bars[i]['h'] - c[i-1]), abs(bars[i]['l'] - c[i-1]))
          for i in range(len(bars) - window, len(bars))]
    return {'close': c[-1], 'ma100': sum(c[-100:]) / 100, 'atr': sum(tr) / window,
            'gap': max(abs(c[i] / c[i-1] - 1) for i in range(len(c) - 90, len(c))),
            'score': momentum_score(c[-90:])}


def ranking(instruments):
    rows = []
    for symbol, r in instruments.items():
        f = features(r['bars'])
        if r['asset'] == 'equity' and r['momentum_member'] and f and f['atr'] > 0:
            rows.append({'symbol': symbol, **f, 'qualified': f['close'] > f['ma100'] and f['gap'] < .15
                         and f['score'] > 0 and f['close'] >= 5})
    rows.sort(key=lambda r: (-r['score'], r['symbol']))
    return {r['symbol']: {**r, 'rank': i, 'keep_cut': int(len(rows) * .2)} for i, r in enumerate(rows)}


def candidates(snapshot, instruments, news, at, settings):
    out = []
    ranks = ranking(instruments)
    spy = [b['c'] for b in instruments['SPY']['bars']]
    regime = spy[-1] > sum(spy[-200:]) / 200
    def add(symbol, playbook, stop, atr, hold, risk):
        out.append({'symbol': symbol, 'layer': 'news' if playbook == 'primary_news' else 'core',
                    'playbook': playbook, 'stop': stop, 'atr': atr, 'holding_days': hold, 'risk_fraction': risk,
                    'cluster': instruments[symbol]['cluster'], 'side': 'BUY'})
    for symbol, row in ranks.items():
        if regime and row['qualified'] and row['rank'] <= row['keep_cut']:
            add(symbol, 'equity_momentum', row['close'] - 3 * row['atr'], row['atr'], 90, .003)
    hunter = snapshot['hunter']
    if (hunter['asof'] == instruments['SPY']['bars'][-1]['t'][:10]
            and fresh(hunter['generated'], at, 4 * 86400)):
        for row in hunter['candidates']:
            symbol = row.get('symbol')
            if symbol not in instruments or instruments[symbol]['asset'] != 'equity':
                continue
            if (row.get('two_layer') is True and isinstance(row.get('rev_accel_pts'), (float, int))
                    and row['rev_accel_pts'] >= 10 and any(code in str(row.get('setup', '')).replace('+', ' ').split() for code in ('S2', 'S3', 'S4'))
                    and isinstance(row.get('signal_close'), (float, int)) and row['signal_close'] > 0):
                f = features(instruments[symbol]['bars'])
                if f and row['signal_close'] * .85 < f['close'] <= row['signal_close'] * 1.1:
                    add(symbol, 'hunter_runner', f['close'] * .8, f['atr'], 252, .003)
    for symbol in settings['etf_universe']:
        row = instruments.get(symbol)
        if not row or row['asset'] != 'etf':
            continue
        f = features(row['bars'], 100)
        closes = [b['c'] for b in row['bars']]
        if f and sum(closes[-50:]) / 50 > f['ma100'] and closes[-1] >= max(closes[-50:]):
            add(symbol, 'etf_trend', f['close'] - 3 * f['atr'], f['atr'], 252, .0025)
    for event in sorted(news, key=lambda e: (e['sentiment'] != 'negative', e['symbol'])):
        symbol = event['symbol']
        if (event['primary'] and event['timely'] and event['liquid_small_mid']
                and event['code'] == 'beat_and_raise'):
            f = features(instruments[symbol]['bars'])
            if f:
                add(symbol, 'primary_news', f['close'] - 3 * f['atr'], f['atr'], 5, .0025)
    # One immutable proposal per symbol; core has priority, no stacking layers.
    unique = {}
    for row in out:
        unique.setdefault(row['symbol'], row)
    return list(unique.values()), ranks


def plan_id(plan):
    return digest({k: v for k, v in plan.items() if k != 'plan_id'})

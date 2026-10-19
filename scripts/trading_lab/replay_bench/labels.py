"""Outcomes after the fact: designed fill times, SPY-relative returns, tiered costs, R units."""
from datetime import timedelta
import math
import random
import statistics

from .grant import ReplayError, instant, iso
from . import marketdata as md, snapshot as sn

LAST_SESSION = '2026-10-16'


def tier(dollar_volume, costs, crypto=False):
    table = costs['spread_slippage_per_side']
    if crypto:
        return table['crypto']
    if dollar_volume > 1e9:
        return table['dollar_volume_gt_1B']
    if dollar_volume > 1e8:
        return table['100M_to_1B']
    return table['10M_to_100M']


def daily_stats(rows, T):
    """vol_20d and 20-session median dollar volume from bars completed before T (causal inputs to sizing/costs)."""
    done = sorted((r for r in rows if r['close_at'] <= T), key=lambda r: r['bar_open_at'])[-21:]
    if len(done) < 21:
        return None
    closes = [float(r['close']) for r in done]
    daily = [closes[i] / closes[i - 1] - 1 for i in range(1, 21)]
    return {'vol_20d': statistics.stdev(daily), 'r20': closes[-1] / closes[0] - 1,
            'dollar_volume': statistics.median(float(r['close']) * float(r['volume']) for r in done[1:])}


class MinuteAnchors:
    """1-minute bars at the designed fill times, fetched one request per (day, anchor)."""

    def __init__(self, transport):
        self.transport, self.cache = transport, {}

    def stock(self, day, anchor, symbols):
        key = ('s', day, anchor, tuple(sorted(symbols)))
        if key not in self.cache:
            start = instant(f'{day}T{anchor}Z')
            self.cache[key] = md.stock_trades(self.transport, symbols, start)
        return self.cache[key]

    def crypto(self, at):
        """Exact minute open. Missing or stale bars are never invented into a fill."""
        key = ('c', iso(at))
        if key not in self.cache:
            bars, _ = md.crypto_bars(self.transport, list(sn.CRYPTO), iso(at), iso(at + timedelta(seconds=1)),
                                     timeframe='1Min')
            self.cache[key] = {a: next((b for b in v if instant(b['t']) == at), None)
                               for a, v in bars.items()}
        return self.cache[key]


def outcome(signed, rel, cost, vol, h):
    net = signed * rel - cost
    return {'rel': rel, 'net': net, 'gross': signed * rel, 'cost': cost, 'hit': net > 0, 'hit_gross': signed * rel > 0,
            'outperform': rel > 0, 'R': net / (vol * math.sqrt(h)) if vol else None}


def label_universe_day(day, universe_bars, anchors, costs, *, last_day=LAST_SESSION, decision_at=None,
                       crypto_entry=None):
    """1d and 5d outcomes for every universe asset on one day (views and baselines share them).

    Stock entries are anchored to the actual session open (+30 s), never to a fixed UTC clock, so a
    daylight-saving change cannot move a fill before the decision. A fill at or before the decision
    is refused."""
    T = decision_at or instant(day + 'T12:00:00Z')
    session = md.sessions(day, day)[0]
    fee = costs['fee_per_side']
    rows = {a: (md.crypto_rows if a in sn.CRYPTO else md.session_rows)(universe_bars.get(a, []))
            for a in sn.UNIVERSE + sn.BENCHMARKS}
    entry_at = session.open_at + timedelta(seconds=30)
    crypto_start = crypto_entry or T + timedelta(minutes=15)
    if entry_at <= T or crypto_start <= T:
        raise ReplayError('ENTRY_NOT_AFTER_DECISION', day)
    entry_anchor = entry_at.strftime('%H:%M:%S')
    exit_at = session.close_at - timedelta(minutes=5)
    exit_anchor = exit_at.strftime('%H:%M:%S')
    symbols = list(sn.STOCKS + sn.ETFS + ('SPY',))
    entry_bars, exit_bars = anchors.stock(day, entry_anchor, symbols), anchors.stock(day, exit_anchor, symbols)
    out = {}

    def stock_return(symbol):
        a, b = entry_bars.get(symbol), exit_bars.get(symbol)
        return (b['p'] / a['p'] - 1) if a and b else None

    spy_1d = stock_return('SPY')
    following = md.sessions(day, (instant(day + 'T00:00:00Z') + timedelta(days=12)).date().isoformat())
    spy_stats = daily_stats(rows['SPY'], T)
    for asset in sn.UNIVERSE:
        stats = daily_stats(rows[asset], T)
        if stats is None:
            continue
        crypto = asset in sn.CRYPTO
        cost = 2 * (fee + tier(stats['dollar_volume'], costs, crypto))
        cell = {'stats': stats, 'cost': cost, 'cluster': sn.CLUSTERS[asset],
                'momentum_up': stats['r20'] > (0 if crypto else spy_stats['r20'])}
        if crypto:
            start = crypto_start
            if (start + timedelta(days=1)).date().isoformat() <= last_day:
                a, b = anchors.crypto(start).get(asset), anchors.crypto(start + timedelta(days=1)).get(asset)
                if a and b:
                    cell['1d'] = {'rel': b['o'] / a['o'] - 1, 'abs': b['o'] / a['o'] - 1}
        else:
            r = stock_return(asset)
            if r is not None and spy_1d is not None:
                cell['1d'] = {'rel': r - spy_1d, 'abs': r}
            if len(following) >= 5 and following[4].session_date <= last_day:
                by_day = {x['bar_open_at'].date().isoformat(): x for x in rows[asset]}
                spy_day = {x['bar_open_at'].date().isoformat(): x for x in rows['SPY']}
                first, last = day, following[4].session_date
                if all(k in d for d in (by_day, spy_day) for k in (first, last)):
                    if entry_bars.get(asset) and entry_bars.get('SPY'):
                        ra = by_day[last]['close'] / entry_bars[asset]['p'] - 1
                        rs = spy_day[last]['close'] / entry_bars['SPY']['p'] - 1
                        cell['5d'] = {'rel': ra - rs, 'abs': ra}
        out[asset] = cell
    out['_spy'] = {'1d': spy_1d}
    return out


def score_view(view, cell, horizon):
    if view['view'] == 'ABSTAIN' or cell is None or horizon not in cell:
        return None
    h = 1 if horizon == '1d' else 5
    signed = 1 if view['view'] == 'UP' else -1
    result = outcome(signed, cell[horizon]['rel'], cell['cost'], cell['stats']['vol_20d'], h)
    result.update(p=view['p_outperform'], cluster=cell['cluster'], direction=view['view'])
    return result


def baselines_universe(day, cells, horizon, seed):
    rng = random.Random(f'{seed}-{day}-{horizon}')
    out = {'always_up': [], 'random': [], 'momentum20': []}
    for asset in sorted(a for a in cells if not a.startswith('_')):
        cell = cells[asset]
        if horizon not in cell:
            continue
        h = 1 if horizon == '1d' else 5
        choices = {'always_up': 1, 'random': rng.choice((1, -1)), 'momentum20': 1 if cell['momentum_up'] else -1}
        for name, signed in choices.items():
            result = outcome(signed, cell[horizon]['rel'], cell['cost'], cell['stats']['vol_20d'], h)
            result.update(p=None, cluster=cell['cluster'], direction='UP' if signed > 0 else 'DOWN')
            out[name].append(result)
    return out


def label_news_view(view, day, bars, costs, spy_rows, *, horizon=None, signed=None,
                    decision_at=None, entry_prices=None, last_day=LAST_SESSION):
    """c/d: enter at the decision session's open, exit at the close of session h; SPY over the same sessions."""
    h = horizon or view['holding_sessions']
    T = decision_at or instant(day + 'T12:00:00Z')
    rows = md.session_rows(bars.get(view['symbol'], []))
    stats = daily_stats(rows, T)
    following = md.sessions(day, (instant(day + 'T00:00:00Z') + timedelta(days=21)).date().isoformat())
    if stats is None or len(following) < h:
        return {'state': 'UNLABELED'}
    last = following[h - 1].session_date
    if last > last_day:
        return {'state': 'PENDING', 'exit_session': last}
    by_day = {r['bar_open_at'].date().isoformat(): r for r in rows}
    spy = {r['bar_open_at'].date().isoformat(): r for r in spy_rows}
    if not all(k in d for d in (by_day, spy) for k in (day, last)):
        return {'state': 'UNLABELED'}
    entry, spy_entry = ((entry_prices[view['symbol']], entry_prices['SPY']) if entry_prices else
                        (by_day[day]['open'], spy[day]['open']))
    ra = by_day[last]['close'] / entry - 1
    rs = spy[last]['close'] / spy_entry - 1
    if signed is None:
        signed = 1 if view['view'] == 'UP' else -1
    cost = 2 * (costs['fee_per_side'] + tier(stats['dollar_volume'], costs))
    result = outcome(signed, ra - rs, cost, stats['vol_20d'], h)
    result.update(state='LABELED', abs=ra, h=h, momentum_up=stats['r20'] > daily_stats(spy_rows, T)['r20'])
    return result


def marked_path(day, asset, rows, spy_rows, horizon, signed, cost, *, entry_prices=None):
    """Daily changes of asset minus SPY, on fixed entry notional, with entry/exit costs.

    Returns (exit session, daily P&L). No outcome at the next close is booked on the decision day.
    A missing interior mark invalidates the whole path.
    """
    following = md.sessions(day, (instant(day + 'T00:00:00Z') + timedelta(days=21)).date().isoformat())[:horizon]
    if len(following) != horizon:
        raise ReplayError('INCOMPLETE_MARK_PATH')
    prices = {r['bar_open_at'].date().isoformat(): r for r in rows}
    spy = {r['bar_open_at'].date().isoformat(): r for r in spy_rows}
    days = [s.session_date for s in following]
    if any(d not in table for d in days for table in (prices, spy)):
        raise ReplayError('INCOMPLETE_MARK_PATH', asset)
    entry, spy_entry = ((entry_prices[asset], entry_prices['SPY']) if entry_prices else
                        (prices[day]['open'], spy[day]['open']))
    previous = 0.0
    path = {}
    for index, d in enumerate(days):
        value = signed * (prices[d]['close'] / entry - spy[d]['close'] / spy_entry)
        path[d] = value - previous - (cost / 2 if index == 0 else 0) - (cost / 2 if index == horizon - 1 else 0)
        previous = value
    return days[-1], path


def crypto_path(day, asset, anchors, signed, cost, *, decision_at=None):
    start = (decision_at or instant(day + 'T12:00:00Z')) + timedelta(minutes=15)
    middle = (start + timedelta(days=1)).replace(hour=0, minute=0)
    end = start + timedelta(days=1)
    a, m, b = (anchors.crypto(at).get(asset) for at in (start, middle, end))
    if not all((a, m, b)):
        raise ReplayError('MISSING_CRYPTO_MARK')
    return {day: signed * (m['o'] - a['o']) / a['o'] - cost / 2,
            end.date().isoformat(): signed * (b['o'] - m['o']) / a['o'] - cost / 2}

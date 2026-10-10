"""Daily long/cash portfolio engine for Part A.

Timing: targets are computed from closes <= t (rebalance day t) and traded at the
close of t+1. A position therefore earns the return of day d only if it was
traded at the close of d-1 or earlier. Costs are charged on traded notional at
the trade close.
"""

import numpy as np
import pandas as pd

from .protocol import DAYS_YEAR, REGIMES, STABLE_BASES, assert_cutoff

SIDE_COST = 0.0025 + 0.0005  # fee + slippage, per unit of notional traded
TREND_DAYS = 200
VOL_DAYS = 30
LIQ_DAYS = 90


def rebalance_days(index, freq):
    """Signal days: last calendar day of each month / each ISO week present in the index."""
    idx = pd.DatetimeIndex(index)
    key = idx.tz_localize(None).to_period("M" if freq == "monthly" else "W")
    return idx[pd.Series(key, index=idx).ne(pd.Series(key, index=idx).shift(-1)).to_numpy()][:-1]


def simulate(close, targets, side_cost=SIDE_COST):
    """close: DataFrame of daily closes. targets: {signal_day: Series of weights}.

    Returns (daily net returns, weights held after each day's trade, daily turnover).
    A missing close after an asset's last candle is carried at its last price.
    """
    assert_cutoff(close.index)
    close = close.ffill()  # delisted: last price; leading NaN stay NaN (never held)
    rets = close.pct_change().fillna(0.0).to_numpy()
    cols = list(close.columns)
    days = close.index
    pending = {}
    for sig_day, w in targets.items():
        i = days.get_loc(sig_day)
        if i + 1 < len(days):
            pending[i + 1] = w.reindex(cols).fillna(0.0).to_numpy()
    w = np.zeros(len(cols))
    net, held, turn = np.zeros(len(days)), np.zeros((len(days), len(cols))), np.zeros(len(days))
    for i in range(len(days)):
        gross = float(w @ rets[i])
        w = w * (1 + rets[i]) / (1 + gross)  # drift (cash drifts to the remainder)
        cost = 0.0
        if i in pending:
            tgt = pending[i]
            traded = np.abs(tgt - w).sum()
            cost = traded * side_cost
            turn[i] = traded
            w = tgt
        net[i] = (1 + gross) * (1 - cost) - 1
        held[i] = w
    return (pd.Series(net, days), pd.DataFrame(held, days, cols), pd.Series(turn, days))


def _inv_vol(close, day, assets):
    vol = close[assets].pct_change().loc[:day].tail(VOL_DAYS).std()
    inv = 1.0 / vol.replace(0, np.nan)
    inv = inv.dropna()
    return inv / inv.sum() if len(inv) else inv


def trend_ok(close, day, assets, ma=TREND_DAYS):
    hist = close[assets].loc[:day]
    ok = hist.count() >= ma
    return ok & (hist.iloc[-1] > hist.tail(ma).mean())


def available(close, day, assets, min_hist):
    hist = close[assets].loc[:day]
    return [a for a in assets if hist[a].count() >= min_hist and not np.isnan(hist[a].iloc[-1])]


def make_targets(close, assets, freq="monthly", weighting="inv_vol", ma=None):
    """ma=None: no trend filter. Filtered-out assets go to cash (weights not renormalised)."""
    targets = {}
    for day in rebalance_days(close.index, freq):
        pool = available(close, day, assets, max(VOL_DAYS + 1, ma or 1))
        if not pool:
            targets[day] = pd.Series(dtype=float)
            continue
        w = pd.Series(1.0 / len(pool), pool) if weighting == "equal" else _inv_vol(close, day, pool)
        if ma:
            w = w[trend_ok(close, day, list(w.index), ma).reindex(w.index).fillna(False)]
        targets[day] = w
    return targets


def buy_hold_targets(close, asset):
    first = close[asset].first_valid_index()
    return {first: pd.Series({asset: 1.0})}


def is_stable(product):
    return product.split("-")[0].upper() in STABLE_BASES


def universe_top(close, volume, day, n=4, min_hist=TREND_DAYS):
    """Top-n non-stable products by trailing 90d dollar volume, using bars <= day only.

    An asset needs `min_hist` candles of history and a price on `day`.
    """
    c, v = close.loc[:day], volume.loc[:day]
    ok = [a for a in c.columns if not is_stable(a) and c[a].count() >= min_hist and not np.isnan(c[a].iloc[-1])]
    dollar = (c[ok] * v[ok]).tail(LIQ_DAYS).sum()
    return list(dollar.sort_values(ascending=False, kind="stable").head(n).index)


def universe_targets(close, volume, freq="monthly", n=4, ma=TREND_DAYS, weighting="inv_vol"):
    targets = {}
    for day in rebalance_days(close.index, freq):
        top = universe_top(close, volume, day, n)
        if not top:
            targets[day] = pd.Series(dtype=float)
            continue
        w = pd.Series(1.0 / len(top), top) if weighting == "equal" else _inv_vol(close, day, top)
        if ma:
            w = w[trend_ok(close, day, list(w.index), ma).reindex(w.index).fillna(False)]
        targets[day] = w
    return targets


def _cagr(r):
    years = len(r) / DAYS_YEAR
    return (1 + r).prod() ** (1 / years) - 1 if years > 0 and (1 + r).prod() > 0 else -1.0


def metrics(r, held=None, turn=None):
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    dd = eq / eq.cummax() - 1
    vol = r.std() * np.sqrt(DAYS_YEAR)
    down = np.sqrt((np.minimum(r, 0) ** 2).mean()) * np.sqrt(DAYS_YEAR)
    cagr = _cagr(r)
    years = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
    out = {
        "CAGR": cagr, "vol": vol,
        "Sharpe": r.mean() * DAYS_YEAR / vol if vol > 0 else np.nan,
        "Sortino": r.mean() * DAYS_YEAR / down if down > 0 else np.nan,
        "maxDD": dd.min(), "Calmar": cagr / -dd.min() if dd.min() < 0 else np.nan,
        "worst_year": years.min(),
    }
    if held is not None:
        out["time_invested"] = (held.sum(axis=1).loc[r.index] > 1e-9).mean()
    if turn is not None:
        out["turnover_per_year"] = turn.loc[r.index].sum() / (len(r) / DAYS_YEAR)  # traded notional / equity
    return out


def regime_table(r):
    rows = {}
    for name, start, end in REGIMES:
        seg = r.loc[start:end]
        rows[name] = {k: v for k, v in metrics(seg).items() if k in ("CAGR", "Sharpe", "maxDD")} if len(seg) > 30 else {}
    return pd.DataFrame(rows).T

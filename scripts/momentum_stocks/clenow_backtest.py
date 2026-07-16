#!/usr/bin/env python3
"""Clenow 'Stocks on the Move' momentum backtest — honest edition.

Rules (from the book, ch. 7-11):
  * Universe: S&P 500 constituents (current set — survivorship caveat).
  * Trade only on Wednesdays (all calcs on daily data).
  * Momentum score = annualized exp-regression slope over 90d * R^2.
  * Disqualify if below 100d MA, or a >15% single-day gap in last 90d.
  * Position size (risk parity): shares = equity * risk_factor / ATR20,
    risk_factor = 0.001 (10 bps daily impact target).
  * Regime filter: only OPEN new positions if index (SPY) > its 200d MA.
  * Sell weekly if a held name drops out of the top `hold_pct` of the ranking,
    or falls below its 100d MA.
  * Rebalance every 2nd Wednesday back to target risk sizes.

Costs: per-side bps applied on every share traded.
Outputs: metrics + equity curve for the portfolio and SPY buy&hold.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kyo/HyprL")
PRICES = ROOT / "data/momentum/prices"

# ---- parameters (all canonical Clenow values; do not tune casually) ----
REG_WINDOW = 90          # regression lookback (trading days)
MA_TREND = 100           # per-stock trend filter
MA_INDEX = 200           # index regime filter
ATR_WINDOW = 20
GAP_MAX = 0.15           # disqualify >15% move in last 90d
RISK_FACTOR = 0.001      # 10 bps target daily impact
HOLD_PCT = 0.20          # keep while in top 20% of ranking
INITIAL_EQUITY = 100_000.0
COST_BPS = 5.0           # per-side cost in basis points (commission+slippage)
REBALANCE_EVERY = 2      # rebalance sizes every 2nd trading Wednesday
MIN_PRICE = 5.0          # skip sub-$5 names (penny/post-bankruptcy junk)
MAX_WEIGHT = 0.15        # cap any single position at 15% of equity


MAX_DAILY_RET = 0.60   # drop tickers with any |daily return| > this = data artifact


def load_prices(quality_filter: bool = True) -> dict[str, pd.DataFrame]:
    data = {}
    dropped = 0
    for f in PRICES.glob("*.parquet"):
        sym = f.stem
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        if sym in ("SPY", "_GSPC"):
            data[sym] = df
            continue
        if quality_filter:
            c = df["Close"].astype(float)
            c = c[c > 0]
            if len(c) < 60:
                dropped += 1
                continue
            r = c.pct_change().dropna()
            if r.abs().max() > MAX_DAILY_RET or (c.max() / c.median()) > 200:
                dropped += 1
                continue
        data[sym] = df
    if quality_filter and dropped:
        print(f"[data-quality: dropped {dropped} corrupted tickers, kept {len(data)}]")
    return data


def load_membership() -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """price-key -> list of (start, end) intervals of S&P 500 membership."""
    import csv
    m: dict[str, list] = {}
    path = ROOT / "data/momentum/sp500_membership.csv"
    if not path.exists():
        return m
    for r in csv.DictReader(open(path)):
        key = r["ticker"].replace(".", "-")
        start = pd.Timestamp(r["start_date"]) if r["start_date"] else pd.Timestamp("1990-01-01")
        end = pd.Timestamp(r["end_date"]) if r["end_date"] else pd.Timestamp("2100-01-01")
        m.setdefault(key, []).append((start, end))
    return m


def is_member(membership, sym, date) -> bool:
    ivs = membership.get(sym)
    if ivs is None:
        return False
    return any(s <= date <= e for s, e in ivs)


def momentum_score(closes: np.ndarray) -> tuple[float, float]:
    """Annualized exponential-regression slope * R^2 over the window."""
    y = np.log(closes)
    x = np.arange(len(y))
    # linear fit on log prices
    slope, intercept = np.polyfit(x, y, 1)
    # R^2
    yhat = slope * x + intercept
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    annualized = (math.exp(slope) ** 250) - 1
    return annualized * r2, annualized


def atr(df: pd.DataFrame, n: int) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


@dataclass
class Position:
    shares: float
    entry: float


@dataclass
class Portfolio:
    cash: float = INITIAL_EQUITY
    pos: dict[str, Position] = field(default_factory=dict)


def rolling_momentum(close: pd.Series, window: int) -> pd.Series:
    """Vectorized annualized exp-regression slope * R^2 for every bar.

    Equivalent to fitting np.polyfit on log(price) over each trailing `window`,
    annualizing exp(slope)^250 - 1, and multiplying by R^2 — but done in one
    sliding-window matrix op instead of a per-bar loop.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    px = close.values.astype(float)
    y = np.where(px > 0, np.log(px), np.nan)  # guard non-positive prices
    n = window
    if len(y) < n:
        return pd.Series(np.nan, index=close.index)
    W = sliding_window_view(y, n)              # (T-n+1, n)
    valid = ~np.isnan(W).any(axis=1)           # drop windows containing NaN/≤0
    x = np.arange(n)
    xc = x - x.mean()
    sxx = (xc ** 2).sum()
    ybar = W.mean(axis=1)
    slope = (W * xc).sum(axis=1) / sxx         # since sum(xc)=0
    yhat = slope[:, None] * xc + ybar[:, None]
    ss_res = ((W - yhat) ** 2).sum(axis=1)
    ss_tot = ((W - ybar[:, None]) ** 2).sum(axis=1)
    r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
    ann = np.exp(slope) ** 250 - 1
    score = ann * r2
    score = np.where(valid, score, np.nan)
    out = np.full(len(y), np.nan)
    out[n - 1:] = score
    return pd.Series(out, index=close.index)


def precompute(data: dict[str, pd.DataFrame], reg_window: int = REG_WINDOW):
    """Per-symbol rolling series needed on decision days (vectorized score)."""
    feats = {}
    for sym, df in data.items():
        c = df["Close"]
        if len(c) < reg_window + 5:
            continue
        ma100 = c.rolling(MA_TREND).mean()
        atr20 = atr(df, ATR_WINDOW)
        ret1 = c.pct_change().abs()
        max_gap = ret1.rolling(reg_window).max()
        score = rolling_momentum(c, reg_window)
        feats[sym] = dict(close=c, ma100=ma100, atr20=atr20,
                          max_gap=max_gap, score=score)
    return feats


def run(data=None, feats=None, index_close=None, index_ma200=None,
        win_start=None, win_end=None, verbose=True, membership=None):
    if data is None:
        data = load_prices()
    idx_key = "_GSPC" if "_GSPC" in data else "SPY"
    if index_close is None:
        index_close = data[idx_key]["Close"]
        index_ma200 = index_close.rolling(MA_INDEX).mean()

    stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
    if feats is None:
        feats = precompute(stocks)

    # master trading calendar = index calendar
    cal = index_close.index
    # warmup
    start_i = max(REG_WINDOW, MA_TREND, MA_INDEX) + 5
    if win_start is not None:
        ws = pd.Timestamp(win_start)
        cand = cal.searchsorted(ws)
        start_i = max(start_i, cand)
    win_end_ts = pd.Timestamp(win_end) if win_end else None

    pf = Portfolio()
    equity_curve = []
    reb_counter = 0
    n_trades = 0
    cost_paid = 0.0

    def price_on(sym, date):
        s = feats.get(sym)
        if s is None:
            return None
        c = s["close"]
        if date in c.index and not math.isnan(c.loc[date]):
            return float(c.loc[date])
        return None

    def mtm(date):
        val = pf.cash
        for sym, p in pf.pos.items():
            px = price_on(sym, date)
            if px:
                val += p.shares * px
        return val

    for i in range(start_i, len(cal)):
        date = cal[i]
        if win_end_ts is not None and date > win_end_ts:
            break
        # daily equity mark
        eq = mtm(date)
        equity_curve.append((date, eq))

        if date.weekday() != 2:  # only trade Wednesdays (0=Mon..2=Wed)
            continue

        # --- build ranking on this date ---
        ranked = []
        for sym, s in feats.items():
            if membership is not None and not is_member(membership, sym, date):
                continue
            c = s["close"]
            if date not in c.index:
                continue
            loc = c.index.get_loc(date)
            if loc < REG_WINDOW:
                continue
            score = s["score"].iloc[loc]
            if math.isnan(score):
                continue
            px = float(c.iloc[loc])
            ma100 = s["ma100"].iloc[loc]
            gap = s["max_gap"].iloc[loc]
            a20 = s["atr20"].iloc[loc]
            if math.isnan(ma100) or math.isnan(a20) or a20 <= 0:
                continue
            qualified = (px > ma100) and (gap < GAP_MAX) and (score > 0) \
                and (px >= MIN_PRICE)
            ranked.append((sym, score, px, a20, qualified))
        if not ranked:
            continue
        ranked.sort(key=lambda r: r[1], reverse=True)
        rank_of = {r[0]: k for k, r in enumerate(ranked)}
        n_uni = len(ranked)
        keep_cut = int(n_uni * HOLD_PCT)

        index_ok = (not math.isnan(index_ma200.loc[date])) and \
                   (float(index_close.loc[date]) > float(index_ma200.loc[date]))

        # --- SELL: names that fell out of top pct or below MA100 ---
        for sym in list(pf.pos.keys()):
            r = rank_of.get(sym, 10**9)
            below_ma = True
            s = feats.get(sym)
            if s is not None and date in s["close"].index:
                loc = s["close"].index.get_loc(date)
                below_ma = float(s["close"].iloc[loc]) < s["ma100"].iloc[loc]
            if r > keep_cut or below_ma:
                px = price_on(sym, date)
                if px:
                    proceeds = pf.pos[sym].shares * px
                    c = proceeds * COST_BPS / 1e4
                    pf.cash += proceeds - c
                    cost_paid += c
                    n_trades += 1
                del pf.pos[sym]

        reb_counter += 1
        do_reb = (reb_counter % REBALANCE_EVERY == 0)

        # --- REBALANCE existing to target size ---
        if do_reb:
            for sym, p in list(pf.pos.items()):
                info = next((x for x in ranked if x[0] == sym), None)
                if not info:
                    continue
                _, _, px, a20, _ = info
                target_sh = math.floor(eq * RISK_FACTOR / a20)
                cap_sh = math.floor(MAX_WEIGHT * eq / px)
                target_sh = min(target_sh, cap_sh)
                delta = target_sh - p.shares
                if delta == 0:
                    continue
                trade_val = abs(delta) * px
                c = trade_val * COST_BPS / 1e4
                if delta > 0 and trade_val + c > pf.cash:
                    continue  # not enough cash to add
                pf.cash -= delta * px + c
                cost_paid += c
                n_trades += 1
                p.shares = target_sh

        # --- BUY new names from top, if regime is risk-on ---
        if index_ok:
            for sym, score, px, a20, qualified in ranked:
                if sym in pf.pos or not qualified:
                    continue
                if rank_of[sym] > keep_cut:
                    break  # only buy from the eligible top band
                target_sh = math.floor(eq * RISK_FACTOR / a20)
                cap_sh = math.floor(MAX_WEIGHT * eq / px)
                target_sh = min(target_sh, cap_sh)
                if target_sh <= 0:
                    continue
                cost_val = target_sh * px
                c = cost_val * COST_BPS / 1e4
                if cost_val + c > pf.cash:
                    continue  # skip if can't afford; keep scanning
                pf.cash -= cost_val + c
                cost_paid += c
                n_trades += 1
                pf.pos[sym] = Position(target_sh, px)

    # ---- metrics ----
    eqs = pd.Series({d: v for d, v in equity_curve})
    if len(eqs) < 30:
        return {"error": "too few bars", "bars": len(eqs)}
    rets = eqs.pct_change().dropna()
    ann = (eqs.iloc[-1] / eqs.iloc[0]) ** (252 / len(eqs)) - 1
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std()
    sortino = rets.mean() / downside * math.sqrt(252) if downside and downside > 0 else 0
    peak = eqs.cummax()
    dd = (eqs - peak) / peak
    maxdd = -dd.min()
    calmar = ann / maxdd if maxdd > 0 else 0

    # SPY buy & hold on same window
    spy = data["SPY"]["Close"].reindex(eqs.index).ffill()
    spy_ret = spy.pct_change().dropna()
    spy_ann = (spy.iloc[-1] / spy.iloc[0]) ** (252 / len(spy)) - 1
    spy_sharpe = spy_ret.mean() / spy_ret.std() * math.sqrt(252) if spy_ret.std() > 0 else 0
    spy_peak = spy.cummax()
    spy_maxdd = -((spy - spy_peak) / spy_peak).min()

    out = {
        "period": [str(eqs.index[0].date()), str(eqs.index[-1].date())],
        "bars": len(eqs),
        "strategy": {
            "final": round(eqs.iloc[-1], 0),
            "total_return_pct": round((eqs.iloc[-1] / eqs.iloc[0] - 1) * 100, 2),
            "annualized_pct": round(ann * 100, 2),
            "sharpe": round(sharpe, 2),
            "sortino": round(sortino, 2),
            "max_dd_pct": round(maxdd * 100, 2),
            "calmar": round(calmar, 2),
            "trades": n_trades,
            "cost_paid": round(cost_paid, 0),
        },
        "spy_buyhold": {
            "total_return_pct": round((spy.iloc[-1] / spy.iloc[0] - 1) * 100, 2),
            "annualized_pct": round(spy_ann * 100, 2),
            "sharpe": round(spy_sharpe, 2),
            "max_dd_pct": round(spy_maxdd * 100, 2),
        },
    }
    if verbose:
        print(json.dumps(out, indent=2))
        eqs.to_frame("equity").to_csv(ROOT / "data/momentum/equity_curve.csv")
        json.dump(out, open(ROOT / "data/momentum/backtest_result.json", "w"), indent=2)
    out["_eqs"] = eqs
    return out


def subperiods(pit=False):
    """Run full period + regime sub-periods reusing loaded data."""
    data = load_prices()
    idx_key = "_GSPC" if "_GSPC" in data else "SPY"
    index_close = data[idx_key]["Close"]
    index_ma200 = index_close.rolling(MA_INDEX).mean()
    stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
    feats = precompute(stocks)
    membership = load_membership() if pit else None
    if pit:
        print(f"[point-in-time membership: {len(membership)} tickers, "
              f"{len(feats)} with price data]")

    windows = [
        ("FULL 2007-2026", None, None),
        ("2008 crisis 07-09", "2007-01-01", "2009-12-31"),
        ("recovery 10-14", "2010-01-01", "2014-12-31"),
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("covid 19-21", "2019-01-01", "2021-12-31"),
        ("recent 22-26", "2022-01-01", "2026-07-11"),
    ]
    rows = []
    for name, ws, we in windows:
        r = run(data=data, feats=feats, index_close=index_close,
                index_ma200=index_ma200, win_start=ws, win_end=we, verbose=False,
                membership=membership)
        s = r.get("strategy", {})
        b = r.get("spy_buyhold", {})
        rows.append((name, s.get("annualized_pct"), s.get("sharpe"),
                     s.get("max_dd_pct"), s.get("total_return_pct"),
                     b.get("sharpe"), b.get("max_dd_pct")))
    print(f"\n{'Period':<20}{'AnnRet%':>9}{'Sharpe':>8}{'MaxDD%':>8}{'TotRet%':>9}"
          f"{'|SPYshp':>8}{'SPYdd%':>8}")
    print("-" * 78)
    shps = []
    for name, ann, shp, dd, tot, sshp, sdd in rows:
        shps.append(shp if name != "FULL 2007-2026" and shp is not None else None)
        print(f"{name:<20}{ann:>9}{shp:>8}{dd:>8}{tot:>9}{sshp:>8}{sdd:>8}")
    sub = [s for s in shps if s is not None]
    if sub:
        mean = sum(sub) / len(sub)
        std = (sum((x - mean) ** 2 for x in sub) / len(sub)) ** 0.5
        print("-" * 78)
        print(f"sub-period Sharpe: mean={mean:.2f} std={std:.2f} "
              f"CoV={std/mean*100:.0f}% min={min(sub):.2f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "sub":
        subperiods(pit=False)
    elif len(sys.argv) > 1 and sys.argv[1] == "pit":
        subperiods(pit=True)
    else:
        run()

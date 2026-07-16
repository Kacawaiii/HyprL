#!/usr/bin/env python3
"""Connors 2-period RSI mean-reversion — stock portfolio version.

Book: Connors & Alvarez, *Short Term Trading Strategies That Work*, ch. 9.
Rules (adapted from the SPY strategy to a stock basket):
  * Universe: point-in-time S&P 500 members with price data, price >= $5.
  * Trend filter: stock above its 200-day MA.
  * Entry: 2-period RSI (Wilder) closes below RSI_ENTRY (test 5 and 10).
          Rank candidates by lowest RSI, take the most oversold first.
  * Buy on close, equal weight WEIGHT_PER per position, max MAX_POS concurrent.
  * Exit: close > 5-day MA (Connors), OR max hold MAX_HOLD days (safety),
          OR stock leaves the index.
  * Costs: COST_BPS per side.
This is daily-frequency and orthogonal to cross-sectional momentum.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import clenow_backtest as cb

ROOT = cb.ROOT

# --- parameters ---
RSI_ENTRY = 10.0
RSI_PERIOD = 2
MA_TREND = 200
MA_EXIT = 5
MAX_POS = 10
WEIGHT_PER = 0.10        # 10% equity per position
MAX_HOLD = 10            # safety time-stop (trading days)
MIN_PRICE = 5.0
COST_BPS = 5.0
INITIAL_EQUITY = 100_000.0


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = ag / al.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100)  # no losses -> RSI 100 (not oversold)


def precompute(stocks, rsi_entry=RSI_ENTRY):
    feats = {}
    for sym, df in stocks.items():
        c = df["Close"].astype(float)
        if len(c) < MA_TREND + 5:
            continue
        feats[sym] = dict(
            close=c,
            ma200=c.rolling(MA_TREND).mean(),
            ma5=c.rolling(MA_EXIT).mean(),
            rsi=wilder_rsi(c, RSI_PERIOD),
        )
    return feats


@dataclass
class Pos:
    shares: float
    entry: float
    held: int = 0


def run(data, feats, membership, win_start=None, win_end=None, rsi_entry=RSI_ENTRY):
    idx_key = "_GSPC" if "_GSPC" in data else "SPY"
    index_close = data[idx_key]["Close"]
    cal = index_close.index
    start_i = MA_TREND + 5
    if win_start:
        start_i = max(start_i, cal.searchsorted(pd.Timestamp(win_start)))
    we = pd.Timestamp(win_end) if win_end else None

    cash = INITIAL_EQUITY
    pos: dict[str, Pos] = {}
    curve = []
    n_trades = 0

    def px(sym, date):
        s = feats.get(sym)
        if s and date in s["close"].index:
            v = s["close"].loc[date]
            return float(v) if not math.isnan(v) else None
        return None

    for i in range(start_i, len(cal)):
        date = cal[i]
        if we is not None and date > we:
            break
        # mark to market
        eq = cash + sum((p.shares * (px(s, date) or p.entry)) for s, p in pos.items())
        curve.append((date, eq))

        # --- exits first ---
        for sym in list(pos.keys()):
            s = feats.get(sym)
            p = pos[sym]
            p.held += 1
            price = px(sym, date)
            if price is None:
                continue
            ma5 = s["ma5"].loc[date] if date in s["ma5"].index else np.nan
            leave = (membership is not None and not cb.is_member(membership, sym, date))
            if price > ma5 or p.held >= MAX_HOLD or leave:
                proceeds = p.shares * price
                c = proceeds * COST_BPS / 1e4
                cash += proceeds - c
                n_trades += 1
                del pos[sym]

        # --- entries ---
        if len(pos) < MAX_POS:
            cands = []
            for sym, s in feats.items():
                if sym in pos:
                    continue
                if membership is not None and not cb.is_member(membership, sym, date):
                    continue
                if date not in s["close"].index:
                    continue
                price = float(s["close"].loc[date])
                ma200 = s["ma200"].loc[date]
                rsi = s["rsi"].loc[date]
                if math.isnan(ma200) or math.isnan(rsi):
                    continue
                if price >= MIN_PRICE and price > ma200 and rsi < rsi_entry:
                    cands.append((rsi, sym, price))
            cands.sort()  # most oversold first
            for rsi, sym, price in cands:
                if len(pos) >= MAX_POS:
                    break
                alloc = WEIGHT_PER * eq
                shares = math.floor(alloc / price)
                if shares <= 0:
                    continue
                cost_val = shares * price
                c = cost_val * COST_BPS / 1e4
                if cost_val + c > cash:
                    continue
                cash -= cost_val + c
                n_trades += 1
                pos[sym] = Pos(shares, price)

    eqs = pd.Series({d: v for d, v in curve})
    if len(eqs) < 30:
        return {"error": "few bars"}
    rets = eqs.pct_change().dropna()
    ann = (eqs.iloc[-1] / eqs.iloc[0]) ** (252 / len(eqs)) - 1
    sharpe = rets.mean() / rets.std() * math.sqrt(252) if rets.std() > 0 else 0
    dn = rets[rets < 0].std()
    sortino = rets.mean() / dn * math.sqrt(252) if dn and dn > 0 else 0
    peak = eqs.cummax()
    mdd = -((eqs - peak) / peak).min()
    return dict(final=round(eqs.iloc[-1]), ann=round(ann * 100, 2),
                sharpe=round(sharpe, 2), sortino=round(sortino, 2),
                maxdd=round(mdd * 100, 2), trades=n_trades,
                total=round((eqs.iloc[-1] / eqs.iloc[0] - 1) * 100, 2), _eqs=eqs)


def main():
    data = cb.load_prices()
    stocks = {s: d for s, d in data.items() if s not in ("_GSPC",)}
    membership = cb.load_membership()
    windows = [
        ("FULL 2007-2026", None, None),
        ("2008 crisis 07-09", "2007-01-01", "2009-12-31"),
        ("recovery 10-14", "2010-01-01", "2014-12-31"),
        ("2015-2018", "2015-01-01", "2018-12-31"),
        ("covid 19-21", "2019-01-01", "2021-12-31"),
        ("recent 22-26", "2022-01-01", "2026-07-11"),
    ]
    for entry in [5.0, 10.0]:
        feats = precompute(stocks, rsi_entry=entry)
        print(f"\n===== Connors RSI(2) < {entry:.0f}  (PIT members, max {MAX_POS} pos, "
              f"{int(WEIGHT_PER*100)}% each) =====")
        print(f"{'Period':<20}{'AnnRet%':>9}{'Sharpe':>8}{'Sortino':>9}{'MaxDD%':>8}"
              f"{'TotRet%':>9}{'Trades':>8}")
        print("-" * 71)
        shps = []
        for name, ws, wend in windows:
            r = run(data, feats, membership, ws, wend, rsi_entry=entry)
            if "error" in r:
                continue
            if name != "FULL 2007-2026":
                shps.append(r["sharpe"])
            print(f"{name:<20}{r['ann']:>9}{r['sharpe']:>8}{r['sortino']:>9}"
                  f"{r['maxdd']:>8}{r['total']:>9}{r['trades']:>8}")
        mean = sum(shps) / len(shps)
        std = (sum((x - mean) ** 2 for x in shps) / len(shps)) ** 0.5
        print("-" * 71)
        print(f"sub-period Sharpe: mean={mean:.2f} std={std:.2f} "
              f"CoV={std/mean*100:.0f}% min={min(shps):.2f}")


if __name__ == "__main__":
    main()

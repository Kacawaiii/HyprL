#!/usr/bin/env python3
"""Turtle (HyprL V3/V4 params) on prop-tradeable instruments, honest daily MTM.

Replicates the EA's frozen params from V4_full_turtle.xlsx:
  InpEntryPeriod=60, InpExitPeriod=20, InpATRPeriod=20,
  InpRiskPercent=1.5, InpATRStopMult=3.0, InpTrailDistance=0.8
Long AND short (turtle trades both ways), no pyramiding by default (baseline was
the best risk-adjusted config: DD 22.9% vs 35.7% with pyramiding).

Unlike the MT5 tester's event-sampled curve, this records equity EVERY day, so
Sharpe / drawdown / the prop daily-loss rule are measured on what actually
happened intra-trade — the exact bug DECISION_NOTE_v2 caught in the crypto v1.

Outputs a daily equity curve for the prop-challenge Monte Carlo.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kyo/HyprL")
PROP = ROOT / "data/momentum/prop"

ENTRY_N, EXIT_N, ATR_N = 60, 20, 20
RISK_PCT = 0.015          # 1.5% per trade (the EA's InpRiskPercent)
STOP_ATR = 3.0
TRAIL_ATR = 0.8
COST_BPS = 3.0            # FX/gold spread+commission per side (~0.6-3bps typical)
INITIAL = 100_000.0
MAX_TOTAL_RISK = 0.04     # InpMaxTotalRisk=4.0


def atr(df, n):
    h, l, c = df["High"], df["Low"], df["Close"]
    pc = c.shift(1)
    tr = pd.concat([(h - l), (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def load():
    data = {}
    for f in sorted(PROP.glob("*.parquet")):
        df = pd.read_parquet(f)
        df.index = pd.to_datetime(df.index)
        data[f.stem] = df
    return data


@dataclass
class Pos:
    side: int
    units: float      # position size in "units of instrument"
    entry: float
    stop: float
    peak: float       # best close since entry (for trailing)


def run(data, risk_mult=1.0, win_start=None, win_end=None):
    feats = {}
    for s, df in data.items():
        c = df["Close"]
        feats[s] = dict(
            c=c,
            hi=c.rolling(ENTRY_N).max().shift(1),   # prior N-day high (exclusive)
            lo=c.rolling(ENTRY_N).min().shift(1),
            xhi=c.rolling(EXIT_N).max().shift(1),
            xlo=c.rolling(EXIT_N).min().shift(1),
            atr=atr(df, ATR_N),
        )
    # master calendar: union of all instrument dates
    cal = sorted(set().union(*[set(f["c"].index) for f in feats.values()]))
    cal = pd.DatetimeIndex(cal)
    start_i = ENTRY_N + ATR_N + 5
    if win_start:
        start_i = max(start_i, cal.searchsorted(pd.Timestamp(win_start)))
    we = pd.Timestamp(win_end) if win_end else None

    cash = INITIAL
    pos: dict[str, Pos] = {}
    curve = []
    n_trades = 0

    def px(s, d):
        c = feats[s]["c"]
        if d in c.index:
            v = float(c.loc[d])
            return v if not math.isnan(v) else None
        return None

    for i in range(start_i, len(cal)):
        d = cal[i]
        if we is not None and d > we:
            break

        # --- MTM equity (every single day) ---
        eq = cash + sum(p.side * p.units * ((px(s, d) or p.entry) - p.entry)
                        for s, p in pos.items())
        curve.append((d, eq))
        if eq <= 0:
            break

        # --- manage open positions ---
        for s, p in list(pos.items()):
            cp = px(s, d)
            a = feats[s]["atr"].get(d, np.nan)
            if cp is None or math.isnan(a):
                continue
            # trailing stop (InpTrailDistance=0.8 ATR from best close)
            if p.side == 1:
                p.peak = max(p.peak, cp)
                p.stop = max(p.stop, p.peak - TRAIL_ATR * a)
            else:
                p.peak = min(p.peak, cp)
                p.stop = min(p.stop, p.peak + TRAIL_ATR * a)
            xhi = feats[s]["xhi"].get(d, np.nan)
            xlo = feats[s]["xlo"].get(d, np.nan)
            hit_stop = (cp <= p.stop) if p.side == 1 else (cp >= p.stop)
            hit_exit = (not math.isnan(xlo) and cp <= xlo) if p.side == 1 else \
                       (not math.isnan(xhi) and cp >= xhi)
            if hit_stop or hit_exit:
                cash += p.side * p.units * (cp - p.entry)
                cash -= abs(p.units * cp) * COST_BPS / 1e4
                del pos[s]
                n_trades += 1

        # --- entries (Donchian 60 breakout, both directions) ---
        open_risk = sum(abs(p.units * (p.entry - p.stop)) for p in pos.values())
        for s in feats:
            if s in pos:
                continue
            cp = px(s, d)
            f = feats[s]
            hi, lo, a = f["hi"].get(d, np.nan), f["lo"].get(d, np.nan), f["atr"].get(d, np.nan)
            if cp is None or any(math.isnan(v) for v in (hi, lo, a)) or a <= 0:
                continue
            side = 1 if cp > hi else (-1 if cp < lo else 0)
            if side == 0:
                continue
            risk_per_unit = STOP_ATR * a
            risk_budget = eq * RISK_PCT * risk_mult
            units = risk_budget / risk_per_unit
            if units <= 0:
                continue
            # portfolio heat cap (InpMaxTotalRisk)
            if open_risk + risk_budget > eq * MAX_TOTAL_RISK * risk_mult:
                continue
            open_risk += risk_budget
            stop = cp - side * STOP_ATR * a
            cash -= abs(units * cp) * COST_BPS / 1e4
            pos[s] = Pos(side, units, cp, stop, cp)

    eqs = pd.Series({d: v for d, v in curve})
    return eqs, n_trades


def metrics(eqs):
    r = eqs.pct_change().dropna()
    yrs = (eqs.index[-1] - eqs.index[0]).days / 365.25
    ann = (eqs.iloc[-1] / eqs.iloc[0]) ** (1 / yrs) - 1
    sh = r.mean() / r.std() * math.sqrt(252) if r.std() > 0 else 0
    mdd = -((eqs - eqs.cummax()) / eqs.cummax()).min()
    return ann * 100, sh, r.std() * math.sqrt(252) * 100, mdd * 100


if __name__ == "__main__":
    data = load()
    print(f"instruments: {len(data)}  ({', '.join(data)})")
    print(f"\n{'risk mult':>10}{'risk/trade%':>12}{'AnnRet%':>9}{'Sharpe':>8}"
          f"{'Vol%':>7}{'MaxDD%':>8}{'trades':>8}")
    print("-" * 62)
    for rm in [1.0, 0.5, 0.33, 0.25]:
        eqs, nt = run(data, risk_mult=rm)
        a, s, v, dd = metrics(eqs)
        print(f"{rm:>10.2f}{RISK_PCT*rm*100:>12.2f}{a:>9.1f}{s:>8.2f}{v:>7.1f}{dd:>8.1f}{nt:>8}")
        if rm == 1.0:
            eqs.to_frame("equity").to_csv(ROOT / "data/momentum/prop_turtle_equity.csv")

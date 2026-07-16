#!/usr/bin/env python3
"""Download S&P 500 daily OHLC data for the Clenow momentum backtest.

Saves one parquet per ticker under data/momentum/prices/ plus SPY (index proxy).
Uses auto-adjusted closes (splits + dividends) so total-return momentum is honest.
Survivorship caveat: this is the *current* S&P 500 constituent set — known bias,
addressed in a later pass with point-in-time membership.
"""
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path("/home/kyo/HyprL")
OUT = ROOT / "data/momentum/prices"
OUT.mkdir(parents=True, exist_ok=True)

START = "2007-01-01"  # covers 2008 crash, 2020, 2022 regimes
END = "2026-07-11"

tickers = json.load(open(ROOT / "data/momentum/sp500_current.json"))
universe = sorted(set(tickers + ["SPY", "^GSPC"]))

print(f"downloading {len(universe)} symbols {START}..{END}")
ok, fail = [], []
# batch download to be gentle and fast
BATCH = 40
for i in range(0, len(universe), BATCH):
    chunk = universe[i:i + BATCH]
    try:
        df = yf.download(chunk, start=START, end=END, progress=False,
                         auto_adjust=True, group_by="ticker", threads=True)
    except Exception as e:
        print("batch fail", chunk[:3], repr(e)[:120])
        fail += chunk
        continue
    for t in chunk:
        try:
            sub = df[t] if len(chunk) > 1 else df
            sub = sub.dropna(subset=["Close"])
            if len(sub) < 300:
                fail.append(t)
                continue
            sub = sub[["Open", "High", "Low", "Close", "Volume"]].copy()
            sub.to_parquet(OUT / f"{t.replace('^', '_')}.parquet")
            ok.append(t)
        except Exception:
            fail.append(t)
    print(f"  {i+len(chunk)}/{len(universe)} done  ok={len(ok)} fail={len(fail)}")
    time.sleep(1)

print(f"DONE ok={len(ok)} fail={len(fail)}")
print("failed:", fail[:30])
json.dump({"ok": ok, "fail": fail}, open(ROOT / "data/momentum/download_status.json", "w"))

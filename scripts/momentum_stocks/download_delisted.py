#!/usr/bin/env python3
"""Download price history for delisted / dropped S&P 500 members (survivorship fix).

Reads the point-in-time membership file, finds tickers we don't already have,
and tries yfinance for each. Many delisted names won't be available — we keep
whatever exists. Saves to the same prices/ dir so the backtest can use them.
"""
import csv
import json
import os
import time
from pathlib import Path

import yfinance as yf

ROOT = Path("/home/kyo/HyprL")
PRICES = ROOT / "data/momentum/prices"
START, END = "2007-01-01", "2026-07-11"

rows = list(csv.DictReader(open(ROOT / "data/momentum/sp500_membership.csv")))
ever = sorted({r["ticker"] for r in rows})
have = {f[:-8] for f in os.listdir(PRICES)}
# yfinance uses '-' for share classes (BRK-B), membership uses '.' sometimes
missing = [t for t in ever if t not in have and t.replace(".", "-") not in have]
print(f"missing tickers to attempt: {len(missing)}")

ok, fail = [], []
for i, t in enumerate(missing):
    yft = t.replace(".", "-")
    try:
        df = yf.download(yft, start=START, end=END, progress=False,
                         auto_adjust=True, threads=False)
        if df is None or len(df) < 200:
            fail.append(t); continue
        if isinstance(df.columns, __import__("pandas").MultiIndex):
            df.columns = [c[0] for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(subset=["Close"])
        if len(df) < 200:
            fail.append(t); continue
        df.to_parquet(PRICES / f"{yft}.parquet")
        ok.append(t)
    except Exception:
        fail.append(t)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(missing)}  recovered={len(ok)}")
        time.sleep(1)

print(f"DONE recovered={len(ok)} unavailable={len(fail)}")
json.dump({"recovered": ok, "unavailable": fail},
          open(ROOT / "data/momentum/delisted_status.json", "w"))

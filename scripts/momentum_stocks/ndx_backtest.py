#!/usr/bin/env python3
"""Run the Clenow momentum engine restricted to the Nasdaq-100 universe.

NOTE: uses CURRENT Nasdaq-100 constituents -> strong survivorship bias (today's
NDX = the tech winners). These numbers are OPTIMISTIC; treat as an upper bound.
Index regime filter still uses SPY (broad-market on/off).
"""
import json
import clenow_backtest as cb

ndx = set(json.load(open(cb.ROOT / "data/momentum/ndx100_current.json")))

data = cb.load_prices()
idx_key = "_GSPC" if "_GSPC" in data else "SPY"
index_close = data[idx_key]["Close"]
index_ma200 = index_close.rolling(cb.MA_INDEX).mean()

stocks = {s: d for s, d in data.items() if s in ndx}
print(f"NDX universe with data: {len(stocks)} tickers")
feats = cb.precompute(stocks)

windows = [
    ("FULL 2007-2026", None, None),
    ("2008 crisis 07-09", "2007-01-01", "2009-12-31"),
    ("recovery 10-14", "2010-01-01", "2014-12-31"),
    ("2015-2018", "2015-01-01", "2018-12-31"),
    ("covid 19-21", "2019-01-01", "2021-12-31"),
    ("recent 22-26", "2022-01-01", "2026-07-11"),
]
print(f"\n{'Period':<20}{'AnnRet%':>9}{'Sharpe':>8}{'MaxDD%':>8}{'TotRet%':>9}"
      f"{'|SPYshp':>8}{'SPYdd%':>8}")
print("-" * 78)
shps = []
for name, ws, we in windows:
    r = cb.run(data=data, feats=feats, index_close=index_close,
               index_ma200=index_ma200, win_start=ws, win_end=we, verbose=False)
    s, b = r["strategy"], r["spy_buyhold"]
    if name != "FULL 2007-2026":
        shps.append(s["sharpe"])
    print(f"{name:<20}{s['annualized_pct']:>9}{s['sharpe']:>8}{s['max_dd_pct']:>8}"
          f"{s['total_return_pct']:>9}{b['sharpe']:>8}{b['max_dd_pct']:>8}")
mean = sum(shps) / len(shps)
std = (sum((x - mean) ** 2 for x in shps) / len(shps)) ** 0.5
print("-" * 78)
print(f"sub-period Sharpe: mean={mean:.2f} std={std:.2f} "
      f"CoV={std/mean*100:.0f}% min={min(shps):.2f}")

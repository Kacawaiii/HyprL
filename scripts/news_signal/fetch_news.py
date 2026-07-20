#!/usr/bin/env python3
"""Fetch point-in-time news history from Alpaca (Benzinga feed) and cache it.

Why this and not the old scraper: `src/hyprl/sentiment/multi_source.py` scrapes
live Reddit/StockTwits/Finviz with no timestamped history, so its score can never
be validated against forward returns — it was only ever a filter nobody tested.
The Alpaca news endpoint gives `created_at` timestamps back to 2016, which is what
makes an honest backtest possible at all.

Fetches the whole feed by time window (not per-symbol: the feed is only ~730
articles/day and each article already carries its `symbols` list, so one pass is
far cheaper than 500 per-symbol queries). Caches one parquet per month, resumable.
"""
from __future__ import annotations
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/news/raw"
OUT.mkdir(parents=True, exist_ok=True)

KEY = os.environ.get("ALPACA_KEY", "")
SECRET = os.environ.get("ALPACA_SECRET", "")
if not KEY or not SECRET:
    sys.exit("ALPACA_KEY / ALPACA_SECRET not set")
HDR = {"APCA-API-KEY-ID": KEY, "APCA-API-SECRET-KEY": SECRET}
URL = "https://data.alpaca.markets/v1beta1/news"

START = os.environ.get("NEWS_START", "2023-01-01")
END = os.environ.get("NEWS_END", "2026-07-16")


def _get(params, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL + "?" + urllib.parse.urlencode(params),
                                         headers=HDR)
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.load(r)
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** i)  # backoff: a transient 429/5xx shouldn't kill the run
    return {}


def fetch_window(start: datetime, end: datetime) -> list[dict]:
    rows, token, pages = [], None, 0
    while True:
        p = {"start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
             "limit": 50, "sort": "asc"}
        if token:
            p["page_token"] = token
        d = _get(p)
        for a in d.get("news", []):
            syms = a.get("symbols") or []
            if not syms:
                continue  # untagged article can't be attributed to a stock
            rows.append({
                "ts": a["created_at"],
                "id": a["id"],
                "headline": a.get("headline") or "",
                "summary": a.get("summary") or "",
                "symbols": ",".join(syms),
                "source": a.get("source") or "",
            })
        token = d.get("next_page_token")
        pages += 1
        if not token:
            break
        if pages > 4000:
            print("  [page cap hit]")
            break
    return rows


def month_iter(start: str, end: str):
    cur = datetime.fromisoformat(start).replace(tzinfo=timezone.utc, day=1)
    last = datetime.fromisoformat(end).replace(tzinfo=timezone.utc)
    while cur < last:
        nxt = (cur.replace(day=28) + timedelta(days=8)).replace(day=1)
        yield cur, min(nxt, last)
        cur = nxt


def main():
    total = 0
    for a, b in month_iter(START, END):
        tag = a.strftime("%Y-%m")
        path = OUT / f"news_{tag}.parquet"
        if path.exists():
            n = len(pd.read_parquet(path))
            total += n
            print(f"{tag}: cached ({n})")
            continue
        t0 = time.time()
        rows = fetch_window(a, b)
        if rows:
            pd.DataFrame(rows).to_parquet(path)
        total += len(rows)
        print(f"{tag}: {len(rows)} articles in {time.time()-t0:.0f}s")
    print(f"TOTAL {total} articles cached in {OUT}")


if __name__ == "__main__":
    main()

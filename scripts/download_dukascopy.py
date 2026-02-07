#!/usr/bin/env python3
"""
Download historical forex data from Dukascopy (multi-threaded).

Downloads tick data via Dukascopy's public API (bi5 compressed format),
aggregates to M30 bars, and saves as Parquet.

Uses ThreadPoolExecutor for 20x speedup (~20 concurrent downloads).

Usage:
    python scripts/download_dukascopy.py --pairs EURUSD,USDJPY,XAUUSD,AUDUSD,USDCAD
    python scripts/download_dukascopy.py --pairs all
"""

import argparse
import lzma
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://datafeed.dukascopy.com/datafeed"

ALL_PAIRS = ["EURUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD"]

POINT_SIZES = {
    "EURUSD": 1e-5, "GBPUSD": 1e-5, "USDJPY": 1e-3, "XAUUSD": 1e-3,
    "AUDUSD": 1e-5, "USDCAD": 1e-5, "NZDUSD": 1e-5, "USDCHF": 1e-5,
}


def _build_url(symbol: str, dt: datetime) -> str:
    return (f"{BASE_URL}/{symbol}/"
            f"{dt.year:04d}/{dt.month - 1:02d}/{dt.day:02d}/"
            f"{dt.hour:02d}h_ticks.bi5")


def _download_one_hour(args: Tuple[str, datetime, float]) -> Tuple[datetime, Optional[bytes]]:
    """Download one hour of bi5 data. Returns (datetime, data_or_None)."""
    url, dt, point_size = args
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=15,
                                headers={"User-Agent": "Mozilla/5.0"})
            if resp.status_code == 200 and len(resp.content) > 0:
                return (dt, resp.content)
            elif resp.status_code == 404:
                return (dt, None)
        except Exception:
            pass
        time.sleep(0.5 * (attempt + 1))
    return (dt, None)


def parse_bi5(data: bytes, base_dt: datetime, point_size: float) -> List[Tuple]:
    """Parse bi5 tick data. Returns list of (time, mid, volume) tuples."""
    try:
        raw = lzma.decompress(data)
    except Exception:
        return []
    if len(raw) == 0 or len(raw) % 20 != 0:
        return []

    n_ticks = len(raw) // 20
    records = []
    for i in range(n_ticks):
        offset = i * 20
        ms, ask_pts, bid_pts, ask_vol, bid_vol = struct.unpack(
            ">IIIff", raw[offset:offset + 20]
        )
        tick_time = base_dt + timedelta(milliseconds=ms)
        mid = (ask_pts + bid_pts) * point_size / 2.0
        vol = ask_vol + bid_vol
        records.append((tick_time, mid, vol))
    return records


def ticks_to_m30(ticks: List[Tuple]) -> pd.DataFrame:
    """Aggregate tick list to M30 OHLCV bars."""
    if not ticks:
        return pd.DataFrame()
    df = pd.DataFrame(ticks, columns=["time", "mid", "volume"])
    df = df.set_index("time")
    bars = df["mid"].resample("30min").agg(
        open="first", high="max", low="min", close="last"
    )
    bars["volume"] = df["volume"].resample("30min").sum()
    bars = bars.dropna(subset=["open"]).reset_index()
    return bars


def download_symbol(
    symbol: str,
    start_date: datetime,
    end_date: datetime,
    output_dir: str = "data/dukascopy",
    max_workers: int = 20,
) -> pd.DataFrame:
    """Download full history for one symbol using multi-threading."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    parquet_path = out_path / f"{symbol}_M30.parquet"

    point_size = POINT_SIZES[symbol]

    # Build list of all hours to download
    hours = []
    current = start_date
    while current < end_date:
        url = _build_url(symbol, current)
        hours.append((url, current, point_size))
        current += timedelta(hours=1)

    total_hours = len(hours)
    print(f"\n{'='*60}")
    print(f"Downloading {symbol}: {start_date.date()} -> {end_date.date()}")
    print(f"Total hours: {total_hours:,} | Workers: {max_workers}")
    print(f"{'='*60}")

    # Download in parallel
    all_ticks = []
    completed = 0
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_download_one_hour, h): h for h in hours}

        for future in as_completed(futures):
            dt, data = future.result()
            completed += 1

            if data is not None:
                ticks = parse_bi5(data, dt, point_size)
                all_ticks.extend(ticks)
            else:
                failed += 1

            if completed % 2000 == 0:
                elapsed = time.time() - t_start
                rate = completed / elapsed
                eta = (total_hours - completed) / rate if rate > 0 else 0
                pct = completed / total_hours * 100
                print(f"  [{pct:5.1f}%] {completed:,}/{total_hours:,} | "
                      f"{len(all_ticks):,} ticks | "
                      f"{rate:.0f} req/s | ETA: {eta/60:.0f}min",
                      flush=True)

    elapsed = time.time() - t_start
    print(f"  Download complete: {elapsed:.0f}s, {completed/elapsed:.0f} req/s")

    if not all_ticks:
        print(f"  WARNING: No data for {symbol}")
        return pd.DataFrame()

    # Aggregate to M30
    print(f"  Aggregating {len(all_ticks):,} ticks to M30...", flush=True)
    bars = ticks_to_m30(all_ticks)

    if bars.empty:
        print(f"  WARNING: No bars generated for {symbol}")
        return pd.DataFrame()

    # Sort and deduplicate
    bars = bars.sort_values("time").drop_duplicates(subset=["time"], keep="last")
    bars = bars.reset_index(drop=True)

    # Save
    bars.to_parquet(parquet_path, index=False)

    print(f"  Done: {len(bars):,} M30 bars")
    print(f"  Range: {bars['time'].min()} -> {bars['time'].max()}")
    print(f"  Saved: {parquet_path}")

    return bars


def main():
    parser = argparse.ArgumentParser(description="Download Dukascopy data (multi-threaded)")
    parser.add_argument("--pairs", type=str, default="all",
                        help="Comma-separated pairs or 'all'")
    parser.add_argument("--start", type=str, default="2015-01-01")
    parser.add_argument("--end", type=str, default="2025-01-01")
    parser.add_argument("--output-dir", type=str, default="data/dukascopy")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of download threads")
    args = parser.parse_args()

    if args.pairs.lower() == "all":
        pairs = ALL_PAIRS
    else:
        pairs = [p.strip().upper() for p in args.pairs.split(",")]

    start = datetime.strptime(args.start, "%Y-%m-%d")
    end = datetime.strptime(args.end, "%Y-%m-%d")

    print("\n" + "=" * 60)
    print("DUKASCOPY DATA DOWNLOAD (Multi-threaded)")
    print("=" * 60)
    print(f"Pairs:   {pairs}")
    print(f"Period:  {start.date()} -> {end.date()}")
    print(f"Workers: {args.workers}")
    print("=" * 60)

    results = {}
    for pair in pairs:
        if pair not in POINT_SIZES:
            print(f"WARNING: Unknown pair {pair}, skipping")
            continue
        df = download_symbol(pair, start, end, args.output_dir, args.workers)
        results[pair] = len(df) if not df.empty else 0

    # Summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(f"{'Pair':<10} {'Bars':>10} {'Status':<10}")
    print("-" * 30)
    for pair, count in results.items():
        status = "OK" if count > 50000 else ("PARTIAL" if count > 0 else "FAILED")
        print(f"{pair:<10} {count:>10,} {status:<10}")


if __name__ == "__main__":
    main()

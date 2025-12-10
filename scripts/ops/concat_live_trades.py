#!/usr/bin/env python3
"""
Concat all per-day live trade CSVs into *_live_all.csv per ticker.

Designed for live/logs layout emitted by run_live_single_ticker_hourly.py:
  live/logs/live_<ticker>/<YYYY-MM-DD>/trades_<TICKER>_live.csv
Output:
  live/logs/live_<ticker>/trades_<TICKER>_live_all.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concat live trade CSVs into *_live_all.csv per ticker.")
    parser.add_argument(
        "--symbols",
        default="NVDA,MSFT,AMD,META,QQQ",
        help="Comma-separated tickers to process (default: NVDA,MSFT,AMD,META,QQQ).",
    )
    parser.add_argument(
        "--log-root",
        type=Path,
        default=Path("live/logs"),
        help="Root directory containing live_<ticker>/ subfolders (default: live/logs).",
    )
    parser.add_argument(
        "--output-suffix",
        default="_live_all.csv",
        help="Suffix for aggregated file (default: _live_all.csv).",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce stdout noise.")
    return parser.parse_args()


def discover_daily_files(symbol: str, log_root: Path) -> List[Path]:
    base = log_root / f"live_{symbol.lower()}"
    if not base.is_dir():
        return []
    return sorted(base.glob("*/trades_*_live.csv"))


def read_rows(path: Path) -> list[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def merge_fieldnames(rows: Iterable[Dict[str, str]]) -> list[str]:
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def dedup_rows(rows: list[Dict[str, str]]) -> list[Dict[str, str]]:
    """Deduplicate rows by full key/value content to avoid double-append on re-runs."""
    seen: set[tuple[tuple[str, str], ...]] = set()
    unique: list[Dict[str, str]] = []
    for row in rows:
        key = tuple(sorted((k, row.get(k, "")) for k in row.keys()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def write_rows(path: Path, fieldnames: list[str], rows: list[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_symbol(symbol: str, log_root: Path, output_suffix: str, quiet: bool) -> None:
    daily_files = discover_daily_files(symbol, log_root)
    if not daily_files:
        if not quiet:
            print(f"[SKIP] {symbol}: no daily trade files under {log_root}/live_{symbol.lower()}")
        return
    all_rows: list[Dict[str, str]] = []
    for path in daily_files:
        try:
            rows = read_rows(path)
        except OSError as exc:
            print(f"[WARN] {symbol}: failed reading {path}: {exc}", file=sys.stderr)
            continue
        all_rows.extend(rows)
    if not all_rows:
        if not quiet:
            print(f"[SKIP] {symbol}: no rows found in daily files.")
        return
    unique_rows = dedup_rows(all_rows)
    fieldnames = merge_fieldnames(unique_rows)
    output_path = log_root / f"live_{symbol.lower()}" / f"trades_{symbol.upper()}{output_suffix}"
    write_rows(output_path, fieldnames, unique_rows)
    if not quiet:
        print(
            f"[OK] {symbol}: {len(unique_rows)} rows -> {output_path} "
            f"(from {len(all_rows)} rows across {len(daily_files)} files)"
        )


def main() -> None:
    args = parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        raise SystemExit("No symbols provided.")
    for symbol in symbols:
        process_symbol(symbol, args.log_root, args.output_suffix, args.quiet)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generic live trade concatenation for any ticker.

Example:
    python scripts/tools/concat_trades_live_generic.py \
      --symbol NVDA \
      --input-dir live/logs/live_nvda \
      --output live/logs/live_nvda/trades_NVDA_live_all.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concat live trade CSVs for a ticker into one deduped file.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol (e.g., NVDA).")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory to search recursively for trade logs.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        help="Filename glob pattern (default: trades_<SYMBOL>_live*.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for the aggregated trades.",
    )
    return parser.parse_args()


def find_trade_files(root: Path, symbol: str, pattern: str | None = None) -> List[Path]:
    glob_pattern = pattern or f"trades_{symbol.upper()}_live*.csv"
    files = list(root.rglob(glob_pattern))
    try:
        files.sort(key=lambda p: (p.stat().st_mtime, p.name))
    except FileNotFoundError:
        files.sort(key=lambda p: p.name)
    return files


def dedupe_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_id" in df.columns:
        key_cols = ["trade_id"]
    else:
        size_col = "position_size" if "position_size" in df.columns else ("size" if "size" in df.columns else None)
        key_candidates = ["ticker", "entry_timestamp", "direction", size_col, "entry_price", "exit_price"]
        key_cols = [c for c in key_candidates if c and c in df.columns]
    if not key_cols:
        return df
    return df.drop_duplicates(subset=key_cols, keep="last")


def write_empty(output: Path, columns: Optional[List[str]] = None) -> None:
    cols = columns or [
        "ticker",
        "entry_timestamp",
        "exit_timestamp",
        "direction",
        "probability_up",
        "threshold",
        "entry_price",
        "stop_price",
        "take_profit_price",
        "trailing_stop_activation_price",
        "trailing_stop_distance_price",
        "exit_price",
        "exit_reason",
        "position_size",
        "pnl",
        "return_pct",
        "equity_after",
        "risk_amount",
        "expected_pnl",
        "risk_profile",
        "effective_long_threshold",
        "effective_short_threshold",
        "regime_name",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(columns=cols).to_csv(output, index=False)


def concat_trades(symbol: str, input_dir: Path, output: Path, pattern: str | None = None) -> None:
    files = find_trade_files(input_dir, symbol, pattern)
    if not files:
        write_empty(output)
        print(f"[OPS] No trade files found under {input_dir} for {symbol}. Wrote empty CSV to {output}.")
        return

    frames: List[pd.DataFrame] = []
    print(f"[OPS] Found {len(files)} trade files for {symbol} under {input_dir} (pattern={pattern or 'default'}).")
    for path in files:
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            frames.append(df)
        except Exception as exc:  # pragma: no cover - defensive read
            print(f"[WARN] Failed to read {path}: {exc}")
            continue

    if not frames:
        write_empty(output)
        print(f"[OPS] All files empty/unreadable for {symbol}. Wrote empty CSV to {output}.")
        return

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = dedupe_frame(merged)
    if "entry_timestamp" in merged.columns:
        sort_ts = pd.to_datetime(merged["entry_timestamp"])
        merged = merged.assign(_sort_ts=sort_ts).sort_values("_sort_ts").drop(columns="_sort_ts")

    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)
    removed = before - len(merged)
    print(f"[OPS] Aggregated {len(files)} files -> {len(merged)} rows into {output} (dedup removed {removed})")


def main() -> None:
    args = parse_args()
    concat_trades(symbol=args.symbol.upper(), input_dir=args.input_dir, output=args.output, pattern=args.pattern)


if __name__ == "__main__":
    main()

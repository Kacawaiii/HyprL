#!/usr/bin/env python3
"""
Concatenate all NVDA live trade logs into a single deduplicated CSV, sorted by entry_timestamp.

Usage:
    python scripts/tools/concat_trades_nvda_live.py \
      --input-dir live/logs/live_nvda \
      --output live/logs/live_nvda/trades_NVDA_live_all.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concat NVDA live trade CSVs into one deduped file.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory to search recursively for trades_NVDA_live*.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path for the aggregated trades.",
    )
    return parser.parse_args()


def find_trade_files(root: Path) -> List[Path]:
    return sorted(root.rglob("trades_NVDA_live*.csv"))


def dedupe_frame(df: pd.DataFrame) -> pd.DataFrame:
    # Prefer trade_id if present; otherwise use a tuple of stable fields.
    if "trade_id" in df.columns:
        key_cols = ["trade_id"]
    else:
        key_cols = [
            "entry_timestamp",
            "direction",
            "position_size",
            "entry_price",
            "stop_price" if "stop_price" in df.columns else None,
            "take_profit_price" if "take_profit_price" in df.columns else None,
        ]
        key_cols = [c for c in key_cols if c is not None and c in df.columns]
    if not key_cols:
        return df
    deduped = df.drop_duplicates(subset=key_cols, keep="last")
    return deduped


def write_empty(output: Path, columns: Optional[List[str]] = None) -> None:
    cols = columns or [
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
    pd.DataFrame(columns=cols).to_csv(output, index=False)


def main() -> None:
    args = parse_args()
    files = find_trade_files(args.input_dir)
    if not files:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_empty(args.output)
        print(f"[OPS] No trade files found under {args.input_dir}. Wrote empty CSV to {args.output}.")
        return

    frames: List[pd.DataFrame] = []
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
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_empty(args.output)
        print(f"[OPS] All files empty/unreadable. Wrote empty CSV to {args.output}.")
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = dedupe_frame(merged)
    if "entry_timestamp" in merged.columns:
        merged["entry_timestamp"] = pd.to_datetime(merged["entry_timestamp"])
        merged = merged.sort_values("entry_timestamp")
        merged["entry_timestamp"] = merged["entry_timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"[OPS] Aggregated {len(files)} files -> {len(merged)} rows into {args.output}")


if __name__ == "__main__":
    main()

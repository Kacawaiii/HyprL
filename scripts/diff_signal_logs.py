#!/usr/bin/env python3
"""Diff HyprL signal trace CSVs to explain backtest vs replay decisions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

JOIN_KEYS = ["timestamp", "symbol"]
DEFAULT_DISPLAY_COLUMNS = [
    "timestamp",
    "symbol",
    "decision_bt",
    "decision_replay",
    "reason_bt",
    "reason_replay",
    "direction_bt",
    "direction_replay",
    "probability_up_bt",
    "probability_up_replay",
    "expected_pnl_bt",
    "expected_pnl_replay",
    "min_ev_bt",
    "min_ev_replay",
    "trend_ok_bt",
    "trend_ok_replay",
    "sentiment_ok_bt",
    "sentiment_ok_replay",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diff two signal trace CSV logs.")
    parser.add_argument("--backtest-log", required=True, help="CSV emitted by run_backtest --signal-log")
    parser.add_argument("--replay-log", required=True, help="CSV emitted by run_live_replay --signal-log")
    parser.add_argument(
        "--output",
        help="Optional path to export the merged dataframe with classification columns.",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=5,
        help="Number of sample rows to display for BT-only / Replay-only decision deltas.",
    )
    return parser.parse_args()


def _load_log(path: str | Path, suffix: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].str.upper()
    else:
        df["symbol"] = "UNKNOWN"
    if "decision" in df.columns:
        decision_series = df["decision"].copy()
    else:
        decision_series = pd.Series([None] * len(df))
    df["accepted"] = decision_series.fillna("").astype(str).str.lower().eq("emit")
    keep_cols = [col for col in df.columns if col not in JOIN_KEYS]
    renamed = {col: f"{col}_{suffix}" for col in keep_cols}
    return df[JOIN_KEYS + keep_cols].rename(columns=renamed)


def _summarize(merged: pd.DataFrame, head: int) -> None:
    merged["accepted_bt"] = merged["accepted_bt"].fillna(False)
    merged["accepted_replay"] = merged["accepted_replay"].fillna(False)

    merged["bt_only"] = merged["accepted_bt"] & ~merged["accepted_replay"]
    merged["replay_only"] = merged["accepted_replay"] & ~merged["accepted_bt"]

    total = len(merged)
    both = int((merged["accepted_bt"] == merged["accepted_replay"]).sum())
    bt_only_count = int(merged["bt_only"].sum())
    replay_only_count = int(merged["replay_only"].sum())

    print("=== Summary ===")
    print(f"Rows compared : {total}")
    print(f"BT-only accepts: {bt_only_count}")
    print(f"Replay-only accepts: {replay_only_count}")
    print(f"Matching decisions: {both}")

    _print_examples(merged, merged["bt_only"], "Backtest accepted / Replay denied", head)
    _print_examples(merged, merged["replay_only"], "Replay accepted / Backtest denied", head)


def _print_examples(df: pd.DataFrame, mask: pd.Series, title: str, head: int) -> None:
    subset = df.loc[mask]
    print(f"\n=== {title} (showing up to {head}) ===")
    if subset.empty:
        print("No rows in this bucket.")
        return
    columns = [col for col in DEFAULT_DISPLAY_COLUMNS if col in subset.columns]
    display = subset.sort_values("timestamp")[columns].head(head)
    pd.set_option("display.max_columns", None)
    print(display.to_string(index=False))


def main() -> None:
    args = _parse_args()
    backtest = _load_log(args.backtest_log, "bt")
    replay = _load_log(args.replay_log, "replay")
    merged = pd.merge(backtest, replay, on=JOIN_KEYS, how="outer", indicator=True)
    _summarize(merged, args.head)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(output_path, index=False)
        print(f"Merged diff exported to {output_path}")


if __name__ == "__main__":
    main()

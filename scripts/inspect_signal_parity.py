#!/usr/bin/env python3
from __future__ import annotations

"""Inspect parity signal logs for feature/training discrepancies."""

import argparse
from pathlib import Path

import pandas as pd


def load_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def summarize_diffs(bt: pd.DataFrame, replay: pd.DataFrame) -> None:
    merged = bt.merge(
        replay,
        on=["timestamp", "symbol"],
        how="inner",
        suffixes=("_bt", "_replay"),
    )
    if merged.empty:
        print("No overlapping rows between logs")
        return
    merged["feature_idx_diff"] = merged["feature_idx_bt"] - merged["feature_idx_replay"]
    merged["design_rows_diff"] = merged["design_rows_bt"] - merged["design_rows_replay"]
    idx_mismatch = merged[merged["feature_idx_diff"] != 0]
    design_mismatch = merged[merged["design_rows_diff"] != 0]
    sig_mismatch = merged[
        merged["feature_signature_bt"].fillna("") != merged["feature_signature_replay"].fillna("")
    ]
    print(f"Total overlapping rows: {len(merged)}")
    print(f"Feature index mismatches : {len(idx_mismatch)}")
    print(f"Design row mismatches   : {len(design_mismatch)}")
    print(f"Signature mismatches    : {len(sig_mismatch)}")
    if not idx_mismatch.empty:
        print("\nSample feature_idx diffs:")
        print(idx_mismatch[["timestamp", "feature_idx_bt", "feature_idx_replay"]].head())
    if not design_mismatch.empty:
        print("\nSample design_rows diffs:")
        print(design_mismatch[["timestamp", "design_rows_bt", "design_rows_replay"]].head())
    if not sig_mismatch.empty:
        print("\nSample feature_signature diffs:")
        print(
            sig_mismatch[
                [
                    "timestamp",
                    "feature_signature_bt",
                    "feature_signature_replay",
                ]
            ].head()
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare backtest vs replay signal logs for feature parity.")
    parser.add_argument("--backtest-log", type=Path, required=True)
    parser.add_argument("--replay-log", type=Path, required=True)
    args = parser.parse_args()
    bt = load_log(args.backtest_log)
    replay = load_log(args.replay_log)
    summarize_diffs(bt, replay)


if __name__ == "__main__":
    main()

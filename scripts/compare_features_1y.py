#!/usr/bin/env python3
from __future__ import annotations

"""Compare backtest vs replay feature columns over a 1y signal log."""

import argparse
from pathlib import Path

import pandas as pd

from hyprl.strategy.core import FEATURE_COLUMNS


def _parse_signature(series: pd.Series) -> pd.DataFrame:
    records: list[dict[str, float]] = []
    for value in series.fillna(""):
        entry: dict[str, float] = {}
        if value:
            parts = value.split("|")
            for part in parts:
                if "=" not in part:
                    continue
                key, raw = part.split("=", 1)
                try:
                    entry[key] = float(raw)
                except ValueError:
                    entry[key] = float("nan")
        records.append(entry)
    df = pd.DataFrame(records)
    return df


def expand_features(df: pd.DataFrame) -> pd.DataFrame:
    expanded = _parse_signature(df["feature_signature"])
    for col in FEATURE_COLUMNS:
        if col not in expanded:
            expanded[col] = float("nan")
    expanded = expanded[FEATURE_COLUMNS]
    expanded.index = df.index
    return expanded


def compare_features(bt_path: Path, replay_path: Path) -> None:
    bt = pd.read_csv(bt_path, parse_dates=["timestamp"])
    rp = pd.read_csv(replay_path, parse_dates=["timestamp"])
    merged = bt.merge(
        rp,
        on=["timestamp", "symbol"],
        suffixes=("_bt", "_replay"),
    )
    if merged.empty:
        print("No overlapping rows between BT and replay logs.")
        return
    features_bt = expand_features(merged[["feature_signature_bt"]].rename(columns={"feature_signature_bt": "feature_signature"}))
    features_rp = expand_features(merged[["feature_signature_replay"]].rename(columns={"feature_signature_replay": "feature_signature"}))
    diffs = (features_bt - features_rp).abs()
    max_per_col = diffs.max()
    global_max = float(max_per_col.max())
    print(f"[Features 1Y] overlap_rows={len(merged)}")
    print("max_abs_diff_per_column:")
    for col in FEATURE_COLUMNS:
        value = max_per_col.get(col, float("nan"))
        print(f"  {col}: {value:.6e}")
    print(f"global_max_abs_diff={global_max:.6e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare BT vs replay features over 1y signal logs.")
    parser.add_argument("--bt-log", type=Path, required=True)
    parser.add_argument("--replay-log", type=Path, required=True)
    args = parser.parse_args()
    compare_features(args.bt_log, args.replay_log)


if __name__ == "__main__":
    main()

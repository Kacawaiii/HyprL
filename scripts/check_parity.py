#!/usr/bin/env python3
from __future__ import annotations

"""Parity gate for NVDA backtest vs replay signal logs."""

import argparse
from pathlib import Path
from typing import Dict

import pandas as pd

DEFAULT_THRESHOLDS = {
    "bt_only_ratio": 0.01,
    "replay_only_ratio": 0.01,
    "aggregate_ratio": 0.02,
    "prob_mean": 5e-5,
    "prob_max": 1e-4,
    "position_max": 1e-3,
    "risk_max": 1e-2,
    "ev_max": 1e-3,
}

MICRO_THRESHOLDS = {
    "bt_only_ratio": 0.2,
    "replay_only_ratio": 0.2,
    "aggregate_ratio": 0.3,
    "prob_mean": 1e-4,
    "prob_max": 1e-4,
    "position_max": 1e-6,
    "risk_max": 1e-6,
    "ev_max": 1e-6,
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Parity diff missing: {path}")
    return pd.read_csv(path)


def _coerce(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _only_mask(df: pd.DataFrame, col: str) -> pd.Series:
    series = df.get(col)
    if series is None:
        return pd.Series(False, index=df.index)
    if series.dtype == bool:
        return series.fillna(False)
    return series.fillna(0).astype(str).str.lower().isin(["1", "true", "yes"])


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    total = len(df)
    if total == 0:
        raise SystemExit("Parity diff is empty")
    bt_mask = _only_mask(df, "bt_only")
    rp_mask = _only_mask(df, "replay_only")
    bt_only = bt_mask.sum()
    replay_only = rp_mask.sum()
    mismatch = (
        (df.get("decision_bt", "").fillna("") != df.get("decision_replay", "").fillna(""))
        & df.get("_merge", "").eq("both")
    ).sum()
    prob_diff = (_coerce(df.get("probability_up_bt")) - _coerce(df.get("probability_up_replay"))).abs().dropna()
    position_diff = (
        _coerce(df.get("position_size_bt")) - _coerce(df.get("position_size_replay"))
    ).abs().dropna()
    risk_diff = (_coerce(df.get("risk_amount_bt")) - _coerce(df.get("risk_amount_replay"))).abs().dropna()
    ev_diff = (_coerce(df.get("expected_pnl_bt")) - _coerce(df.get("expected_pnl_replay"))).abs().dropna()
    aggregate = (bt_only + replay_only + mismatch) / total
    return {
        "total": float(total),
        "bt_only_ratio": float(bt_only / total),
        "replay_only_ratio": float(replay_only / total),
        "aggregate_ratio": float(aggregate),
        "prob_mean": float(prob_diff.mean()) if not prob_diff.empty else 0.0,
        "prob_max": float(prob_diff.max()) if not prob_diff.empty else 0.0,
        "position_max": float(position_diff.max()) if not position_diff.empty else 0.0,
        "risk_max": float(risk_diff.max()) if not risk_diff.empty else 0.0,
        "ev_max": float(ev_diff.max()) if not ev_diff.empty else 0.0,
        "ev_mean": float(ev_diff.mean()) if not ev_diff.empty else 0.0,
    }


def _dump_only_samples(df: pd.DataFrame, col: str, path: Path, limit: int = 25) -> None:
    mask = _only_mask(df, col)
    subset = df[mask]
    if subset.empty:
        path.write_text("timestamp,symbol\n")
        return
    subset = subset[["timestamp", "symbol", "decision_bt", "decision_replay", "reason_bt", "reason_replay"]].head(limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)


def _evaluate(label: str, metrics: Dict[str, float], thresholds: Dict[str, float]) -> list[str]:
    failures: list[str] = []
    for key, limit in thresholds.items():
        value = metrics.get(key, 0.0)
        if value > limit:
            failures.append(f"{label}:{key}={value:.6g} exceeds {limit}")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description="Check NVDA parity diff metrics against acceptance gates.")
    parser.add_argument(
        "--diff",
        type=Path,
        default=Path("data/signals/nvda_signal_diff.csv"),
        help="Path to the 1y NVDA diff CSV.",
    )
    parser.add_argument(
        "--microscope-diff",
        type=Path,
        default=Path("data/parity/nvda_signal_diff_MICRO.csv"),
        help="Path to the microscope diff CSV.",
    )
    args = parser.parse_args()

    macro_df = _load_csv(args.diff)
    micro_df = _load_csv(args.microscope_diff)
    _dump_only_samples(macro_df, "bt_only", Path("data/parity/nvda_bt_only_samples.csv"))
    _dump_only_samples(macro_df, "replay_only", Path("data/parity/nvda_replay_only_samples.csv"))
    macro_metrics = compute_metrics(macro_df)
    micro_metrics = compute_metrics(micro_df)

    print("[parity] 1y metrics:")
    for key in sorted(macro_metrics):
        print(f"  - {key}: {macro_metrics[key]:.6g}")
    print("[parity] microscope metrics:")
    for key in sorted(micro_metrics):
        print(f"  - {key}: {micro_metrics[key]:.6g}")

    failures = []
    failures.extend(_evaluate("macro", macro_metrics, DEFAULT_THRESHOLDS))
    failures.extend(_evaluate("micro", micro_metrics, MICRO_THRESHOLDS))
    if failures:
        for reason in failures:
            print("[FAIL]", reason)
        raise SystemExit(1)
    print("[OK] parity metrics satisfy thresholds")


if __name__ == "__main__":
    main()

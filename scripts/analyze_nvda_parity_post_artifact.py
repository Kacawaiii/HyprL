#!/usr/bin/env python3
from __future__ import annotations

"""Analyze NVDA parity drifts after freezing the ProbabilityModel artifact."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _prob_stats(df: pd.DataFrame) -> dict[str, float]:
    both = df[df["_merge"] == "both"].copy()
    mask = both["probability_up_bt"].notna() & both["probability_up_replay"].notna()
    diffs = (both.loc[mask, "probability_up_bt"] - both.loc[mask, "probability_up_replay"]).abs()
    if diffs.empty:
        return {
            "mean": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
        }
    return {
        "mean": float(diffs.mean()),
        "max": float(diffs.max()),
        "p50": float(np.percentile(diffs, 50)),
        "p90": float(np.percentile(diffs, 90)),
        "p99": float(np.percentile(diffs, 99)),
    }


def _ev_stats(df: pd.DataFrame) -> dict[str, float]:
    both = df[df["_merge"] == "both"].copy()
    mask = both["expected_pnl_bt"].notna() & both["expected_pnl_replay"].notna()
    diffs = (both.loc[mask, "expected_pnl_bt"] - both.loc[mask, "expected_pnl_replay"]).abs()
    if diffs.empty:
        return {"mean": float("nan"), "max": float("nan")}
    return {"mean": float(diffs.mean()), "max": float(diffs.max())}


def _ratio(df: pd.DataFrame, col: str) -> float:
    total = len(df)
    if total == 0:
        return 0.0
    return float(df[col].sum()) / total


def analyze(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    stats = _prob_stats(df)
    ev = _ev_stats(df)
    stats["bt_only_ratio"] = _ratio(df, "bt_only")
    stats["replay_only_ratio"] = _ratio(df, "replay_only")
    stats["rows"] = len(df)
    stats["ev_mean"] = ev["mean"]
    stats["ev_max"] = ev["max"]
    return stats


def format_report(name: str, stats: dict[str, object]) -> str:
    return (
        f"[{name}] rows={stats['rows']} \n"
        f"  prob mean={stats['mean']:.6f} max={stats['max']:.6f} "
        f"p50={stats['p50']:.6f} p90={stats['p90']:.6f} p99={stats['p99']:.6f}\n"
        f"  EV mean={stats['ev_mean']:.6f} max={stats['ev_max']:.6f}\n"
        f"  bt_only={stats['bt_only_ratio']*100:.2f}% replay_only={stats['replay_only_ratio']*100:.2f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze NVDA parity drift after artifact.")
    parser.add_argument("--diff", type=Path, default=Path("data/signals/nvda_signal_diff.csv"))
    parser.add_argument(
        "--micro-diff",
        type=Path,
        default=Path("data/parity/nvda_signal_diff_MICRO.csv"),
    )
    args = parser.parse_args()
    macro_stats = analyze(args.diff)
    micro_stats = analyze(args.micro_diff)
    print(format_report("NVDA 1Y", macro_stats))
    print(format_report("NVDA MICRO", micro_stats))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

PROBABILITY_BINS = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]


def load_trades(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        parse_dates=["entry_timestamp", "exit_timestamp"],
    )
    required = {
        "pnl",
        "return_pct",
        "probability_up",
        "threshold",
        "direction",
        "entry_price",
        "exit_price",
        "equity_after",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in trades CSV: {sorted(missing)}")
    return df


def _max_streak(values: pd.Series, positive: bool) -> int:
    streak = 0
    best = 0
    comparator = (lambda v: v > 0) if positive else (lambda v: v <= 0)
    for value in values:
        if comparator(value):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def compute_basic_stats(df: pd.DataFrame) -> dict[str, float]:
    n_trades = len(df)
    if n_trades == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "best_trade": 0.0,
            "worst_trade": 0.0,
            "max_consec_wins": 0,
            "max_consec_losses": 0,
        }
    wins = df["pnl"] > 0
    losses = ~wins
    win_rate = float(wins.mean()) if n_trades else 0.0
    avg_win = float(df.loc[wins, "pnl"].mean()) if wins.any() else 0.0
    avg_loss = float(df.loc[losses, "pnl"].mean()) if losses.any() else 0.0
    expectancy = win_rate * avg_win + (1.0 - win_rate) * avg_loss
    positive_sum = float(df.loc[wins, "pnl"].sum())
    negative_sum = float(df.loc[losses, "pnl"].sum())
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0 else float("inf")
    best_trade = float(df["pnl"].max())
    worst_trade = float(df["pnl"].min())
    max_consec_wins = _max_streak(df["pnl"], positive=True)
    max_consec_losses = _max_streak(df["pnl"], positive=False)
    avg_expected_pnl = float(df["expected_pnl"].mean()) if "expected_pnl" in df else 0.0
    return {
        "n_trades": int(n_trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "expectancy": float(expectancy),
        "profit_factor": float(profit_factor),
        "best_trade": best_trade,
        "worst_trade": worst_trade,
        "max_consec_wins": max_consec_wins,
        "max_consec_losses": max_consec_losses,
        "avg_expected_pnl": avg_expected_pnl,
    }


def describe_returns(df: pd.DataFrame) -> pd.Series:
    if "return_pct" not in df:
        raise ValueError("Trades DataFrame must include 'return_pct'.")
    return df["return_pct"].describe()


def compute_calibration(
    df: pd.DataFrame, bins: Iterable[float]
) -> list[dict[str, float | int | str]]:
    if "probability_up" not in df or df.empty:
        return []
    bins_list = list(bins)
    if len(bins_list) < 2:
        raise ValueError("Need at least two bin edges for calibration.")
    probabilities = df["probability_up"].astype(float)
    filtered = df.loc[probabilities >= bins_list[0]].copy()
    if filtered.empty:
        return []
    categories = pd.cut(
        filtered["probability_up"],
        bins=bins_list,
        include_lowest=True,
        right=False,
    )
    calibration: list[dict[str, float | int | str]] = []
    grouped = filtered.groupby(categories, observed=True, dropna=True)
    for interval, group in grouped:
        if interval is None or group.empty:
            continue
        right_edge = min(interval.right, 1.0)
        right_bracket = "]" if interval.right >= 1.0 else ")"
        label = f"[{interval.left:.2f}, {right_edge:.2f}{right_bracket}"
        calibration.append(
            {
                "bin": label,
                "count": int(len(group)),
                "mean_prob": float(group["probability_up"].mean()),
                "emp_win_rate": float((group["pnl"] > 0).mean()),
                "mean_return": float(group["return_pct"].mean()),
            }
        )
    return calibration


def analyze_trades(df: pd.DataFrame, bins: Iterable[float] | None = None) -> dict[str, object]:
    stats = compute_basic_stats(df)
    returns = describe_returns(df)
    calibration_bins = list(bins) if bins is not None else PROBABILITY_BINS
    calibration = compute_calibration(df, calibration_bins)
    return {
        "basic_stats": stats,
        "return_distribution": returns,
        "calibration": calibration,
    }


def _print_basic_stats(stats: dict[str, float]) -> None:
    print("=== Basic stats ===")
    print(f"Trades: {stats['n_trades']}")
    print(f"Win rate: {stats['win_rate'] * 100:.2f}%")
    print(f"Avg win (pnl): {stats['avg_win']:.2f}")
    print(f"Avg loss (pnl): {stats['avg_loss']:.2f}")
    print(f"Expectancy (pnl): {stats['expectancy']:.2f}")
    print(f"Profit factor: {stats['profit_factor']:.2f}")
    print(f"Best trade: {stats['best_trade']:.2f} | Worst trade: {stats['worst_trade']:.2f}")
    print(
        f"Max consecutive wins: {stats['max_consec_wins']} | "
        f"Max consecutive losses: {stats['max_consec_losses']}"
    )
    print(f"Average expected pnl: {stats['avg_expected_pnl']:.2f}")


def _print_return_distribution(return_stats: pd.Series) -> None:
    print("\n=== Return distribution ===")
    print(return_stats.to_string())


def _print_calibration(calibration: list[dict[str, float | int | str]]) -> None:
    print("\n=== Calibration by probability bin ===")
    if not calibration:
        print("No trades in specified probability range.")
        return
    print("bin\tcount\tmean_prob\temp_win_rate\tmean_return")
    for entry in calibration:
        print(
            f"{entry['bin']}\t{entry['count']}\t"
            f"{entry['mean_prob']:.3f}\t{entry['emp_win_rate']:.3f}\t"
            f"{entry['mean_return']:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze HyprL backtest trade logs.")
    parser.add_argument("--trades", required=True, type=Path, help="Path to CSV exported by run_backtest.")
    parser.add_argument(
        "--bins",
        help='Optional probability bins (e.g. "0.4,0.5,0.6,0.7,0.8,0.9,1.01").',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_trades(args.trades)
    custom_bins = None
    if args.bins:
        custom_bins = [float(chunk.strip()) for chunk in args.bins.split(",") if chunk.strip()]
        if len(custom_bins) < 2:
            raise ValueError("Calibration bins require at least two edges.")
    results = analyze_trades(df, bins=custom_bins)
    _print_basic_stats(results["basic_stats"])  # type: ignore[arg-type]
    _print_return_distribution(results["return_distribution"])  # type: ignore[arg-type]
    _print_calibration(results["calibration"])  # type: ignore[arg-type]


if __name__ == "__main__":
    main()

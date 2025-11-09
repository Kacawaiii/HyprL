#!/usr/bin/env python
from __future__ import annotations

import argparse

from hyprl.backtest.runner import BacktestConfig, sweep_thresholds
from hyprl.risk.manager import RiskConfig

DEFAULT_THRESHOLDS = [0.4, 0.45, 0.5, 0.55, 0.6]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a HyprL backtest threshold sweep.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--period", help="yfinance period string, e.g. 1y.")
    parser.add_argument("--start", help="Explicit start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Explicit end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1h", help="yfinance interval (default: 1h).")
    parser.add_argument("--initial-balance", type=float, default=10_000.0, help="Starting capital.")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic model fits (default: 42).",
    )
    parser.add_argument(
        "--thresholds",
        help='Comma-separated thresholds (e.g. "0.4,0.45,0.5"); defaults to 0.4:0.6 step 0.05. '
        "0.40 is the recommended baseline for AAPL 1y.",
    )
    args = parser.parse_args()
    if not args.period and not (args.start and args.end):
        parser.error("Provide --period or both --start and --end.")
    return args


def parse_thresholds(threshold_arg: str | None) -> list[float]:
    if not threshold_arg:
        return DEFAULT_THRESHOLDS
    values = []
    for chunk in threshold_arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("No valid thresholds parsed from --thresholds argument.")
    return values


def main() -> None:
    args = parse_args()
    thresholds = parse_thresholds(args.thresholds)
    risk_cfg = RiskConfig(balance=args.initial_balance)
    config = BacktestConfig(
        ticker=args.ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_balance=args.initial_balance,
        threshold=thresholds[0],
        risk=risk_cfg,
        random_state=args.seed,
    )
    summaries = sweep_thresholds(config, thresholds)
    header = "Threshold | Return % | Sharpe | Max DD % | Trades | Win %"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        sharpe_display = f"{summary.sharpe_ratio:.2f}" if summary.sharpe_ratio is not None else "n/a"
        print(
            f"{summary.threshold:9.2f} | "
            f"{summary.total_return:8.2f} | "
            f"{sharpe_display:>6} | "
            f"{summary.max_drawdown * 100.0:8.2f} | "
            f"{summary.n_trades:6d} | "
            f"{summary.win_rate * 100.0:6.2f}"
        )


if __name__ == "__main__":
    main()

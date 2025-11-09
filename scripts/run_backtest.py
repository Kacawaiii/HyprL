#!/usr/bin/env python
from __future__ import annotations

import argparse

from hyprl.backtest.runner import BacktestConfig, run_backtest
from hyprl.risk.manager import RiskConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-ticker HyprL backtest.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--period", help="yfinance period string, e.g. 1y.")
    parser.add_argument("--start", help="Explicit start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Explicit end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1h", help="yfinance interval (default: 1h).")
    parser.add_argument("--initial-balance", type=float, default=10_000.0, help="Starting capital.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.4,
        help="Decision threshold for long vs short (default: 0.40).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the backtest model (default: 42).",
    )
    args = parser.parse_args()
    if not args.period and not (args.start and args.end):
        parser.error("Provide --period or both --start and --end.")
    return args


def main() -> None:
    args = parse_args()
    risk_cfg = RiskConfig(balance=args.initial_balance)
    config = BacktestConfig(
        ticker=args.ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_balance=args.initial_balance,
        threshold=args.threshold,
        risk=risk_cfg,
        random_state=args.seed,
    )
    result = run_backtest(config)
    total_return_pct = (
        (result.final_balance / args.initial_balance - 1.0) * 100.0 if args.initial_balance > 0 else 0.0
    )
    sharpe_display = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "n/a"
    print(f"Final balance: ${result.final_balance:,.2f}")
    print(f"Total return: {total_return_pct:.2f}%")
    print(f"Trades: {result.n_trades} | Win rate: {result.win_rate * 100.0:.2f}%")
    print(f"Max drawdown: {result.max_drawdown * 100.0:.2f}%")
    print(f"Sharpe ratio: {sharpe_display}")


if __name__ == "__main__":
    main()

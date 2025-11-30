#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from hyprl.backtest.runner import BacktestResult, run_backtest
from hyprl.execution.broker import PaperBroker
from hyprl.execution.engine import run_paper_trading_session
from hyprl.execution.logging import LiveLogger
from hyprl.execution.paper import (
    PaperBuildParams,
    build_backtest_config_from_row,
)
from hyprl.portfolio.core import compute_portfolio_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a paper trading session based on Supersearch output.")
    parser.add_argument("--tickers", required=True, help="Comma-separated list of tickers to trade.")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--config-csv", required=True, help="CSV exported by run_supersearch.")
    parser.add_argument("--config-index", type=int, default=0, help="Row index in the CSV to replay.")
    parser.add_argument("--engine", choices=["auto", "python", "native"], default="auto")
    parser.add_argument("--session-id", help="Optional session identifier.")
    parser.add_argument("--model-type", default="logistic")
    parser.add_argument("--calibration", default="none")
    parser.add_argument("--default-long-threshold", type=float, default=0.6)
    parser.add_argument("--default-short-threshold", type=float, default=0.4)
    parser.add_argument("--mode", choices=["replay"], default="replay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if not tickers:
        raise SystemExit("No tickers provided")
    try:
        df = pd.read_csv(args.config_csv)
    except FileNotFoundError:
        raise SystemExit(f"Config CSV not found: {args.config_csv}")
    if args.config_index < 0 or args.config_index >= len(df):
        raise SystemExit("config-index out of range")
    row = df.iloc[args.config_index]
    capital_share = args.initial_balance / len(tickers)
    params = PaperBuildParams(
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        model_type=args.model_type,
        calibration=args.calibration,
        default_long_threshold=args.default_long_threshold,
        default_short_threshold=args.default_short_threshold,
    )
    ticker_results: Dict[str, BacktestResult] = {}
    print(f"Paper trading mode={args.mode} engine={args.engine}")
    for ticker in tickers:
        cfg = build_backtest_config_from_row(row, ticker=ticker, capital_share=capital_share, params=params)
        ticker_results[ticker] = run_backtest(cfg)
    broker = PaperBroker(args.initial_balance)
    session_id = args.session_id or f"paper_{int(time.time())}"
    logger = LiveLogger(session_id)
    equity_series = run_paper_trading_session(ticker_results, broker, logger)
    if equity_series.empty:
        print("No trades executed.")
        return
    stats = compute_portfolio_stats(
        portfolio_equity=equity_series,
        initial_balance=args.initial_balance,
        seed=None,
        bootstrap_runs=128,
    )
    summary = {
        "session": session_id,
        "final_balance": stats["final_balance"],
        "return_pct": stats["return_pct"],
        "profit_factor": stats["profit_factor"],
        "sharpe": stats["sharpe"],
        "max_drawdown_pct": stats["max_drawdown_pct"],
    }
    print("Paper trading session complete:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

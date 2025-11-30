#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

os.environ.setdefault("YF_DISABLE_CURL", "1")

from hyprl import AnalysisConfig, AnalysisPipeline
from hyprl.configs import load_ticker_settings, load_long_threshold
from hyprl.labels.amplitude import LabelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intraday probability analysis combining SMA trend, RSI, and news sentiment."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to analyse.")
    parser.add_argument(
        "--interval",
        default="5m",
        help="Interval supported by yfinance (default: 5m).",
    )
    parser.add_argument(
        "--period",
        default="5d",
        help="Historical period to download (default: 5d).",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "xgboost"],
        default="logistic",
        help="Probability model backend (default: logistic).",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Optional probability calibration method.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to dump latest feature row and metrics as JSON.",
    )
    parser.add_argument(
        "--label-mode",
        choices=["binary", "amplitude"],
        default="binary",
        help="Labeling mode for on-the-fly training.",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=4,
        help="Look-ahead bars when label-mode=amplitude.",
    )
    parser.add_argument(
        "--label-threshold-pct",
        type=float,
        default=1.5,
        help="Percent move threshold for BIG_UP/BIG_DOWN (default 1.5).",
    )
    parser.add_argument(
        "--label-neutral-strategy",
        choices=["ignore", "keep"],
        default="ignore",
        help="Handling of NEUTRAL amplitude labels.",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=40,
        help="Minimum class samples when label-mode=amplitude.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_ticker_settings(args.ticker, args.interval) or {}
    threshold = load_long_threshold(settings)
    label_cfg = LabelConfig(
        mode=args.label_mode,
        horizon=args.label_horizon,
        threshold_pct=args.label_threshold_pct,
        neutral_strategy=args.label_neutral_strategy,
        min_samples_per_class=args.min_samples_per_class,
    )
    config = AnalysisConfig(
        ticker=args.ticker,
        interval=args.interval,
        period=args.period,
        model_type=args.model_type,
        calibration=args.calibration,
        threshold=threshold,
        label=label_cfg,
    )
    pipeline = AnalysisPipeline(config)
    result = pipeline.run()

    print(f"Ticker: {config.ticker}")
    print(f"News sentiment score: {result.news_sentiment:.3f}")
    print(f"Probability price increases next interval: {result.latest_probability_up:.3%}")
    print(f"Probability price decreases next interval: {result.latest_probability_down:.3%}")
    print(f"Predicted direction @ threshold {config.threshold:.2f}: {result.predicted_direction}")
    print("Latest feature snapshot:")
    latest_series = result.latest_row[
        [
            "price",
            "sma_short",
            "sma_long",
            "rsi_raw",
            "trend_ratio",
            "rsi_normalized",
            "volatility",
            "sentiment_score",
        ]
    ]
    print(latest_series.to_string())
    print("Backtest-style diagnostics:")
    for key, value in result.metrics.items():
        print(f"  {key}: {value:.4f}")
    if result.risk:
        print("Risk plan:")
        print(f"  Position size: {result.risk.position_size}")
        print(f"  Stop price: {result.risk.stop_price:.3f}")
        print(f"  Take profit: {result.risk.take_profit_price:.3f}")
        print(f"  R:R multiple: {result.risk.rr_multiple:.2f}")

    if args.output:
        payload = {
            "ticker": config.ticker,
            "news_sentiment": result.news_sentiment,
            "probability_up": result.latest_probability_up,
            "probability_down": result.latest_probability_down,
            "predicted_direction": result.predicted_direction,
            "metrics": result.metrics,
            "latest_row": result.latest_row.to_dict(),
            "risk": result.risk.to_dict() if result.risk else None,
        }
        args.output.write_text(json.dumps(payload, indent=2, default=_serialize_floats))
        print(f"\nWrote diagnostics to {args.output}")


def _serialize_floats(obj):
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return obj.isoformat()
    if isinstance(obj, float):
        return round(obj, 8)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


if __name__ == "__main__":
    main()

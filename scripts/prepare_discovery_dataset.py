#!/usr/bin/env python3
"""Download price data and prepare feature datasets for expansion candidates."""

from __future__ import annotations

import argparse
import pathlib
import re

import pandas as pd

from hyprl.data.market import MarketDataFetcher
from hyprl.features.equity_v2 import compute_equity_v2_features

_PERIOD_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([a-zA-Z]+)\s*$")


def _parse_period_days(period: str) -> float | None:
    match = _PERIOD_RE.match(period)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"d", "day", "days"}:
        return value
    if unit in {"wk", "w", "week", "weeks"}:
        return value * 7.0
    if unit in {"mo", "mos", "month", "months"}:
        return value * 30.0
    if unit in {"y", "yr", "year", "years"}:
        return value * 365.0
    return None


def _span_days(prices: pd.DataFrame) -> float:
    if prices.empty:
        return 0.0
    start = prices.index.min()
    end = prices.index.max()
    return float((end - start).total_seconds() / 86400.0)


def _print_dataset_stats(
    ticker: str,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    period: str,
) -> None:
    start = prices.index.min() if not prices.empty else None
    end = prices.index.max() if not prices.empty else None
    span_days = _span_days(prices)
    expected_days = _parse_period_days(period) if period else None
    ratio = span_days / expected_days if expected_days else None
    print(
        f"[DATA] {ticker} rows={len(prices)} features={len(features)} "
        f"span_days={span_days:.1f} start={start} end={end}"
    )
    if expected_days:
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        print(f"[DATA] expected_days={expected_days:.1f} span_ratio={ratio_str}")


def _validate_span(
    ticker: str,
    prices: pd.DataFrame,
    period: str,
    min_rows: int,
    min_span_days: float,
    min_span_ratio: float,
) -> None:
    if min_rows > 0 and len(prices) < min_rows:
        raise SystemExit(
            f"[ERROR] {ticker} fetched {len(prices)} bars (< min_rows={min_rows})."
        )
    span_days = _span_days(prices)
    if min_span_days > 0 and span_days < min_span_days:
        raise SystemExit(
            f"[ERROR] {ticker} span {span_days:.1f}d (< min_span_days={min_span_days})."
        )
    expected_days = _parse_period_days(period) if period else None
    if expected_days:
        ratio = span_days / expected_days if expected_days else 0.0
        if ratio < 0.7:
            print(f"[WARN] {ticker} span ratio low: {ratio:.2f} for period {period}.")
    if min_span_ratio > 0:
        if expected_days is None:
            raise SystemExit(
                f"[ERROR] min_span_ratio={min_span_ratio} requires parseable --period (got {period!r})."
            )
        ratio = span_days / expected_days if expected_days else 0.0
        if ratio < min_span_ratio:
            raise SystemExit(
                f"[ERROR] {ticker} span ratio {ratio:.2f} (< min_span_ratio={min_span_ratio})."
            )


def main():
    parser = argparse.ArgumentParser(description="Prepare expansion datasets.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol.")
    parser.add_argument("--period", default="5y", help="Historical period.")
    parser.add_argument("--interval", default="1h", help="Data interval.")
    parser.add_argument("--output", type=pathlib.Path, help="Output Parquet path.")
    parser.add_argument(
        "--min-rows",
        type=int,
        default=0,
        help="Fail if fetched OHLCV bars are fewer than this.",
    )
    parser.add_argument(
        "--min-span-days",
        type=float,
        default=0.0,
        help="Fail if price span days are fewer than this.",
    )
    parser.add_argument(
        "--min-span-ratio",
        type=float,
        default=0.0,
        help="Fail if span_days / expected_period_days is below this (requires parseable --period).",
    )
    args = parser.parse_args()

    print(f"[INFO] Fetching data for {args.ticker} ({args.period})...")
    fetcher = MarketDataFetcher(args.ticker)
    prices = fetcher.get_prices(interval=args.interval, period=args.period)

    if prices.empty:
        print(f"[ERROR] No data found for {args.ticker}")
        return

    print("[INFO] Computing features (discovery_v3 preset)...")
    features = compute_equity_v2_features(prices)
    if features.empty:
        raise SystemExit(f"[ERROR] No features produced for {args.ticker}; insufficient history.")
    features["close"] = prices["close"]  # Preserve close price for label computation
    features["ts"] = features.index.astype(int) // 10**6  # ms timestamp

    _print_dataset_stats(args.ticker, prices, features, args.period)
    _validate_span(
        args.ticker,
        prices,
        args.period,
        args.min_rows,
        args.min_span_days,
        args.min_span_ratio,
    )

    out_path = args.output or pathlib.Path(f"data/cache/{args.ticker.lower()}_1h_features_v3.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_path)
    print(f"[OK] Saved {len(features)} rows to {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from hyprl.backtest.runner import BacktestConfig, ThresholdSummary, sweep_thresholds
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_short_threshold,
    load_ticker_settings,
)
from hyprl.risk.manager import RiskConfig

DEFAULT_THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75]


def parse_tickers(arg: str) -> list[str]:
    tickers = [token.strip().upper() for token in arg.split(",") if token.strip()]
    if not tickers:
        raise ValueError("Provide at least one ticker via --tickers.")
    return tickers


def parse_thresholds(arg: str | None) -> list[float]:
    if not arg:
        return DEFAULT_THRESHOLDS
    values: list[float] = []
    for chunk in arg.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(float(chunk))
    if not values:
        raise ValueError("No valid thresholds parsed.")
    return values


def select_best_summary(summaries: Iterable[ThresholdSummary]) -> ThresholdSummary:
    summaries = list(summaries)
    if not summaries:
        raise ValueError("No threshold summaries provided.")

    def _is_positive(summary: ThresholdSummary) -> bool:
        sharpe = summary.sharpe_ratio or 0.0
        return summary.total_return > 0 and sharpe > 0

    def _score(summary: ThresholdSummary) -> tuple[float, float, float]:
        sharpe = summary.sharpe_ratio if summary.sharpe_ratio is not None else float("-inf")
        return (summary.alpha_return, summary.total_return, sharpe)

    positive = [s for s in summaries if _is_positive(s)]
    pool = positive if positive else summaries
    return max(pool, key=_score)


def compute_score(summary: ThresholdSummary, tradable: bool) -> float:
    if not tradable:
        return float("-inf")
    sharpe = summary.sharpe_ratio or 0.0
    pf = summary.profit_factor or 0.0
    alpha = summary.alpha_return
    return sharpe + (pf - 1.0) * 2.0 + alpha / 50.0


def build_config(
    ticker: str,
    args: argparse.Namespace,
    short_threshold: float,
    adaptive_overrides: dict[str, object],
) -> BacktestConfig:
    settings = load_ticker_settings(ticker, args.interval) or {}
    model_type = args.model_type or settings.get("model_type", "logistic")
    calibration = args.calibration or settings.get("calibration", "none")
    risk_profile = args.risk_profile or settings.get("default_risk_profile") or "normal"
    risk_params = get_risk_settings(settings, risk_profile)
    risk_cfg = RiskConfig(balance=args.initial_balance, **risk_params)
    overrides = adaptive_overrides or None
    adaptive_cfg = get_adaptive_config(settings, overrides)
    min_ev_multiple = float(settings.get("min_ev_multiple", 0.0))
    enable_trend_filter = bool(settings.get("enable_trend_filter", False))
    trend_long_min = float(settings.get("trend_long_min", 0.0))
    trend_short_min = float(settings.get("trend_short_min", 0.0))
    return BacktestConfig(
        ticker=ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_balance=args.initial_balance,
        long_threshold=args.thresholds_list[0],
        short_threshold=short_threshold,
        model_type=model_type,
        calibration=calibration,
        risk=risk_cfg,
        risk_profile=risk_profile,
        risk_profiles=settings.get("risk_profiles", {}),
        adaptive=adaptive_cfg,
        random_state=args.seed,
        min_ev_multiple=min_ev_multiple,
        enable_trend_filter=enable_trend_filter,
        trend_long_min=trend_long_min,
        trend_short_min=trend_short_min,
    )


def summarize_ticker(
    ticker: str,
    args: argparse.Namespace,
    thresholds: list[float],
) -> dict[str, float | int | str]:
    settings = load_ticker_settings(ticker, args.interval)
    tradable = settings.get("tradable", True)
    if not tradable:
        note = settings.get("note", "no edge on reference window.")
        print(f"[WARN] {ticker} config marked non-tradable: {note}")
    short_threshold = args.short_threshold
    if short_threshold is None:
        short_threshold = settings.get("short_threshold")
    if short_threshold is None:
        short_threshold = load_short_threshold(settings)
    adaptive_overrides: dict[str, object] = {"enable": False}
    adaptive_requested = bool(args.adaptive) and not args.force_non_adaptive
    if adaptive_requested:
        adaptive_overrides["enable"] = True
    if args.adaptive_lookback is not None:
        adaptive_overrides["lookback_trades"] = args.adaptive_lookback
    if args.adaptive_default_regime:
        adaptive_overrides["default_regime"] = args.adaptive_default_regime
    config = build_config(ticker, args, short_threshold, adaptive_overrides)
    summaries = sweep_thresholds(config, thresholds)
    best = select_best_summary(summaries)
    sharpe = best.sharpe_ratio if best.sharpe_ratio is not None else float("nan")
    return {
        "ticker": ticker,
        "tradable": tradable,
        "adaptive": bool(config.adaptive.enable),
        "short_threshold": short_threshold,
        "best_long_threshold": best.threshold,
        "strategy_return_pct": best.total_return,
        "strategy_ann_pct": (best.annualized_return or 0.0) * 100.0,
        "benchmark_return_pct": best.benchmark_return,
        "alpha_pct": best.alpha_return,
        "profit_factor": best.profit_factor if best.profit_factor is not None else float("nan"),
        "sharpe": sharpe,
        "max_drawdown_pct": best.max_drawdown * 100.0,
        "n_trades": best.n_trades,
        "win_rate_pct": best.win_rate * 100.0,
        "expectancy": best.expectancy,
        "score": compute_score(best, tradable),
        "best_regime": best.final_regime or "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep thresholds across a universe of tickers.")
    parser.add_argument("--tickers", required=True, help="Comma-separated tickers, e.g. AAPL,MSFT,SPY.")
    parser.add_argument("--period", help="yfinance period string, e.g. 1y.")
    parser.add_argument("--start", help="Explicit start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Explicit end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1h", help="yfinance interval (default: 1h).")
    parser.add_argument("--initial-balance", type=float, default=10_000.0, help="Starting capital per ticker.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the probability model.")
    parser.add_argument(
        "--short-threshold",
        type=float,
        help="Short-entry threshold; defaults to ticker config or 0.40.",
    )
    parser.add_argument(
        "--thresholds",
        help='Comma-separated long thresholds (e.g. "0.6,0.65,0.7"); defaults to 0.55–0.75 step 0.05.',
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to store the summary table.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "xgboost"],
        default=None,
        help="Probability model backend (default from config or logistic).",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default=None,
        help="Probability calibration method (default from config or none).",
    )
    parser.add_argument(
        "--risk-profile",
        help="Override ticker risk profile for all runs.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Force adaptive mode for all tickers (default: off).",
    )
    parser.add_argument(
        "--force-non-adaptive",
        action="store_true",
        help="Scoreboard mode: ignore adaptive presets and run baseline only.",
    )
    parser.add_argument(
        "--adaptive-lookback",
        type=int,
        help="Override adaptive lookback window (trades).",
    )
    parser.add_argument(
        "--adaptive-default-regime",
        help="Override adaptive default regime name.",
    )
    args = parser.parse_args()
    if not args.period and not (args.start and args.end):
        parser.error("Provide --period or both --start and --end.")
    args.thresholds_list = parse_thresholds(args.thresholds)
    args.ticker_list = parse_tickers(args.tickers)
    return args


def main() -> None:
    args = parse_args()
    rows: list[dict[str, float | int | str]] = []
    for ticker in args.ticker_list:
        try:
            rows.append(summarize_ticker(ticker, args, args.thresholds_list))
        except Exception as exc:  # pragma: no cover - surface failure per ticker
            print(f"[WARN] Failed to process {ticker}: {exc}")
    if not rows:
        print("No results to display.")
        return
    df = pd.DataFrame(rows)
    df_sorted = df.sort_values(by="score", ascending=False)
    rows_sorted = df_sorted.to_dict("records")
    header = (
        "Ticker | Tradable | Adaptive | Short | Best Long | Strat % | Ann % | Bench % | Alpha % | PF | Sharpe | Max DD % | Trades | Win % | Exp | Score | Best Regime"
    )
    print(header)
    print("-" * len(header))
    for row in rows_sorted:
        sharpe_display = (
            f"{row['sharpe']:.2f}" if isinstance(row["sharpe"], float) and pd.notna(row["sharpe"]) else "n/a"
        )
        tradable_flag = "YES" if row.get("tradable", True) else "NO"
        adaptive_flag = "YES" if row.get("adaptive") else "NO"
        print(
            f"{row['ticker']:>6} | "
            f"{tradable_flag:^8} | "
            f"{adaptive_flag:^8} | "
            f"{row['short_threshold']:6.2f} | "
            f"{row['best_long_threshold']:9.2f} | "
            f"{row['strategy_return_pct']:7.2f} | "
            f"{row['strategy_ann_pct']:6.2f} | "
            f"{row['benchmark_return_pct']:7.2f} | "
            f"{row['alpha_pct']:7.2f} | "
            f"{row['profit_factor']:6.2f} | "
            f"{sharpe_display:>6} | "
            f"{row['max_drawdown_pct']:9.2f} | "
            f"{row['n_trades']:6d} | "
            f"{row['win_rate_pct']:6.2f} | "
            f"{row['expectancy']:7.2f} | "
            f"{row['score']:7.3f} | "
            f"{row.get('best_regime', ''):>11}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved summary to {args.output}")


if __name__ == "__main__":
    main()

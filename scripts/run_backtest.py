#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from hyprl.backtest.runner import BacktestConfig, run_backtest
from hyprl.labels.amplitude import LabelConfig
from hyprl.risk.manager import RiskConfig
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
)
from hyprl.snapshots import save_snapshot, make_backup_zip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-ticker HyprL backtest.")
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g. AAPL.")
    parser.add_argument("--period", help="yfinance period string, e.g. 1y.")
    parser.add_argument("--start", help="Explicit start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Explicit end date (YYYY-MM-DD).")
    parser.add_argument("--interval", default="1h", help="yfinance interval (default: 1h).")
    parser.add_argument("--initial-balance", type=float, default=10_000.0, help="Starting capital.")
    parser.add_argument(
        "--long-threshold",
        type=float,
        help="Probability threshold to enter a long position (default: 0.60).",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        help="Probability threshold to enter a short position (default: ticker config or 0.40).",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "random_forest", "xgboost"],
        default=None,
        help="Probability model backend (default pulled from config or logistic).",
    )
    parser.add_argument(
        "--calibration",
        choices=["none", "platt", "isotonic"],
        default=None,
        help="Probability calibration method (default pulled from config or none).",
    )
    parser.add_argument(
        "--export-trades",
        type=Path,
        help="Optional path to CSV file where per-trade logs will be written.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the backtest model (default: 42).",
    )
    parser.add_argument(
        "--risk-profile",
        help="Override ticker risk profile (e.g. safe/normal/aggressive).",
    )
    parser.add_argument("--risk-pct", type=float, help="Override risk_pct.")
    parser.add_argument("--atr-multiplier", type=float, help="Override ATR multiplier.")
    parser.add_argument("--reward-multiple", type=float, help="Override reward multiple.")
    parser.add_argument("--min-position-size", type=int, help="Override minimum position size.")
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Enable adaptive risk/threshold adjustments.",
    )
    parser.add_argument("--adaptive-lookback", type=int, help="Adaptive lookback window (trades).")
    parser.add_argument(
        "--adaptive-default-regime",
        help="Override adaptive default regime (e.g. safe/normal/aggressive).",
    )
    parser.add_argument(
        "--adaptive-regimes-json",
        help="JSON dictionary overriding adaptive.regimes (expert use).",
    )
    parser.add_argument("--min-ev-multiple", type=float, help="Minimum EV multiple of risk (e.g., 0.2).")
    parser.add_argument(
        "--enable-trend-filter",
        action="store_true",
        help="Require rolling_return trend confirmation before trades.",
    )
    parser.add_argument("--trend-long-min", type=float, help="Minimum rolling_return for long trades.")
    parser.add_argument("--trend-short-min", type=float, help="Minimum |rolling_return| for short trades.")
    parser.add_argument(
        "--label-mode",
        choices=["binary", "amplitude"],
        default="binary",
        help="Labeling mode for model training (default: binary).",
    )
    parser.add_argument(
        "--label-horizon",
        type=int,
        default=4,
        help="Look-ahead horizon in bars when label-mode=amplitude.",
    )
    parser.add_argument(
        "--label-threshold-pct",
        type=float,
        default=1.5,
        help="Percent move required to tag BIG_UP/BIG_DOWN (default: 1.5).",
    )
    parser.add_argument(
        "--label-neutral-strategy",
        choices=["ignore", "keep"],
        default="ignore",
        help="Whether to discard NEUTRAL amplitude labels (default: ignore).",
    )
    parser.add_argument(
        "--min-samples-per-class",
        type=int,
        default=40,
        help="Minimum BIG_UP/BIG_DOWN samples required when label-mode=amplitude.",
    )
    args = parser.parse_args()
    if not args.period and not (args.start and args.end):
        parser.error("Provide --period or both --start and --end.")
    return args


def _resolve_thresholds(args: argparse.Namespace, settings: dict) -> tuple[float, float]:
    settings = settings or {}
    short_threshold = args.short_threshold
    if short_threshold is None:
        short_threshold = settings.get("short_threshold")
    if short_threshold is None:
        short_threshold = load_short_threshold(settings)

    long_threshold = args.long_threshold
    if long_threshold is None:
        long_threshold = settings.get("long_threshold")
    if long_threshold is None:
        long_threshold = load_long_threshold(settings, default=0.6)
    if short_threshold is None:
        short_threshold = 0.4
    return float(long_threshold), float(short_threshold)


def main() -> None:
    args = parse_args()
    settings = load_ticker_settings(args.ticker, args.interval)
    if settings.get("tradable") is False:
        note = settings.get("note", "no edge on reference window.")
        print(f"[WARN] {args.ticker} config marked non-tradable: {note}")
    long_threshold, short_threshold = _resolve_thresholds(args, settings)
    if not (0.0 < short_threshold < 1.0 and 0.0 < long_threshold < 1.0):
        raise ValueError("Thresholds must be between 0 and 1.")
    if long_threshold < short_threshold:
        raise ValueError("long-threshold must be greater than or equal to short-threshold.")
    model_type = args.model_type or settings.get("model_type", "logistic")
    calibration = args.calibration or settings.get("calibration", "none")
    risk_profile = args.risk_profile or settings.get("default_risk_profile") or "normal"
    risk_params = get_risk_settings(settings, risk_profile)
    if args.risk_pct is not None:
        risk_params["risk_pct"] = args.risk_pct
    if args.atr_multiplier is not None:
        risk_params["atr_multiplier"] = args.atr_multiplier
    if args.reward_multiple is not None:
        risk_params["reward_multiple"] = args.reward_multiple
    if args.min_position_size is not None:
        risk_params["min_position_size"] = args.min_position_size
    risk_cfg = RiskConfig(balance=args.initial_balance, **risk_params)
    adaptive_overrides: dict[str, object] = {}
    if args.adaptive:
        adaptive_overrides["enable"] = True
    if args.adaptive_lookback is not None:
        adaptive_overrides["lookback_trades"] = args.adaptive_lookback
    if args.adaptive_default_regime:
        adaptive_overrides["default_regime"] = args.adaptive_default_regime
    if args.adaptive_regimes_json:
        try:
            adaptive_overrides["regimes"] = json.loads(args.adaptive_regimes_json)
        except json.JSONDecodeError as exc:
            raise ValueError("--adaptive-regimes-json must be valid JSON") from exc
    adaptive_cfg = get_adaptive_config(settings, adaptive_overrides)
    min_ev_multiple = (
        args.min_ev_multiple if args.min_ev_multiple is not None else float(settings.get("min_ev_multiple", 0.0))
    )
    enable_trend_filter = bool(settings.get("enable_trend_filter", False))
    if args.enable_trend_filter:
        enable_trend_filter = True
    trend_long_min = (
        args.trend_long_min
        if args.trend_long_min is not None
        else float(settings.get("trend_long_min", 0.0))
    )
    trend_short_min = (
        args.trend_short_min
        if args.trend_short_min is not None
        else float(settings.get("trend_short_min", 0.0))
    )
    label_cfg = LabelConfig(
        mode=args.label_mode,
        horizon=args.label_horizon,
        threshold_pct=args.label_threshold_pct,
        neutral_strategy=args.label_neutral_strategy,
        min_samples_per_class=args.min_samples_per_class,
    )
    config = BacktestConfig(
        ticker=args.ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_balance=args.initial_balance,
        long_threshold=long_threshold,
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
        label=label_cfg,
    )
    result = run_backtest(config)
    strategy_return_pct = (
        (result.final_balance / args.initial_balance - 1.0) * 100.0 if args.initial_balance > 0 else 0.0
    )
    sharpe_display = f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "n/a"
    print(f"Strategy final balance: ${result.final_balance:,.2f}")
    print(f"Strategy return: {strategy_return_pct:.2f}%")
    print(f"Benchmark (buy & hold) final balance: ${result.benchmark_final_balance:,.2f}")
    print(f"Benchmark return: {result.benchmark_return:.2f}%")
    print(f"Trades: {result.n_trades} | Win rate: {result.win_rate * 100.0:.2f}%")
    print(f"Max drawdown: {result.max_drawdown * 100.0:.2f}%")
    print(f"Sharpe ratio: {sharpe_display}")
    if result.annualized_return is not None:
        print(f"Annualized return: {result.annualized_return * 100.0:.2f}%")
    if result.annualized_benchmark_return is not None:
        print(f"Benchmark annualized return: {result.annualized_benchmark_return * 100.0:.2f}%")
    if result.annualized_volatility is not None:
        print(f"Annualized volatility: {result.annualized_volatility * 100.0:.2f}%")
    if result.sortino_ratio is not None:
        print(f"Sortino ratio: {result.sortino_ratio:.2f}")
    if result.profit_factor is not None:
        print(f"Profit factor: {result.profit_factor:.2f}")
    print(f"Expectancy (pnl/trade): {result.expectancy:.2f}")
    print(f"Average R-multiple: {result.avg_r_multiple:.3f}")
    print(f"Average expected pnl: {result.avg_expected_pnl:.2f}")
    if result.brier_score is not None:
        print(f"Brier score: {result.brier_score:.4f}")
    if result.log_loss is not None:
        print(f"Log-loss: {result.log_loss:.4f}")
    print(
        f"Long trades: {result.long_trades} (win {result.long_win_rate * 100.0:.2f}%), "
        f"Short trades: {result.short_trades} (win {result.short_win_rate * 100.0:.2f}%)"
    )
    print(
        f"Final regime: {result.final_risk_profile} | "
        f"Long threshold: {result.final_long_threshold:.2f} | "
        f"Short threshold: {result.final_short_threshold:.2f} | "
        f"Adaptive changes: {result.adaptive_profile_changes}"
    )
    if result.regime_usage:
        total_trades = sum(result.regime_usage.values())
        print("Regime usage:")
        for regime, count in sorted(result.regime_usage.items(), key=lambda item: (-item[1], item[0])):
            pct = (count / total_trades * 100.0) if total_trades else 0.0
            print(f"  - {regime}: {count} trades ({pct:.1f}%)")
    if result.regime_transitions:
        transitions = ", ".join(f"{t['regime']}@{t['trade']}" for t in result.regime_transitions)
        print(f"Regime transitions: {transitions}")
    snapshot_zip = None
    if args.export_trades:
        trade_path = args.export_trades
        records = [asdict(trade) for trade in result.trades]
        df = pd.DataFrame.from_records(records)
        parent = trade_path.parent
        if parent and not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(trade_path, index=False)
        print(f"Exported {len(records)} trades to {trade_path}")
        snapshot_dir = save_snapshot(config, result, trades_path=trade_path)
        snapshot_zip = make_backup_zip(snapshot_dir)
        print(f"Snapshot saved to {snapshot_dir}")
        print(f"Backup zip: {snapshot_zip}")


if __name__ == "__main__":
    main()

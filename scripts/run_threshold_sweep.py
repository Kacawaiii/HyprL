#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from hyprl.backtest.runner import BacktestConfig, sweep_thresholds
from hyprl.risk.manager import RiskConfig
from hyprl.supercalc import prepare_supercalc_dataset, _build_signal_series  # type: ignore
from hyprl.native.supercalc import run_backtest_native, native_available  # type: ignore
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
)

DEFAULT_LONG_THRESHOLDS = [0.55, 0.6, 0.65, 0.7, 0.75]


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
        help="Override ticker risk profile when constructing configs.",
    )
    parser.add_argument(
        "--thresholds",
        help='Comma-separated long thresholds (e.g. "0.6,0.65,0.7"); defaults to 0.55:0.75 step 0.05 with ticker short baseline.'
        " AAPL 1h short baseline remains 0.40.",
    )
    parser.add_argument(
        "--short-threshold",
        type=float,
        help="Override short threshold (default pulled from ticker settings).",
    )
    parser.add_argument(
        "--engine",
        choices=["python", "native"],
        default="python",
        help="Engine to use for sweep (python refit or native fast).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast path (native dataset reuse) when engine=native.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        help="Optional model artifact path override (used for native fast path).",
    )
    parser.add_argument(
        "--feature-columns",
        type=str,
        help="Optional feature column JSON override (used for native fast path).",
    )
    args = parser.parse_args()
    if not args.period and not (args.start and args.end):
        parser.error("Provide --period or both --start and --end.")
    return args


def parse_thresholds(threshold_arg: str | None) -> list[float]:
    if not threshold_arg:
        return DEFAULT_LONG_THRESHOLDS
    path = Path(threshold_arg)
    values: list[float] = []
    if path.suffix.lower() in {".yaml", ".yml", ".json"} and not path.exists():
        raise ValueError(f"Threshold file not found: {path}")
    if path.exists():
        try:
            content = path.read_text()
            try:
                parsed = json.loads(content)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                values = [float(x) for x in parsed]
            else:
                for line in content.splitlines():
                    line = line.strip().strip(",")
                    if not line:
                        continue
                    try:
                        values.append(float(line))
                    except ValueError:
                        continue
        except Exception:
            values = []
    else:
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
    settings = load_ticker_settings(args.ticker, args.interval) or {}
    if settings.get("tradable") is False:
        note = settings.get("note", "no edge on reference window.")
        print(f"[WARN] {args.ticker} config marked non-tradable: {note}")
    short_threshold = args.short_threshold
    if short_threshold is None:
        short_threshold = settings.get("short_threshold")
    if short_threshold is None:
        short_threshold = load_short_threshold(settings)
    short_threshold = float(short_threshold)
    thresholds = parse_thresholds(args.thresholds)
    risk_profile = args.risk_profile or settings.get("default_risk_profile") or "normal"
    risk_params = get_risk_settings(settings, risk_profile)
    risk_cfg = RiskConfig(balance=args.initial_balance, **risk_params)
    model_type = args.model_type or settings.get("model_type", "logistic")
    calibration = args.calibration or settings.get("calibration", "none")
    adaptive_cfg = get_adaptive_config(settings, None)
    min_ev_multiple = float(settings.get("min_ev_multiple", 0.0))
    enable_trend_filter = bool(settings.get("enable_trend_filter", False))
    trend_long_min = float(settings.get("trend_long_min", 0.0))
    trend_short_min = float(settings.get("trend_short_min", 0.0))
    model_settings = settings.get("model", {}) or {}
    model_artifact = args.artifact or model_settings.get("artifact")
    model_feature_columns = None
    if args.feature_columns:
        model_feature_columns = json.loads(Path(args.feature_columns).read_text())
    else:
        model_feature_columns = model_settings.get("feature_columns") or []
    feature_preset = model_settings.get("preset")
    config = BacktestConfig(
        ticker=args.ticker,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        initial_balance=args.initial_balance,
        long_threshold=thresholds[0] if thresholds else load_long_threshold(settings),
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
        model_artifact_path=model_artifact,
        model_feature_columns=model_feature_columns,
        feature_preset=feature_preset,
    )
    dataset = None
    use_native = args.engine == "native"
    if use_native and args.fast:
        if not native_available():
            raise SystemExit("Native engine not available; build hyprl_supercalc or use --engine python.")
        dataset = prepare_supercalc_dataset(
            config,
            fast=True,
            use_cache_features=True,
            use_cache_prices=True,
        )
    summaries = sweep_thresholds(config, thresholds, engine=args.engine, dataset=dataset)
    header = "Threshold | Strat % | Ann % | Bench % | Alpha % | PF | Sharpe | Max DD % | Trades | Win % | Exp"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        sharpe_display = f"{summary.sharpe_ratio:.2f}" if summary.sharpe_ratio is not None else "n/a"
        print(
            f"{summary.threshold:9.2f} | "
            f"{summary.total_return:8.2f} | "
            f"{(summary.annualized_return or 0.0) * 100.0:6.2f} | "
            f"{summary.benchmark_return:8.2f} | "
            f"{summary.alpha_return:8.2f} | "
            f"{(summary.profit_factor or 0.0):6.2f} | "
            f"{sharpe_display:>6} | "
            f"{summary.max_drawdown * 100.0:8.2f} | "
            f"{summary.n_trades:6d} | "
            f"{summary.win_rate * 100.0:6.2f} | "
            f"{summary.expectancy:7.2f}"
        )


if __name__ == "__main__":
    main()

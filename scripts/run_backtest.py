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
    load_cli_config,
)
from hyprl.utils.strategy_id import StrategyIdentity, compute_strategy_id
from hyprl.snapshots import save_snapshot, make_backup_zip
from hyprl.supercalc import prepare_supercalc_dataset, _build_signal_series  # type: ignore
from hyprl.native.supercalc import run_backtest_native, native_available  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-ticker HyprL backtest.")
    parser.add_argument("--config", type=str, help="Path to YAML config file (e.g. configs/NVDA-1h.yaml).")
    parser.add_argument("--ticker", help="Ticker symbol, e.g. AAPL.")
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
    parser.add_argument(
        "--trailing-stop-activation",
        type=float,
        help="Activation threshold for trailing stop (in R).",
    )
    parser.add_argument(
        "--trailing-stop-distance",
        type=float,
        help="Trailing distance (in R).",
    )
    parser.add_argument(
        "--signal-log",
        type=Path,
        help="Optional CSV path to log per-bar signal diagnostics.",
    )
    parser.add_argument(
        "--feature-preset",
        help="Feature preset to use when building features (e.g., nvda_v2, equity_v2).",
    )
    parser.add_argument(
        "--model-artifact",
        type=Path,
        help="Optional ProbabilityModel artifact used for inference (shared between BT/live).",
    )
    parser.add_argument(
        "--engine",
        choices=["auto", "python", "native"],
        default="auto",
        help="Backtest engine to use (default: auto prefers native when available).",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print coarse timing breakdown (dataset/signals/native/export) for perf tuning.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Fast mode: skip signal logging and snapshot/backup to speed up iterations.",
    )
    parser.add_argument(
        "--cache-prices",
        action="store_true",
        help="Cache OHLCV fetch to data/cache/ohlcv_<ticker>_<interval>_<period>.parquet when period is used.",
    )
    parser.add_argument(
        "--cache-features",
        action="store_true",
        help="Cache feature frame to data/cache/features_<hash>.parquet for faster reruns.",
    )
    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args()
    args._defaults = defaults
    return args


def _assign_if_default(args: argparse.Namespace, defaults: dict, attr: str, value) -> None:
    if value is None:
        return
    if not hasattr(args, attr):
        return
    if getattr(args, attr) == defaults.get(attr):
        setattr(args, attr, value)


def _apply_backtest_cli_config(args: argparse.Namespace) -> None:
    config_path = getattr(args, "config", None)
    if not config_path:
        return
    cfg = load_cli_config(config_path)
    defaults = getattr(args, "_defaults", {})

    _assign_if_default(args, defaults, "ticker", cfg.get("ticker"))
    _assign_if_default(args, defaults, "period", cfg.get("period"))
    _assign_if_default(args, defaults, "start", cfg.get("start"))
    _assign_if_default(args, defaults, "end", cfg.get("end"))
    _assign_if_default(args, defaults, "interval", cfg.get("interval"))
    _assign_if_default(args, defaults, "initial_balance", cfg.get("initial_balance"))

    model_cfg = cfg.get("model", {}) or {}
    _assign_if_default(args, defaults, "model_type", model_cfg.get("type"))
    _assign_if_default(args, defaults, "model_artifact", model_cfg.get("artifact"))
    _assign_if_default(args, defaults, "calibration", model_cfg.get("calibration"))
    _assign_if_default(args, defaults, "seed", model_cfg.get("seed"))
    if not getattr(args, "model_feature_columns", None):
        try:
            setattr(args, "model_feature_columns", list(model_cfg.get("feature_columns") or []))
        except Exception:
            setattr(args, "model_feature_columns", [])

    thresholds_cfg = cfg.get("thresholds", {}) or {}
    _assign_if_default(args, defaults, "long_threshold", thresholds_cfg.get("long"))
    _assign_if_default(args, defaults, "short_threshold", thresholds_cfg.get("short"))

    risk_cfg = cfg.get("risk", {}) or {}
    _assign_if_default(args, defaults, "risk_pct", risk_cfg.get("risk_pct"))
    _assign_if_default(args, defaults, "atr_multiplier", risk_cfg.get("atr_multiplier"))
    _assign_if_default(args, defaults, "reward_multiple", risk_cfg.get("reward_multiple"))
    _assign_if_default(args, defaults, "min_position_size", risk_cfg.get("min_position_size"))

    trailing_cfg = cfg.get("trailing", {}) or {}
    if trailing_cfg.get("enabled"):
        _assign_if_default(args, defaults, "trailing_stop_activation", trailing_cfg.get("stop_activation"))
        _assign_if_default(args, defaults, "trailing_stop_distance", trailing_cfg.get("stop_distance"))

    _assign_if_default(args, defaults, "signal_log", cfg.get("signal_log"))
    replay_cfg = cfg.get("replay", {}) or {}
    _assign_if_default(args, defaults, "signal_log", replay_cfg.get("signal_log"))

    # Feature preset (model-level takes precedence over top-level)
    preset = cfg.get("feature_preset") or model_cfg.get("preset")
    if not hasattr(args, "feature_preset"):
        setattr(args, "feature_preset", None)
    if getattr(args, "feature_preset") is None and preset:
        setattr(args, "feature_preset", preset)


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
    import time

    t0 = time.time()
    args = parse_args()
    _apply_backtest_cli_config(args)
    if not args.ticker:
        raise SystemExit("Provide --ticker or specify 'ticker' in the config file.")
    if not args.period and not (args.start and args.end):
        raise SystemExit("Provide --period or both --start and --end (via CLI or config).")
    settings = load_ticker_settings(args.ticker, args.interval)
    if settings.get("tradable") is False:
        note = settings.get("note", "no edge on reference window.")
        print(f"[WARN] {args.ticker} config marked non-tradable: {note}")
    long_threshold, short_threshold = _resolve_thresholds(args, settings)
    if not (0.0 < short_threshold < 1.0 and 0.0 < long_threshold < 1.0):
        raise ValueError("Thresholds must be between 0 and 1.")
    if long_threshold < short_threshold:
        raise ValueError("long-threshold must be greater than or equal to short-threshold.")
    model_cfg = settings.get("model", {}) or {}
    model_type = args.model_type or model_cfg.get("type") or settings.get("model_type", "logistic")
    calibration = args.calibration or model_cfg.get("calibration") or settings.get("calibration", "none")
    model_feature_columns = getattr(args, "model_feature_columns", None) or model_cfg.get("feature_columns") or []
    feature_preset = getattr(args, "feature_preset", None) or model_cfg.get("preset") or settings.get("feature_preset")
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
    if args.trailing_stop_activation is not None:
        risk_params["trailing_stop_activation"] = args.trailing_stop_activation
    if args.trailing_stop_distance is not None:
        risk_params["trailing_stop_distance"] = args.trailing_stop_distance
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
        signal_log_path=str(args.signal_log) if args.signal_log else None,
        model_artifact_path=str(args.model_artifact) if args.model_artifact else None,
        model_feature_columns=list(model_feature_columns),
        feature_preset=feature_preset,
    )
    model_id = str(args.model_artifact) if args.model_artifact else model_type
    trailing_enabled = bool(risk_cfg.trailing_stop_activation or risk_cfg.trailing_stop_distance)
    identity = StrategyIdentity(
        tickers=(args.ticker,),
        interval=args.interval,
        model_id=model_id,
        label_mode=args.label_mode,
        label_horizon=int(args.label_horizon) if args.label_horizon is not None else None,
        feature_set_id=getattr(config, "feature_preset", None),
        risk_profile=risk_profile,
        risk_pct=risk_cfg.risk_pct,
        tp_multiple=risk_cfg.reward_multiple,
        sl_multiple=risk_cfg.atr_multiplier,
        trailing=trailing_enabled,
        execution_mode="backtest",
    )
    strategy_id = compute_strategy_id(identity)
    strategy_label = f"{args.ticker}_{args.interval}_{model_id}"
    engine = (args.engine or "auto").lower()
    use_native = engine in ("auto", "native") and native_available()
    result = None
    used_native = False
    if use_native:
        try:
            t_dataset_start = time.time()
            ds_timings: dict[str, float] = {}
            dataset = prepare_supercalc_dataset(
                config,
                timings=ds_timings,
                use_cache_prices=args.cache_prices,
                use_cache_features=args.cache_features,
                fast=args.fast,
            )
            t_dataset_end = time.time()
            signal, _, _ = _build_signal_series(dataset, config)
            t_signal_end = time.time()
            result = run_backtest_native(
                dataset.prices,
                signal,
                config,
                export_trades_path=args.export_trades,
                strategy_id=strategy_id,
                strategy_label=strategy_label,
                session_id=config.group or "backtest",
                source_type="backtest",
            )
            t_engine_end = time.time()
            used_native = True
        except Exception as exc:
            if engine == "native":
                raise
            print(f"[WARN] Native engine failed ({type(exc).__name__}: {exc}); falling back to python backtest.")
    if result is None:
        t_dataset_start = time.time()
        result = run_backtest(config)
        t_dataset_end = t_dataset_start
        t_signal_end = t_dataset_end
        t_engine_end = time.time()
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
    if result.exit_reason_counts:
        total_exits = sum(result.exit_reason_counts.values())
        print("Exit reasons:")
        for reason, count in sorted(result.exit_reason_counts.items(), key=lambda item: (-item[1], item[0])):
            pct = (count / total_exits * 100.0) if total_exits else 0.0
            print(f"  - {reason}: {count} ({pct:.1f}%)")
    snapshot_zip = None
    if args.export_trades:
        trade_path = args.export_trades
        if not used_native:
            rows = []
            def _normalize_exit_reason(reason: str) -> str:
                mapping = {
                    "stop_or_take": "stop_loss",
                    "stop_loss": "stop_loss",
                    "take_profit": "take_profit",
                    "trailing_stop": "trailing_stop",
                    "end_of_data": "time_exit",
                }
                return mapping.get(reason, reason or "unknown")
            for trade in result.trades:
                rows.append(
                    {
                        "entry_ts": trade.entry_timestamp,
                        "exit_ts": trade.exit_timestamp,
                        "symbol": args.ticker.upper(),
                        "side": trade.direction.upper(),
                        "qty": trade.position_size,
                        "entry_price": trade.entry_price,
                        "exit_price": trade.exit_price,
                        "pnl": trade.pnl,
                        "pnl_pct": trade.return_pct,
                        "exit_reason": _normalize_exit_reason(trade.exit_reason),
                        "strategy_id": strategy_id,
                        "strategy_label": strategy_label,
                        "source_type": "backtest",
                        "session_id": config.group or "backtest",
                    }
                )
            df = pd.DataFrame.from_records(rows)
            parent = trade_path.parent
            if parent and not parent.exists():
                parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(trade_path, index=False)
            print(f"Exported {len(rows)} trades to {trade_path}")
        else:
            if not trade_path.exists():
                print(f"[WARN] Native backtest indicated export_trades_path but file missing: {trade_path}")
        if trade_path.exists() and not args.fast:
            snapshot_dir = save_snapshot(config, result, trades_path=trade_path)
            snapshot_zip = make_backup_zip(snapshot_dir)
            print(f"Snapshot saved to {snapshot_dir}")
            print(f"Backup zip: {snapshot_zip}")
    t_end = time.time()
    if getattr(args, "timing", False):
        td = t_dataset_end - t_dataset_start if "t_dataset_end" in locals() else 0.0
        ts = t_signal_end - t_dataset_end if "t_signal_end" in locals() else 0.0
        te = t_engine_end - t_signal_end if "t_engine_end" in locals() else 0.0
        texport = t_end - t_engine_end if "t_engine_end" in locals() else 0.0
        print(
            f"[TIMING] dataset: {td:.2f}s | signals: {ts:.2f}s | engine: {te:.2f}s | export+snapshot: {texport:.2f}s | total: {t_end - t0:.2f}s"
        )
        if "ds_timings" in locals():
            print(
                "[TIMING-dataset] "
                f"ohlcv={ds_timings.get('ohlcv',0.0):.2f}s | "
                f"features={ds_timings.get('features',0.0):.2f}s | "
                f"model={ds_timings.get('model',0.0):.2f}s | "
                f"total={ds_timings.get('total_dataset',0.0):.2f}s"
            )


if __name__ == "__main__":
    main()

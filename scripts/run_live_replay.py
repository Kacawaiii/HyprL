#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Sequence

import pandas as pd

from hyprl.backtest.runner import BacktestConfig
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
    load_cli_config,
)
from hyprl.data.market import MarketDataFetcher
from hyprl.live.broker import PaperBrokerImpl, TradeRecordLive
from hyprl.live.risk import LiveRiskConfig, LiveRiskManager
from hyprl.live.strategy_engine import StrategyEngine
from hyprl.live.types import Bar
from hyprl.logging.signals import SignalTraceWriter
from hyprl.risk.manager import RiskConfig
from hyprl.risk.kelly import KellyParams, compute_kelly_risk_pct
from hyprl.strategy import prepare_feature_frame


class _ReplayClock:
    def __init__(self, current: date) -> None:
        self._current = current

    def set(self, current: date) -> None:
        self._current = current

    def __call__(self) -> date:  # pragma: no cover - trivial
        return self._current


def _default_trade_log(symbol: str, interval: str, tag: str) -> Path:
    sanitized = interval.replace("/", "-")
    safe_tag = tag.replace("/", "-")
    return Path("data") / "live" / f"trades_{symbol.upper()}_{sanitized}_{safe_tag}.csv"


def _bars_from_prices(symbol: str, df: pd.DataFrame) -> list[Bar]:
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol.upper(),
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
        )
    return bars


def _resolve_thresholds(args: argparse.Namespace, settings: dict) -> tuple[float, float]:
    short_threshold = args.short_threshold if args.short_threshold is not None else settings.get("short_threshold")
    if short_threshold is None:
        short_threshold = load_short_threshold(settings)
    long_threshold = args.long_threshold if args.long_threshold is not None else settings.get("long_threshold")
    if long_threshold is None:
        long_threshold = load_long_threshold(settings, default=0.6)
    if short_threshold is None:
        short_threshold = 0.4
    if not (0.0 < float(short_threshold) < 1.0 and 0.0 < float(long_threshold) < 1.0):
        raise ValueError("Thresholds must be between 0 and 1.")
    if float(long_threshold) < float(short_threshold):
        raise ValueError("long-threshold must be >= short-threshold.")
    return float(long_threshold), float(short_threshold)


def _build_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    if getattr(args, "config", None):
        settings = load_cli_config(args.config)
    else:
        settings = load_ticker_settings(args.symbol, args.interval)
    long_threshold, short_threshold = _resolve_thresholds(args, settings)
    model_type = args.model_type or settings.get("model_type", "logistic")
    calibration = args.calibration or settings.get("calibration", "none")
    risk_profile = args.risk_profile or settings.get("default_risk_profile") or "normal"
    risk_params = get_risk_settings(settings, risk_profile)
    overrides = {
        "risk_pct": args.risk_pct,
        "atr_multiplier": args.atr_multiplier,
        "reward_multiple": args.reward_multiple,
        "min_position_size": args.min_position_size,
        "trailing_stop_activation": args.trailing_stop_activation,
        "trailing_stop_distance": args.trailing_stop_distance,
    }
    for key, value in overrides.items():
        if value is not None:
            risk_params[key] = value
    risk_cfg = RiskConfig(balance=args.initial_balance, **risk_params)
    adaptive_cfg = get_adaptive_config(settings, {})
    min_ev_multiple = float(settings.get("min_ev_multiple", 0.0))
    enable_trend_filter = bool(settings.get("enable_trend_filter", False))
    trend_long_min = float(settings.get("trend_long_min", 0.0))
    trend_short_min = float(settings.get("trend_short_min", 0.0))
    dyn_cfg = (settings.get("risk", {}) or {}).get("dynamic_sizing", {}) or {}
    dyn_mode = args.dynamic_sizing_mode or dyn_cfg.get("type") or dyn_cfg.get("mode")
    dyn_base = args.dynamic_base_risk_pct or dyn_cfg.get("base_risk_pct")
    dyn_lookback = args.dynamic_lookback or dyn_cfg.get("lookback_trades") or dyn_cfg.get("lookback") or 50
    dyn_min_trades = args.dynamic_min_trades or dyn_cfg.get("min_trades") or 10
    dyn_max_mult = args.dynamic_max_multiplier or dyn_cfg.get("max_multiplier") or 2.0
    dyn_min_mult = args.dynamic_min_multiplier or dyn_cfg.get("min_multiplier") or 0.25
    feature_preset = (settings.get("features", {}) or {}).get("preset")
    model_cfg = (settings.get("model", {}) or {})
    if not feature_preset:
        feature_preset = model_cfg.get("preset")
    model_feature_columns = list(model_cfg.get("feature_columns", []) or [])
    mtf_frames: list[str] = []
    if settings.get("multi_timeframes_enabled"):
        frames = settings.get("multi_timeframes_frames")
        if isinstance(frames, str):
            mtf_frames = [tok.strip() for tok in frames.split(",") if tok.strip()]
        elif isinstance(frames, (list, tuple)):
            mtf_frames = [str(tok) for tok in frames]
    fusion_method = settings.get("fusion_method") or (settings.get("fusion") or {}).get("method") or "mean"
    fusion_weights = settings.get("fusion_weights") or (settings.get("fusion") or {}).get("weights") or {}
    return BacktestConfig(
        ticker=args.symbol,
        period=args.lookback,
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
        model_artifact_path=str(getattr(args, "model_artifact", None)) if getattr(args, "model_artifact", None) else None,
        dynamic_sizing_mode=dyn_mode,
        dynamic_sizing_base_pct=float(dyn_base) if dyn_base is not None else None,
        dynamic_sizing_lookback=int(dyn_lookback),
        dynamic_sizing_min_trades=int(dyn_min_trades),
        dynamic_sizing_max_multiplier=float(dyn_max_mult),
        dynamic_sizing_min_multiplier=float(dyn_min_mult),
        feature_preset=feature_preset,
        model_feature_columns=model_feature_columns,
        multi_timeframes=mtf_frames,
        fusion_method=fusion_method,
        fusion_weights=fusion_weights,
    )


def _finalize_open_positions(broker: PaperBrokerImpl, bar: Bar) -> None:
    for symbol in list(broker.positions.keys()):
        broker.close_position(symbol, "end_of_data", bar)


def _trade_stats(trades: Sequence[TradeRecordLive]) -> dict[str, float]:
    pnl_values = [float(t.pnl) for t in trades if t.pnl is not None]
    gross_profit = sum(v for v in pnl_values if v > 0)
    gross_loss = -sum(v for v in pnl_values if v < 0)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    wins = sum(1 for v in pnl_values if v > 0)
    win_rate = wins / len(pnl_values) if pnl_values else 0.0
    return {
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "total_pnl": sum(pnl_values),
    }


def _kelly_params_from_config(config: BacktestConfig) -> KellyParams | None:
    mode = (config.dynamic_sizing_mode or "").lower()
    if mode != "kelly":
        return None
    base_pct = config.dynamic_sizing_base_pct or config.risk.risk_pct
    if base_pct <= 0:
        return None
    return KellyParams(
        lookback_trades=config.dynamic_sizing_lookback,
        base_risk_pct=base_pct,
        min_trades=max(1, int(config.dynamic_sizing_min_trades)),
        max_multiplier=float(config.dynamic_sizing_max_multiplier),
        min_multiplier=float(config.dynamic_sizing_min_multiplier),
    )


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay historical bars through the live strategy stack.")
    parser.add_argument("--config", type=str, help="Path to YAML config file (e.g. configs/NVDA-1h.yaml).")
    parser.add_argument("--symbol")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--lookback", default="1y")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--long-threshold", type=float)
    parser.add_argument("--short-threshold", type=float)
    parser.add_argument("--model-type")
    parser.add_argument("--calibration", choices=["none", "platt", "isotonic"])
    parser.add_argument("--risk-profile")
    parser.add_argument("--risk-pct", type=float)
    parser.add_argument("--atr-multiplier", type=float)
    parser.add_argument("--reward-multiple", type=float)
    parser.add_argument("--min-position-size", type=float)
    parser.add_argument("--dynamic-sizing-mode")
    parser.add_argument("--dynamic-base-risk-pct", type=float)
    parser.add_argument("--dynamic-lookback", type=int)
    parser.add_argument("--dynamic-min-trades", type=int)
    parser.add_argument("--dynamic-max-multiplier", type=float)
    parser.add_argument("--dynamic-min-multiplier", type=float)
    parser.add_argument("--trailing-stop-activation", type=float)
    parser.add_argument("--trailing-stop-distance", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-bars", type=int, help="Override required history (defaults to indicator min history).")
    parser.add_argument("--max-bars", type=int, help="Optional cap on total bars (after warmup).")
    parser.add_argument("--trade-log", type=Path)
    parser.add_argument("--tag", default="replay")
    parser.add_argument("--max-daily-loss-pct", type=float, default=0.03)
    parser.add_argument("--max-position-notional-pct", type=float, default=0.2)
    parser.add_argument("--max-notional-per-trade", type=float, default=0.0)
    parser.add_argument("--max-gross-exposure-pct", type=float, default=1.0)
    parser.add_argument("--model-artifact", type=Path, help="Optional ProbabilityModel artifact path.")
    parser.add_argument(
        "--disable-risk-gating",
        action="store_true",
        help="Allow trades even if LiveRiskManager would block (diagnostic mode).",
    )
    parser.add_argument(
        "--signal-log",
        type=Path,
        help="Optional CSV path to log per-bar signal diagnostics during replay.",
    )
    parser.add_argument(
        "--parity-mode",
        action="store_true",
        help="Force feature parity with backtest and bypass live notional/exposure clamps.",
    )
    parser.add_argument("--verbose", action="store_true")
    defaults = {action.dest: action.default for action in parser._actions}
    args = parser.parse_args(args=args)
    args._defaults = defaults
    return args


def _assign_if_default(args: argparse.Namespace, defaults: dict, attr: str, value) -> None:
    if value is None or not hasattr(args, attr):
        return
    if getattr(args, attr) == defaults.get(attr):
        setattr(args, attr, value)


def _apply_replay_cli_config(args: argparse.Namespace) -> None:
    config_path = getattr(args, "config", None)
    if not config_path:
        return
    cfg = load_cli_config(config_path)
    defaults = getattr(args, "_defaults", {})

    _assign_if_default(args, defaults, "symbol", cfg.get("symbol") or cfg.get("ticker"))
    _assign_if_default(args, defaults, "interval", cfg.get("interval"))
    _assign_if_default(args, defaults, "lookback", cfg.get("lookback") or cfg.get("period"))
    _assign_if_default(args, defaults, "start", cfg.get("start"))
    _assign_if_default(args, defaults, "end", cfg.get("end"))
    _assign_if_default(args, defaults, "initial_balance", cfg.get("initial_balance"))

    model_cfg = cfg.get("model", {}) or {}
    _assign_if_default(args, defaults, "model_type", model_cfg.get("type"))
    _assign_if_default(args, defaults, "model_artifact", model_cfg.get("artifact"))
    _assign_if_default(args, defaults, "calibration", model_cfg.get("calibration"))
    _assign_if_default(args, defaults, "seed", model_cfg.get("seed"))

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

    dyn_cfg = (cfg.get("risk", {}) or {}).get("dynamic_sizing", {}) or {}
    if dyn_cfg.get("enabled"):
        _assign_if_default(args, defaults, "dynamic_sizing_mode", dyn_cfg.get("type") or dyn_cfg.get("mode"))
        _assign_if_default(args, defaults, "dynamic_base_risk_pct", dyn_cfg.get("base_risk_pct"))
        _assign_if_default(args, defaults, "dynamic_lookback", dyn_cfg.get("lookback_trades") or dyn_cfg.get("lookback"))
        _assign_if_default(args, defaults, "dynamic_min_trades", dyn_cfg.get("min_trades"))
        _assign_if_default(args, defaults, "dynamic_max_multiplier", dyn_cfg.get("max_multiplier"))
        _assign_if_default(args, defaults, "dynamic_min_multiplier", dyn_cfg.get("min_multiplier"))

    replay_cfg = cfg.get("replay", {}) or {}
    _assign_if_default(args, defaults, "parity_mode", replay_cfg.get("parity_mode"))
    _assign_if_default(args, defaults, "signal_log", replay_cfg.get("signal_log"))
    _assign_if_default(args, defaults, "tag", replay_cfg.get("tag"))
    _assign_if_default(args, defaults, "trade_log", replay_cfg.get("trade_log"))
def _required_history(config: BacktestConfig) -> int:
    return max(config.sma_long_window, config.rsi_window, config.atr_window) + 5


def main(args: list[str] | None = None) -> None:
    args = parse_args() if args is None else parse_args(args=args)
    _apply_replay_cli_config(args)
    if not args.symbol:
        raise SystemExit("Provide --symbol or set 'symbol' in the config file.")
    if not args.lookback and not (args.start and args.end):
        raise SystemExit("Provide --lookback or both --start and --end (via CLI or config).")
    if (args.start and not args.end) or (args.end and not args.start):
        raise SystemExit("Provide both --start and --end when specifying explicit dates.")
    config = _build_backtest_config(args)
    min_history = _required_history(config)
    warmup_bars = args.warmup_bars if args.warmup_bars is not None else min_history
    if warmup_bars < min_history:
        raise SystemExit(
            f"warmup-bars ({warmup_bars}) must be >= required history ({min_history})."
        )
    args.warmup_bars = warmup_bars
    trade_log_path = args.trade_log or _default_trade_log(args.symbol, args.interval, args.tag)
    trade_log_path.parent.mkdir(parents=True, exist_ok=True)
    signal_logger: SignalTraceWriter | None = None
    if args.signal_log:
        signal_logger = SignalTraceWriter(args.signal_log, source="replay", symbol=args.symbol)
    fetcher = MarketDataFetcher(args.symbol)
    period = args.lookback if not (args.start or args.end) else None
    prices = fetcher.get_prices(interval=args.interval, period=period, start=args.start, end=args.end)
    bars = _bars_from_prices(args.symbol, prices)
    if args.max_bars is not None and args.max_bars > 0:
        if args.max_bars <= warmup_bars:
            raise SystemExit("max-bars must be greater than warmup-bars")
        bars = bars[-args.max_bars :]
    if len(bars) <= warmup_bars:
        raise SystemExit("Not enough bars to satisfy warmup window.")

    warmup_cutoff = warmup_bars
    trace_callback = signal_logger.log if signal_logger else None
    feature_cache = None
    if args.parity_mode:
        try:
            feature_cache = prepare_feature_frame(prices, config)
        except Exception as exc:  # pragma: no cover - defensive parity fallback
            print(f"[WARN] Failed to precompute parity features: {exc}")
    engine = StrategyEngine(config=config, trace_callback=trace_callback, precomputed_features=feature_cache)
    engine.warmup(bars[:warmup_cutoff])
    broker = PaperBrokerImpl(
        cash=config.initial_balance,
        commission_pct=config.commission_pct,
        slippage_pct=config.slippage_pct,
        trade_log_path=trade_log_path,
    )
    clock = _ReplayClock(bars[warmup_cutoff - 1].timestamp.date())
    kelly_params = _kelly_params_from_config(config)
    risk_manager = LiveRiskManager(
        LiveRiskConfig(
            max_daily_loss_pct=args.max_daily_loss_pct,
            max_position_notional_pct=args.max_position_notional_pct,
            max_notional_per_trade=args.max_notional_per_trade,
            max_gross_exposure_pct=args.max_gross_exposure_pct,
        ),
        broker=broker,
        clock=clock,
        parity_mode=args.parity_mode,
    )

    replay_bars = bars[warmup_cutoff:]
    symbol_upper = args.symbol.upper()
    parity_bypass_counts: Counter[str] = Counter()
    try:
        for idx, bar in enumerate(replay_bars, start=warmup_cutoff):
            clock.set(bar.timestamp.date())
            broker.mark_to_market(bar)
            if broker.exited_on_bar(symbol_upper, bar.timestamp):
                if args.verbose:
                    print(f"[{idx}] {bar.timestamp} skip entry due to same-bar exit")
                continue
            open_positions = broker.get_positions()
            equity = broker.get_balance()
            dynamic_risk_pct = None
            if kelly_params is not None:
                closed_pnls = [float(t.pnl) for t in broker.trades if t.pnl is not None]
                dynamic_risk_pct = compute_kelly_risk_pct(closed_pnls, kelly_params)
            signals = engine.on_bar(bar, open_positions, equity=equity, risk_pct_override=dynamic_risk_pct)
            if args.verbose:
                print(f"[{idx}] {bar.timestamp} signals={len(signals)} equity={equity:.2f}")
            for signal in signals:
                decision = risk_manager.allow_trade(signal, price=bar.close)
                if decision.bypassed and args.parity_mode and decision.reason:
                    parity_bypass_counts[decision.reason] += 1
                    if args.verbose:
                        print(f"  parity bypass: {decision.reason}")
                if not decision.allowed:
                    if signal_logger:
                        signal_logger.log(
                            timestamp=bar.timestamp,
                            decision="risk_override" if args.disable_risk_gating else "risk_blocked",
                            reason=decision.reason or "unknown",
                            direction=signal.side,
                            probability_up=signal.probability_up,
                            long_threshold=signal.long_threshold,
                            short_threshold=signal.short_threshold,
                            expected_pnl=signal.expected_pnl,
                            position_size=signal.size,
                            risk_amount=signal.risk_amount,
                            equity=equity,
                        )
                    if not args.disable_risk_gating:
                        if args.verbose:
                            print(f"  blocked: {decision.reason}")
                        continue
                    if args.verbose:
                        print(f"  risk override (ignored): {decision.reason}")
                broker.submit_signal(signal, bar)
                if args.verbose:
                    print(f"  executed: {signal.reason} size={signal.size:.4f} stop={signal.stop_price:.4f}")

    finally:
        if signal_logger:
            signal_logger.close()

    _finalize_open_positions(broker, bars[-1])
    closed_trades = [trade for trade in broker.trades if not trade.is_open()]
    if not closed_trades:
        print("No trades executed during replay.")
        return

    stats = _trade_stats(closed_trades)
    exit_counts = Counter(trade.exit_reason for trade in closed_trades if trade.exit_reason)
    total_trades = len(closed_trades)

    print("Replay complete")
    print(f"  Trades           : {total_trades}")
    print(f"  Profit factor    : {stats['profit_factor']:.3f}")
    print(f"  Win rate         : {stats['win_rate']*100:.2f}%")
    print(f"  Total PnL        : {stats['total_pnl']:.2f}")
    print(f"  Final balance    : {broker.get_balance():.2f}")
    print(f"  Trade log        : {trade_log_path}")
    if exit_counts:
        mix = ", ".join(f"{k}:{exit_counts[k]/total_trades*100:.1f}%" for k in sorted(exit_counts))
        print(f"  Exit mix         : {mix}")
        if args.parity_mode and parity_bypass_counts:
            details = ", ".join(f"{reason}:{count}" for reason, count in parity_bypass_counts.items())
            print(f"  Parity bypasses  : {details}")


if __name__ == "__main__":
    main()

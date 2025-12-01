#!/usr/bin/env python3
from __future__ import annotations

"""Generate NVDA microscope fixtures (backtest vs replay) for parity tests."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import pandas as pd

from hyprl.backtest.runner import BacktestConfig, BacktestResult, run_backtest
from hyprl.configs import (
    get_adaptive_config,
    get_risk_settings,
    load_long_threshold,
    load_short_threshold,
    load_ticker_settings,
)
from hyprl.data.market import MarketDataFetcher
from hyprl.labels.amplitude import LabelConfig
from hyprl.live.broker import PaperBrokerImpl
from hyprl.live.risk import LiveRiskConfig, LiveRiskManager
from hyprl.live.strategy_engine import StrategyEngine
from hyprl.live.types import Bar
from hyprl.logging.signals import SignalTraceWriter
from hyprl.risk.manager import RiskConfig
from hyprl.strategy import prepare_feature_frame
from scripts import run_live_replay as replay_mod

PARITY_DIR = Path("data") / "parity"
PARITY_DIR.mkdir(parents=True, exist_ok=True)

SYMBOL = "NVDA"
INTERVAL = "1h"
START = "2025-01-15"
END = "2025-01-20"
INITIAL_BALANCE = 10_000.0
SEED = 42
MODEL_ARTIFACT_PATH: str | None = None

BT_LOG = PARITY_DIR / "nvda_backtest_signal_log_MICRO.csv"
REPLAY_LOG = PARITY_DIR / "nvda_replay_signal_log_MICRO.csv"
DIFF_LOG = PARITY_DIR / "nvda_signal_diff_MICRO.csv"
REPLAY_TRADES = PARITY_DIR / "nvda_replay_trades_MICRO.csv"


def _resolve_thresholds(settings: dict[str, Any]) -> tuple[float, float]:
    long_threshold = settings.get("long_threshold")
    short_threshold = settings.get("short_threshold")
    if long_threshold is None:
        long_threshold = load_long_threshold(settings, default=0.6)
    if short_threshold is None:
        short_threshold = load_short_threshold(settings)
    if short_threshold is None:
        short_threshold = 0.4
    if not (0.0 < float(short_threshold) < 1.0 and 0.0 < float(long_threshold) < 1.0):
        raise ValueError("Thresholds must be between 0 and 1.")
    if float(long_threshold) < float(short_threshold):
        raise ValueError("long-threshold must be >= short-threshold.")
    return float(long_threshold), float(short_threshold)


def _build_bt_config(signal_log_path: Path) -> BacktestConfig:
    settings = load_ticker_settings(SYMBOL, INTERVAL)
    long_threshold, short_threshold = _resolve_thresholds(settings)
    risk_profile = settings.get("default_risk_profile") or "normal"
    risk_params = get_risk_settings(settings, risk_profile)
    risk_cfg = RiskConfig(balance=INITIAL_BALANCE, **risk_params)
    adaptive_cfg = get_adaptive_config(settings, {})
    label_cfg = LabelConfig()
    return BacktestConfig(
        ticker=SYMBOL,
        start=START,
        end=END,
        interval=INTERVAL,
        initial_balance=INITIAL_BALANCE,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        model_type=settings.get("model_type", "logistic"),
        calibration=settings.get("calibration", "none"),
        risk=risk_cfg,
        risk_profile=risk_profile,
        risk_profiles=settings.get("risk_profiles", {}),
        adaptive=adaptive_cfg,
        random_state=SEED,
        min_ev_multiple=float(settings.get("min_ev_multiple", 0.0)),
        enable_trend_filter=bool(settings.get("enable_trend_filter", False)),
        trend_long_min=float(settings.get("trend_long_min", 0.0)),
        trend_short_min=float(settings.get("trend_short_min", 0.0)),
        label=label_cfg,
        signal_log_path=str(signal_log_path),
        model_artifact_path=MODEL_ARTIFACT_PATH,
    )


def run_backtest_with_signals() -> BacktestResult:
    config = _build_bt_config(BT_LOG)
    BT_LOG.parent.mkdir(parents=True, exist_ok=True)
    return run_backtest(config)


def _args_for_replay(signal_log: Path) -> SimpleNamespace:
    return SimpleNamespace(
        symbol=SYMBOL,
        interval=INTERVAL,
        lookback=None,
        start=START,
        end=END,
        initial_balance=INITIAL_BALANCE,
        long_threshold=None,
        short_threshold=None,
        model_type=None,
        calibration=None,
        risk_profile=None,
        risk_pct=None,
        atr_multiplier=None,
        reward_multiple=None,
        min_position_size=None,
        trailing_stop_activation=None,
        trailing_stop_distance=None,
        seed=SEED,
        warmup_bars=None,
        max_bars=None,
        trade_log=REPLAY_TRADES,
        tag="microscope",
        max_daily_loss_pct=0.03,
        max_position_notional_pct=0.2,
        max_gross_exposure_pct=1.0,
        disable_risk_gating=False,
        signal_log=signal_log,
        parity_mode=True,
        verbose=False,
        model_artifact=MODEL_ARTIFACT_PATH,
    )


@dataclass
class ReplayStats:
    trades: int
    profit_factor: float
    win_rate: float
    total_pnl: float


def _bars_from_prices(symbol: str, df: pd.DataFrame) -> list[Bar]:
    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row.get("volume", 0.0)),
            )
        )
    return bars


def _load_equity_map(path: Path) -> dict[pd.Timestamp, float]:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    eq_map: dict[pd.Timestamp, float] = {}
    for _, row in df.iterrows():
        eq = row.get("equity")
        if pd.isna(eq):
            continue
        ts = pd.Timestamp(row["timestamp"]).tz_convert("UTC") if row["timestamp"].tzinfo else pd.Timestamp(row["timestamp"]).tz_localize("UTC")
        eq_map[ts] = float(eq)
    return eq_map


def _run_replay_signal_log(parity_equity: dict[pd.Timestamp, float]) -> ReplayStats:
    args = _args_for_replay(REPLAY_LOG)
    config = replay_mod._build_backtest_config(args)  # type: ignore[attr-defined]
    min_history = replay_mod._required_history(config)  # type: ignore[attr-defined]
    warmup_bars = max(args.warmup_bars or min_history, min_history)
    args.warmup_bars = warmup_bars
    replay_log = SignalTraceWriter(args.signal_log, source="replay", symbol=SYMBOL)
    fetcher = MarketDataFetcher(args.symbol)
    prices = fetcher.get_prices(interval=args.interval, start=args.start, end=args.end)
    if prices.empty:
        raise RuntimeError("No price data for replay window")
    bars = _bars_from_prices(args.symbol.upper(), prices)
    if len(bars) <= warmup_bars:
        raise RuntimeError("Not enough bars for warmup window")
    feature_cache = prepare_feature_frame(prices, config)
    engine = StrategyEngine(
        config=config,
        trace_callback=replay_log.log,
        precomputed_features=feature_cache,
        parity_equity=parity_equity,
    )
    engine.warmup(bars[:warmup_bars])
    broker = PaperBrokerImpl(
        cash=config.initial_balance,
        commission_pct=config.commission_pct,
        slippage_pct=config.slippage_pct,
        trade_log_path=args.trade_log,
    )
    clock = replay_mod._ReplayClock(bars[warmup_bars - 1].timestamp.date())  # type: ignore[attr-defined]
    risk_manager = LiveRiskManager(
        LiveRiskConfig(
            max_daily_loss_pct=args.max_daily_loss_pct,
            max_position_notional_pct=args.max_position_notional_pct,
            max_gross_exposure_pct=args.max_gross_exposure_pct,
        ),
        broker=broker,
        clock=clock,
        parity_mode=True,
    )
    symbol_upper = args.symbol.upper()
    try:
        for idx in range(warmup_bars, len(bars)):
            bar = bars[idx]
            clock.set(bar.timestamp.date())
            broker.mark_to_market(bar)
            open_positions = broker.get_positions()
            signals = engine.on_bar(bar, open_positions, equity=broker.get_balance())
            for signal in signals:
                decision = risk_manager.allow_trade(signal, price=bar.close)
                if not decision.allowed:
                    replay_log.log(
                        timestamp=bar.timestamp,
                        decision="risk_blocked",
                        reason=decision.reason or "risk_manager_block",
                        direction=signal.side,
                        probability_up=signal.probability_up,
                        long_threshold=signal.long_threshold,
                        short_threshold=signal.short_threshold,
                        expected_pnl=signal.expected_pnl,
                        position_size=signal.size,
                        risk_amount=signal.risk_amount,
                        equity=broker.get_balance(),
                    )
                    continue
                broker.submit_signal(signal, bar)
    finally:
        replay_log.close()
    replay_mod._finalize_open_positions(broker, bars[-1])  # type: ignore[attr-defined]
    closed_trades = [trade for trade in broker.trades if not trade.is_open()]
    stats = replay_mod._trade_stats(closed_trades)  # type: ignore[attr-defined]
    return ReplayStats(
        trades=len(closed_trades),
        profit_factor=float(stats["profit_factor"]),
        win_rate=float(stats["win_rate"]),
        total_pnl=float(stats["total_pnl"]),
    )


def _merge_signal_logs(start_ts: str, end_ts: str) -> dict[str, float]:
    start_dt = pd.to_datetime(start_ts).tz_localize("UTC")
    end_dt = pd.to_datetime(end_ts).tz_localize("UTC")
    bt = pd.read_csv(BT_LOG, parse_dates=["timestamp"])
    replay = pd.read_csv(REPLAY_LOG, parse_dates=["timestamp"])
    bt = bt[(bt["timestamp"] >= start_dt) & (bt["timestamp"] <= end_dt)]
    replay = replay[(replay["timestamp"] >= start_dt) & (replay["timestamp"] <= end_dt)]
    bt = bt.rename(columns={col: f"{col}_bt" for col in bt.columns if col not in {"timestamp", "symbol"}})
    replay = replay.rename(columns={col: f"{col}_replay" for col in replay.columns if col not in {"timestamp", "symbol"}})
    merged = bt.merge(replay, on=["timestamp", "symbol"], how="outer", indicator=True).sort_values("timestamp")
    merged["accepted_bt"] = merged.get("decision_bt").fillna("").eq("emit")
    merged["accepted_replay"] = merged.get("decision_replay").fillna("").eq("emit")
    merged["bt_only"] = merged["accepted_bt"] & ~merged["accepted_replay"]
    merged["replay_only"] = merged["accepted_replay"] & ~merged["accepted_bt"]
    ordered_cols = [
        "timestamp",
        "symbol",
        "source_bt",
        "decision_bt",
        "reason_bt",
        "direction_bt",
        "probability_up_bt",
        "long_threshold_bt",
        "short_threshold_bt",
        "expected_pnl_bt",
        "min_ev_bt",
        "trend_ok_bt",
        "sentiment_ok_bt",
        "atr_value_bt",
        "position_size_bt",
        "risk_amount_bt",
        "equity_bt",
        "meta_bt",
        "accepted_bt",
        "source_replay",
        "decision_replay",
        "reason_replay",
        "direction_replay",
        "probability_up_replay",
        "long_threshold_replay",
        "short_threshold_replay",
        "expected_pnl_replay",
        "min_ev_replay",
        "trend_ok_replay",
        "sentiment_ok_replay",
        "atr_value_replay",
        "position_size_replay",
        "risk_amount_replay",
        "equity_replay",
        "meta_replay",
        "accepted_replay",
        "_merge",
        "bt_only",
        "replay_only",
    ]
    for col in ordered_cols:
        if col not in merged.columns:
            merged[col] = None
    merged = merged[ordered_cols]
    DIFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(DIFF_LOG, index=False)
    total_rows = len(merged)
    mismatch = int(merged["bt_only"].sum() + merged["replay_only"].sum())
    prob_cols = merged[["probability_up_bt", "probability_up_replay"]].dropna()
    prob_diff = (prob_cols["probability_up_bt"] - prob_cols["probability_up_replay"]).abs()
    max_prob_diff = float(prob_diff.max()) if not prob_diff.empty else 0.0
    mismatch_pct = (mismatch / total_rows * 100.0) if total_rows else 0.0
    return {
        "rows": total_rows,
        "bt_only": float(merged["bt_only"].sum()),
        "replay_only": float(merged["replay_only"].sum()),
        "mismatch_pct": mismatch_pct,
        "max_prob_diff": max_prob_diff,
    }


def main() -> None:
    global START, END, MODEL_ARTIFACT_PATH
    parser = argparse.ArgumentParser(description="Generate NVDA microscope parity fixtures.")
    parser.add_argument("--start", default=START)
    parser.add_argument("--end", default=END)
    parser.add_argument("--model-artifact", type=Path, help="Optional ProbabilityModel artifact shared by BT/replay.")
    args = parser.parse_args()
    START = args.start
    END = args.end
    MODEL_ARTIFACT_PATH = str(args.model_artifact) if args.model_artifact else None
    print(f"[parity] Running NVDA backtest {START} → {END}")
    bt_result = run_backtest_with_signals()
    print(
        f"[parity] Backtest trades={bt_result.n_trades} pf={bt_result.profit_factor or 0:.3f} "
        f"expectancy={bt_result.expectancy:.2f}"
    )
    print("[parity] Running NVDA replay")
    equity_map = _load_equity_map(BT_LOG)
    replay_stats = _run_replay_signal_log(equity_map)
    print(
        f"[parity] Replay trades={replay_stats.trades} pf={replay_stats.profit_factor:.3f} "
        f"pnl={replay_stats.total_pnl:.2f}"
    )
    metrics = _merge_signal_logs(START, END)
    print("[parity] Signal diff written to", DIFF_LOG)
    print(
        f"[parity] rows={metrics['rows']} bt_only={metrics['bt_only']:.0f} "
        f"replay_only={metrics['replay_only']:.0f} mismatch_pct={metrics['mismatch_pct']:.2f}% "
        f"max_prob_diff={metrics['max_prob_diff']:.6f}"
    )


if __name__ == "__main__":
    main()

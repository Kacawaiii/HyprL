#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from hyprl.backtest.runner import BacktestConfig
from hyprl.configs import get_adaptive_config, get_risk_settings, load_cli_config
from hyprl.data.market import MarketDataFetcher
from hyprl.live.broker import PaperBrokerImpl, TradeRecordLive, OpenTradeState
from hyprl.live.portfolio import PortfolioRunStats, build_portfolio_summary
from hyprl.live.types import Bar, Position, TradeSignal
from hyprl.risk.manager import RiskConfig
from hyprl.risk.portfolio import PortfolioRiskLimits, PortfolioRiskManager
from hyprl.risk.kelly import KellyParams, compute_kelly_risk_pct
from hyprl.risk.guards import RiskGuardConfig, RiskGuardMonitor
from hyprl.risk.sizing import clamp_position_size
from hyprl.execution.algos import ExecutionSlice, TWAPExecutor
from hyprl.strategy.core import AdaptiveState, decide_signals_on_bar, initial_regime_name, build_multiframe_feature_set
from hyprl.strategy import prepare_feature_frame, expected_trade_pnl

DEFAULT_SIGNAL_LOG = Path("live/logs/live_signals.jsonl")
DEFAULT_TRADES_JSON = Path("live/logs/live_trades.jsonl")
DEFAULT_TRADES_CSV = Path("live/logs/live_trades.csv")
DEFAULT_STATE_FILE = Path("live/state/live_state.json")
DEFAULT_SUMMARY_FILE = Path("live/dashboard/live_summary.json")
STATE_TEMPLATE = {"last_run": None, "last_run_map": {}, "cash": None, "equity": None, "positions": []}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the NVDA 1h live loop (XGB + trailing).")
    parser.add_argument("--config", required=True, help="YAML config file (e.g. configs/NVDA-1h.yaml).")
    parser.add_argument(
        "--config-list",
        nargs="+",
        help="Optional additional YAML configs for multi-ticker portfolio run.",
    )
    parser.add_argument("--state-file", type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--signal-log", type=Path, default=DEFAULT_SIGNAL_LOG)
    parser.add_argument("--trades-json", type=Path, default=DEFAULT_TRADES_JSON)
    parser.add_argument("--trade-log", type=Path, default=None, help="CSV trade log path (defaults to config/live).")
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY_FILE)
    parser.add_argument("--dry-run", action="store_true", help="Generate signals without mutating broker state.")
    parser.add_argument("--backfill", action="store_true", help="Replay the loop over a historical window.")
    parser.add_argument("--start", type=str, help="Start date for backfill (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date for backfill (YYYY-MM-DD)")
    args = parser.parse_args()
    if args.backfill and (not args.start or not args.end):
        parser.error("--backfill requires --start and --end.")
    return args


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def load_state(path: Path, initial_cash: float) -> Dict[str, Any]:
    if not path.exists():
        state = dict(STATE_TEMPLATE)
        state["cash"] = initial_cash
        state["equity"] = initial_cash
        return state
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    for key, default in STATE_TEMPLATE.items():
        data.setdefault(key, default)
    if data.get("cash") is None:
        data["cash"] = initial_cash
    if data.get("equity") is None:
        data["equity"] = float(data.get("cash", initial_cash))
    if "last_run_map" not in data:
        data["last_run_map"] = {}
    return data


def flatten_open_positions(
    broker: PaperBrokerImpl,
    bars_lookup: Dict[str, Bar],
    *,
    reason: ExitReason = "end_of_window",
) -> None:
    """Force-close all open positions at the provided last bar prices (backfill/replay only)."""
    symbols = list(broker.positions.keys())
    for symbol in symbols:
        bar = bars_lookup.get(symbol.upper())
        if bar is None:
            continue
        print(f"[DEBUG] flatten_open_positions symbol={symbol} exit_price={bar.close}")
        broker.mark_to_market(bar)
        broker.close_position(symbol, reason=reason, bar=bar, exit_price=bar.close)


def save_state(path: Path, state: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def build_portfolio_limits(cfg: Dict[str, Any]) -> PortfolioRiskLimits:
    live_cfg = cfg.get("live", {}) or {}
    portfolio_cfg = (live_cfg.get("portfolio_risk") or cfg.get("portfolio_risk") or {}) or {}

    def _opt_float(key: str, default: float | None) -> float | None:
        value = portfolio_cfg.get(key, default)
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    max_positions = portfolio_cfg.get("max_positions")
    return PortfolioRiskLimits(
        max_total_risk_pct=float(portfolio_cfg.get("max_total_risk_pct", 0.05)),
        max_ticker_risk_pct=_opt_float("max_ticker_risk_pct", 0.03),
        max_group_risk_pct=_opt_float("max_group_risk_pct", 0.04),
        max_positions=int(max_positions) if max_positions is not None else None,
    )


def build_backtest_config(cfg: Dict[str, Any]) -> BacktestConfig:
    ticker = cfg.get("ticker") or cfg.get("symbol")
    if not ticker:
        raise SystemExit("Config must define 'ticker'.")
    interval = cfg.get("interval", "1h")
    mtf_cfg = cfg.get("multi_timeframes") or {}
    fusion_cfg = cfg.get("fusion") or {}
    # Legacy flattened keys
    mtf_enabled_flat = bool(cfg.get("multi_timeframes_enabled", False))
    mtf_frames_flat = cfg.get("multi_timeframes_frames")
    if isinstance(mtf_frames_flat, str):
        mtf_frames_flat = [frame.strip() for frame in mtf_frames_flat.split(",") if frame.strip()]

    if isinstance(mtf_cfg, dict):
        mtf_enabled = bool(mtf_cfg.get("enabled", mtf_enabled_flat))
        mtf_frames = mtf_cfg.get("frames") or mtf_frames_flat or []
    else:
        mtf_enabled = bool(mtf_cfg) or mtf_enabled_flat
        mtf_frames = mtf_cfg if isinstance(mtf_cfg, list) else (mtf_frames_flat or [])

    fusion_method = str(fusion_cfg.get("method", cfg.get("fusion_method", "mean")))
    fusion_weights = fusion_cfg.get("weights", cfg.get("fusion_weights", {})) or {}
    risk_profile = cfg.get("risk_profile") or cfg.get("default_risk_profile") or "normal"
    risk_settings = get_risk_settings(cfg, risk_profile)
    group = cfg.get("group") or cfg.get("sector") or (cfg.get("meta", {}) or {}).get("group")
    risk_cfg = RiskConfig(
        balance=float(cfg.get("initial_balance", 10_000.0)),
        **risk_settings,
    )
    trailing_cfg = cfg.get("trailing", {}) or {}
    if trailing_cfg.get("enabled"):
        risk_cfg.trailing_stop_activation = trailing_cfg.get("stop_activation")
        risk_cfg.trailing_stop_distance = trailing_cfg.get("stop_distance")
    thresholds = cfg.get("thresholds", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    risk_cfg_block = cfg.get("risk", {}) or {}
    dynamic_cfg = risk_cfg_block.get("dynamic_sizing", {}) or cfg.get("dynamic_sizing", {}) or {}
    live_exec_cfg = cfg.get("live", {}) or {}
    live_risk_cfg = live_exec_cfg.get("risk", {}) or {}
    exec_cfg = cfg.get("execution", {}) or {}
    twap_cfg = exec_cfg.get("twap", {}) or {}
    guards_cfg = risk_cfg_block.get("guards", {}) or cfg.get("guards", {}) or {}
    features_cfg = cfg.get("features", {}) or {}
    return BacktestConfig(
        ticker=ticker,
        period=cfg.get("period", "1y"),
        start=cfg.get("start"),
        end=cfg.get("end"),
        interval=interval,
        initial_balance=float(cfg.get("initial_balance", 10_000.0)),
        long_threshold=float(thresholds.get("long", 0.55)),
        short_threshold=float(thresholds.get("short", 0.40)),
        model_type=model_cfg.get("type", "logistic"),
        calibration=model_cfg.get("calibration", "none"),
        risk=risk_cfg,
        risk_profile=risk_profile,
        risk_profiles=cfg.get("risk_profiles", {}),
        adaptive=get_adaptive_config(cfg, {}),
        random_state=model_cfg.get("seed", 42),
        min_ev_multiple=float(cfg.get("min_ev_multiple", 0.0)),
        enable_trend_filter=bool(cfg.get("enable_trend_filter", False)),
        trend_long_min=float(cfg.get("trend_long_min", 0.0)),
        trend_short_min=float(cfg.get("trend_short_min", 0.0)),
        model_artifact_path=model_cfg.get("artifact"),
        multi_timeframes=list(mtf_frames) if mtf_enabled else [],
        fusion_method=fusion_method,
        fusion_weights=dict(fusion_weights) if fusion_weights else {},
        group=group,
        feature_preset=features_cfg.get("preset"),
        dynamic_sizing_mode=str(dynamic_cfg.get("mode")) if dynamic_cfg.get("mode") else None,
        dynamic_sizing_base_pct=float(dynamic_cfg.get("base_risk_pct", 0.0)) or None,
        dynamic_sizing_lookback=int(dynamic_cfg.get("lookback", 50)),
        dynamic_sizing_min_trades=int(dynamic_cfg.get("min_trades", 10)),
        dynamic_sizing_max_multiplier=float(dynamic_cfg.get("max_multiplier", 2.0)),
        dynamic_sizing_min_multiplier=float(dynamic_cfg.get("min_multiplier", 0.25)),
        execution_algo=(exec_cfg.get("algo") or live_exec_cfg.get("execution_algo") or cfg.get("execution_algo")),
        execution_slices=int(twap_cfg.get("num_slices", live_exec_cfg.get("execution_slices", 4))),
        execution_horizon_sec=int(twap_cfg.get("total_seconds", live_exec_cfg.get("execution_horizon_sec", 300))),
        max_notional_per_trade=float(live_exec_cfg.get("max_notional_per_trade", cfg.get("max_notional_per_trade", 0.0))) or None,
        max_position_notional_pct=float(live_risk_cfg.get("max_position_notional_pct", cfg.get("max_position_notional_pct", 0.0))) or None,
        guards_config=guards_cfg,
    )


def build_bars_map(ticker: str, prices: pd.DataFrame) -> Dict[pd.Timestamp, Bar]:
    bars: Dict[pd.Timestamp, Bar] = {}
    for ts, row in prices.iterrows():
        bars[ts] = Bar(
            symbol=ticker.upper(),
            timestamp=ts.to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row.get("volume", 0.0)),
        )
    return bars


def log_signal(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def log_trade_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _load_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _serialize_open_position(symbol: str, pos: Position, record: TradeRecordLive, state: Optional[OpenTradeState]) -> Dict[str, Any]:
    entry = {
        "symbol": symbol,
        "side": pos.side,
        "size": pos.size,
        "avg_price": pos.avg_price,
        "unrealized_pnl": pos.unrealized_pnl,
        "trade": {
            "direction": record.direction,
            "entry_timestamp": record.entry_timestamp.isoformat(),
            "probability_up": record.probability_up,
            "threshold": record.threshold,
            "entry_price": record.entry_price,
            "stop_price": record.stop_price,
            "take_profit_price": record.take_profit_price,
            "trailing_stop_activation_price": record.trailing_stop_activation_price,
            "trailing_stop_distance_price": record.trailing_stop_distance_price,
            "position_size": record.position_size,
            "risk_amount": record.risk_amount,
            "expected_pnl": record.expected_pnl,
            "risk_profile": record.risk_profile,
            "effective_long_threshold": record.effective_long_threshold,
            "effective_short_threshold": record.effective_short_threshold,
            "regime_name": record.regime_name,
        },
        "open_state": None,
    }
    if state is not None:
        entry["open_state"] = {
            "stop_price": state.stop_price,
            "take_profit_price": state.take_profit_price,
            "trailing_activation": state.trailing_activation,
            "trailing_distance": state.trailing_distance,
            "current_stop_price": state.current_stop_price,
            "trailing_engaged": state.trailing_engaged,
            "highest_price": state.highest_price,
            "lowest_price": state.lowest_price,
        }
    return entry


def broker_to_state(broker: PaperBrokerImpl, *, last_run: str, last_run_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    positions_data: List[Dict[str, Any]] = []
    for symbol, pos in broker.positions.items():
        open_state = broker._open_states.get(symbol)
        record = open_state.record if open_state else broker._fallback_open_trade(symbol)
        if record is None:
            continue
        positions_data.append(_serialize_open_position(symbol, pos, record, open_state))
    state = {
        "last_run": last_run,
        "cash": broker.cash,
        "equity": broker.get_balance(),
        "positions": positions_data,
    }
    if last_run_map is not None:
        state["last_run_map"] = last_run_map
    return state


def broker_from_state(
    state: Dict[str, Any],
    *,
    initial_cash: float,
    commission_pct: float,
    slippage_pct: float,
    trade_log_path: Optional[Path],
) -> PaperBrokerImpl:
    cash = float(state.get("cash", initial_cash))
    broker = PaperBrokerImpl(
        cash=cash,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        trade_log_path=trade_log_path,
    )
    for entry in state.get("positions", []):
        symbol = entry.get("symbol")
        if not symbol:
            continue
        symbol = str(symbol).upper()
        position = Position(
            symbol=symbol,
            side=entry.get("side", "long"),
            size=float(entry.get("size", 0.0)),
            avg_price=float(entry.get("avg_price", 0.0)),
            unrealized_pnl=float(entry.get("unrealized_pnl", 0.0)),
        )
        broker.positions[symbol] = position
        trade_info = entry.get("trade", {})
        record = TradeRecordLive(
            symbol=symbol,
            direction=trade_info.get("direction", entry.get("side", "long")),
            entry_timestamp=_load_datetime(trade_info.get("entry_timestamp")) or datetime.now(timezone.utc),
            probability_up=float(trade_info.get("probability_up", 0.0)),
            threshold=float(trade_info.get("threshold", 0.0)),
            entry_price=float(trade_info.get("entry_price", position.avg_price)),
            stop_price=float(trade_info.get("stop_price", position.avg_price)),
            take_profit_price=float(trade_info.get("take_profit_price", position.avg_price)),
            trailing_stop_activation_price=trade_info.get("trailing_stop_activation_price"),
            trailing_stop_distance_price=trade_info.get("trailing_stop_distance_price"),
            position_size=float(trade_info.get("position_size", position.size)),
            risk_amount=float(trade_info.get("risk_amount", 0.0)),
            expected_pnl=float(trade_info.get("expected_pnl", 0.0)),
            risk_profile=trade_info.get("risk_profile"),
            effective_long_threshold=float(trade_info.get("effective_long_threshold", 0.0)),
            effective_short_threshold=float(trade_info.get("effective_short_threshold", 0.0)),
            regime_name=trade_info.get("regime_name"),
        )
        broker.trades.append(record)
        open_state_info = entry.get("open_state")
        if open_state_info:
            open_state = OpenTradeState(
                record=record,
                stop_price=float(open_state_info.get("stop_price", record.stop_price)),
                take_profit_price=float(open_state_info.get("take_profit_price", record.take_profit_price)),
                trailing_activation=open_state_info.get("trailing_activation"),
                trailing_distance=open_state_info.get("trailing_distance"),
                direction=record.direction,
            )
            open_state.current_stop_price = float(open_state_info.get("current_stop_price", open_state.stop_price))
            open_state.trailing_engaged = bool(open_state_info.get("trailing_engaged", False))
            open_state.highest_price = float(open_state_info.get("highest_price", record.entry_price))
            open_state.lowest_price = float(open_state_info.get("lowest_price", record.entry_price))
            broker._open_states[symbol] = open_state
        else:
            broker._open_states[symbol] = OpenTradeState(
                record=record,
                stop_price=float(record.stop_price),
                take_profit_price=float(record.take_profit_price),
                trailing_activation=record.trailing_stop_activation_price,
                trailing_distance=record.trailing_stop_distance_price,
                direction=record.direction,
            )
    return broker


def summarize_positions(broker: PaperBrokerImpl) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for pos in broker.get_positions():
        summary.append(
            {
                "symbol": pos.symbol,
                "side": pos.side,
                "size": pos.size,
                "avg_price": pos.avg_price,
                "unrealized_pnl": pos.unrealized_pnl,
            }
        )
    return summary


def compute_live_metrics(trade_log_path: Path, fallback_equity: float) -> Dict[str, float]:
    metrics = {
        "equity": fallback_equity,
        "net_pnl": 0.0,
        "pf": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "trade_count": 0,
        "win_rate": 0.0,
    }
    if not trade_log_path.exists():
        return metrics
    df = pd.read_csv(trade_log_path)
    if df.empty or "pnl" not in df.columns:
        return metrics
    closed = df.dropna(subset=["exit_timestamp"])
    if closed.empty:
        return metrics
    if "exit_timestamp" in closed.columns:
        try:
            closed = closed.sort_values("exit_timestamp")
        except Exception:
            pass
    pnl = closed["pnl"].astype(float)
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    metrics["pf"] = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    total_pnl = pnl.sum()
    metrics["net_pnl"] = total_pnl
    equity_end = float(fallback_equity)
    equity_start = equity_end - total_pnl
    equity_curve = equity_start + pnl.cumsum()
    equity_prev = equity_curve.shift(1, fill_value=equity_start)
    returns = pnl / equity_prev.replace(0, pd.NA)
    returns = returns.dropna()
    if len(returns) > 1 and returns.std(ddof=1) > 0:
        metrics["sharpe"] = returns.mean() / returns.std(ddof=1)
    metrics["equity"] = equity_curve.iloc[-1]
    peaks = equity_curve.cummax()
    drawdowns = peaks - equity_curve
    if not drawdowns.empty and peaks.max() > 0:
        metrics["max_drawdown"] = float((drawdowns.max() / peaks.max()) * 100.0)
    metrics["trade_count"] = len(closed)
    wins = (pnl > 0).sum()
    metrics["win_rate"] = float(wins / len(closed)) * 100.0
    return metrics


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


def compute_dynamic_risk_pct(broker: PaperBrokerImpl, config: BacktestConfig) -> float | None:
    params = _kelly_params_from_config(config)
    if params is None:
        return None
    closed_pnls = [float(t.pnl) for t in broker.trades if getattr(t, "pnl", None) is not None]
    return compute_kelly_risk_pct(closed_pnls, params)


def update_summary(summary_path: Path, payload: Dict[str, Any]) -> None:
    ensure_parent(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_timestamp(value: str, tz) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if tz is not None:
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    return ts


def select_feature_timestamps(index: pd.DatetimeIndex, start: Optional[str], end: Optional[str]) -> List[pd.Timestamp]:
    timestamps = index
    if start:
        start_ts = normalize_timestamp(start, index.tz)
        timestamps = timestamps[timestamps >= start_ts]
    if end:
        end_ts = normalize_timestamp(end, index.tz)
        timestamps = timestamps[timestamps <= end_ts]
    return list(timestamps)


def run_for_timestamp(
    ts: pd.Timestamp,
    *,
    backtest_config: BacktestConfig,
    features: pd.DataFrame,
    prices: pd.DataFrame,
    bars_map: Dict[pd.Timestamp, Bar],
    broker: PaperBrokerImpl,
    state: Dict[str, Any],
    signal_log: Path,
    trades_json: Path,
    trade_csv: Path,
    summary_file: Path,
    max_notional: float,
    dry_run: bool,
    force: bool,
    multiframe_features: Dict[str, pd.DataFrame] | None = None,
    fusion_method: str = "mean",
    fusion_weights: Dict[str, float] | None = None,
    portfolio_risk: PortfolioRiskManager | None = None,
    portfolio_stats: PortfolioRunStats | None = None,
    risk_guard: RiskGuardMonitor | None = None,
) -> Dict[str, Any]:
    bar = bars_map.get(ts)
    if bar is None:
        return state
    bar_hour = bar.timestamp.replace(minute=0, second=0, microsecond=0)
    last_run_map = state.get("last_run_map", {})
    ticker = backtest_config.ticker.upper()
    last_run_str = last_run_map.get(ticker) or state.get("last_run")
    if not force and last_run_str:
        last_run_dt = datetime.fromisoformat(last_run_str)
        last_hour = last_run_dt.replace(minute=0, second=0, microsecond=0)
        if bar_hour <= last_hour:
            print("[LIVE] Already processed this bar; skipping.")
            return state

    broker.mark_to_market(bar)

    try:
        current_idx = features.index.get_loc(ts)
    except KeyError:
        print(f"[LIVE] Timestamp {ts} missing in feature index; skipping.")
        return state

    min_history = max(backtest_config.sma_long_window, backtest_config.rsi_window, backtest_config.atr_window) + 5
    if current_idx < min_history:
        print(f"[LIVE] Insufficient history for {ts}; skipping.")
        return state
    risk_pct_override = compute_dynamic_risk_pct(broker, backtest_config)

    adaptive_state = AdaptiveState(regime_name=initial_regime_name(backtest_config.adaptive, backtest_config.risk_profile))
    decision = decide_signals_on_bar(
        config=backtest_config,
        features=features,
        prices=prices,
        current_idx=current_idx,
        equity=broker.get_balance(),
        adaptive_cfg=backtest_config.adaptive,
        adaptive_state=adaptive_state,
        trace_log=None,
        multiframe_features=multiframe_features,
        fusion_method=fusion_method,
        fusion_weights=fusion_weights,
        risk_pct_override=risk_pct_override,
    )

    timestamp_iso = datetime.now(timezone.utc).isoformat()
    signal_payload = {
        "timestamp": timestamp_iso,
        "symbol": backtest_config.ticker,
        "decision": "flat",
        "probability_up": None,
        "threshold": None,
        "size": 0.0,
        "reason": "",
        "equity": broker.get_balance(),
    }
    executed_payload: Optional[Dict[str, Any]] = None
    existing_symbols = {pos.symbol.upper() for pos in broker.get_positions()}
    portfolio_block_reason: Optional[str] = None
    twap_plan: Optional[List[Dict[str, float]]] = None

    if decision is not None:
        signal_payload.update(
            {
                "decision": decision.direction,
                "probability_up": decision.probability_up,
                "threshold": decision.threshold,
            }
        )
        if risk_guard and risk_guard.should_stop_trading():
            signal_payload["reason"] = risk_guard.state.reason or "guard_triggered"
        elif ticker in existing_symbols:
            signal_payload["reason"] = "position_open"
        else:
            position_size = float(decision.risk_plan.position_size)
            entry_price = float(decision.entry_price)
            risk_amount = float(decision.risk_plan.risk_amount)
            orig_position_size = position_size
            orig_risk_amount = risk_amount
            notional = abs(position_size * entry_price)
            clamped_size, clamped_risk = clamp_position_size(
                entry_price=entry_price,
                stop_price=decision.risk_plan.stop_price,
                position_size=position_size,
                equity=broker.get_balance(),
                max_notional=max_notional,
                max_notional_pct=backtest_config.max_position_notional_pct,
            )
            if clamped_size != position_size:
                position_size = clamped_size
                risk_amount = clamped_risk
                notional = abs(position_size * entry_price)
                decision.risk_plan.position_size = position_size
                decision.risk_plan.risk_amount = risk_amount
                decision.expected_pnl = expected_trade_pnl(decision.risk_plan, decision.probability_up)

            if portfolio_risk is not None and position_size > 0 and risk_amount > 0:
                risk_decision = portfolio_risk.evaluate(
                    ticker,
                    proposed_position_size=position_size,
                    risk_amount=risk_amount,
                    equity=broker.get_balance(),
                    broker=broker,
                        min_position_size=float(backtest_config.risk.min_position_size),
                )
                if not risk_decision.allowed:
                    portfolio_block_reason = risk_decision.reason or "portfolio_risk_blocked"
                    if portfolio_stats is not None:
                        portfolio_stats.record_blocked(
                            symbol=ticker,
                            reason=portfolio_block_reason,
                            risk_amount=risk_amount,
                        )
                else:
                    if risk_decision.adjusted:
                        position_size = risk_decision.position_size
                        risk_amount = risk_decision.risk_amount
                        notional = abs(position_size * entry_price)
                        decision.risk_plan.position_size = position_size
                        decision.risk_plan.risk_amount = risk_amount
                        decision.expected_pnl = expected_trade_pnl(decision.risk_plan, decision.probability_up)
                        if portfolio_stats is not None:
                            scale = risk_decision.position_size / max(orig_position_size, 1e-9)
                            portfolio_stats.record_scaled(
                                symbol=ticker,
                                scale=scale,
                        risk_before=orig_risk_amount,
                        risk_after=risk_amount,
                    )
            if portfolio_block_reason:
                signal_payload["reason"] = portfolio_block_reason
            elif position_size > 0:
                signal_payload["size"] = position_size
                signal_payload["reason"] = decision.direction
                slices = None
                if (backtest_config.execution_algo or "").lower() == "twap":
                    executor = TWAPExecutor(
                        num_slices=max(1, int(backtest_config.execution_slices)),
                        total_seconds=float(backtest_config.execution_horizon_sec),
                    )
                    slices = executor.build_slices(position_size)
                    twap_plan = [{"delay_sec": s.delay_sec, "qty": s.qty} for s in slices]
                    signal_payload["execution_algo"] = "twap"
                    signal_payload["twap_plan"] = twap_plan
                if not dry_run and not (risk_guard and risk_guard.should_stop_trading()):
                    slice_list = slices if slices is not None else [ExecutionSlice(delay_sec=0.0, qty=position_size)]
                    for idx_slice, s in enumerate(slice_list):
                        scale = s.qty / position_size if position_size > 0 else 0.0
                        trade_signal = TradeSignal(
                            symbol=ticker,
                            side=decision.direction,
                            size=s.qty,
                            reason=f"strategy:{decision.direction}:{decision.threshold:.3f}:slice{idx_slice}",
                            timestamp=bar.timestamp,
                            probability_up=decision.probability_up,
                            threshold=decision.threshold,
                            entry_price=entry_price,
                            expected_pnl=decision.expected_pnl * scale,
                            risk_amount=decision.risk_plan.risk_amount * scale,
                            long_threshold=decision.long_threshold,
                            short_threshold=decision.short_threshold,
                            stop_price=decision.risk_plan.stop_price,
                            take_profit_price=decision.risk_plan.take_profit_price,
                            trailing_stop_activation_price=decision.risk_plan.trailing_stop_activation_price,
                            trailing_stop_distance_price=decision.risk_plan.trailing_stop_distance_price,
                            risk_profile=decision.profile_name,
                            regime_name=decision.regime_name,
                        )
                        broker.submit_signal(trade_signal, bar)
                        if portfolio_stats is not None:
                            portfolio_stats.record_executed(
                                symbol=ticker,
                                expected_pnl=trade_signal.expected_pnl,
                                probability_up=trade_signal.probability_up,
                                direction=trade_signal.side,
                                threshold=trade_signal.threshold,
                            )
                        executed_payload = {
                            "timestamp": timestamp_iso,
                            "symbol": trade_signal.symbol,
                            "side": trade_signal.side,
                            "size": trade_signal.size,
                            "entry_price": trade_signal.entry_price,
                            "reason": trade_signal.reason,
                            "execution_algo": "twap" if slices is not None else None,
                            "twap_plan": twap_plan,
                        }
            else:
                signal_payload["reason"] = "position_size_zero"
    else:
        signal_payload["reason"] = "no_signal"

    log_signal(signal_log, signal_payload)
    if executed_payload is not None and not dry_run:
        log_trade_json(trades_json, executed_payload)

    if not dry_run:
        last_run_map[ticker] = ts.isoformat()
        new_state = broker_to_state(broker, last_run=ts.isoformat(), last_run_map=last_run_map)
    else:
        new_state = dict(state)
        new_state["last_run"] = state.get("last_run")
        new_state["last_run_map"] = last_run_map

    metrics = compute_live_metrics(trade_csv, fallback_equity=broker.get_balance())
    summary_payload = {
        "last_update": timestamp_iso,
        "equity": metrics["equity"],
        "net_pnl": metrics["net_pnl"],
        "pf": metrics["pf"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "trades": metrics["trade_count"],
        "win_rate": metrics["win_rate"],
        "open_positions": len(broker.get_positions()),
        "positions": summarize_positions(broker),
    }
    update_summary(summary_file, summary_payload)
    return new_state


def run_backfill(
    args: argparse.Namespace,
    cfg: Dict[str, Any],
    backtest_config: BacktestConfig,
    state: Dict[str, Any],
    *,
    prices: pd.DataFrame,
    features: pd.DataFrame,
    bars_map: Dict[pd.Timestamp, Bar],
    broker: PaperBrokerImpl,
    max_notional: float,
    trade_log_path: Path,
    multiframe_features: Dict[pd.Timestamp, pd.DataFrame] | Dict[str, pd.DataFrame] | None = None,
    portfolio_risk: PortfolioRiskManager | None = None,
    portfolio_stats: PortfolioRunStats | None = None,
    risk_guard: RiskGuardMonitor | None = None,
) -> Dict[str, Any]:
    timestamps = select_feature_timestamps(features.index, args.start, args.end)
    for ts in timestamps:
        state = run_for_timestamp(
            ts,
            backtest_config=backtest_config,
            features=features,
            prices=prices,
            bars_map=bars_map,
            broker=broker,
            state=state,
            signal_log=args.signal_log,
            trades_json=args.trades_json,
            trade_csv=trade_log_path,
            summary_file=args.summary_file,
            max_notional=max_notional,
            dry_run=args.dry_run,
            force=True,
            multiframe_features=multiframe_features,
            fusion_method=backtest_config.fusion_method,
            fusion_weights=backtest_config.fusion_weights,
            portfolio_risk=portfolio_risk,
            portfolio_stats=portfolio_stats,
            risk_guard=risk_guard,
        )
        if portfolio_stats is not None:
            portfolio_stats.record_risk(
                equity_ref=backtest_config.initial_balance,
                open_risk=broker.get_open_risk_amounts(),
            )
    return state


def run_portfolio(args: argparse.Namespace, config_paths: List[str]) -> None:
    if not config_paths:
        raise SystemExit("No configs provided for portfolio run.")
    loaded_configs: List[tuple[Dict[str, Any], BacktestConfig]] = []
    for path in config_paths:
        cfg = load_cli_config(path)
        bt_cfg = build_backtest_config(cfg)
        loaded_configs.append((cfg, bt_cfg))
    # Initialize state/broker from the first config balance as default.
    initial_balance = loaded_configs[0][1].initial_balance
    state = load_state(args.state_file, initial_balance)
    live_cfg0 = loaded_configs[0][0].get("live", {}) or {}
    commission_pct = float(live_cfg0.get("commission_pct", 0.0005))
    slippage_pct = float(live_cfg0.get("slippage_pct", 0.0005))
    portfolio_trade_log = Path(args.trade_log) if args.trade_log else Path(
        live_cfg0.get("trade_log", "live/logs/live_trades_portfolio.csv")
    )
    portfolio_signal_log = Path(args.signal_log) if args.signal_log else Path(
        live_cfg0.get("signal_log", "live/logs/live_signals_portfolio.jsonl")
    )
    portfolio_trades_json = Path(args.trades_json) if args.trades_json else Path(
        live_cfg0.get("trades_json", "live/logs/live_trades_portfolio.json")
    )
    portfolio_summary = Path(args.summary_file) if args.summary_file else Path(
        live_cfg0.get("summary_file", "live/dashboard/live_summary_portfolio.json")
    )
    broker = broker_from_state(
        state,
        initial_cash=initial_balance,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        trade_log_path=None if args.dry_run else portfolio_trade_log,
    )
    portfolio_limits = build_portfolio_limits(loaded_configs[0][0])
    group_map = {
        bt_cfg.ticker.upper(): (bt_cfg.group or bt_cfg.ticker)
        for _, bt_cfg in loaded_configs
    }
    portfolio_risk = PortfolioRiskManager(portfolio_limits, group_map=group_map)
    run_stats = PortfolioRunStats()
    session_ts = datetime.now(timezone.utc)
    portfolio_summary_file = Path(live_cfg0.get("portfolio_summary_dir", "live/logs/portfolio")) / session_ts.strftime(
        "%Y-%m-%d"
    ) / "summary.json"
    guard_config = loaded_configs[0][1].guards_config or {}
    guard_cfg = guard_config.get("guards", guard_config) if isinstance(guard_config, dict) else {}
    guard_monitor = None
    if guard_cfg.get("enabled"):
        guard_monitor = RiskGuardMonitor(
            RiskGuardConfig(
                max_drawdown_pct=float(guard_cfg.get("max_drawdown_pct", 0.2)),
                min_pf=float(guard_cfg.get("min_pf", 1.0)),
                lookback_trades=int(guard_cfg.get("lookback_trades", 50)),
                max_consec_losses=guard_cfg.get("max_consec_losses"),
            ),
            equity_start=initial_balance,
        )

    prepared = []
    for cfg, backtest_config in loaded_configs:
        portfolio_risk.register_symbol(backtest_config.ticker, backtest_config.group)
        live_cfg = cfg.get("live", {}) or {}
        max_notional = float(live_cfg.get("max_notional_per_trade", 50_000))
        commission_pct = float(live_cfg.get("commission_pct", 0.0005))
        slippage_pct = float(live_cfg.get("slippage_pct", 0.0005))
        trade_log_path = portfolio_trade_log
        signal_log_path = portfolio_signal_log
        trades_json_path = portfolio_trades_json
        summary_path = portfolio_summary

        fetcher = MarketDataFetcher(backtest_config.ticker)
        prices = fetcher.get_prices(
            interval=backtest_config.interval,
            period=backtest_config.period if not args.backfill else None,
            start=args.start if args.backfill else backtest_config.start,
            end=args.end if args.backfill else backtest_config.end,
        )
        prices = prices.sort_index()
        if prices.empty:
            print(f"[LIVE] No price data for {backtest_config.ticker}; skipping.")
            continue
        min_history = max(
            backtest_config.atr_window,
            backtest_config.sma_long_window,
            backtest_config.rsi_window,
            72,  # nvda_v2 feature preset uses ATR72
        ) + 5
        if len(prices) < min_history:
            raise SystemExit(
                f"Not enough bars ({len(prices)}) for required history ({min_history}). "
                f"Use an earlier --start or a wider backfill window."
            )
        multiframe_features = None
        if backtest_config.multi_timeframes:
            multiframe_features = build_multiframe_feature_set(
                prices, backtest_config, backtest_config.multi_timeframes
            )
            features = multiframe_features.get("base", pd.DataFrame())
        else:
            features = prepare_feature_frame(prices, backtest_config)
        if features.empty:
            print(f"[LIVE] Empty feature frame for {backtest_config.ticker}; skipping.")
            continue
        bars_map = build_bars_map(backtest_config.ticker, prices)
        last_bar = bars_map[max(bars_map)] if bars_map else None
        timestamps = (
            select_feature_timestamps(features.index, args.start, args.end)
            if args.backfill
            else [features.index[-1]]
        )
        prepared.append(
            {
                "cfg": cfg,
                "backtest_config": backtest_config,
                "prices": prices,
                "features": features,
                "bars_map": bars_map,
                "multiframe_features": multiframe_features,
                "max_notional": max_notional,
                "trade_log_path": trade_log_path,
                "signal_log_path": signal_log_path,
                "trades_json_path": trades_json_path,
                "summary_path": summary_path,
                "timestamps": timestamps,
                "timestamp_set": set(timestamps),
                "last_bar": last_bar,
            }
        )

    if not prepared:
        print("[LIVE] No prepared tickers for portfolio run; exiting.")
        return

    if args.backfill:
        all_ts = sorted({ts for item in prepared for ts in item["timestamps"]})
        for ts in all_ts:
            for item in prepared:
                if ts not in item["timestamp_set"]:
                    continue
                # Record risk usage before processing this bar to capture open exposure.
                open_risk = broker.get_open_risk_amounts()
                run_stats.record_risk(
                    equity_ref=initial_balance,
                    open_risk=open_risk,
                )
                state = run_for_timestamp(
                    ts,
                    backtest_config=item["backtest_config"],
                    features=item["features"],
                    prices=item["prices"],
                    bars_map=item["bars_map"],
                    broker=broker,
                    state=state,
                    signal_log=item["signal_log_path"],
                    trades_json=item["trades_json_path"],
                    trade_csv=item["trade_log_path"],
                    summary_file=item["summary_path"],
                    max_notional=item["max_notional"],
                    dry_run=args.dry_run,
                    force=True,
                    multiframe_features=item["multiframe_features"],
                    fusion_method=item["backtest_config"].fusion_method,
                    fusion_weights=item["backtest_config"].fusion_weights,
                    portfolio_risk=portfolio_risk,
                    portfolio_stats=run_stats,
                    risk_guard=guard_monitor,
                )
                run_stats.record_risk(
                    equity_ref=initial_balance,
                    open_risk=broker.get_open_risk_amounts(),
                )
            if not args.dry_run:
                save_state(args.state_file, state)
    else:
        for item in prepared:
            ts = item["timestamps"][0]
            open_risk = broker.get_open_risk_amounts()
            run_stats.record_risk(
                equity_ref=initial_balance,
                open_risk=open_risk,
            )
            state = run_for_timestamp(
                ts,
                backtest_config=item["backtest_config"],
                features=item["features"],
                prices=item["prices"],
                bars_map=item["bars_map"],
                broker=broker,
                state=state,
                signal_log=item["signal_log_path"],
                trades_json=item["trades_json_path"],
                trade_csv=item["trade_log_path"],
                summary_file=item["summary_path"],
                max_notional=item["max_notional"],
                dry_run=args.dry_run,
                force=False,
                multiframe_features=item["multiframe_features"],
                fusion_method=item["backtest_config"].fusion_method,
                fusion_weights=item["backtest_config"].fusion_weights,
                portfolio_risk=portfolio_risk,
                portfolio_stats=run_stats,
                risk_guard=guard_monitor,
            )
            run_stats.record_risk(
                equity_ref=initial_balance,
                open_risk=broker.get_open_risk_amounts(),
            )
            if not args.dry_run:
                save_state(args.state_file, state)

    tickers = ",".join(bt.ticker for _, bt in loaded_configs)
    guard_state = None
    if guard_monitor is not None:
        guard_state = {
            "equity_start": guard_monitor.state.equity_start,
            "equity_current": guard_monitor.state.equity_current,
            "consec_losses": guard_monitor.state.consec_losses,
            "is_triggered": guard_monitor.state.is_triggered,
            "reason": guard_monitor.state.reason,
        }
    if args.backfill:
        bars_lookup = {
            item["backtest_config"].ticker.upper(): item["last_bar"]
            for item in prepared
            if item.get("last_bar") is not None
        }
        flatten_open_positions(broker, bars_lookup)
        run_stats.record_risk(
            equity_ref=initial_balance,
            open_risk=broker.get_open_risk_amounts(),
        )
    summary_payload = build_portfolio_summary(
        broker=broker,
        tickers=[bt.ticker for _, bt in loaded_configs],
        stats=run_stats,
        summary_path=portfolio_summary_file,
        guard_state=guard_state,
        equity_ref=initial_balance,
        portfolio_limits=portfolio_limits,
    )
    print(f"[LIVE] Completed portfolio run for {tickers} at {datetime.now(timezone.utc).isoformat()}.")
    print(
        f"[LIVE] Portfolio risk used {summary_payload['risk_used_pct_total']:.2%} "
        f"blocked={summary_payload['counts']['blocked']} scaled={summary_payload['counts']['scaled']}"
    )

def main() -> None:
    args = parse_args()
    config_paths = [args.config]
    if args.config_list:
        config_paths.extend(args.config_list)
        run_portfolio(args, config_paths)
        return
    cfg = load_cli_config(args.config)
    backtest_config = build_backtest_config(cfg)
    live_cfg = cfg.get("live", {}) or {}
    max_notional = float(live_cfg.get("max_notional_per_trade", 50_000))
    commission_pct = float(live_cfg.get("commission_pct", 0.0005))
    slippage_pct = float(live_cfg.get("slippage_pct", 0.0005))
    trade_log_path = Path(args.trade_log) if args.trade_log else Path(live_cfg.get("trade_log", DEFAULT_TRADES_CSV))
    portfolio_limits = build_portfolio_limits(cfg)
    portfolio_risk = PortfolioRiskManager(
        portfolio_limits,
        {backtest_config.ticker.upper(): backtest_config.group or backtest_config.ticker},
    )
    run_stats = PortfolioRunStats()
    session_ts = datetime.now(timezone.utc)
    portfolio_summary_file = Path(live_cfg.get("portfolio_summary_dir", "live/logs/portfolio")) / session_ts.strftime(
        "%Y-%m-%d"
    ) / "summary.json"

    fetcher = MarketDataFetcher(backtest_config.ticker)
    prices = fetcher.get_prices(
        interval=backtest_config.interval,
        period=backtest_config.period if not args.backfill else None,
        start=args.start if args.backfill else backtest_config.start,
        end=args.end if args.backfill else backtest_config.end,
    )
    prices = prices.sort_index()
    if prices.empty:
        raise SystemExit("No price data available for live run.")
    min_history = max(
        backtest_config.atr_window,
        backtest_config.sma_long_window,
        backtest_config.rsi_window,
        72,
    ) + 5
    if len(prices) < min_history:
        raise SystemExit(
            f"Not enough bars ({len(prices)}) for required history ({min_history}). "
            f"Use an earlier --start or widen the backfill window."
        )
    multiframe_features = None
    if backtest_config.multi_timeframes:
        multiframe_features = build_multiframe_feature_set(prices, backtest_config, backtest_config.multi_timeframes)
        features = multiframe_features.get("base", pd.DataFrame())
    else:
        features = prepare_feature_frame(prices, backtest_config)
    if features.empty:
        raise SystemExit("Feature frame empty; cannot produce live signal.")
    bars_map = build_bars_map(backtest_config.ticker, prices)

    state = load_state(args.state_file, backtest_config.initial_balance)
    broker = broker_from_state(
        state,
        initial_cash=backtest_config.initial_balance,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
        trade_log_path=None if args.dry_run else trade_log_path,
    )

    if args.backfill:
        state = run_backfill(
            args,
            cfg,
            backtest_config,
            state,
            prices=prices,
            features=features,
            bars_map=bars_map,
            broker=broker,
            max_notional=max_notional,
            trade_log_path=trade_log_path,
            multiframe_features=multiframe_features,
            portfolio_risk=portfolio_risk,
            portfolio_stats=run_stats,
        )
        last_bar = bars_map[max(bars_map)] if bars_map else None
        if last_bar is not None:
            flatten_open_positions(
                broker, {backtest_config.ticker.upper(): last_bar}, reason="end_of_window"
            )
            run_stats.record_risk(
                equity_ref=backtest_config.initial_balance,
                open_risk=broker.get_open_risk_amounts(),
            )
        if not args.dry_run:
            save_state(args.state_file, state)
        build_portfolio_summary(
            broker=broker,
            tickers=[backtest_config.ticker],
            stats=run_stats,
            summary_path=portfolio_summary_file,
            equity_ref=backtest_config.initial_balance,
            portfolio_limits=portfolio_limits,
        )
        print(f"[LIVE] Backfill completed for {args.start} → {args.end}.")
        return

    run_stats.record_risk(
        equity_ref=broker.get_balance(),
        open_risk=broker.get_open_risk_amounts(),
    )
    ts = features.index[-1]
    state = run_for_timestamp(
        ts,
        backtest_config=backtest_config,
        features=features,
        prices=prices,
        bars_map=bars_map,
        broker=broker,
        state=state,
        signal_log=args.signal_log,
        trades_json=args.trades_json,
        trade_csv=trade_log_path,
        summary_file=args.summary_file,
        max_notional=max_notional,
        dry_run=args.dry_run,
        force=False,
        multiframe_features=multiframe_features,
        fusion_method=backtest_config.fusion_method,
        fusion_weights=backtest_config.fusion_weights,
        portfolio_risk=portfolio_risk,
        portfolio_stats=run_stats,
    )
    build_portfolio_summary(
        broker=broker,
        tickers=[backtest_config.ticker],
        stats=run_stats,
        summary_path=portfolio_summary_file,
        equity_ref=backtest_config.initial_balance,
        portfolio_limits=portfolio_limits,
    )
    if not args.dry_run:
        save_state(args.state_file, state)

    print(
        f"[LIVE] Completed run for {backtest_config.ticker} at "
        f"{datetime.now(timezone.utc).isoformat()}."
    )


if __name__ == "__main__":
    main()

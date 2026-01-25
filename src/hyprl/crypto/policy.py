"""Policy-based crypto trading signals (trend/momentum + risk)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any, Optional

import math
import numpy as np
import pandas as pd

TIMEFRAME_MINUTES: dict[str, int] = {
    "1Min": 1,
    "5Min": 5,
    "15Min": 15,
    "1Hour": 60,
    "1Day": 1440,
}


def _bars_per_hour(timeframe: str) -> int:
    minutes = TIMEFRAME_MINUTES.get(timeframe, 60)
    if minutes <= 0:
        return 1
    if minutes >= 60:
        return 1
    return max(1, int(round(60 / minutes)))


@dataclass
class PolicyConfig:
    timeframe: str = "1Hour"

    z_entry: float = 0.30
    rsi_min: float = 52.0
    vol24_max: float = 0.025

    atr_period_hours: int = 14
    atr_stop_k: float = 2.5
    atr_tp_k: float = 3.0
    use_take_profit: bool = True

    base_pct: float = 0.05
    max_position_pct: float = 0.10
    regime_mult_bull: float = 1.0
    regime_mult_bear: float = 0.0
    regime_mult_neutral: float = 0.5
    vol_scale_ref: float = 0.015

    ema_entry_hours: int = 20
    ema_fast_hours: int = 50
    ema_slow_hours: int = 200
    slope_lookback_hours: int = 12
    slope_neutral_min: float = 0.0
    allow_neutral_bullish: bool = False
    rsi_period_hours: int = 14
    mom_z_window_hours: int = 240

    fees_bps: float = 6.0
    slip_bps: float = 4.0
    edge_buffer_bps: float = 5.0

    max_daily_dd_pct: float = 0.01
    cooldown_seconds: int = 2 * 3600
    max_orders_day: int = 4


@dataclass
class PolicyState:
    day: Optional[date] = None
    day_start_equity: Optional[float] = None
    orders_today: int = 0
    last_trade_ts: Optional[float] = None
    trailing_stop: Optional[float] = None
    entry_price: Optional[float] = None
    peak_price: Optional[float] = None
    in_position: bool = False


@dataclass
class PolicyDecision:
    action: str
    size_pct: float
    stop_loss: float
    take_profit: float
    probability: float
    confidence: float
    reason: str


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.ewm(alpha=1 / period, adjust=False).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _zscore(x: pd.Series, window: int = 240) -> pd.Series:
    mu = x.rolling(window).mean()
    sd = x.rolling(window).std(ddof=0).replace(0.0, np.nan)
    return (x - mu) / sd


def _slope(series: pd.Series, lookback: int = 12) -> pd.Series:
    return (series - series.shift(lookback)) / float(lookback)


def _as_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        ts = pd.to_datetime(value, utc=True)
    except Exception:
        return datetime.now(timezone.utc)
    if isinstance(ts, pd.Timestamp):
        return ts.to_pydatetime()
    return datetime.now(timezone.utc)


def _as_float(value: Any, default: float) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(val):
        return default
    return val


def _ensure_day(state: PolicyState, now: datetime, equity: Optional[float]) -> None:
    day = now.date()
    if state.day != day:
        state.day = day
        state.day_start_equity = equity
        state.orders_today = 0
        state.last_trade_ts = None
        state.trailing_stop = None
        state.entry_price = None
        state.peak_price = None
        state.in_position = False


def can_open_trade(cfg: PolicyConfig, state: PolicyState, now: datetime, equity: Optional[float]) -> tuple[bool, str]:
    _ensure_day(state, now, equity)
    if state.day_start_equity is not None and equity is not None:
        if equity < state.day_start_equity * (1.0 - cfg.max_daily_dd_pct):
            return False, "daily_dd_cap"
    if state.orders_today >= cfg.max_orders_day:
        return False, "max_orders_day"
    if state.last_trade_ts is not None and (now.timestamp() - state.last_trade_ts) < cfg.cooldown_seconds:
        return False, "cooldown"
    return True, "ok"


def compute_policy_frame(df: pd.DataFrame, cfg: PolicyConfig) -> pd.DataFrame:
    required = {"close", "high", "low"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise ValueError(f"Missing columns: {missing}")

    x = df.copy()
    if "timestamp" in x.columns:
        x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True, errors="coerce")
    else:
        x["timestamp"] = pd.to_datetime(x.index, utc=True, errors="coerce")
    x = x.sort_values("timestamp")

    bars_per_hour = _bars_per_hour(cfg.timeframe)

    def _bars(hours: int) -> int:
        return max(1, int(round(hours * bars_per_hour)))

    close = x["close"].astype(float)
    high = x["high"].astype(float)
    low = x["low"].astype(float)

    x["ret_1h"] = close.pct_change(_bars(1))
    x["ret_12h"] = close.pct_change(_bars(12))
    x["ret_24h"] = close.pct_change(_bars(24))

    x["ema20"] = _ema(close, _bars(cfg.ema_entry_hours))
    x["ema50"] = _ema(close, _bars(cfg.ema_fast_hours))
    x["ema200"] = _ema(close, _bars(cfg.ema_slow_hours))
    x["slope50"] = _slope(x["ema50"], _bars(cfg.slope_lookback_hours))

    x["rsi14"] = _rsi(close, _bars(cfg.rsi_period_hours))
    x["atr"] = _atr(high, low, close, _bars(cfg.atr_period_hours))

    mom_raw = (x["ret_12h"].fillna(0.0) + x["ret_24h"].fillna(0.0))
    x["mom_z"] = _zscore(mom_raw, _bars(cfg.mom_z_window_hours))

    vol_window = _bars(24)
    x["vol_24h"] = x["ret_1h"].rolling(vol_window).std(ddof=0)

    bull = (x["ema50"] > x["ema200"]) & (x["slope50"] > 0)
    bear = (x["ema50"] < x["ema200"]) & (x["slope50"] < 0)
    x["regime"] = np.where(bull, "BULL", np.where(bear, "BEAR", "NEUTRAL"))

    x["edge_bps"] = np.maximum(x["mom_z"] - cfg.z_entry, 0.0) * x["vol_24h"].fillna(0.0) * 10000.0

    return x


def _momentum_probability(mom_z: float) -> float:
    scaled = math.tanh(mom_z / 2.0)
    prob = 0.5 + 0.25 * scaled
    return min(0.99, max(0.01, prob))


def decide_row(
    row: Any,
    cfg: PolicyConfig,
    state: PolicyState,
    equity: Optional[float] = None,
) -> PolicyDecision:
    now = _as_datetime(getattr(row, "timestamp", None))
    close = _as_float(getattr(row, "close", None), 0.0)
    atr = _as_float(getattr(row, "atr", None), 0.0)
    vol24 = _as_float(getattr(row, "vol_24h", None), 0.0)
    edge_bps = _as_float(getattr(row, "edge_bps", None), 0.0)
    mom_z = _as_float(getattr(row, "mom_z", None), -999.0)
    rsi14 = _as_float(getattr(row, "rsi14", None), 0.0)
    ema20 = _as_float(getattr(row, "ema20", None), close)
    regime = str(getattr(row, "regime", "NEUTRAL"))

    probability = _momentum_probability(mom_z)
    size_pct = 0.0
    stop_loss = close
    take_profit = close

    def _size_pct() -> float:
        z_scale = min(2.0, max(0.0, mom_z)) / 2.0
        vol_scale = min(1.0, cfg.vol_scale_ref / max(vol24, 1e-6))
        regime_mult = cfg.regime_mult_bull if regime == "BULL" else cfg.regime_mult_bear
        size = cfg.base_pct * z_scale * regime_mult * vol_scale
        return float(np.clip(size, 0.0, cfg.max_position_pct))

    reason_parts = [
        f"regime={regime}",
        f"mom_z={mom_z:.2f}",
        f"rsi={rsi14:.0f}",
        f"vol24={vol24:.4f}",
        f"edge_bps={edge_bps:.1f}",
    ]

    if state.in_position:
        if atr > 0:
            state.peak_price = close if state.peak_price is None else max(state.peak_price, close)
            trail = state.peak_price - cfg.atr_stop_k * atr
            state.trailing_stop = trail if state.trailing_stop is None else max(state.trailing_stop, trail)

        stop_loss = state.trailing_stop if state.trailing_stop is not None else close
        if cfg.use_take_profit and state.entry_price is not None and atr > 0:
            take_profit = state.entry_price + cfg.atr_tp_k * atr
        else:
            take_profit = close

        if state.trailing_stop is not None and close <= state.trailing_stop:
            state.in_position = False
            state.entry_price = None
            state.trailing_stop = None
            state.peak_price = None
            state.last_trade_ts = now.timestamp()
            reason_parts.append("exit=trailing_stop")
            return PolicyDecision(
                action="SELL",
                size_pct=0.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                probability=probability,
                confidence=0.0,
                reason="policy: " + ", ".join(reason_parts),
            )

        if cfg.use_take_profit and state.entry_price is not None and atr > 0:
            tp = state.entry_price + cfg.atr_tp_k * atr
            if close >= tp:
                state.in_position = False
                state.entry_price = None
                state.trailing_stop = None
                state.peak_price = None
                state.last_trade_ts = now.timestamp()
                reason_parts.append("exit=take_profit")
                return PolicyDecision(
                    action="SELL",
                    size_pct=0.0,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    probability=probability,
                    confidence=0.0,
                    reason="policy: " + ", ".join(reason_parts),
                )

        if regime != "BULL" or mom_z <= cfg.z_entry or rsi14 <= cfg.rsi_min or close <= ema20:
            state.in_position = False
            state.entry_price = None
            state.trailing_stop = None
            state.peak_price = None
            state.last_trade_ts = now.timestamp()
            reason_parts.append("exit=regime")
            return PolicyDecision(
                action="SELL",
                size_pct=0.0,
                stop_loss=stop_loss,
                take_profit=take_profit,
                probability=probability,
                confidence=0.0,
                reason="policy: " + ", ".join(reason_parts),
            )

        size_pct = _size_pct()
        confidence = min(1.0, size_pct / cfg.max_position_pct) if cfg.max_position_pct > 0 else 0.0
        reason_parts.append("action=hold")
        return PolicyDecision(
            action="HOLD",
            size_pct=size_pct,
            stop_loss=stop_loss,
            take_profit=take_profit,
            probability=probability,
            confidence=confidence,
            reason="policy: " + ", ".join(reason_parts),
        )

    min_edge = cfg.fees_bps + cfg.slip_bps + cfg.edge_buffer_bps
    no_trade = vol24 > cfg.vol24_max or edge_bps < min_edge

    entry_ok, entry_reason = can_open_trade(cfg, state, now, equity)
    if (
        entry_ok
        and not no_trade
        and regime == "BULL"
        and mom_z > cfg.z_entry
        and rsi14 > cfg.rsi_min
        and close > ema20
        and atr > 0
    ):
        size_pct = _size_pct()
        if size_pct > 0:
            state.in_position = True
            state.entry_price = close
            state.peak_price = close
            state.trailing_stop = close - cfg.atr_stop_k * atr
            state.orders_today += 1
            state.last_trade_ts = now.timestamp()
            stop_loss = state.trailing_stop
            if cfg.use_take_profit:
                take_profit = close + cfg.atr_tp_k * atr
            confidence = min(1.0, size_pct / cfg.max_position_pct) if cfg.max_position_pct > 0 else 0.0
            reason_parts.append("action=buy")
            return PolicyDecision(
                action="BUY",
                size_pct=size_pct,
                stop_loss=stop_loss,
                take_profit=take_profit,
                probability=probability,
                confidence=confidence,
                reason="policy: " + ", ".join(reason_parts),
            )

    blocker = None
    if not entry_ok:
        blocker = entry_reason
    elif no_trade:
        blocker = "vol24" if vol24 > cfg.vol24_max else "edge_bps"
    elif regime != "BULL":
        blocker = "regime"
    elif mom_z <= cfg.z_entry:
        blocker = "mom_z"
    elif rsi14 <= cfg.rsi_min:
        blocker = "rsi"
    elif close <= ema20:
        blocker = "ema20"
    elif atr <= 0:
        blocker = "atr"

    reason_parts.append("action=hold")
    if blocker:
        reason_parts.append(f"blocker={blocker}")
    return PolicyDecision(
        action="HOLD",
        size_pct=0.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        probability=probability,
        confidence=0.0,
        reason="policy: " + ", ".join(reason_parts),
    )


def simulate_policy(frame: pd.DataFrame, cfg: PolicyConfig) -> tuple[Optional[PolicyDecision], PolicyState]:
    state = PolicyState()
    last_decision: Optional[PolicyDecision] = None
    for row in frame.itertuples(index=False):
        last_decision = decide_row(row, cfg, state, equity=None)
    return last_decision, state

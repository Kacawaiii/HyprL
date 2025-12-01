from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

Side = Literal["long", "short"]
ExitReason = Literal["stop_loss", "take_profit", "trailing_stop", "end_of_data", "manual", "end_of_window"]


@dataclass(slots=True)
class Bar:
    """Single OHLCV bar aligned with backtest data semantics."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(slots=True)
class TradeSignal:
    """Desired trade action emitted by the strategy engine."""

    symbol: str
    side: Side
    size: float
    reason: str
    timestamp: datetime
    probability_up: float
    threshold: float
    entry_price: float
    expected_pnl: float
    risk_amount: float
    long_threshold: float
    short_threshold: float
    stop_price: float
    take_profit_price: float
    trailing_stop_activation_price: float | None
    trailing_stop_distance_price: float | None
    risk_profile: str | None = None
    regime_name: str | None = None


@dataclass(slots=True)
class Position:
    """Current live/paper position snapshot."""

    symbol: str
    side: Side
    size: float
    avg_price: float
    unrealized_pnl: float = 0.0

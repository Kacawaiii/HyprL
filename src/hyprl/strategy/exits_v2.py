"""
Exit Management V2 - Partial profits and time-based exits.

Features:
1. Partial profit taking at milestones
2. Time-based exit for stagnant trades
3. Profit protection rules
4. Trailing exit logic
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, List


@dataclass
class ExitConfig:
    """Configuration for exit management."""
    # Partial profit taking
    partial_enabled: bool = True
    partial_r_levels: tuple = (1.5, 2.5)  # Take partial at these R-multiples
    partial_pct: float = 0.5  # Close 50% at each level

    # Time-based exit
    time_exit_enabled: bool = True
    max_hold_hours: int = 24  # Maximum hold time
    stagnant_exit_hours: int = 6  # Exit if no progress after X hours
    stagnant_threshold_r: float = 0.3  # Must be up at least 0.3R to avoid stagnant exit

    # Profit protection
    protect_profit_enabled: bool = True
    protect_trigger_r: float = 2.0  # Start protecting at 2R
    protect_min_r: float = 1.0  # Never give back below 1R once triggered

    # End of day
    eod_exit_minutes: int = 15  # Exit X minutes before close


@dataclass
class PositionState:
    """State tracking for a position."""
    symbol: str
    entry_price: float
    entry_time: datetime
    initial_stop: float
    initial_qty: int
    current_qty: int
    is_long: bool
    highest_r: float = 0.0  # Highest R-multiple reached
    lowest_r: float = 0.0  # Lowest R-multiple (for shorts tracking)
    partials_taken: List[float] = field(default_factory=list)  # R-levels where partial was taken
    protect_triggered: bool = False


@dataclass
class ExitSignal:
    """Signal to exit all or part of a position."""
    action: str  # "close_all", "close_partial", "hold"
    qty_to_close: int
    reason: str
    urgency: str  # "immediate", "market", "limit"
    limit_price: Optional[float] = None


def compute_r_multiple(
    state: PositionState,
    current_price: float,
) -> float:
    """Compute current R-multiple for position."""
    risk_per_share = abs(state.entry_price - state.initial_stop)

    if risk_per_share <= 0:
        return 0.0

    if state.is_long:
        pnl = current_price - state.entry_price
    else:
        pnl = state.entry_price - current_price

    return pnl / risk_per_share


def check_partial_profit(
    state: PositionState,
    current_r: float,
    config: ExitConfig,
) -> Optional[ExitSignal]:
    """Check if we should take partial profit."""
    if not config.partial_enabled:
        return None

    if state.current_qty <= 1:
        return None  # Can't take partial from 1 share

    for r_level in config.partial_r_levels:
        if r_level not in state.partials_taken and current_r >= r_level:
            qty_to_close = int(state.current_qty * config.partial_pct)
            qty_to_close = max(1, qty_to_close)

            if qty_to_close > 0:
                return ExitSignal(
                    action="close_partial",
                    qty_to_close=qty_to_close,
                    reason=f"partial_profit_{r_level}R",
                    urgency="market",
                )

    return None


def check_time_exit(
    state: PositionState,
    current_r: float,
    now: Optional[datetime] = None,
    config: ExitConfig = None,
) -> Optional[ExitSignal]:
    """Check if position should be exited due to time."""
    if config is None:
        config = ExitConfig()

    if not config.time_exit_enabled:
        return None

    if now is None:
        now = datetime.now(timezone.utc)

    age_hours = (now - state.entry_time).total_seconds() / 3600

    # Maximum hold time
    if age_hours >= config.max_hold_hours:
        return ExitSignal(
            action="close_all",
            qty_to_close=state.current_qty,
            reason="max_hold_time",
            urgency="market",
        )

    # Stagnant exit - position not making progress
    if age_hours >= config.stagnant_exit_hours:
        if current_r < config.stagnant_threshold_r:
            return ExitSignal(
                action="close_all",
                qty_to_close=state.current_qty,
                reason="stagnant_trade",
                urgency="market",
            )

    return None


def check_profit_protection(
    state: PositionState,
    current_r: float,
    config: ExitConfig,
) -> Optional[ExitSignal]:
    """Check if profit protection should trigger exit."""
    if not config.protect_profit_enabled:
        return None

    # Update highest R
    if current_r > state.highest_r:
        state.highest_r = current_r

    # Check if protection should trigger
    if state.highest_r >= config.protect_trigger_r:
        state.protect_triggered = True

    # If protection triggered and we're giving back too much
    if state.protect_triggered:
        if current_r < config.protect_min_r:
            return ExitSignal(
                action="close_all",
                qty_to_close=state.current_qty,
                reason="profit_protection",
                urgency="immediate",
            )

    return None


def check_eod_exit(
    now: Optional[datetime] = None,
    market_close_time: Optional[datetime] = None,
    config: ExitConfig = None,
) -> bool:
    """Check if we're in EOD exit window."""
    if config is None:
        config = ExitConfig()

    if now is None:
        now = datetime.now(timezone.utc)

    if market_close_time is None:
        # Default US market close: 21:00 UTC (4 PM ET)
        market_close_time = now.replace(hour=21, minute=0, second=0, microsecond=0)

    minutes_to_close = (market_close_time - now).total_seconds() / 60

    return 0 < minutes_to_close <= config.eod_exit_minutes


def evaluate_exit(
    state: PositionState,
    current_price: float,
    now: Optional[datetime] = None,
    market_close_time: Optional[datetime] = None,
    config: Optional[ExitConfig] = None,
) -> ExitSignal:
    """
    Evaluate all exit conditions and return appropriate action.

    Priority:
    1. EOD exit (highest)
    2. Profit protection
    3. Time-based exit
    4. Partial profit
    5. Hold (no action)
    """
    if config is None:
        config = ExitConfig()

    if now is None:
        now = datetime.now(timezone.utc)

    current_r = compute_r_multiple(state, current_price)

    # 1. EOD exit
    if check_eod_exit(now, market_close_time, config):
        return ExitSignal(
            action="close_all",
            qty_to_close=state.current_qty,
            reason="eod_exit",
            urgency="immediate",
        )

    # 2. Profit protection
    protection_signal = check_profit_protection(state, current_r, config)
    if protection_signal:
        return protection_signal

    # 3. Time-based exit
    time_signal = check_time_exit(state, current_r, now, config)
    if time_signal:
        return time_signal

    # 4. Partial profit
    partial_signal = check_partial_profit(state, current_r, config)
    if partial_signal:
        return partial_signal

    # 5. Hold
    return ExitSignal(
        action="hold",
        qty_to_close=0,
        reason="no_exit_trigger",
        urgency="none",
    )


def create_position_state(
    symbol: str,
    entry_price: float,
    stop_price: float,
    qty: int,
    is_long: bool,
    entry_time: Optional[datetime] = None,
) -> PositionState:
    """Create initial position state for exit tracking."""
    if entry_time is None:
        entry_time = datetime.now(timezone.utc)

    return PositionState(
        symbol=symbol,
        entry_price=entry_price,
        entry_time=entry_time,
        initial_stop=stop_price,
        initial_qty=qty,
        current_qty=qty,
        is_long=is_long,
        highest_r=0.0,
        lowest_r=0.0,
        partials_taken=[],
        protect_triggered=False,
    )

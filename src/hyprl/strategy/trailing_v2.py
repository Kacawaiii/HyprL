"""
Intelligent Trailing Stop V2 - ATR-based dynamic trailing.

Features:
1. ATR-based trailing distance (adapts to volatility)
2. Breakeven trigger after reaching 1R profit
3. Step-up trailing (lock in profits at milestones)
4. Time-decay (tighten stop for old positions)
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional


@dataclass
class TrailingConfig:
    """Configuration for intelligent trailing stop."""
    # ATR-based distance
    atr_multiplier: float = 2.0  # Trail at 2x ATR
    min_trail_pct: float = 0.005  # Minimum 0.5% trail
    max_trail_pct: float = 0.03  # Maximum 3% trail

    # Breakeven settings
    breakeven_trigger_r: float = 1.0  # Move to breakeven at 1R profit
    breakeven_buffer_pct: float = 0.001  # 0.1% above entry for longs

    # Step-up milestones (R-multiples)
    step_up_enabled: bool = True
    step_up_milestones: tuple = (1.5, 2.0, 3.0)  # Lock at these R levels
    step_up_lock_pct: float = 0.5  # Lock 50% of profit at each milestone

    # Time decay
    time_decay_enabled: bool = True
    time_decay_start_hours: int = 6  # Start tightening after 6 hours
    time_decay_rate: float = 0.1  # Tighten by 10% per hour after start


@dataclass
class TrailingState:
    """Current state of trailing stop for a position."""
    entry_price: float
    current_stop: float
    highest_price: float  # For longs
    lowest_price: float  # For shorts
    entry_time: datetime
    risk_amount: float  # Original risk (entry - initial stop)
    is_long: bool
    breakeven_triggered: bool = False
    current_milestone_idx: int = -1  # Which step-up milestone we're at


@dataclass
class TrailingUpdate:
    """Result of trailing stop update."""
    new_stop: float
    trail_distance: float
    trail_pct: float
    reason: str
    should_close: bool = False


def compute_atr_trail_distance(
    current_atr: float,
    current_price: float,
    config: TrailingConfig,
) -> float:
    """Compute trailing distance based on ATR."""
    # ATR-based distance
    atr_distance = current_atr * config.atr_multiplier

    # Convert to percentage
    atr_pct = atr_distance / current_price if current_price > 0 else 0

    # Clamp to min/max
    clamped_pct = max(config.min_trail_pct, min(config.max_trail_pct, atr_pct))

    return current_price * clamped_pct


def compute_time_decay_factor(
    entry_time: datetime,
    now: Optional[datetime] = None,
    config: TrailingConfig = None,
) -> float:
    """
    Compute tightening factor based on position age.

    Returns multiplier < 1.0 to tighten the trail.
    """
    if config is None:
        config = TrailingConfig()

    if not config.time_decay_enabled:
        return 1.0

    if now is None:
        now = datetime.now(timezone.utc)

    age_hours = (now - entry_time).total_seconds() / 3600

    if age_hours < config.time_decay_start_hours:
        return 1.0

    # Hours past decay start
    decay_hours = age_hours - config.time_decay_start_hours

    # Tighten by rate per hour
    decay_factor = 1.0 - (decay_hours * config.time_decay_rate)

    # Minimum factor of 0.5 (can't tighten more than 50%)
    return max(0.5, decay_factor)


def update_trailing_stop(
    state: TrailingState,
    current_price: float,
    current_atr: float,
    config: Optional[TrailingConfig] = None,
    now: Optional[datetime] = None,
) -> TrailingUpdate:
    """
    Update trailing stop based on current market conditions.

    Args:
        state: Current trailing state
        current_price: Current market price
        current_atr: Current ATR value
        config: Trailing configuration
        now: Current time (for time decay)

    Returns:
        TrailingUpdate with new stop level
    """
    if config is None:
        config = TrailingConfig()

    if now is None:
        now = datetime.now(timezone.utc)

    # Calculate current P&L in R-multiples
    if state.is_long:
        pnl = current_price - state.entry_price
        pnl_r = pnl / state.risk_amount if state.risk_amount > 0 else 0
    else:
        pnl = state.entry_price - current_price
        pnl_r = pnl / state.risk_amount if state.risk_amount > 0 else 0

    # Check for stop hit
    if state.is_long and current_price <= state.current_stop:
        return TrailingUpdate(
            new_stop=state.current_stop,
            trail_distance=0,
            trail_pct=0,
            reason="stop_hit",
            should_close=True,
        )
    elif not state.is_long and current_price >= state.current_stop:
        return TrailingUpdate(
            new_stop=state.current_stop,
            trail_distance=0,
            trail_pct=0,
            reason="stop_hit",
            should_close=True,
        )

    # Compute base trail distance from ATR
    base_trail = compute_atr_trail_distance(current_atr, current_price, config)

    # Apply time decay
    time_factor = compute_time_decay_factor(state.entry_time, now, config)
    trail_distance = base_trail * time_factor

    # Calculate new potential stop
    if state.is_long:
        # Update highest price
        new_highest = max(state.highest_price, current_price)

        # Potential stop based on trailing from high
        potential_stop = new_highest - trail_distance

        # Check breakeven trigger
        if not state.breakeven_triggered and pnl_r >= config.breakeven_trigger_r:
            breakeven_stop = state.entry_price + (state.entry_price * config.breakeven_buffer_pct)
            potential_stop = max(potential_stop, breakeven_stop)
            reason = "breakeven_triggered"
        else:
            reason = "trail_update"

        # Check step-up milestones
        if config.step_up_enabled:
            for idx, milestone in enumerate(config.step_up_milestones):
                if pnl_r >= milestone and idx > state.current_milestone_idx:
                    # Lock in portion of profit
                    profit = current_price - state.entry_price
                    locked_profit = profit * config.step_up_lock_pct
                    milestone_stop = state.entry_price + locked_profit
                    potential_stop = max(potential_stop, milestone_stop)
                    reason = f"milestone_{milestone}R"

        # Never lower the stop
        new_stop = max(state.current_stop, potential_stop)

    else:
        # Short position - mirror logic
        new_lowest = min(state.lowest_price, current_price)

        potential_stop = new_lowest + trail_distance

        if not state.breakeven_triggered and pnl_r >= config.breakeven_trigger_r:
            breakeven_stop = state.entry_price - (state.entry_price * config.breakeven_buffer_pct)
            potential_stop = min(potential_stop, breakeven_stop)
            reason = "breakeven_triggered"
        else:
            reason = "trail_update"

        if config.step_up_enabled:
            for idx, milestone in enumerate(config.step_up_milestones):
                if pnl_r >= milestone and idx > state.current_milestone_idx:
                    profit = state.entry_price - current_price
                    locked_profit = profit * config.step_up_lock_pct
                    milestone_stop = state.entry_price - locked_profit
                    potential_stop = min(potential_stop, milestone_stop)
                    reason = f"milestone_{milestone}R"

        # Never raise the stop (for shorts)
        new_stop = min(state.current_stop, potential_stop)

    trail_pct = trail_distance / current_price if current_price > 0 else 0

    return TrailingUpdate(
        new_stop=new_stop,
        trail_distance=trail_distance,
        trail_pct=trail_pct,
        reason=reason,
        should_close=False,
    )


def create_initial_state(
    entry_price: float,
    stop_price: float,
    is_long: bool,
    entry_time: Optional[datetime] = None,
) -> TrailingState:
    """Create initial trailing state for a new position."""
    if entry_time is None:
        entry_time = datetime.now(timezone.utc)

    risk_amount = abs(entry_price - stop_price)

    return TrailingState(
        entry_price=entry_price,
        current_stop=stop_price,
        highest_price=entry_price,
        lowest_price=entry_price,
        entry_time=entry_time,
        risk_amount=risk_amount,
        is_long=is_long,
        breakeven_triggered=False,
        current_milestone_idx=-1,
    )

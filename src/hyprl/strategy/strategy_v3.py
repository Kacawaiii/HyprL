"""
Strategy V3 - Unified strategy with all enhancements.

Combines:
1. Dynamic conviction-based sizing
2. ATR-based trailing stops
3. Partial profit taking
4. Signal quality filtering
5. Time-based exit management
6. Enhanced features

Usage:
    from hyprl.strategy.strategy_v3 import StrategyV3, StrategyV3Config

    strategy = StrategyV3(config)
    result = strategy.evaluate_signal(signal_data)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

from .sizing_v2 import SizingConfig, compute_dynamic_size, SizingResult
from .trailing_v2 import TrailingConfig, TrailingState, update_trailing_stop, create_initial_state
from .exits_v2 import ExitConfig, PositionState, evaluate_exit, create_position_state, ExitSignal
from .signal_quality import QualityConfig, evaluate_signal_quality, QualityResult


@dataclass
class StrategyV3Config:
    """Master configuration for Strategy V3."""
    # Sub-configs
    sizing: SizingConfig = field(default_factory=SizingConfig)
    trailing: TrailingConfig = field(default_factory=TrailingConfig)
    exits: ExitConfig = field(default_factory=ExitConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Strategy-level settings
    enabled: bool = True
    min_probability: float = 0.55  # Minimum probability to consider
    max_positions: int = 3
    position_cooldown_seconds: int = 300  # 5 min between trades on same symbol


@dataclass
class SignalEvaluation:
    """Result of signal evaluation."""
    should_trade: bool
    direction: str  # "long", "short", "flat"
    size: int
    entry_price: float
    stop_price: float
    take_profit_price: float
    quality_score: float
    sizing_result: Optional[SizingResult] = None
    quality_result: Optional[QualityResult] = None
    reasons: List[str] = field(default_factory=list)


@dataclass
class PositionUpdate:
    """Update for an existing position."""
    symbol: str
    action: str  # "hold", "update_stop", "close_partial", "close_all"
    new_stop: Optional[float] = None
    qty_to_close: int = 0
    reason: str = ""


class StrategyV3:
    """
    Strategy V3 - Production-ready strategy with all enhancements.
    """

    def __init__(self, config: Optional[StrategyV3Config] = None):
        self.config = config or StrategyV3Config()
        self._positions: Dict[str, PositionState] = {}
        self._trailing_states: Dict[str, TrailingState] = {}
        self._last_trade_time: Dict[str, datetime] = {}

    def evaluate_signal(
        self,
        symbol: str,
        probability: float,
        threshold_long: float,
        threshold_short: float,
        entry_price: float,
        stop_price: float,
        take_profit_price: float,
        equity: float,
        current_volume: float,
        avg_volume: float,
        atr: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        sma_short: Optional[float] = None,
        sma_long: Optional[float] = None,
        regime_mult: float = 1.0,
        now: Optional[datetime] = None,
    ) -> SignalEvaluation:
        """
        Evaluate a trading signal with all quality filters and sizing.

        Returns:
            SignalEvaluation with trade recommendation
        """
        if now is None:
            now = datetime.now(timezone.utc)

        reasons = []

        # Determine direction
        if probability >= threshold_long:
            direction = "long"
        elif probability <= (1 - threshold_short):
            direction = "short"
        else:
            return SignalEvaluation(
                should_trade=False,
                direction="flat",
                size=0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                quality_score=0.0,
                reasons=["no_signal"],
            )

        # Check cooldown
        if symbol in self._last_trade_time:
            elapsed = (now - self._last_trade_time[symbol]).total_seconds()
            if elapsed < self.config.position_cooldown_seconds:
                reasons.append(f"cooldown_{int(self.config.position_cooldown_seconds - elapsed)}s")
                return SignalEvaluation(
                    should_trade=False,
                    direction=direction,
                    size=0,
                    entry_price=entry_price,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    quality_score=0.0,
                    reasons=reasons,
                )

        # Quality check
        atr_pct = atr / entry_price if entry_price > 0 else 0
        quality_result = evaluate_signal_quality(
            signal_direction=direction,
            current_price=entry_price,
            current_volume=current_volume,
            avg_volume=avg_volume,
            atr_pct=atr_pct,
            bid=bid,
            ask=ask,
            sma_short=sma_short,
            sma_long=sma_long,
            now=now,
            config=self.config.quality,
        )

        if quality_result.recommendation == "skip":
            reasons.append("quality_skip")
            for check in quality_result.checks:
                if not check.passed:
                    reasons.append(check.reason)

            return SignalEvaluation(
                should_trade=False,
                direction=direction,
                size=0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                quality_score=quality_result.score,
                quality_result=quality_result,
                reasons=reasons,
            )

        # Dynamic sizing
        threshold = threshold_long if direction == "long" else threshold_short
        sizing_result = compute_dynamic_size(
            equity=equity,
            entry_price=entry_price,
            stop_price=stop_price,
            probability=probability,
            threshold=threshold,
            current_atr_pct=atr_pct,
            regime_mult=regime_mult,
            config=self.config.sizing,
        )

        if sizing_result.shares <= 0:
            reasons.append(sizing_result.reason)
            return SignalEvaluation(
                should_trade=False,
                direction=direction,
                size=0,
                entry_price=entry_price,
                stop_price=stop_price,
                take_profit_price=take_profit_price,
                quality_score=quality_result.score,
                sizing_result=sizing_result,
                quality_result=quality_result,
                reasons=reasons,
            )

        # Reduce size if quality suggests
        final_size = sizing_result.shares
        if quality_result.recommendation == "reduce_size":
            final_size = max(1, int(final_size * 0.5))
            reasons.append("size_reduced_quality")

        reasons.append("signal_approved")

        return SignalEvaluation(
            should_trade=True,
            direction=direction,
            size=final_size,
            entry_price=entry_price,
            stop_price=stop_price,
            take_profit_price=take_profit_price,
            quality_score=quality_result.score,
            sizing_result=sizing_result,
            quality_result=quality_result,
            reasons=reasons,
        )

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        stop_price: float,
        qty: int,
        is_long: bool,
        entry_time: Optional[datetime] = None,
    ) -> None:
        """Register a new position for tracking."""
        if entry_time is None:
            entry_time = datetime.now(timezone.utc)

        # Create position state for exit tracking
        self._positions[symbol] = create_position_state(
            symbol=symbol,
            entry_price=entry_price,
            stop_price=stop_price,
            qty=qty,
            is_long=is_long,
            entry_time=entry_time,
        )

        # Create trailing state
        self._trailing_states[symbol] = create_initial_state(
            entry_price=entry_price,
            stop_price=stop_price,
            is_long=is_long,
            entry_time=entry_time,
        )

        # Update last trade time
        self._last_trade_time[symbol] = entry_time

    def update_position(
        self,
        symbol: str,
        current_price: float,
        current_atr: float,
        now: Optional[datetime] = None,
        market_close_time: Optional[datetime] = None,
    ) -> PositionUpdate:
        """
        Update position state and check for exit conditions.

        Returns:
            PositionUpdate with recommended action
        """
        if now is None:
            now = datetime.now(timezone.utc)

        if symbol not in self._positions:
            return PositionUpdate(
                symbol=symbol,
                action="hold",
                reason="position_not_found",
            )

        position = self._positions[symbol]
        trailing = self._trailing_states.get(symbol)

        # Check exit conditions
        exit_signal = evaluate_exit(
            state=position,
            current_price=current_price,
            now=now,
            market_close_time=market_close_time,
            config=self.config.exits,
        )

        if exit_signal.action == "close_all":
            return PositionUpdate(
                symbol=symbol,
                action="close_all",
                qty_to_close=position.current_qty,
                reason=exit_signal.reason,
            )

        if exit_signal.action == "close_partial":
            # Update position quantity
            position.current_qty -= exit_signal.qty_to_close
            if exit_signal.reason.startswith("partial_profit"):
                r_level = float(exit_signal.reason.split("_")[-1].replace("R", ""))
                position.partials_taken.append(r_level)

            return PositionUpdate(
                symbol=symbol,
                action="close_partial",
                qty_to_close=exit_signal.qty_to_close,
                reason=exit_signal.reason,
            )

        # Update trailing stop
        if trailing:
            trail_update = update_trailing_stop(
                state=trailing,
                current_price=current_price,
                current_atr=current_atr,
                config=self.config.trailing,
                now=now,
            )

            if trail_update.should_close:
                return PositionUpdate(
                    symbol=symbol,
                    action="close_all",
                    qty_to_close=position.current_qty,
                    reason="trailing_stop_hit",
                )

            if trail_update.new_stop != trailing.current_stop:
                trailing.current_stop = trail_update.new_stop
                # Update highest/lowest prices
                if position.is_long:
                    trailing.highest_price = max(trailing.highest_price, current_price)
                else:
                    trailing.lowest_price = min(trailing.lowest_price, current_price)

                return PositionUpdate(
                    symbol=symbol,
                    action="update_stop",
                    new_stop=trail_update.new_stop,
                    reason=trail_update.reason,
                )

        return PositionUpdate(
            symbol=symbol,
            action="hold",
            reason="no_action_needed",
        )

    def close_position(self, symbol: str) -> None:
        """Remove position from tracking."""
        self._positions.pop(symbol, None)
        self._trailing_states.pop(symbol, None)

    def get_position_state(self, symbol: str) -> Optional[PositionState]:
        """Get current position state."""
        return self._positions.get(symbol)

    def get_all_positions(self) -> Dict[str, PositionState]:
        """Get all tracked positions."""
        return self._positions.copy()


# Default instance
_default_strategy: Optional[StrategyV3] = None


def get_default_strategy() -> StrategyV3:
    """Get or create default strategy instance."""
    global _default_strategy
    if _default_strategy is None:
        _default_strategy = StrategyV3()
    return _default_strategy

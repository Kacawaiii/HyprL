"""Multi-Timeframe Strategy combining 1h and 15min signals.

Strategy Logic:
- 1h model: Primary direction (validated edge)
- 15min model: Entry timing and confirmation

Signal Combinations:
- 1h long + 15min long → STRONG_LONG (1.5x size)
- 1h long + 15min flat → NORMAL_LONG (1.0x size)
- 1h long + 15min short → WEAK_LONG (0.5x size) or SKIP
- 1h flat + 15min long → OPPORTUNISTIC_LONG (0.7x size)
- Both flat → NO_SIGNAL
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
from datetime import datetime


class SignalStrength(Enum):
    STRONG = "strong"
    NORMAL = "normal"
    WEAK = "weak"
    OPPORTUNISTIC = "opportunistic"
    NONE = "none"


@dataclass
class MTFSignal:
    """Multi-timeframe signal result."""
    direction: str  # "long", "short", "flat"
    strength: SignalStrength
    size_multiplier: float
    prob_1h: float
    prob_15m: float
    decision_1h: str
    decision_15m: str
    reason: str
    timestamp: datetime


@dataclass
class MTFConfig:
    """Multi-timeframe strategy configuration."""
    # 1h thresholds (existing)
    threshold_1h_long: float = 0.60
    threshold_1h_short: float = 0.40

    # 15min thresholds (tighter since higher accuracy)
    threshold_15m_long: float = 0.55
    threshold_15m_short: float = 0.45

    # Size multipliers
    strong_mult: float = 1.5      # Both timeframes agree
    normal_mult: float = 1.0      # 1h signal, 15m neutral
    weak_mult: float = 0.5        # 1h signal, 15m disagrees
    opportunistic_mult: float = 0.7  # 15m only signal

    # Skip conflicting signals?
    skip_conflicting: bool = False

    # Allow 15min-only trades?
    allow_15m_only: bool = True


def get_decision(prob: float, threshold_long: float, threshold_short: float) -> str:
    """Convert probability to decision."""
    if prob >= threshold_long:
        return "long"
    elif prob <= threshold_short:
        return "short"
    return "flat"


def evaluate_mtf_signal(
    prob_1h: float,
    prob_15m: float,
    config: Optional[MTFConfig] = None,
    timestamp: Optional[datetime] = None,
) -> MTFSignal:
    """Evaluate multi-timeframe signal.

    Args:
        prob_1h: Probability from 1h model
        prob_15m: Probability from 15min model
        config: Strategy configuration
        timestamp: Signal timestamp

    Returns:
        MTFSignal with combined decision
    """
    if config is None:
        config = MTFConfig()

    if timestamp is None:
        timestamp = datetime.now()

    # Get individual decisions
    decision_1h = get_decision(prob_1h, config.threshold_1h_long, config.threshold_1h_short)
    decision_15m = get_decision(prob_15m, config.threshold_15m_long, config.threshold_15m_short)

    # === LONG signals ===
    if decision_1h == "long":
        if decision_15m == "long":
            # Both agree → STRONG LONG
            return MTFSignal(
                direction="long",
                strength=SignalStrength.STRONG,
                size_multiplier=config.strong_mult,
                prob_1h=prob_1h,
                prob_15m=prob_15m,
                decision_1h=decision_1h,
                decision_15m=decision_15m,
                reason="1h_long_15m_confirm",
                timestamp=timestamp,
            )
        elif decision_15m == "flat":
            # 1h long, 15m neutral → NORMAL LONG
            return MTFSignal(
                direction="long",
                strength=SignalStrength.NORMAL,
                size_multiplier=config.normal_mult,
                prob_1h=prob_1h,
                prob_15m=prob_15m,
                decision_1h=decision_1h,
                decision_15m=decision_15m,
                reason="1h_long_15m_neutral",
                timestamp=timestamp,
            )
        else:  # decision_15m == "short"
            # 1h long, 15m short → CONFLICT
            if config.skip_conflicting:
                return MTFSignal(
                    direction="flat",
                    strength=SignalStrength.NONE,
                    size_multiplier=0.0,
                    prob_1h=prob_1h,
                    prob_15m=prob_15m,
                    decision_1h=decision_1h,
                    decision_15m=decision_15m,
                    reason="conflict_skip",
                    timestamp=timestamp,
                )
            return MTFSignal(
                direction="long",
                strength=SignalStrength.WEAK,
                size_multiplier=config.weak_mult,
                prob_1h=prob_1h,
                prob_15m=prob_15m,
                decision_1h=decision_1h,
                decision_15m=decision_15m,
                reason="1h_long_15m_conflict",
                timestamp=timestamp,
            )

    # === SHORT signals ===
    elif decision_1h == "short":
        if decision_15m == "short":
            # Both agree → STRONG SHORT
            return MTFSignal(
                direction="short",
                strength=SignalStrength.STRONG,
                size_multiplier=config.strong_mult,
                prob_1h=prob_1h,
                prob_15m=prob_15m,
                decision_1h=decision_1h,
                decision_15m=decision_15m,
                reason="1h_short_15m_confirm",
                timestamp=timestamp,
            )
        elif decision_15m == "flat":
            # 1h short, 15m neutral → NORMAL SHORT
            return MTFSignal(
                direction="short",
                strength=SignalStrength.NORMAL,
                size_multiplier=config.normal_mult,
                prob_1h=prob_1h,
                prob_15m=prob_15m,
                decision_1h=decision_1h,
                decision_15m=decision_15m,
                reason="1h_short_15m_neutral",
                timestamp=timestamp,
            )
        else:  # decision_15m == "long"
            # 1h short, 15m long → CONFLICT
            if config.skip_conflicting:
                return MTFSignal(
                    direction="flat",
                    strength=SignalStrength.NONE,
                    size_multiplier=0.0,
                    prob_1h=prob_1h,
                    prob_15m=prob_15m,
                    decision_1h=decision_1h,
                    decision_15m=decision_15m,
                    reason="conflict_skip",
                    timestamp=timestamp,
                )
            return MTFSignal(
                direction="short",
                strength=SignalStrength.WEAK,
                size_multiplier=config.weak_mult,
                prob_1h=prob_1h,
                prob_15m=prob_15m,
                decision_1h=decision_1h,
                decision_15m=decision_15m,
                reason="1h_short_15m_conflict",
                timestamp=timestamp,
            )

    # === 1h FLAT ===
    else:  # decision_1h == "flat"
        if config.allow_15m_only:
            if decision_15m == "long":
                # 15m-only long → OPPORTUNISTIC
                return MTFSignal(
                    direction="long",
                    strength=SignalStrength.OPPORTUNISTIC,
                    size_multiplier=config.opportunistic_mult,
                    prob_1h=prob_1h,
                    prob_15m=prob_15m,
                    decision_1h=decision_1h,
                    decision_15m=decision_15m,
                    reason="15m_only_long",
                    timestamp=timestamp,
                )
            elif decision_15m == "short":
                # 15m-only short → OPPORTUNISTIC
                return MTFSignal(
                    direction="short",
                    strength=SignalStrength.OPPORTUNISTIC,
                    size_multiplier=config.opportunistic_mult,
                    prob_1h=prob_1h,
                    prob_15m=prob_15m,
                    decision_1h=decision_1h,
                    decision_15m=decision_15m,
                    reason="15m_only_short",
                    timestamp=timestamp,
                )

        # Both flat → NO SIGNAL
        return MTFSignal(
            direction="flat",
            strength=SignalStrength.NONE,
            size_multiplier=0.0,
            prob_1h=prob_1h,
            prob_15m=prob_15m,
            decision_1h=decision_1h,
            decision_15m=decision_15m,
            reason="no_signal",
            timestamp=timestamp,
        )


def should_exit_early(
    current_position: str,  # "long" or "short"
    prob_15m: float,
    config: Optional[MTFConfig] = None,
) -> tuple[bool, str]:
    """Check if 15min signal suggests early exit.

    Args:
        current_position: Current position direction
        prob_15m: Current 15min probability
        config: Strategy configuration

    Returns:
        (should_exit, reason)
    """
    if config is None:
        config = MTFConfig()

    decision_15m = get_decision(prob_15m, config.threshold_15m_long, config.threshold_15m_short)

    if current_position == "long" and decision_15m == "short":
        return True, "15m_reversal_signal"

    if current_position == "short" and decision_15m == "long":
        return True, "15m_reversal_signal"

    return False, ""

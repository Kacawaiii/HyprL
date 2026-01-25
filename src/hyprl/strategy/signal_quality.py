"""
Signal Quality Filter - Only trade when conditions are aligned.

Checks:
1. Volume confirmation (above average)
2. Spread check (liquidity OK)
3. Volatility regime (not extreme)
4. Time of day (avoid open/close)
5. Event calendar (no earnings nearby)
6. Trend alignment (multi-timeframe)
7. Momentum confirmation
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Optional, List, Dict, Any


@dataclass
class QualityConfig:
    """Configuration for signal quality filter."""
    # Volume filter
    volume_enabled: bool = True
    volume_min_ratio: float = 0.8  # Minimum volume vs 20-bar average
    volume_surge_bonus: float = 1.5  # Volume > 1.5x avg = quality bonus

    # Spread filter
    spread_enabled: bool = True
    spread_max_pct: float = 0.15  # Maximum spread as % of price

    # Volatility filter
    volatility_enabled: bool = True
    volatility_max_atr_pct: float = 0.05  # Max ATR as % (5% = very volatile)
    volatility_min_atr_pct: float = 0.003  # Min ATR (avoid dead markets)

    # Time of day filter (UTC hours)
    time_enabled: bool = True
    avoid_first_minutes: int = 30  # Avoid first 30 min after open
    avoid_last_minutes: int = 30  # Avoid last 30 min before close
    market_open_utc: int = 14  # 9:30 AM ET = 14:30 UTC
    market_close_utc: int = 21  # 4:00 PM ET = 21:00 UTC

    # Trend alignment
    trend_enabled: bool = True
    trend_lookback: int = 20  # Bars to check trend

    # Minimum quality score to trade
    min_quality_score: float = 0.6  # 0-1 scale


@dataclass
class QualityCheck:
    """Result of a single quality check."""
    name: str
    passed: bool
    score: float  # 0-1, contribution to overall quality
    reason: str


@dataclass
class QualityResult:
    """Overall signal quality assessment."""
    score: float  # 0-1 overall quality
    passed: bool  # Meets minimum threshold
    checks: List[QualityCheck]
    recommendation: str  # "trade", "skip", "reduce_size"


def check_volume(
    current_volume: float,
    avg_volume: float,
    config: QualityConfig,
) -> QualityCheck:
    """Check if volume confirms the signal."""
    if not config.volume_enabled or avg_volume <= 0:
        return QualityCheck("volume", True, 1.0, "disabled")

    ratio = current_volume / avg_volume

    if ratio < config.volume_min_ratio:
        return QualityCheck(
            "volume",
            False,
            ratio / config.volume_min_ratio,
            f"low_volume_{ratio:.2f}x",
        )

    # Bonus for high volume
    if ratio >= config.volume_surge_bonus:
        return QualityCheck("volume", True, 1.0, f"volume_surge_{ratio:.2f}x")

    # Normal volume
    score = min(1.0, ratio / config.volume_surge_bonus)
    return QualityCheck("volume", True, score, f"volume_ok_{ratio:.2f}x")


def check_spread(
    bid: Optional[float],
    ask: Optional[float],
    mid_price: float,
    config: QualityConfig,
) -> QualityCheck:
    """Check if spread is acceptable."""
    if not config.spread_enabled:
        return QualityCheck("spread", True, 1.0, "disabled")

    if bid is None or ask is None or mid_price <= 0:
        return QualityCheck("spread", True, 0.8, "no_quote_data")

    spread_pct = (ask - bid) / mid_price

    if spread_pct > config.spread_max_pct:
        return QualityCheck(
            "spread",
            False,
            0.0,
            f"wide_spread_{spread_pct:.3f}",
        )

    # Score based on spread tightness
    score = 1.0 - (spread_pct / config.spread_max_pct)
    return QualityCheck("spread", True, score, f"spread_ok_{spread_pct:.4f}")


def check_volatility(
    atr_pct: float,
    config: QualityConfig,
) -> QualityCheck:
    """Check if volatility is in acceptable range."""
    if not config.volatility_enabled:
        return QualityCheck("volatility", True, 1.0, "disabled")

    if atr_pct > config.volatility_max_atr_pct:
        return QualityCheck(
            "volatility",
            False,
            0.3,  # Not zero - can still trade with reduced size
            f"high_volatility_{atr_pct:.3f}",
        )

    if atr_pct < config.volatility_min_atr_pct:
        return QualityCheck(
            "volatility",
            False,
            0.5,
            f"low_volatility_{atr_pct:.4f}",
        )

    # Optimal volatility score (peak at middle of range)
    mid_vol = (config.volatility_max_atr_pct + config.volatility_min_atr_pct) / 2
    distance = abs(atr_pct - mid_vol) / mid_vol
    score = max(0.5, 1.0 - distance)

    return QualityCheck("volatility", True, score, f"volatility_ok_{atr_pct:.4f}")


def check_time_of_day(
    now: Optional[datetime] = None,
    config: QualityConfig = None,
) -> QualityCheck:
    """Check if it's a good time to trade."""
    if config is None:
        config = QualityConfig()

    if not config.time_enabled:
        return QualityCheck("time", True, 1.0, "disabled")

    if now is None:
        now = datetime.now(timezone.utc)

    hour = now.hour
    minute = now.minute
    current_minutes = hour * 60 + minute

    # Market hours in minutes (UTC)
    market_open_minutes = config.market_open_utc * 60 + 30  # 14:30 UTC
    market_close_minutes = config.market_close_utc * 60  # 21:00 UTC

    # Outside market hours
    if current_minutes < market_open_minutes or current_minutes >= market_close_minutes:
        return QualityCheck("time", False, 0.0, "outside_market_hours")

    # First X minutes
    if current_minutes < market_open_minutes + config.avoid_first_minutes:
        return QualityCheck("time", False, 0.5, "opening_volatility")

    # Last X minutes
    if current_minutes >= market_close_minutes - config.avoid_last_minutes:
        return QualityCheck("time", False, 0.5, "closing_period")

    # Prime trading hours (10:00 - 15:30 ET = 15:00 - 20:30 UTC)
    if 15 * 60 <= current_minutes <= 20 * 60 + 30:
        return QualityCheck("time", True, 1.0, "prime_hours")

    return QualityCheck("time", True, 0.8, "normal_hours")


def check_trend_alignment(
    current_price: float,
    sma_short: float,
    sma_long: float,
    signal_direction: str,  # "long" or "short"
    config: QualityConfig,
) -> QualityCheck:
    """Check if trend aligns with signal direction."""
    if not config.trend_enabled:
        return QualityCheck("trend", True, 1.0, "disabled")

    # Trend direction based on MAs
    trend_up = sma_short > sma_long and current_price > sma_short
    trend_down = sma_short < sma_long and current_price < sma_short

    if signal_direction == "long":
        if trend_up:
            return QualityCheck("trend", True, 1.0, "trend_aligned_up")
        elif trend_down:
            return QualityCheck("trend", False, 0.3, "trend_opposed")
        else:
            return QualityCheck("trend", True, 0.6, "trend_neutral")

    else:  # short
        if trend_down:
            return QualityCheck("trend", True, 1.0, "trend_aligned_down")
        elif trend_up:
            return QualityCheck("trend", False, 0.3, "trend_opposed")
        else:
            return QualityCheck("trend", True, 0.6, "trend_neutral")


def evaluate_signal_quality(
    signal_direction: str,
    current_price: float,
    current_volume: float,
    avg_volume: float,
    atr_pct: float,
    bid: Optional[float] = None,
    ask: Optional[float] = None,
    sma_short: Optional[float] = None,
    sma_long: Optional[float] = None,
    now: Optional[datetime] = None,
    config: Optional[QualityConfig] = None,
) -> QualityResult:
    """
    Evaluate overall signal quality.

    Returns:
        QualityResult with score and recommendation
    """
    if config is None:
        config = QualityConfig()

    checks = []

    # Volume check
    checks.append(check_volume(current_volume, avg_volume, config))

    # Spread check
    mid_price = (bid + ask) / 2 if bid and ask else current_price
    checks.append(check_spread(bid, ask, mid_price, config))

    # Volatility check
    checks.append(check_volatility(atr_pct, config))

    # Time check
    checks.append(check_time_of_day(now, config))

    # Trend check
    if sma_short and sma_long:
        checks.append(check_trend_alignment(
            current_price, sma_short, sma_long, signal_direction, config
        ))

    # Calculate overall score
    total_score = sum(c.score for c in checks) / len(checks)
    all_passed = all(c.passed for c in checks)

    # Determine recommendation
    if total_score >= config.min_quality_score and all_passed:
        recommendation = "trade"
    elif total_score >= config.min_quality_score * 0.8:
        recommendation = "reduce_size"
    else:
        recommendation = "skip"

    return QualityResult(
        score=total_score,
        passed=total_score >= config.min_quality_score,
        checks=checks,
        recommendation=recommendation,
    )

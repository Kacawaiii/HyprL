"""
Rule-based filter for ML signals.
Prevents buying falling knives and selling bottoms.

Use: Wrap ML signals with sanity checks based on trend/momentum/volatility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class FilterResult:
    allowed: bool
    original_direction: str
    filtered_direction: str
    reason: str
    trend: str
    rsi: float
    momentum: float
    volatility: float


def compute_filter_indicators(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> Optional[dict]:
    """Compute indicators for filtering."""

    n = len(closes)
    if n < 50:
        return None

    price = closes[-1]

    # SMAs
    sma_20 = np.mean(closes[-20:]) if n >= 20 else price
    sma_50 = np.mean(closes[-50:]) if n >= 50 else price
    sma_200 = np.mean(closes[-200:]) if n >= 200 else sma_50

    # RSI 14
    if n >= 15:
        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50

    # ATR 14
    if n >= 15:
        tr_values = []
        for i in range(-min(14, n-1), 0):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            tr_values.append(max(high_low, high_close, low_close))
        atr = np.mean(tr_values)
    else:
        atr = 0

    # Momentum: ROC 5 and ROC 10
    roc_5 = (closes[-1] - closes[-6]) / closes[-6] * 100 if n >= 6 else 0
    roc_10 = (closes[-1] - closes[-11]) / closes[-11] * 100 if n >= 11 else 0

    # Volatility (normalized)
    if n >= 21:
        returns = np.diff(closes[-21:]) / closes[-21:-1]
        volatility = np.std(returns) * 100  # en %
    else:
        volatility = 1.0

    # Trend detection
    if price > sma_50 > sma_200:
        trend = "bull"
    elif price < sma_50 < sma_200:
        trend = "bear"
    elif price > sma_50:
        trend = "weak_bull"
    elif price < sma_50:
        trend = "weak_bear"
    else:
        trend = "neutral"

    # Is price crashing right now? (more lenient thresholds)
    is_crashing = roc_5 < -5 and roc_10 < -8  # Was -3/-5
    is_pumping = roc_5 > 5 and roc_10 > 8  # Was 3/5

    # Is price at extreme?
    is_oversold = rsi < 30
    is_overbought = rsi > 70

    # Distance from SMA
    dist_sma20_pct = (price - sma_20) / sma_20 * 100
    dist_sma50_pct = (price - sma_50) / sma_50 * 100

    return {
        "price": price,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi": rsi,
        "atr": atr,
        "roc_5": roc_5,
        "roc_10": roc_10,
        "volatility": volatility,
        "trend": trend,
        "is_crashing": is_crashing,
        "is_pumping": is_pumping,
        "is_oversold": is_oversold,
        "is_overbought": is_overbought,
        "dist_sma20_pct": dist_sma20_pct,
        "dist_sma50_pct": dist_sma50_pct,
    }


def filter_ml_signal(
    ml_direction: str,
    ml_probability: float,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    min_prob: float = 0.52,
    require_trend_alignment: bool = False,  # Less strict
    block_falling_knife: bool = True,
    block_overbought_longs: bool = True,
    block_oversold_shorts: bool = True,
) -> FilterResult:
    """
    Filter ML signal with rule-based sanity checks.

    Blocks:
    - Longs when price is crashing (falling knife)
    - Longs when overbought and extended
    - Shorts when oversold
    - Any trade against the main trend
    - Low probability signals
    """

    ind = compute_filter_indicators(closes, highs, lows)

    if ind is None:
        return FilterResult(
            allowed=False,
            original_direction=ml_direction,
            filtered_direction="flat",
            reason="insufficient_data",
            trend="unknown",
            rsi=50,
            momentum=0,
            volatility=0,
        )

    trend = ind["trend"]
    rsi = ind["rsi"]
    roc_5 = ind["roc_5"]
    volatility = ind["volatility"]

    reasons = []
    blocked = False

    # Check 1: Minimum probability
    if ml_probability < min_prob:
        reasons.append(f"low_prob({ml_probability:.2f}<{min_prob})")
        blocked = True

    # Check 2: Long specific filters
    if ml_direction == "long" and not blocked:

        # Block falling knife: buying while crashing
        if block_falling_knife and ind["is_crashing"]:
            reasons.append(f"falling_knife(roc5={roc_5:.1f}%)")
            blocked = True

        # Block overbought longs (only at extremes)
        if block_overbought_longs and rsi > 80 and ind["dist_sma20_pct"] > 5:
            reasons.append(f"overbought_extended(rsi={rsi:.0f})")
            blocked = True

        # Block longs in bear trend
        if require_trend_alignment and trend in ("bear", "weak_bear"):
            reasons.append(f"against_trend({trend})")
            blocked = True

    # Check 3: Short specific filters
    if ml_direction == "short" and not blocked:

        # Block selling the bottom
        if block_oversold_shorts and ind["is_oversold"]:
            reasons.append(f"oversold(rsi={rsi:.0f})")
            blocked = True

        # Block shorts when pumping
        if ind["is_pumping"]:
            reasons.append(f"pumping(roc5={roc_5:.1f}%)")
            blocked = True

        # Block shorts in bull trend
        if require_trend_alignment and trend in ("bull", "weak_bull"):
            reasons.append(f"against_trend({trend})")
            blocked = True

    # Check 4: Extreme volatility filter (more lenient for crypto)
    if volatility > 8.0 and not blocked:  # >8% daily vol = crazy
        reasons.append(f"extreme_vol({volatility:.1f}%)")
        blocked = True

    # Determine final direction
    if blocked:
        filtered_direction = "flat"
        reason = "BLOCKED: " + ", ".join(reasons)
    else:
        filtered_direction = ml_direction
        reason = f"ALLOWED: trend={trend}, rsi={rsi:.0f}, mom={roc_5:.1f}%"

    return FilterResult(
        allowed=not blocked,
        original_direction=ml_direction,
        filtered_direction=filtered_direction,
        reason=reason,
        trend=trend,
        rsi=rsi,
        momentum=roc_5,
        volatility=volatility,
    )


def should_exit_position(
    position_side: Literal["long", "short"],
    entry_price: float,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    profit_pct: float,
) -> tuple[bool, str]:
    """
    Rule-based exit signals.

    Exit long if:
    - Trend flips to bear
    - RSI > 80 (extreme overbought)
    - Price crashes > 3% from recent high

    Exit short if:
    - Trend flips to bull
    - RSI < 20 (extreme oversold)
    """

    ind = compute_filter_indicators(closes, highs, lows)
    if ind is None:
        return False, "no_data"

    trend = ind["trend"]
    rsi = ind["rsi"]
    price = ind["price"]

    if position_side == "long":
        # Exit on trend flip
        if trend == "bear":
            return True, f"trend_flip_bear"

        # Exit on extreme overbought with profit
        if rsi > 80 and profit_pct > 1.0:
            return True, f"overbought_exit(rsi={rsi:.0f})"

        # Exit on breakdown below SMA50
        if price < ind["sma_50"] and profit_pct > 0:
            return True, f"sma50_breakdown"

    elif position_side == "short":
        # Exit on trend flip
        if trend == "bull":
            return True, f"trend_flip_bull"

        # Exit on extreme oversold with profit
        if rsi < 20 and profit_pct > 1.0:
            return True, f"oversold_exit(rsi={rsi:.0f})"

        # Exit on breakout above SMA50
        if price > ind["sma_50"] and profit_pct > 0:
            return True, f"sma50_breakout"

    return False, "hold"

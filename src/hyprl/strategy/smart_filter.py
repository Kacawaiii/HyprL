"""
Smart filter for ML signals - Focus on high-impact blocks only.

Only blocks trades that are statistically likely to fail:
1. Buying during a crash (falling knife)
2. Selling during a pump (catching falling short)
3. Trading against strong momentum

Less restrictive than rule_filter.py - only blocks the worst cases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class SmartFilterResult:
    allowed: bool
    direction: str
    reason: str
    momentum_5: float
    momentum_20: float
    rsi: float
    trend_strength: float


def compute_momentum_indicators(closes: np.ndarray) -> dict:
    """Compute momentum-based indicators."""

    n = len(closes)
    if n < 50:
        return {}

    price = closes[-1]

    # Momentum at different lookbacks
    mom_5 = (price - closes[-6]) / closes[-6] * 100 if n >= 6 else 0
    mom_10 = (price - closes[-11]) / closes[-11] * 100 if n >= 11 else 0
    mom_20 = (price - closes[-21]) / closes[-21] * 100 if n >= 21 else 0

    # RSI 14
    if n >= 15:
        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) + 1e-10
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    else:
        rsi = 50

    # Trend strength: how far from mean
    sma_50 = np.mean(closes[-50:])
    trend_strength = (price - sma_50) / sma_50 * 100

    # Consecutive down/up bars
    recent = closes[-6:]
    down_bars = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])
    up_bars = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])

    # Acceleration (momentum of momentum)
    mom_accel = mom_5 - mom_10  # Positive = accelerating up, negative = accelerating down

    return {
        "price": price,
        "mom_5": mom_5,
        "mom_10": mom_10,
        "mom_20": mom_20,
        "rsi": rsi,
        "trend_strength": trend_strength,
        "down_bars": down_bars,
        "up_bars": up_bars,
        "mom_accel": mom_accel,
    }


def smart_filter(
    ml_direction: str,
    ml_probability: float,
    closes: np.ndarray,
    min_prob: float = 0.52,
) -> SmartFilterResult:
    """
    Smart filter - only block high-risk trades.

    Blocks LONG if:
    - Crashing hard: mom_5 < -6% AND mom_10 < -10% (falling knife)
    - 5 out of 5 last bars are down (capitulation)
    - RSI < 15 AND still falling (extreme panic)

    Blocks SHORT if:
    - Pumping hard: mom_5 > 6% AND mom_10 > 10%
    - 5 out of 5 last bars are up
    - RSI > 85 AND still rising

    Everything else: ALLOW (let ML decide)
    """

    ind = compute_momentum_indicators(closes)

    if not ind:
        return SmartFilterResult(
            allowed=False,
            direction="flat",
            reason="insufficient_data",
            momentum_5=0,
            momentum_20=0,
            rsi=50,
            trend_strength=0,
        )

    mom_5 = ind["mom_5"]
    mom_10 = ind["mom_10"]
    mom_20 = ind["mom_20"]
    rsi = ind["rsi"]
    trend_strength = ind["trend_strength"]
    down_bars = ind["down_bars"]
    up_bars = ind["up_bars"]
    mom_accel = ind["mom_accel"]

    blocked = False
    reason = "allowed"

    # Low probability filter
    if ml_probability < min_prob:
        blocked = True
        reason = f"low_prob({ml_probability:.2f})"

    # LONG specific blocks
    elif ml_direction == "long":

        # Falling knife: hard crash in progress
        if mom_5 < -6 and mom_10 < -10:
            blocked = True
            reason = f"falling_knife(mom5={mom_5:.1f}%, mom10={mom_10:.1f}%)"

        # Capitulation: 5 down bars in a row
        elif down_bars >= 5:
            blocked = True
            reason = f"capitulation({down_bars}_down_bars)"

        # Extreme panic with continued selling
        elif rsi < 15 and mom_accel < -2:
            blocked = True
            reason = f"extreme_panic(rsi={rsi:.0f}, accel={mom_accel:.1f})"

    # SHORT specific blocks
    elif ml_direction == "short":

        # Pump in progress
        if mom_5 > 6 and mom_10 > 10:
            blocked = True
            reason = f"pump_in_progress(mom5={mom_5:.1f}%, mom10={mom_10:.1f}%)"

        # Melt-up: 5 up bars in a row
        elif up_bars >= 5:
            blocked = True
            reason = f"meltup({up_bars}_up_bars)"

        # Extreme FOMO with continued buying
        elif rsi > 85 and mom_accel > 2:
            blocked = True
            reason = f"extreme_fomo(rsi={rsi:.0f}, accel={mom_accel:.1f})"

    if not blocked:
        reason = f"allowed(mom5={mom_5:.1f}%, rsi={rsi:.0f})"

    return SmartFilterResult(
        allowed=not blocked,
        direction="flat" if blocked else ml_direction,
        reason=reason,
        momentum_5=mom_5,
        momentum_20=mom_20,
        rsi=rsi,
        trend_strength=trend_strength,
    )

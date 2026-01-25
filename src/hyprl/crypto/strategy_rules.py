"""
Pure rule-based crypto strategy - No ML, just math.

Strategy: Trend Following with Pullback Entry
- Trade in direction of trend (SMA 50/200)
- Enter on pullbacks (RSI < 40 in uptrend, > 60 in downtrend)
- ATR-based stops and targets
- No overfitting, no training needed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class RuleSignal:
    direction: Literal["long", "short", "flat"]
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    reason: str


def compute_indicators(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> dict:
    """Compute all indicators from OHLC data."""

    n = len(closes)
    if n < 200:
        return {}

    # SMAs
    sma_20 = np.mean(closes[-20:])
    sma_50 = np.mean(closes[-50:])
    sma_200 = np.mean(closes[-200:])

    # RSI 14
    deltas = np.diff(closes[-15:])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains) if len(gains) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 1e-10
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # ATR 14
    tr_values = []
    for i in range(-14, 0):
        high_low = highs[i] - lows[i]
        high_close = abs(highs[i] - closes[i - 1])
        low_close = abs(lows[i] - closes[i - 1])
        tr_values.append(max(high_low, high_close, low_close))
    atr = np.mean(tr_values)

    # Momentum (ROC 10)
    roc_10 = (closes[-1] - closes[-11]) / closes[-11] * 100

    # Volatility (std of returns)
    returns = np.diff(closes[-21:]) / closes[-21:-1]
    volatility = np.std(returns)

    # Distance from SMA 50 (mean reversion signal)
    dist_sma50_pct = (closes[-1] - sma_50) / sma_50 * 100

    return {
        "price": closes[-1],
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "rsi": rsi,
        "atr": atr,
        "roc_10": roc_10,
        "volatility": volatility,
        "dist_sma50_pct": dist_sma50_pct,
    }


def detect_trend(ind: dict) -> Literal["bull", "bear", "neutral"]:
    """Detect market trend."""
    price = ind["price"]
    sma_50 = ind["sma_50"]
    sma_200 = ind["sma_200"]

    # Strong bull: price > SMA50 > SMA200
    if price > sma_50 > sma_200:
        return "bull"

    # Strong bear: price < SMA50 < SMA200
    if price < sma_50 < sma_200:
        return "bear"

    return "neutral"


def generate_signal(
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atr_stop_mult: float = 2.0,
    atr_target_mult: float = 3.0,
    rsi_oversold: float = 35,
    rsi_overbought: float = 65,
    allow_shorts: bool = False,
) -> RuleSignal:
    """
    Generate trading signal based on rules.

    Rules:
    1. Trend: Determined by SMA 50/200 alignment
    2. Entry: Pullback to SMA 20 zone + RSI confirmation
    3. Stop: ATR-based (2x ATR default)
    4. Target: ATR-based (3x ATR default) = 1.5 R/R
    """

    ind = compute_indicators(closes, highs, lows)

    if not ind:
        return RuleSignal(
            direction="flat",
            entry_price=0,
            stop_loss=0,
            take_profit=0,
            confidence=0,
            reason="insufficient_data",
        )

    price = ind["price"]
    atr = ind["atr"]
    rsi = ind["rsi"]
    sma_20 = ind["sma_20"]
    sma_50 = ind["sma_50"]
    trend = detect_trend(ind)

    # Calculate distances
    dist_to_sma20_pct = abs(price - sma_20) / sma_20 * 100

    # Default: flat
    direction = "flat"
    confidence = 0.0
    reasons = []

    # LONG conditions
    if trend == "bull":
        reasons.append(f"trend=BULL")

        # Pullback entry: price near SMA20 + RSI not overbought
        pullback_ok = dist_to_sma20_pct < 3.0  # Within 3% of SMA20
        rsi_ok = rsi < rsi_overbought
        momentum_ok = ind["roc_10"] > -5  # Not crashing

        if pullback_ok and rsi_ok and momentum_ok:
            direction = "long"
            confidence = min(0.8, 0.5 + (rsi_overbought - rsi) / 100)
            reasons.append(f"pullback_entry")
        elif rsi < rsi_oversold:
            # Oversold bounce in uptrend
            direction = "long"
            confidence = 0.7
            reasons.append(f"oversold_bounce")
        else:
            reasons.append(f"wait_pullback")

    # SHORT conditions (if allowed)
    elif trend == "bear" and allow_shorts:
        reasons.append(f"trend=BEAR")

        pullback_ok = dist_to_sma20_pct < 3.0
        rsi_ok = rsi > rsi_oversold
        momentum_ok = ind["roc_10"] < 5

        if pullback_ok and rsi_ok and momentum_ok:
            direction = "short"
            confidence = min(0.8, 0.5 + (rsi - rsi_oversold) / 100)
            reasons.append(f"pullback_entry")
        elif rsi > rsi_overbought:
            direction = "short"
            confidence = 0.7
            reasons.append(f"overbought_fade")
        else:
            reasons.append(f"wait_pullback")

    else:
        reasons.append(f"trend={trend.upper()}")
        reasons.append("no_trade_zone")

    # Calculate stops and targets
    if direction == "long":
        stop_loss = price - (atr * atr_stop_mult)
        take_profit = price + (atr * atr_target_mult)
    elif direction == "short":
        stop_loss = price + (atr * atr_stop_mult)
        take_profit = price - (atr * atr_target_mult)
    else:
        stop_loss = price
        take_profit = price

    # Add indicator values to reason
    reasons.append(f"rsi={rsi:.0f}")
    reasons.append(f"atr={atr:.2f}")
    reasons.append(f"dist_sma20={dist_to_sma20_pct:.1f}%")

    return RuleSignal(
        direction=direction,
        entry_price=price,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        reason=", ".join(reasons),
    )


# Convenience function matching existing interface
def get_rule_based_signal(
    symbol: str,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
) -> dict:
    """Get signal in dict format compatible with existing bridge."""

    signal = generate_signal(
        closes=closes,
        highs=highs,
        lows=lows,
        allow_shorts=False,  # Crypto shorts disabled
    )

    # Convert to probability-like score for compatibility
    if signal.direction == "long":
        prob = 0.5 + (signal.confidence * 0.3)  # 0.5-0.8 range
    elif signal.direction == "short":
        prob = 0.5 - (signal.confidence * 0.3)  # 0.2-0.5 range
    else:
        prob = 0.5

    return {
        "symbol": symbol,
        "direction": signal.direction,
        "probability": prob,
        "confidence": signal.confidence,
        "entry_price": signal.entry_price,
        "stop_loss": signal.stop_loss,
        "take_profit": signal.take_profit,
        "reason": signal.reason,
    }

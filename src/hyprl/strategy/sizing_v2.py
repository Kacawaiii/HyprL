"""
Dynamic Position Sizing V2 - Conviction-based sizing.

Adjusts position size based on:
1. Signal probability (conviction)
2. Current volatility (ATR)
3. Account equity
4. Regime multiplier
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SizingConfig:
    """Configuration for dynamic sizing."""
    base_risk_pct: float = 0.01  # Base risk per trade (1%)
    min_conviction_mult: float = 0.5  # Multiplier for low conviction
    max_conviction_mult: float = 1.5  # Multiplier for high conviction
    conviction_low_threshold: float = 0.55  # Below this = low conviction
    conviction_high_threshold: float = 0.70  # Above this = high conviction
    vol_scaling_enabled: bool = True  # Scale by volatility
    vol_target: float = 0.15  # Target annualized volatility (15%)
    min_size_usd: float = 100.0  # Minimum position size
    max_size_pct: float = 0.20  # Maximum position size as % of equity


@dataclass
class SizingResult:
    """Result of position sizing calculation."""
    shares: int
    notional: float
    risk_amount: float
    conviction_mult: float
    vol_mult: float
    regime_mult: float
    reason: str


def compute_conviction_multiplier(
    probability: float,
    threshold: float,
    config: SizingConfig,
) -> float:
    """
    Compute sizing multiplier based on signal conviction.

    Higher probability = higher conviction = larger position.
    """
    # Distance from threshold (how confident is the signal?)
    if probability >= 0.5:
        # Long signal
        edge = probability - threshold
    else:
        # Short signal
        edge = (1 - probability) - (1 - threshold)

    # Normalize edge to multiplier
    if edge <= 0:
        return config.min_conviction_mult

    # Scale edge to conviction range
    # Edge of 0.1+ = high conviction
    if probability > config.conviction_high_threshold or probability < (1 - config.conviction_high_threshold):
        return config.max_conviction_mult
    elif probability < config.conviction_low_threshold and probability > (1 - config.conviction_low_threshold):
        return config.min_conviction_mult
    else:
        # Linear interpolation
        return 1.0


def compute_volatility_multiplier(
    current_atr_pct: float,
    config: SizingConfig,
) -> float:
    """
    Compute sizing multiplier based on current volatility.

    Higher volatility = smaller position (inverse scaling).
    """
    if not config.vol_scaling_enabled or current_atr_pct <= 0:
        return 1.0

    # Annualize ATR (assuming hourly bars, ~252 trading days, ~7 hours/day)
    annualized_vol = current_atr_pct * (252 * 7) ** 0.5

    # Target / Actual = multiplier
    # High vol = smaller position, Low vol = larger position
    if annualized_vol > 0:
        mult = config.vol_target / annualized_vol
        # Clamp to reasonable range
        return max(0.3, min(2.0, mult))

    return 1.0


def compute_dynamic_size(
    equity: float,
    entry_price: float,
    stop_price: float,
    probability: float,
    threshold: float,
    current_atr_pct: float = 0.0,
    regime_mult: float = 1.0,
    config: Optional[SizingConfig] = None,
) -> SizingResult:
    """
    Compute optimal position size using conviction-based sizing.

    Args:
        equity: Current account equity
        entry_price: Expected entry price
        stop_price: Stop loss price
        probability: Model probability (0-1)
        threshold: Decision threshold
        current_atr_pct: Current ATR as percentage of price
        regime_mult: Regime-based multiplier (from macro detector)
        config: Sizing configuration

    Returns:
        SizingResult with shares and metadata
    """
    if config is None:
        config = SizingConfig()

    # Base risk amount
    base_risk = equity * config.base_risk_pct

    # Conviction multiplier
    conviction_mult = compute_conviction_multiplier(probability, threshold, config)

    # Volatility multiplier
    vol_mult = compute_volatility_multiplier(current_atr_pct, config)

    # Combined multiplier
    total_mult = conviction_mult * vol_mult * regime_mult

    # Adjusted risk
    adjusted_risk = base_risk * total_mult

    # Risk per share (distance to stop)
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share <= 0:
        return SizingResult(
            shares=0,
            notional=0.0,
            risk_amount=0.0,
            conviction_mult=conviction_mult,
            vol_mult=vol_mult,
            regime_mult=regime_mult,
            reason="invalid_stop",
        )

    # Shares based on risk
    shares_float = adjusted_risk / risk_per_share
    notional = shares_float * entry_price

    # Apply max size constraint
    max_notional = equity * config.max_size_pct
    if notional > max_notional:
        shares_float = max_notional / entry_price
        notional = max_notional

    # Apply min size constraint
    if notional < config.min_size_usd:
        return SizingResult(
            shares=0,
            notional=0.0,
            risk_amount=0.0,
            conviction_mult=conviction_mult,
            vol_mult=vol_mult,
            regime_mult=regime_mult,
            reason="below_min_size",
        )

    # Round to integer shares
    shares = int(shares_float)
    if shares < 1:
        shares = 1

    final_notional = shares * entry_price
    final_risk = shares * risk_per_share

    return SizingResult(
        shares=shares,
        notional=final_notional,
        risk_amount=final_risk,
        conviction_mult=conviction_mult,
        vol_mult=vol_mult,
        regime_mult=regime_mult,
        reason="ok",
    )

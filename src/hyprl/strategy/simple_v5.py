"""
HYPRL v5 Simple Strategy
========================
Règles claires, pas de ML complexe.
"""

import pandas as pd
import numpy as np


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute only 4 essential features."""
    df = df.copy()
    
    # 1. Momentum 3h
    df['ret_3h'] = df['close'].pct_change(3)
    
    # 2. RSI-14
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 3. ATR normalized
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift(1))
    low_close = abs(df['low'] - df['close'].shift(1))
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14'] = tr.rolling(14).mean()
    df['atr_norm'] = df['atr_14'] / df['close']
    
    # 4. Volume ratio
    df['vol_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    return df


def generate_signal(
    row: pd.Series,
    long_rsi_below: float = 45,
    long_momentum_above: float = 0.004,
    short_rsi_above: float = 65,
    short_momentum_below: float = -0.004,
    min_volume_ratio: float = 0.8
) -> tuple[str, float]:
    """
    Generate trading signal from a single row.

    Args:
        row: DataFrame row with computed features
        long_rsi_below: RSI threshold for long signals (default: 45)
        long_momentum_above: Momentum threshold for long signals (default: 0.4%)
        short_rsi_above: RSI threshold for short signals (default: 65)
        short_momentum_below: Momentum threshold for short signals (default: -0.4%)
        min_volume_ratio: Minimum volume ratio (default: 0.8)

    Returns:
        (direction, confidence) where direction is 'long', 'short', or 'flat'
    """
    rsi = row.get('rsi_14', 50)
    momentum = row.get('ret_3h', 0)
    vol_ratio = row.get('vol_ratio', 1)

    # LONG: RSI below threshold + positive momentum + decent volume
    if rsi < long_rsi_below and momentum > long_momentum_above and vol_ratio > min_volume_ratio:
        # Confidence based on how far RSI is below threshold
        rsi_score = (long_rsi_below - rsi) / long_rsi_below
        mom_score = min(1.0, momentum / 0.01)  # Scale momentum
        confidence = min(1.0, (rsi_score + mom_score) / 2)
        return 'long', confidence

    # SHORT: RSI above threshold + negative momentum + decent volume
    if rsi > short_rsi_above and momentum < short_momentum_below and vol_ratio > min_volume_ratio:
        # Confidence based on how far RSI is above threshold
        rsi_score = (rsi - short_rsi_above) / (100 - short_rsi_above)
        mom_score = min(1.0, abs(momentum) / 0.01)  # Scale momentum
        confidence = min(1.0, (rsi_score + mom_score) / 2)
        return 'short', confidence

    return 'flat', 0.0


def calculate_position_size(
    equity: float,
    entry_price: float,
    atr: float,
    risk_pct: float = 0.02,
    max_position_pct: float = 0.25,
    atr_multiplier: float = 1.5,
) -> tuple[float, float, float]:
    """
    Calculate position size based on ATR risk.
    
    Returns:
        (shares, stop_price_distance, position_value)
    """
    stop_distance = atr * atr_multiplier
    risk_amount = equity * risk_pct
    
    # Shares based on risk
    shares = risk_amount / stop_distance
    position_value = shares * entry_price
    
    # Cap at max position
    if position_value > equity * max_position_pct:
        position_value = equity * max_position_pct
        shares = position_value / entry_price
    
    return shares, stop_distance, position_value

"""
Enhanced Features V3 - Additional predictive features.

New features:
1. VWAP distance - Price relative to volume-weighted average
2. Overnight gap - Gap from previous close
3. Order flow imbalance proxy - Volume on up vs down bars
4. Sector momentum - Relative strength vs sector/market
5. Intraday patterns - Time-of-day effects
6. Microstructure - High-low range patterns
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_vwap(prices: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Compute Volume-Weighted Average Price.

    Args:
        prices: DataFrame with high, low, close, volume
        window: Rolling window for VWAP

    Returns:
        VWAP series
    """
    typical_price = (prices["high"] + prices["low"] + prices["close"]) / 3
    volume = prices.get("volume", pd.Series(1, index=prices.index))

    vwap = (typical_price * volume).rolling(window).sum() / volume.rolling(window).sum()
    return vwap


def compute_vwap_distance(prices: pd.DataFrame, window: int = 24) -> pd.Series:
    """
    Compute distance from VWAP as percentage.

    Positive = price above VWAP (bullish)
    Negative = price below VWAP (bearish)
    """
    vwap = compute_vwap(prices, window)
    close = prices["close"]

    distance = (close - vwap) / vwap
    return distance.replace([np.inf, -np.inf], np.nan)


def compute_overnight_gap(prices: pd.DataFrame) -> pd.Series:
    """
    Compute overnight gap from previous close.

    For intraday data, this detects gaps between sessions.
    """
    close = prices["close"]
    open_price = prices["open"]

    # Gap = (Open - Previous Close) / Previous Close
    prev_close = close.shift(1)
    gap = (open_price - prev_close) / prev_close

    return gap.replace([np.inf, -np.inf], np.nan)


def compute_order_flow_imbalance(prices: pd.DataFrame, window: int = 10) -> pd.Series:
    """
    Proxy for order flow imbalance using volume on up vs down bars.

    Positive = more volume on up bars (buying pressure)
    Negative = more volume on down bars (selling pressure)
    """
    close = prices["close"]
    volume = prices.get("volume", pd.Series(1, index=prices.index))

    # Determine bar direction
    bar_direction = np.sign(close - close.shift(1))

    # Volume on up bars vs down bars
    up_volume = (volume * (bar_direction > 0)).rolling(window).sum()
    down_volume = (volume * (bar_direction < 0)).rolling(window).sum()
    total_volume = up_volume + down_volume

    # Imbalance ratio
    imbalance = (up_volume - down_volume) / total_volume.replace(0, np.nan)

    return imbalance


def compute_relative_strength(
    prices: pd.DataFrame,
    benchmark: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Compute relative strength vs benchmark.

    Positive = outperforming benchmark
    Negative = underperforming benchmark
    """
    asset_returns = prices["close"].pct_change(window)
    bench_returns = benchmark["close"].pct_change(window)

    # Align indices
    bench_aligned = bench_returns.reindex(asset_returns.index, method="ffill")

    relative_strength = asset_returns - bench_aligned
    return relative_strength


def compute_intraday_position(prices: pd.DataFrame) -> pd.Series:
    """
    Compute position within the day's range.

    0 = at day's low
    1 = at day's high
    0.5 = middle of range

    For hourly data, uses rolling 8-bar (1 trading day) window.
    """
    high = prices["high"].rolling(8).max()
    low = prices["low"].rolling(8).min()
    close = prices["close"]

    day_range = high - low
    position = (close - low) / day_range.replace(0, np.nan)

    return position.clip(0, 1)


def compute_range_expansion(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Detect range expansion (breakout potential).

    Values > 1 indicate range is expanding (trending)
    Values < 1 indicate range is contracting (consolidation)
    """
    current_range = prices["high"] - prices["low"]
    avg_range = current_range.rolling(window).mean()

    expansion = current_range / avg_range.replace(0, np.nan)
    return expansion


def compute_close_location(prices: pd.DataFrame) -> pd.Series:
    """
    Where did price close relative to bar's range?

    1 = closed at high (bullish)
    0 = closed at low (bearish)
    0.5 = closed at midpoint
    """
    high = prices["high"]
    low = prices["low"]
    close = prices["close"]

    bar_range = high - low
    location = (close - low) / bar_range.replace(0, np.nan)

    return location.clip(0, 1)


def compute_momentum_divergence(prices: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Detect momentum divergence (price vs RSI direction).

    Positive = bullish divergence (price down, momentum up)
    Negative = bearish divergence (price up, momentum down)
    """
    from ta.momentum import RSIIndicator

    close = prices["close"]
    rsi = RSIIndicator(close=close, window=window).rsi()

    # 5-bar price change
    price_change = close.diff(5)
    rsi_change = rsi.diff(5)

    # Divergence = when they move in opposite directions
    divergence = -np.sign(price_change) * rsi_change

    return divergence


def compute_enhanced_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all enhanced V3 features.

    Args:
        prices: DataFrame with OHLCV data

    Returns:
        DataFrame with enhanced features
    """
    feats = pd.DataFrame(index=prices.index)

    # VWAP features
    feats["vwap_dist_24"] = compute_vwap_distance(prices, window=24)
    feats["vwap_dist_8"] = compute_vwap_distance(prices, window=8)

    # Gap feature
    feats["overnight_gap"] = compute_overnight_gap(prices)

    # Order flow proxy
    feats["flow_imbalance_10"] = compute_order_flow_imbalance(prices, window=10)
    feats["flow_imbalance_5"] = compute_order_flow_imbalance(prices, window=5)

    # Intraday patterns
    feats["intraday_position"] = compute_intraday_position(prices)
    feats["close_location"] = compute_close_location(prices)

    # Range analysis
    feats["range_expansion"] = compute_range_expansion(prices, window=14)

    # Momentum divergence
    feats["momentum_div"] = compute_momentum_divergence(prices, window=14)

    # Clean up
    feats = feats.replace([np.inf, -np.inf], np.nan)

    return feats


# List of new feature names for model training
ENHANCED_FEATURES = [
    "vwap_dist_24",
    "vwap_dist_8",
    "overnight_gap",
    "flow_imbalance_10",
    "flow_imbalance_5",
    "intraday_position",
    "close_location",
    "range_expansion",
    "momentum_div",
]

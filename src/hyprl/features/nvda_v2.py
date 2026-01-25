from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


def _rolling_func(series: pd.Series, window: int, func) -> pd.Series:
    return series.rolling(window=window, min_periods=window).apply(func, raw=True)


def compute_nvda_v2_features(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    volume = prices.get("volume", pd.Series(index=prices.index, dtype=float))

    feats = pd.DataFrame(index=prices.index)
    feats["returns_next"] = close.pct_change().shift(-1)

    # Momentum horizons
    feats["ret_1h"] = close.pct_change(1)
    feats["ret_3h"] = close.pct_change(3)
    feats["ret_6h"] = close.pct_change(6)
    feats["ret_24h"] = close.pct_change(24)

    # ATR stack
    atr_14 = AverageTrueRange(high=high, low=low, close=close, window=14).average_true_range()
    atr_72 = AverageTrueRange(high=high, low=low, close=close, window=72).average_true_range()
    feats["atr_14"] = atr_14
    feats["atr_72"] = atr_72
    feats["atr_14_norm"] = atr_14 / close
    feats["atr_72_norm"] = atr_72 / close

    # RSI stack
    feats["rsi_7"] = RSIIndicator(close=close, window=7).rsi() / 100.0
    feats["rsi_14"] = RSIIndicator(close=close, window=14).rsi() / 100.0
    feats["rsi_21"] = RSIIndicator(close=close, window=21).rsi() / 100.0

    # Volatility regime
    vol_10 = close.pct_change().rolling(window=10, min_periods=10).std()
    vol_30 = close.pct_change().rolling(window=30, min_periods=30).std()
    feats["vol_ratio_10_30"] = vol_10 / vol_30
    feats["vol_regime_high"] = (feats["vol_ratio_10_30"] > 1.0).astype(float)

    # Volume signals
    vol_mean_24 = volume.rolling(window=24, min_periods=24).mean()
    vol_std_24 = volume.rolling(window=24, min_periods=24).std(ddof=0)
    feats["volume_zscore_24"] = (volume - vol_mean_24) / vol_std_24.replace(0, np.nan)
    feats["volume_surge"] = (feats["volume_zscore_24"] > 2.0).astype(float)

    # Range and true range
    feats["range_pct"] = (high - low) / close
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    feats["true_range"] = true_range / close

    # Higher moments on returns
    returns_1h = close.pct_change()
    feats["ret_skew_20"] = _rolling_func(returns_1h, 20, skew)
    feats["ret_kurt_20"] = _rolling_func(returns_1h, 20, kurtosis)

    feats = feats.replace([np.inf, -np.inf], np.nan)
    required = [
        "ret_1h",
        "ret_3h",
        "ret_6h",
        "ret_24h",
        "atr_14",
        "atr_72",
        "atr_14_norm",
        "atr_72_norm",
        "rsi_7",
        "rsi_14",
        "rsi_21",
        "vol_ratio_10_30",
        "vol_regime_high",
        "volume_zscore_24",
        "volume_surge",
        "range_pct",
        "true_range",
        "ret_skew_20",
        "ret_kurt_20",
    ]
    feats = feats.dropna(subset=[c for c in required if c in feats.columns])
    return feats

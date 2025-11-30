from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange


@dataclass(slots=True)
class TechnicalIndicatorCalculator:
    closes: pd.Series

    def sma(self, window: int) -> pd.Series:
        return self.closes.rolling(window=window, min_periods=window).mean()

    def rsi(self, window: int = 14) -> pd.Series:
        indicator = RSIIndicator(close=self.closes, window=window)
        return indicator.rsi()


def compute_feature_frame(
    prices: pd.DataFrame,
    sma_short_window: int,
    sma_long_window: int,
    rsi_window: int,
    atr_window: int,
) -> pd.DataFrame:
    calc = TechnicalIndicatorCalculator(prices["close"])
    close = prices["close"]
    high = prices["high"]
    low = prices["low"]
    features = pd.DataFrame(index=prices.index)
    features["price"] = close
    features["returns_next"] = close.pct_change().shift(-1)

    features["sma_short"] = calc.sma(sma_short_window)
    features["sma_long"] = calc.sma(sma_long_window)
    features["sma_ratio"] = features["sma_short"] / features["sma_long"] - 1.0
    features["trend_ratio"] = features["sma_ratio"].clip(-0.05, 0.05)

    features["ema_short"] = close.ewm(span=sma_short_window, adjust=False).mean()
    features["ema_long"] = close.ewm(span=sma_long_window, adjust=False).mean()
    features["ema_ratio"] = features["ema_short"] / features["ema_long"] - 1.0

    rsi_raw = calc.rsi(rsi_window)
    features["rsi_raw"] = rsi_raw
    features["rsi_normalized"] = rsi_raw / 100.0

    features["volatility"] = close.pct_change().rolling(window=sma_short_window, min_periods=sma_short_window).std()

    atr_indicator = AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=atr_window,
    )
    atr_column = f"atr_{atr_window}"
    features[atr_column] = atr_indicator.average_true_range()
    features["atr_normalized"] = features[atr_column] / close

    features["range_pct"] = (high - low) / close
    features["rolling_return"] = close.pct_change(periods=rsi_window)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna()

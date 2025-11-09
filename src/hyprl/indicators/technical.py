from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

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
    features = pd.DataFrame(index=prices.index)
    features["price"] = prices["close"]
    features["sma_short"] = calc.sma(sma_short_window)
    features["sma_long"] = calc.sma(sma_long_window)
    features["rsi"] = calc.rsi(rsi_window)
    features["returns_next"] = prices["close"].pct_change().shift(-1)
    features["trend_ratio"] = (features["sma_short"] - features["sma_long"]) / features["sma_long"]
    features["trend_ratio"] = features["trend_ratio"].clip(-0.05, 0.05)
    features["rsi_normalized"] = (features["rsi"] - 50.0) / 50.0
    features["volatility"] = prices["close"].pct_change().rolling(window=sma_short_window).std()
    atr_indicator = AverageTrueRange(
        high=prices["high"],
        low=prices["low"],
        close=prices["close"],
        window=atr_window,
    )
    atr_column = f"atr_{atr_window}"
    features[atr_column] = atr_indicator.average_true_range()
    return features.dropna()

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(series: pd.Series, span: int) -> float:
    if len(series) < span:
        return np.nan
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _sma(series: pd.Series, window: int) -> float:
    if len(series) < window:
        return np.nan
    return float(series.rolling(window).mean().iloc[-1])


def _rsi(series: pd.Series, window: int) -> float:
    if len(series) < window + 1:
        return np.nan
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])


def _atr(df: pd.DataFrame, window: int) -> float:
    if len(df) < window + 1:
        return np.nan
    high_low = (df["high"] - df["low"]).abs()
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(window).mean().iloc[-1]
    return float(atr)


def _bollinger(series: pd.Series, window: int, num_std: float) -> tuple[float, float, float]:
    if len(series) < window:
        return np.nan, np.nan, np.nan
    rolling = series.rolling(window)
    mean = rolling.mean().iloc[-1]
    std = rolling.std(ddof=0).iloc[-1]
    upper = mean + num_std * std
    lower = mean - num_std * std
    width = (upper - lower) / mean if mean else np.nan
    return float(upper), float(lower), float(width)


def compute_features(bars: pd.DataFrame) -> dict[str, float]:
    """Compute technical features for the latest completed bar."""

    if bars.empty:
        return {}
    closes = bars["close"]
    high = bars["high"]
    low = bars["low"]
    volume = bars["volume"]

    sma_short = _sma(closes, 14)
    sma_long = _sma(closes, 50)
    rsi_raw = _rsi(closes, 14)
    ema_12 = _ema(closes, 12)
    ema_26 = _ema(closes, 26)
    macd = ema_12 - ema_26 if np.isfinite(ema_12) and np.isfinite(ema_26) else np.nan
    macd_series = closes.ewm(span=12, adjust=False).mean() - closes.ewm(span=26, adjust=False).mean()
    macd_signal = macd_series.ewm(span=9, adjust=False).mean().iloc[-1] if len(macd_series) >= 9 else np.nan
    macd_hist = macd - macd_signal if np.isfinite(macd) and np.isfinite(macd_signal) else np.nan
    bb_upper, bb_lower, bb_width = _bollinger(closes, window=20, num_std=2.0)
    atr = _atr(bars, 14)
    trend_ratio = (closes.iloc[-1] / sma_long - 1.0) if np.isfinite(sma_long) and sma_long else np.nan
    log_returns = np.log(closes / closes.shift()).dropna()
    volatility = float(log_returns.rolling(20).std(ddof=0).iloc[-1]) if len(log_returns) >= 20 else np.nan

    return {
        "sma_short": float(sma_short) if np.isfinite(sma_short) else np.nan,
        "sma_long": float(sma_long) if np.isfinite(sma_long) else np.nan,
        "rsi_raw": float(rsi_raw) if np.isfinite(rsi_raw) else np.nan,
        "macd": float(macd) if np.isfinite(macd) else np.nan,
        "macd_signal": float(macd_signal) if np.isfinite(macd_signal) else np.nan,
        "macd_hist": float(macd_hist) if np.isfinite(macd_hist) else np.nan,
        "bb_upper": float(bb_upper) if np.isfinite(bb_upper) else np.nan,
        "bb_lower": float(bb_lower) if np.isfinite(bb_lower) else np.nan,
        "bb_width": float(bb_width) if np.isfinite(bb_width) else np.nan,
        "atr": float(atr) if np.isfinite(atr) else np.nan,
        "trend_ratio": float(trend_ratio) if np.isfinite(trend_ratio) else np.nan,
        "volatility": float(volatility) if np.isfinite(volatility) else np.nan,
        "close": float(closes.iloc[-1]),
        "open": float(bars["open"].iloc[-1]),
        "volume": float(volume.iloc[-1]),
    }

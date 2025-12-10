from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

pl = pytest.importorskip("polars")
hy_sc = pytest.importorskip("hyprl_supercalc")


def _make_df(length: int) -> "pl.DataFrame":
    rng = np.random.default_rng(42 + length)
    base = 100.0 + rng.standard_normal(length).cumsum()
    open_prices = base + rng.normal(scale=0.2, size=length)
    high = np.maximum(open_prices, base) + rng.random(length)
    low = np.minimum(open_prices, base) - rng.random(length)
    close = base
    volume = rng.integers(1_000, 10_000, size=length).astype(float)
    ts = np.arange(length, dtype=np.int64) * 3_600_000
    return pl.DataFrame(
        {
            "ts": ts,
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def _python_reference(df: "pl.DataFrame") -> dict[str, np.ndarray]:
    pdf = df.to_pandas()
    close = pdf["close"]
    high = pdf["high"]
    low = pdf["low"]

    sma_20 = close.rolling(window=20, min_periods=20).mean().to_numpy()
    ema_20 = close.ewm(span=20, adjust=False).mean().to_numpy()
    rsi_14 = RSIIndicator(close=close, window=14).rsi().to_numpy()

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd_series = ema_12 - ema_26
    macd = macd_series.to_numpy()
    macd_signal = macd_series.ewm(span=9, adjust=False).mean().to_numpy()
    macd_hist = macd - macd_signal

    bb_mid = sma_20
    bb_std = close.rolling(window=20, min_periods=20).std(ddof=0).to_numpy()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std

    atr_series = AverageTrueRange(high=high, low=low, close=close, window=14)
    atr_14 = atr_series.average_true_range().to_numpy()

    sma_50 = close.rolling(window=50, min_periods=50).mean()
    sma_200 = close.rolling(window=200, min_periods=200).mean()
    ratio = (sma_50 / sma_200) - 1.0
    trend_ratio = ratio.clip(-0.05, 0.05).to_numpy()

    log_returns = np.log(close / close.shift(1))
    rolling_vol = log_returns.rolling(window=20, min_periods=20).std(ddof=0).to_numpy()

    return {
        "sma_20": sma_20,
        "ema_20": ema_20,
        "rsi_14": rsi_14,
        "macd": macd,
        "macd_signal": macd_signal,
        "macd_hist": macd_hist,
        "bb_upper_20": bb_upper,
        "bb_mid_20": bb_mid,
        "bb_lower_20": bb_lower,
        "atr_14": atr_14,
        "trend_ratio_50_200": trend_ratio,
        "rolling_vol_20": rolling_vol,
    }


def _assert_allclose(left, right, key: str, rtol=1e-6, atol=1e-8):
    np.testing.assert_allclose(
        np.asarray(left, dtype=float),
        np.asarray(right, dtype=float),
        rtol=rtol,
        atol=atol,
        equal_nan=True,
        err_msg=f"Mismatch on {key}",
    )


@pytest.mark.parametrize("length", [64, 256])
def test_compute_indicators_parity(length: int):
    df = _make_df(length)
    rust = hy_sc.compute_indicators_py(df)
    py = _python_reference(df)

    _assert_allclose(rust["sma_20"], py["sma_20"], "sma_20")
    _assert_allclose(rust["ema_20"], py["ema_20"], "ema_20")
    _assert_allclose(rust["rsi_14"], py["rsi_14"], "rsi_14", rtol=1e-5, atol=1e-5)
    _assert_allclose(rust["macd"], py["macd"], "macd", rtol=1e-5, atol=1e-5)
    _assert_allclose(
        rust["macd_signal"],
        py["macd_signal"],
        "macd_signal",
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_allclose(
        rust["macd_hist"],
        py["macd_hist"],
        "macd_hist",
        rtol=1e-5,
        atol=1e-5,
    )
    _assert_allclose(rust["bb_upper_20"], py["bb_upper_20"], "bb_upper_20")
    _assert_allclose(rust["bb_mid_20"], py["bb_mid_20"], "bb_mid_20")
    _assert_allclose(rust["bb_lower_20"], py["bb_lower_20"], "bb_lower_20")
    _assert_allclose(rust["atr_14"], py["atr_14"], "atr_14", rtol=1e-5, atol=1e-5)
    _assert_allclose(
        rust["trend_ratio_50_200"],
        py["trend_ratio_50_200"],
        "trend_ratio_50_200",
    )
    _assert_allclose(
        rust["rolling_vol_20"],
        py["rolling_vol_20"],
        "rolling_vol_20",
        rtol=1e-5,
        atol=1e-5,
    )

from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.indicators.technical import compute_feature_frame


def _synthetic_prices(rows: int = 256) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 120, rows)
    noise = np.sin(np.linspace(0, 6 * np.pi, rows)) * 0.8
    close = base + noise
    data = {
        "open": close - 0.2,
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "adj_close": close,
        "volume": np.linspace(5_000, 10_000, rows),
    }
    return pd.DataFrame(data, index=index)


def test_compute_feature_frame_emits_expected_columns() -> None:
    prices = _synthetic_prices()

    features = compute_feature_frame(prices, sma_short_window=5, sma_long_window=36, rsi_window=14, atr_window=14)

    expected_columns = {
        "price",
        "sma_short",
        "sma_long",
        "sma_ratio",
        "trend_ratio",
        "ema_short",
        "ema_long",
        "ema_ratio",
        "rsi_raw",
        "rsi_normalized",
        "volatility",
        "atr_14",
        "atr_normalized",
        "range_pct",
        "rolling_return",
        "returns_next",
    }
    assert expected_columns.issubset(set(features.columns))
    assert len(features) > 0


def test_compute_feature_frame_drops_nan_rows() -> None:
    prices = _synthetic_prices()

    features = compute_feature_frame(prices, sma_short_window=5, sma_long_window=36, rsi_window=14, atr_window=14)

    assert not features.isna().any().any()
    numeric_values = features.select_dtypes(include=[np.number]).to_numpy()
    assert np.isfinite(numeric_values).all()

from __future__ import annotations

import pandas as pd
import pytest

from hyprl.labels.amplitude import (
    BIG_DOWN,
    BIG_UP,
    LabelConfig,
    attach_amplitude_labels,
    compute_amplitude_labels,
    compute_symmetric_binary_labels,
    encode_amplitude_target,
    validate_label_support,
)


def _synthetic_prices(n_cycles: int = 4, bars_per_leg: int = 6) -> pd.DataFrame:
    """Zig-zag prices that alternate +1%/-1% moves to create both label classes."""
    closes: list[float] = []
    price = 100.0
    for _ in range(n_cycles):
        for _ in range(bars_per_leg):
            price *= 1.01
            closes.append(price)
        for _ in range(bars_per_leg):
            price *= 0.99
            closes.append(price)
    index = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    close_series = pd.Series(closes, index=index)
    frame = pd.DataFrame(
        {
            "open": close_series,
            "high": close_series * 1.01,
            "low": close_series * 0.99,
            "close": close_series,
            "adj_close": close_series,
            "volume": 1_000_000,
        },
        index=index,
    )
    return frame


def test_compute_amplitude_labels_marks_expected_classes():
    index = pd.date_range("2024-01-01", periods=9, freq="h", tz="UTC")
    close = pd.Series([100, 106, 112, 118, 124, 118, 112, 106, 100], index=index)
    cfg = LabelConfig(mode="amplitude", horizon=2, threshold_pct=4.0, min_samples_per_class=1)
    labels = compute_amplitude_labels(close, cfg.horizon, cfg.threshold_pct)
    assert labels.iloc[0] == BIG_UP
    assert labels.iloc[4] == BIG_DOWN
    assert pd.isna(labels.iloc[-1])


def test_attach_amplitude_labels_truncates_tail_rows():
    prices = _synthetic_prices(bars_per_leg=8)
    cfg = LabelConfig(mode="amplitude", horizon=4, threshold_pct=1.5, min_samples_per_class=1)
    features = pd.DataFrame(
        {
            "price": prices["close"],
            "returns_next": prices["close"].pct_change().shift(-1),
        },
        index=prices.index,
    ).dropna()
    initial_len = len(features)
    augmented = attach_amplitude_labels(features, prices, cfg)
    assert "label_amplitude" in augmented.columns
    assert len(augmented) == initial_len - (cfg.horizon - 1)
    assert set(augmented["label_amplitude"].unique()) >= {BIG_UP, BIG_DOWN}


def test_validate_label_support_requires_balanced_samples():
    prices = _synthetic_prices()
    cfg = LabelConfig(mode="amplitude", horizon=4, threshold_pct=4.0, min_samples_per_class=5)
    features = pd.DataFrame(
        {
            "price": prices["close"],
            "returns_next": prices["close"].pct_change().shift(-1),
        },
        index=prices.index,
    ).dropna()
    augmented = attach_amplitude_labels(features, prices, cfg)
    # threshold_pct=4 forces mostly BIG_DOWN labels → should raise
    with pytest.raises(ValueError):
        validate_label_support(augmented, cfg)


def test_encode_amplitude_target_maps_strings_to_binary():
    series = pd.Series([BIG_UP, BIG_DOWN, "NEUTRAL", None])
    encoded = encode_amplitude_target(series)
    assert encoded.iloc[0] == 1
    assert encoded.iloc[1] == 0
    assert pd.isna(encoded.iloc[2])
    assert pd.isna(encoded.iloc[3])


def test_compute_symmetric_binary_labels_marks_long_short_and_neutral():
    index = pd.date_range("2024-01-01", periods=6, freq="h", tz="UTC")
    close = pd.Series([100.0, 101.0, 101.0, 98.0, 98.0, 99.0], index=index)
    labels = compute_symmetric_binary_labels(close, horizon=1, threshold_pct=0.003)
    # Forward returns: +1.0%, 0.0%, -2.97%, 0.0%, +1.02%, NaN
    assert labels.iloc[0] == 1.0  # long
    assert labels.iloc[2] == 0.0  # short
    assert pd.isna(labels.iloc[1])  # neutral band
    assert pd.isna(labels.iloc[-1])  # tail shifted to NaN

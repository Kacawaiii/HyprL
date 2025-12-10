from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

BIG_UP = "BIG_UP"
BIG_DOWN = "BIG_DOWN"
NEUTRAL = "NEUTRAL"
TRAINABLE_LABELS = (BIG_DOWN, BIG_UP)
LabelMode = Literal["binary", "amplitude"]
NeutralStrategy = Literal["ignore", "keep"]


@dataclass(slots=True)
class LabelConfig:
    mode: LabelMode = "binary"
    horizon: int = 4
    threshold_pct: float = 1.5
    neutral_strategy: NeutralStrategy = "ignore"
    min_samples_per_class: int = 40

    def __post_init__(self) -> None:
        mode = (self.mode or "binary").lower()
        if mode not in {"binary", "amplitude"}:
            raise ValueError(f"Unsupported label mode: {self.mode}")
        object.__setattr__(self, "mode", mode)

        neutral = (self.neutral_strategy or "ignore").lower()
        if neutral not in {"ignore", "keep"}:
            raise ValueError(f"Unsupported neutral strategy: {self.neutral_strategy}")
        object.__setattr__(self, "neutral_strategy", neutral)

        horizon = max(1, int(self.horizon))
        object.__setattr__(self, "horizon", horizon)

        threshold = float(self.threshold_pct)
        if threshold <= 0:
            raise ValueError("label.threshold_pct must be > 0")
        object.__setattr__(self, "threshold_pct", threshold)

        min_samples = max(1, int(self.min_samples_per_class))
        object.__setattr__(self, "min_samples_per_class", min_samples)


def compute_amplitude_labels(prices: pd.DataFrame | pd.Series, horizon: int, threshold_pct: float) -> pd.Series:
    """Return BIG_UP/BIG_DOWN/NEUTRAL labels over the requested horizon."""
    if isinstance(prices, pd.DataFrame):
        close = prices["close"]
    else:
        close = prices
    if close.empty:
        return pd.Series(dtype="object")
    close = close.astype(float)
    horizon = max(1, int(horizon))
    forward = close.shift(-horizon)
    with np.errstate(invalid="ignore", divide="ignore"):
        future_return = forward / close - 1.0
    pct_threshold = float(threshold_pct) / 100.0
    labels = pd.Series(np.full(len(close), NEUTRAL, dtype=object), index=close.index)
    labels[future_return >= pct_threshold] = BIG_UP
    labels[future_return <= -pct_threshold] = BIG_DOWN
    invalid_mask = ~np.isfinite(future_return.to_numpy(dtype=float))
    if invalid_mask.any():
        labels.iloc[np.where(invalid_mask)[0]] = np.nan
    return labels


def attach_amplitude_labels(
    features: pd.DataFrame,
    prices: pd.DataFrame,
    config: LabelConfig,
) -> pd.DataFrame:
    """Attach amplitude labels to the feature frame when requested."""
    if config.mode != "amplitude":
        return features
    amplitude = compute_amplitude_labels(prices, config.horizon, config.threshold_pct)
    features = features.copy()
    features["label_amplitude"] = amplitude.reindex(features.index)
    features = features.dropna(subset=["label_amplitude"])
    return features


def encode_amplitude_target(labels: pd.Series) -> pd.Series:
    mapping = {BIG_DOWN: 0, BIG_UP: 1}
    return labels.map(mapping)


def compute_symmetric_binary_labels(
    prices: pd.DataFrame | pd.Series,
    horizon: int = 6,
    threshold_pct: float = 0.003,
) -> pd.Series:
    """
    Compute 1/0/NaN labels with a neutral band around zero.

    Args:
        prices: Close series or prices dataframe with a 'close' column.
        horizon: Forward horizon (number of bars) to look ahead.
        threshold_pct: Decimal threshold (e.g., 0.003 = 0.3%).

    Returns:
        pd.Series of floats where 1.0 = long, 0.0 = short, NaN = neutral.
    """
    if isinstance(prices, pd.DataFrame):
        close = prices["close"]
    else:
        close = prices
    if close.empty:
        return pd.Series(dtype=float)
    horizon = max(1, int(horizon))
    threshold = float(threshold_pct)
    if threshold <= 0:
        raise ValueError("threshold_pct must be > 0")

    close = close.astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        future_return = close.shift(-horizon) / close - 1.0
    labels = pd.Series(np.full(len(close), np.nan, dtype=float), index=close.index)
    labels[future_return > threshold] = 1.0
    labels[future_return < -threshold] = 0.0

    invalid_mask = ~np.isfinite(future_return.to_numpy(dtype=float))
    if invalid_mask.any():
        labels.iloc[np.where(invalid_mask)[0]] = np.nan
    return labels


def validate_label_support(features: pd.DataFrame, config: LabelConfig) -> None:
    if config.mode != "amplitude":
        return
    if "label_amplitude" not in features.columns:
        raise RuntimeError("Amplitude labels missing from feature frame.")
    min_samples = max(1, int(config.min_samples_per_class))
    labels = features["label_amplitude"].dropna()
    mask = labels.isin(TRAINABLE_LABELS)
    filtered = labels[mask]
    if filtered.empty:
        raise ValueError("No BIG_UP/BIG_DOWN samples available for amplitude labels.")
    big_up = int((filtered == BIG_UP).sum())
    big_down = int((filtered == BIG_DOWN).sum())
    if big_up < min_samples or big_down < min_samples:
        raise ValueError(
            "Insufficient amplitude label support: "
            f"BIG_UP={big_up}, BIG_DOWN={big_down}, required={min_samples}."
        )

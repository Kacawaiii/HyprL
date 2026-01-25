from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression


@dataclass(slots=True)
class QuantileCalibrator:
    """Simple quantile-based calibrator relying on percentile interpolation."""

    pred_bins: np.ndarray
    target_bins: np.ndarray

    def predict(self, values: np.ndarray) -> np.ndarray:
        series = np.asarray(values, dtype=float).reshape(-1)
        calibrated = np.interp(
            series,
            self.pred_bins,
            self.target_bins,
            left=float(self.target_bins[0]),
            right=float(self.target_bins[-1]),
        )
        return np.clip(calibrated, 0.0, 1.0)


def build_quantile_calibrator(preds: np.ndarray, targets: np.ndarray, n_bins: int = 101) -> QuantileCalibrator:
    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    if preds.size != targets.size:
        raise ValueError("preds and targets must share the same length")
    percentiles = np.linspace(0, 100, num=max(2, n_bins))
    pred_bins = np.percentile(preds, percentiles)
    target_bins = np.percentile(targets, percentiles)
    base_min = float(preds.min()) if preds.size else 0.0
    base_max = float(preds.max()) if preds.size else 1.0
    pred_bins[0] = min(pred_bins[0], base_min)
    pred_bins[-1] = max(pred_bins[-1], base_max)
    target_bins = np.clip(target_bins, 0.0, 1.0)
    return QuantileCalibrator(pred_bins=pred_bins, target_bins=target_bins)


def fit_isotonic_calibrator(preds: np.ndarray, targets: np.ndarray) -> IsotonicRegression:
    preds = np.asarray(preds, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(preds, targets)
    return iso


def build_calibrator(method: Literal["isotonic", "quantile"], preds: np.ndarray, targets: np.ndarray):
    method_norm = method.lower()
    if method_norm == "isotonic":
        return fit_isotonic_calibrator(preds, targets)
    if method_norm == "quantile":
        return build_quantile_calibrator(preds, targets)
    raise ValueError(f"Unsupported calibration method: {method}")

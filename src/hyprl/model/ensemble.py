from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


class CalibratedVoting:
    """Blend calibrated probability outputs using a logistic meta learner."""

    def __init__(self, method: Literal["isotonic", "platt"] = "isotonic") -> None:
        self.method = method
        self.calibrators: list[IsotonicRegression | LogisticRegression] = []
        self.meta: LogisticRegression | None = None

    def fit(self, probs_list: list[np.ndarray], y: np.ndarray) -> "CalibratedVoting":
        if not probs_list:
            raise ValueError("CalibratedVoting.fit requires at least one probability array.")
        stacked = np.column_stack(probs_list)
        calibrated = self._calibrate_columns(stacked, y)
        meta = LogisticRegression(max_iter=1000, random_state=42)
        meta.fit(calibrated, y)
        self.meta = meta
        return self

    def predict_proba(self, probs_list: list[np.ndarray]) -> np.ndarray:
        if self.meta is None:
            raise RuntimeError("Call fit() before predict_proba().")
        stacked = np.column_stack(probs_list)
        calibrated = self._apply_calibrators(stacked)
        probs = self.meta.predict_proba(calibrated)[:, 1]
        return np.clip(probs, 0.0, 1.0)

    def _calibrate_columns(self, arr: np.ndarray, y: np.ndarray) -> np.ndarray:
        calibrated_cols = []
        self.calibrators = []
        for col in arr.T:
            if self.method == "isotonic":
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(col, y)
                self.calibrators.append(iso)
                calibrated_cols.append(iso.predict(col))
            else:
                logit = LogisticRegression(max_iter=1000, random_state=42)
                logit.fit(col.reshape(-1, 1), y)
                self.calibrators.append(logit)
                calibrated_cols.append(logit.predict_proba(col.reshape(-1, 1))[:, 1])
        return np.column_stack(calibrated_cols)

    def _apply_calibrators(self, arr: np.ndarray) -> np.ndarray:
        calibrated_cols = []
        for col, calib in zip(arr.T, self.calibrators, strict=False):
            if isinstance(calib, IsotonicRegression):
                calibrated_cols.append(calib.predict(col))
            else:
                calibrated_cols.append(calib.predict_proba(col.reshape(-1, 1))[:, 1])
        return np.column_stack(calibrated_cols)

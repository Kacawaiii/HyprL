from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

try:  # optional dependency
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - optional backend
    XGBClassifier = None  # type: ignore

CalibrationMethod = Literal["none", "platt", "isotonic"]
ModelType = Literal["logistic", "random_forest", "xgboost"]

_CALIBRATOR_CACHE: dict[str, object] = {}


def _coerce_probabilities(probs: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(probs, dtype=float).reshape(-1)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def load_calibrator(path: str | Path) -> object:
    key = str(path)
    cached = _CALIBRATOR_CACHE.get(key)
    if cached is not None:
        return cached
    calibrator = joblib.load(Path(key))
    _CALIBRATOR_CACHE[key] = calibrator
    return calibrator


def apply_calibrator(probs: np.ndarray | list[float], calibrator: object) -> np.ndarray:
    p = _coerce_probabilities(probs)
    if calibrator is None:
        return p

    method: str | None = None
    model: Any | None = None
    if isinstance(calibrator, dict):
        method = str(calibrator.get("method") or "").lower() or None
        model = calibrator.get("model")
    if method == "platt" and model is not None and hasattr(model, "predict_proba"):
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return np.clip(model.predict_proba(logits)[:, 1], 0.0, 1.0)
    if method == "isotonic" and model is not None and hasattr(model, "transform"):
        return np.clip(model.transform(p), 0.0, 1.0)

    if hasattr(calibrator, "transform"):
        return np.clip(calibrator.transform(p), 0.0, 1.0)
    if hasattr(calibrator, "predict_proba"):
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        try:
            return np.clip(calibrator.predict_proba(logits)[:, 1], 0.0, 1.0)
        except Exception:
            return np.clip(calibrator.predict_proba(p.reshape(-1, 1))[:, 1], 0.0, 1.0)
    if callable(calibrator):
        try:
            return np.clip(np.asarray(calibrator(p), dtype=float).reshape(-1), 0.0, 1.0)
        except Exception:
            return p
    return p


def maybe_calibrate(probs: np.ndarray | list[float], calibrator_path: str | Path | None) -> np.ndarray:
    if not calibrator_path:
        return _coerce_probabilities(probs)
    calibrator = load_calibrator(calibrator_path)
    return apply_calibrator(probs, calibrator)


@dataclass(slots=True)
class ProbabilityCalibrator:
    method: CalibrationMethod
    random_state: Optional[int] = None
    platt_model: Optional[LogisticRegression] = None
    isotonic_model: Optional[IsotonicRegression] = None
    is_fitted: bool = False

    def fit(self, raw_probs: np.ndarray, targets: np.ndarray) -> None:
        if self.method == "platt":
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="lbfgs",
                max_iter=200,
                random_state=self.random_state,
            )
            model.fit(raw_probs.reshape(-1, 1), targets)
            self.platt_model = model
            self.is_fitted = True
        elif self.method == "isotonic":
            model = IsotonicRegression(out_of_bounds="clip")
            model.fit(raw_probs, targets)
            self.is_fitted = True
            self.isotonic_model = model
        else:
            self.is_fitted = False

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if self.method == "platt" and self.platt_model is not None and self.is_fitted:
            calibrated = self.platt_model.predict_proba(raw_probs.reshape(-1, 1))[:, 1]
            return calibrated
        if self.method == "isotonic" and self.isotonic_model is not None and self.is_fitted:
            return self.isotonic_model.predict(raw_probs)
        return np.clip(raw_probs, 0.0, 1.0)


@dataclass(slots=True)
class ProbabilityModel:
    scaler: Optional[StandardScaler]
    classifier: ClassifierMixin
    calibrator: ProbabilityCalibrator
    model_type: ModelType

    @classmethod
    def create(
        cls,
        model_type: ModelType = "logistic",
        calibration: CalibrationMethod = "none",
        random_state: int | None = 42,
        *,
        xgb_max_depth: int = 4,
        xgb_estimators: int = 400,
        xgb_eta: float = 0.05,
        xgb_subsample: float = 0.8,
        xgb_colsample: float = 0.8,
    ) -> "ProbabilityModel":
        if random_state is None:
            seed = 42
        else:
            try:
                seed = int(random_state)
            except (TypeError, ValueError):
                seed = 42
        if model_type == "logistic":
            scaler: Optional[StandardScaler] = StandardScaler()
            classifier: ClassifierMixin = LogisticRegression(
                penalty="l2",
                C=0.8,
                max_iter=300,
                solver="lbfgs",
                random_state=seed,
            )
        elif model_type == "random_forest":
            scaler = None
            classifier = RandomForestClassifier(
                n_estimators=300,
                max_depth=6,
                min_samples_leaf=10,
                random_state=seed,
                n_jobs=-1,
            )
        elif model_type == "xgboost":
            if XGBClassifier is None:
                raise ImportError(
                    "XGBoost backend requested but xgboost is not installed. Install with 'pip install xgboost'."
                )
            scaler = StandardScaler()
            classifier = XGBClassifier(
                max_depth=xgb_max_depth,
                n_estimators=xgb_estimators,
                learning_rate=xgb_eta,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=seed,
                seed=seed,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")
        calibrator = ProbabilityCalibrator(method=calibration, random_state=seed)
        return cls(scaler=scaler, classifier=classifier, calibrator=calibrator, model_type=model_type)

    def fit(
        self,
        feature_df: pd.DataFrame,
        target: pd.Series,
        calibration_cv: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the model with proper calibration using cross-validation.

        Args:
            feature_df: Feature matrix
            target: Target labels (0/1)
            calibration_cv: Number of CV folds for calibration (0 = use train preds, not recommended)

        The calibrator is fitted on out-of-fold predictions to avoid overfitting.
        This is critical for reliable probability estimates and position sizing.
        """
        feature_matrix = feature_df.to_numpy(dtype=float)
        target_vector = target.to_numpy(dtype=float)
        feature_scaled = (
            self.scaler.fit_transform(feature_matrix) if self.scaler is not None else feature_matrix
        )

        # Collect out-of-fold predictions for calibration
        if self.calibrator.method != "none" and calibration_cv > 1 and len(target_vector) >= calibration_cv * 10:
            from sklearn.model_selection import KFold
            import copy

            oof_probs = np.zeros(len(target_vector))
            kf = KFold(n_splits=calibration_cv, shuffle=False)  # No shuffle to preserve time order

            for train_idx, val_idx in kf.split(feature_scaled):
                # Clone classifier for this fold
                fold_clf = copy.deepcopy(self.classifier)
                fold_clf.fit(feature_scaled[train_idx], target_vector[train_idx])
                # Get OOF predictions
                fold_probs = fold_clf.predict_proba(feature_scaled[val_idx])
                if fold_probs.ndim == 2:
                    fold_probs = fold_probs[:, 1]
                oof_probs[val_idx] = fold_probs

            # Fit calibrator on OOF predictions (unbiased)
            self.calibrator.fit(oof_probs, target_vector)

        # Now fit final classifier on ALL data
        self.classifier.fit(feature_scaled, target_vector)

        # If no CV calibration, fall back to train predictions (not recommended)
        if self.calibrator.method != "none" and not self.calibrator.is_fitted:
            raw_probs = self.classifier.predict_proba(feature_scaled)[:, 1]
            self.calibrator.fit(raw_probs, target_vector)

        return feature_scaled, target_vector

    def predict_proba(self, feature_df: pd.DataFrame | np.ndarray) -> np.ndarray:
        if hasattr(feature_df, "to_numpy"):
            feature_matrix = feature_df.to_numpy(dtype=float)
        else:
            feature_matrix = np.asarray(feature_df, dtype=float)
        feature_scaled = (
            self.scaler.transform(feature_matrix) if self.scaler is not None else feature_matrix
        )
        raw_probs = self.classifier.predict_proba(feature_scaled)
        if raw_probs.ndim == 2 and raw_probs.shape[1] >= 2:
            raw_probs = raw_probs[:, 1]
        else:
            raw_probs = raw_probs.ravel()
        return self.calibrator.transform(raw_probs)

    def latest_prediction(
        self, feature_df: pd.DataFrame, threshold: float = 0.5
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        if feature_df.empty:
            return None, None, None
        proba = self.predict_proba(feature_df.tail(1))
        probability_up = float(proba[-1])
        probability_down = 1.0 - probability_up
        signal = int(probability_up >= threshold)
        return probability_up, probability_down, signal

    def dump(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load_artifact(cls, path: str | Path) -> "ProbabilityModel":
        obj = joblib.load(Path(path))
        if not isinstance(obj, ProbabilityModel):  # pragma: no cover - defensive
            raise TypeError(f"Artifact at {path} is not a ProbabilityModel")
        return obj

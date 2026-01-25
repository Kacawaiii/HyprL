from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hyprl.meta.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES


def apply_calibration(preds: np.ndarray, calibrator: Any) -> np.ndarray:
    if calibrator is None:
        return np.clip(preds, 0.0, 1.0)
    values = np.asarray(preds, dtype=float).reshape(-1)
    calibrated: np.ndarray | Any
    if hasattr(calibrator, "predict"):
        try:
            calibrated = calibrator.predict(values)
        except TypeError:
            calibrated = calibrator.predict(values.reshape(-1, 1))
    elif callable(calibrator):
        calibrated = calibrator(values)
    else:  # pragma: no cover - defensive
        raise TypeError("Calibrator must expose a predict method or be callable.")
    return np.clip(np.asarray(calibrated, dtype=float).reshape(-1), 0.0, 1.0)


def _build_regressor(model_type: Literal["elasticnet", "rf", "xgb"], random_state: int) -> Any:
    if model_type == "elasticnet":
        return ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
            n_alphas=50,
            random_state=random_state,
            max_iter=5000,
        )
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=400,
            max_depth=8,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "xgb":
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "xgboost is not installed. Install xgboost or use --model elasticnet/rf."
            ) from exc

        return XGBRegressor(
            n_estimators=600,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported meta model type: {model_type}")


def _build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


@dataclass(slots=True)
class MetaRobustnessModel:
    model_type: Literal["elasticnet", "rf", "xgb"] = "rf"
    random_state: int = 42
    feature_names: list[str] = field(default_factory=list)
    pipeline: Pipeline | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MetaRobustnessModel":
        self.feature_names = list(X.columns)
        preprocessor = _build_preprocessor()
        regressor = _build_regressor(self.model_type, self.random_state)
        self.pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("regressor", regressor),
            ]
        )
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame, calibrator: Any | None = None) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("MetaRobustnessModel not fitted.")
        missing = [col for col in self.feature_names if col not in X.columns]
        if missing:
            X = X.copy()
            for column in missing:
                X[column] = 0.0
        ordered = X[self.feature_names]
        preds = self.pipeline.predict(ordered)
        preds = np.clip(preds, 0.0, 1.0)
        return apply_calibration(preds, calibrator)

    def save(self, path: str | Path) -> None:
        if self.pipeline is None:
            raise RuntimeError("MetaRobustnessModel not fitted.")
        payload = {
            "model_type": self.model_type,
            "random_state": self.random_state,
            "feature_names": self.feature_names,
            "pipeline": self.pipeline,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str | Path) -> "MetaRobustnessModel":
        payload = joblib.load(path)
        model = MetaRobustnessModel(
            model_type=payload["model_type"],
            random_state=payload.get("random_state", 42),
            feature_names=payload.get("feature_names", []),
        )
        model.pipeline = payload["pipeline"]
        return model

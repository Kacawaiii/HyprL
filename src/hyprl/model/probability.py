from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class ProbabilityModel:
    scaler: StandardScaler
    classifier: LogisticRegression

    @classmethod
    def create(cls, random_state: int | None = 42) -> "ProbabilityModel":
        scaler = StandardScaler()
        classifier = LogisticRegression(
            penalty="l2",
            C=0.8,
            max_iter=300,
            solver="lbfgs",
            random_state=random_state,
        )
        return cls(scaler=scaler, classifier=classifier)

    def fit(self, feature_df: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        feature_matrix = feature_df.to_numpy(dtype=float)
        target_vector = target.to_numpy(dtype=float)
        feature_scaled = self.scaler.fit_transform(feature_matrix)
        self.classifier.fit(feature_scaled, target_vector)
        return feature_scaled, target_vector

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        feature_matrix = feature_df.to_numpy(dtype=float)
        feature_scaled = self.scaler.transform(feature_matrix)
        proba = self.classifier.predict_proba(feature_scaled)[:, 1]
        return proba

    def latest_prediction(
        self, feature_df: pd.DataFrame
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        if feature_df.empty:
            return None, None, None
        proba = self.predict_proba(feature_df.tail(1))
        probability_up = float(proba[-1])
        probability_down = 1.0 - probability_up
        signal = int(probability_up >= 0.5)
        return probability_up, probability_down, signal

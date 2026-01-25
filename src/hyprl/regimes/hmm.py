from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover
    GaussianHMM = None


class RegimeHMM:
    """GaussianHMM wrapper with pandas-friendly API."""

    def __init__(self, n_states: int = 3, cov_type: str = "full", random_state: int = 42) -> None:
        self.n_states = n_states
        self.cov_type = cov_type
        self.random_state = random_state
        self._model: Optional[GaussianHMM] = None

    def fit(self, X: pd.DataFrame) -> "RegimeHMM":
        if GaussianHMM is None:  # pragma: no cover
            raise ImportError(
                "hmmlearn is required for RegimeHMM. Install with: pip install hmmlearn"
            )
        clean = X.dropna()
        if clean.empty:
            raise ValueError("Input data for RegimeHMM.fit must contain at least one row.")
        model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.cov_type,
            random_state=self.random_state,
            n_iter=200,
        )
        model.fit(clean.values)
        self._model = model
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("RegimeHMM must be fit before predict().")
        clean = X.dropna()
        if clean.empty:
            return pd.Series(dtype="Int64", index=X.index)
        preds = self._model.predict(clean.values)
        out = pd.Series(np.nan, index=X.index, dtype="float64")
        out.loc[clean.index] = preds
        return out.astype("Int64")

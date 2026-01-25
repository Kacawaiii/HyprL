from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


@dataclass(slots=True)
class EvalReport:
    r2: float
    mae: float
    spearman: float
    kendall: float
    n: int


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> float:
    if len(a) < 2:
        return float("nan")
    corr = a.corr(b, method=method)
    return float(corr) if corr == corr else float("nan")


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> EvalReport:
    series_true = pd.Series(y_true)
    series_pred = pd.Series(y_pred)
    report = EvalReport(
        r2=float(r2_score(series_true, series_pred)),
        mae=float(mean_absolute_error(series_true, series_pred)),
        spearman=_safe_corr(series_true, series_pred, method="spearman"),
        kendall=_safe_corr(series_true, series_pred, method="kendall"),
        n=len(series_true),
    )
    return report

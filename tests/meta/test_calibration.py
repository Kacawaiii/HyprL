from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hyprl.meta.calibration import build_calibrator
from hyprl.meta.features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, select_meta_features
from hyprl.meta.model import MetaRobustnessModel, apply_calibration
from hyprl.meta.scoring import evaluate_predictions


def _synthetic_meta_frame(n_rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    data = {column: rng.uniform(0.4, 0.7, size=n_rows) for column in NUMERIC_FEATURES}
    for column in CATEGORICAL_FEATURES:
        data[column] = rng.choice(["off", "normal"], size=n_rows)
    base_score = (
        0.5 * data["pf_backtest"]
        + 0.3 * data["sharpe_backtest"]
        - 0.01 * data["maxdd_backtest"]
        + 0.1 * data["winrate_backtest"]
    )
    noise = rng.normal(scale=0.02, size=n_rows)
    robustness = np.clip(0.3 + base_score + noise, 0.0, 1.0)
    frame = pd.DataFrame(data)
    frame["robustness_score"] = robustness
    return frame


def test_calibration_isotonic_improves_spearman() -> None:
    df = _synthetic_meta_frame()
    X, _ = select_meta_features(df)
    y = df["robustness_score"].to_numpy()
    model = MetaRobustnessModel(model_type="rf", random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X))
    for train_idx, test_idx in kf.split(X):
        fold = MetaRobustnessModel(model_type=model.model_type, random_state=42)
        fold.fit(X.iloc[train_idx], y[train_idx])
        oof[test_idx] = fold.predict(X.iloc[test_idx])
    baseline = evaluate_predictions(y, oof)
    calibrator = build_calibrator("isotonic", oof, y)
    calibrated = apply_calibration(oof, calibrator)
    assert np.all((calibrated >= 0.0) & (calibrated <= 1.0))
    after = evaluate_predictions(y, calibrated)
    assert after.spearman + 0.05 >= baseline.spearman

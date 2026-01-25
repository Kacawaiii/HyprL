from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hyprl.meta.features import select_meta_features
from hyprl.meta.model import MetaRobustnessModel


def test_meta_model_fit_save_load(tmp_path: Path) -> None:
    rows = 10
    df = pd.DataFrame(
        {
            "long_threshold": np.linspace(0.55, 0.6, rows),
            "short_threshold": np.linspace(0.4, 0.35, rows),
            "risk_pct": 0.015,
            "min_ev_multiple": 0.1,
            "trend_filter": ["true"] * rows,
            "sentiment_min": -0.4,
            "sentiment_max": 0.4,
            "sentiment_regime": "off",
            "weighting_scheme": "equal",
            "pf_backtest": np.linspace(1.0, 1.5, rows),
            "sharpe_backtest": np.linspace(0.4, 0.9, rows),
            "maxdd_backtest": np.linspace(10, 25, rows),
            "winrate_backtest": 0.55,
            "equity_vol_backtest": 0.1,
            "trades_backtest": 50,
            "correlation_mean": 0.2,
            "correlation_max": 0.5,
        }
    )
    X, _ = select_meta_features(df)
    y = np.linspace(0.4, 0.9, rows)
    model = MetaRobustnessModel(model_type="rf", random_state=7)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (rows,)

    model_path = tmp_path / "meta.joblib"
    model.save(model_path)
    loaded = MetaRobustnessModel.load(model_path)
    preds_loaded = loaded.predict(X)
    np.testing.assert_allclose(predictions, preds_loaded, atol=1e-6)

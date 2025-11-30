from __future__ import annotations

import pandas as pd

from hyprl.model.probability import ProbabilityModel
from hyprl.rt.features import compute_features


def test_compute_features_and_prediction_threshold() -> None:
    index = pd.date_range("2024-01-01", periods=80, freq="T")
    data = {
        "open": [100 + i * 0.1 for i in range(80)],
        "high": [100 + i * 0.1 + 0.5 for i in range(80)],
        "low": [100 + i * 0.1 - 0.5 for i in range(80)],
        "close": [100 + i * 0.1 for i in range(80)],
        "volume": [1_000 + i for i in range(80)],
    }
    bars = pd.DataFrame(data, index=index)
    feats = compute_features(bars)
    assert feats["sma_short"] == feats["sma_short"]
    feature_df = pd.DataFrame([{k: v for k, v in feats.items() if k not in {"close", "open"}}])
    feature_df["rsi_raw"] = feature_df["rsi_raw"].fillna(50.0)
    model = ProbabilityModel.create()
    design = pd.concat([feature_df] * 5, ignore_index=True)
    target = pd.Series([1, 0, 1, 1, 0])
    model.fit(design, target)
    prob_up = float(model.predict_proba(design.tail(1))[-1])
    assert 0.0 <= prob_up <= 1.0
    _, _, signal = model.latest_prediction(design.tail(1), threshold=0.52)
    assert (signal == 1 and prob_up >= 0.52) or (signal == 0 and prob_up < 0.52)

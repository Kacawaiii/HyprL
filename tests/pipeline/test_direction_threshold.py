from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.model.probability import ProbabilityModel


def test_latest_prediction_obeys_threshold():
    rng = np.random.default_rng(42)
    feature_df = pd.DataFrame(
        {
            "a": rng.normal(size=64),
            "b": rng.normal(size=64),
            "c": rng.normal(size=64),
            "d": rng.normal(size=64),
        }
    )
    target = ((feature_df["a"] + feature_df["b"]) > 0).astype(int)
    model = ProbabilityModel.create()
    model.fit(feature_df, target)
    prob_up, prob_down, signal = model.latest_prediction(feature_df.tail(1), threshold=0.9)
    assert prob_up is not None and prob_down is not None
    assert prob_up < 0.9
    assert signal == 0

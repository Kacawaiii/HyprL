from __future__ import annotations

import pandas as pd

from hyprl.analysis.meta_view import build_meta_diag_frame
from hyprl.meta.model import MetaRobustnessModel


class DummyMetaModel(MetaRobustnessModel):
    def __init__(self, value: float) -> None:
        super().__init__(model_type="rf")
        self.value = value

    def predict(self, X):  # noqa: ANN001
        return [self.value] * len(X)


def test_build_meta_diag_frame_normalizes_scores() -> None:
    df = pd.DataFrame(
        {
            "long_threshold": [0.6, 0.62],
            "short_threshold": [0.4, 0.38],
            "risk_pct": [0.02, 0.018],
            "min_ev_multiple": [0.1, 0.0],
            "sentiment_min": [-0.4, -0.3],
            "sentiment_max": [0.4, 0.5],
            "sentiment_regime": ["off", "off"],
            "portfolio_profit_factor": [1.2, 1.4],
            "portfolio_sharpe": [0.8, 1.0],
            "portfolio_max_drawdown_pct": [20.0, 15.0],
            "win_rate_pct": [55.0, 60.0],
            "portfolio_equity_vol": [0.12, 0.09],
            "n_trades": [60, 80],
            "correlation_mean": [0.2, 0.1],
            "correlation_max": [0.4, 0.35],
            "base_score": [0.5, 0.7],
        }
    )
    diag = build_meta_diag_frame(df, meta_weight=0.5, model=DummyMetaModel(0.9))
    assert "base_score_normalized" in diag.columns
    assert diag["base_score_normalized"].between(0, 1).all()
    assert "final_score" in diag.columns
    assert diag["final_score"].between(0, 1).all()

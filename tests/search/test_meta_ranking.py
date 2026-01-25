from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.search import optimizer as search_optimizer
from hyprl.search.optimizer import CandidateConfig, SearchResult


class DummyModel:
    def __init__(self, preds: list[float]) -> None:
        self.preds = preds
        self.feature_names = []

    def predict(self, X: pd.DataFrame) -> np.ndarray:  # noqa: D401
        return np.asarray(self.preds[: len(X)], dtype=float)


def _make_result(base_score: float = 0.5) -> SearchResult:
    candidate = CandidateConfig(
        long_threshold=0.6,
        short_threshold=0.4,
        risk_pct=0.02,
        min_ev_multiple=0.1,
        trend_filter=False,
        sentiment_min=-0.4,
        sentiment_max=0.4,
        sentiment_regime="off",
    )
    result = SearchResult(
        config=candidate,
        strategy_return_pct=5.0,
        benchmark_return_pct=3.0,
        alpha_pct=2.0,
        profit_factor=1.2,
        sharpe=0.9,
        max_drawdown_pct=15.0,
        expectancy=25.0,
        n_trades=50,
        win_rate_pct=55.0,
        sentiment_stats={},
        expectancy_per_trade=25.0,
        risk_of_ruin=0.1,
        maxdd_p95=20.0,
        pnl_p05=0.0,
        portfolio_return_pct=5.0,
        portfolio_profit_factor=1.2,
        portfolio_sharpe=0.9,
        portfolio_max_drawdown_pct=15.0,
        portfolio_risk_of_ruin=0.1,
        portfolio_maxdd_p95=20.0,
        portfolio_pnl_p05=0.0,
        correlation_mean=0.2,
        correlation_max=0.4,
        portfolio_equity_vol=0.12,
    )
    result.base_score = base_score
    result.final_score = base_score
    return result


def test_apply_meta_scores_respects_weight(monkeypatch) -> None:
    res_hi = _make_result()
    res_lo = _make_result()
    res_list = [res_hi, res_lo]

    def fake_features(df: pd.DataFrame):
        return df.assign(dummy=1.0), ["dummy"]

    monkeypatch.setattr(search_optimizer, "select_meta_features", fake_features)
    model = DummyModel([0.9, 0.1])
    search_optimizer._apply_meta_scores(res_list, model, meta_weight=0.5, weighting_scheme="equal")

    assert res_hi.meta_prediction == 0.9
    assert res_lo.meta_prediction == 0.1
    assert res_hi.final_score > res_lo.final_score

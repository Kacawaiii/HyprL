from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyprl.backtest import runner
from hyprl.labels.amplitude import LabelConfig
from hyprl.risk.manager import RiskConfig


def _zigzag_prices(bars_per_leg: int = 4, cycles: int = 20) -> pd.DataFrame:
    price = 150.0
    closes: list[float] = []
    for _ in range(cycles):
        for _ in range(bars_per_leg):
            price *= 1.01
            closes.append(price)
        for _ in range(bars_per_leg):
            price *= 0.99
            closes.append(price)
    index = pd.date_range("2024-01-01", periods=len(closes), freq="h", tz="UTC")
    close_series = pd.Series(closes, index=index)
    frame = pd.DataFrame(
        {
            "open": close_series,
            "high": close_series * 1.01,
            "low": close_series * 0.99,
            "close": close_series,
            "adj_close": close_series,
            "volume": 1_000_000,
        },
        index=index,
    )
    return frame


class _DeterministicModel:
    def __init__(self, probability: float) -> None:
        self._probability = probability
        self.class_counts: dict[int, int] = {}

    def fit(self, design: pd.DataFrame, target: pd.Series):
        counts = target.value_counts().to_dict()
        self.class_counts = {int(k): int(v) for k, v in counts.items()}
        return design.to_numpy(), target.to_numpy()

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(feature_df), self._probability, dtype=float)


def test_backtest_runs_with_amplitude_labels(monkeypatch):
    prices = _zigzag_prices()

    def _fake_prices(self, **_kwargs):
        return prices.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_prices)

    created_models: list[_DeterministicModel] = []

    def _fake_create(cls, *args, **kwargs):  # noqa: ARG001 - signature mimics classmethod
        model = _DeterministicModel(probability=0.8)
        created_models.append(model)
        return model

    monkeypatch.setattr(runner.ProbabilityModel, "create", classmethod(_fake_create))

    risk_cfg = RiskConfig(balance=25_000.0, risk_pct=0.02, atr_multiplier=1.5, reward_multiple=2.0, min_position_size=1)
    label_cfg = LabelConfig(mode="amplitude", horizon=4, threshold_pct=1.5, neutral_strategy="ignore", min_samples_per_class=5)
    config = runner.BacktestConfig(
        ticker="TEST",
        period="1y",
        interval="1h",
        initial_balance=25_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        model_type="logistic",
        calibration="none",
        risk=risk_cfg,
        random_state=7,
        commission_pct=0.0,
        slippage_pct=0.0,
        sentiment_min=-1.0,
        sentiment_max=1.0,
        sentiment_regime="off",
        label=label_cfg,
    )

    result = runner.run_backtest(config)

    assert result.n_trades > 0
    assert result.trades[0].probability_up == pytest.approx(0.8, rel=1e-6)
    assert any(model.class_counts.get(0, 0) > 0 and model.class_counts.get(1, 0) > 0 for model in created_models)

from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_price_frame(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    trend = np.linspace(100, 105, rows)
    noise = np.sin(np.linspace(0, 12 * np.pi, rows)) * 3.0
    close = trend + noise
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": np.linspace(1_000, 1_800, rows),
        },
        index=index,
    )


class _DummyModel:
    def __init__(self, probability: float):
        self.probability = probability

    def fit(self, feature_df, target):
        return self

    def predict_proba(self, feature_df):
        return np.array([self.probability], dtype=float)


def _patch_probability_model(monkeypatch, value: float) -> None:
    def _fake_create(cls, **kwargs):
        return _DummyModel(value)

    monkeypatch.setattr(runner.ProbabilityModel, "create", classmethod(_fake_create))


def test_flat_band_reduces_trades(monkeypatch):
    prices = _synthetic_price_frame()

    def _fake_get_prices(self, **kwargs):
        return prices.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)
    _patch_probability_model(monkeypatch, 0.55)
    base_kwargs = dict(
        ticker="TEST",
        period="6mo",
        initial_balance=30_000.0,
        risk=RiskConfig(balance=30_000.0, risk_pct=0.04, atr_multiplier=1.0, reward_multiple=1.5, min_position_size=8),
    )

    cfg_always = runner.BacktestConfig(long_threshold=0.5, short_threshold=0.5, **base_kwargs)
    cfg_neutral = runner.BacktestConfig(long_threshold=0.65, short_threshold=0.45, **base_kwargs)

    result_always = runner.run_backtest(cfg_always)
    result_neutral = runner.run_backtest(cfg_neutral)

    assert result_always.n_trades > 0
    assert result_neutral.n_trades < result_always.n_trades
    assert result_neutral.final_balance >= 0

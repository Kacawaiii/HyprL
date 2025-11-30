from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_price_frame(rows: int = 200) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 110, rows)
    noise = np.sin(np.linspace(0, 6 * np.pi, rows)) * 0.8
    close = base + noise
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.6,
            "low": close - 0.6,
            "close": close,
            "adj_close": close,
            "volume": np.linspace(1_000, 1_800, rows),
        },
        index=index,
    )


class _ConstantModel:
    def __init__(self, probability: float):
        self._probability = probability

    def fit(self, *_args, **_kwargs):
        return None, None

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(feature_df), self._probability, dtype=float)


def _patch_model(monkeypatch: pytest.MonkeyPatch, probability: float) -> None:
    def _create(cls, **_kwargs):
        return _ConstantModel(probability)

    monkeypatch.setattr(runner.ProbabilityModel, "create", classmethod(_create))


def test_backtest_reports_extended_metrics(monkeypatch):
    prices = _synthetic_price_frame()
    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", lambda self, **kwargs: prices.copy())
    _patch_model(monkeypatch, 0.8)
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=20_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        risk=RiskConfig(
            balance=20_000.0,
            risk_pct=0.05,
            atr_multiplier=1.0,
            reward_multiple=1.5,
            min_position_size=5,
        ),
        random_state=123,
    )
    result = runner.run_backtest(cfg)
    assert result.n_trades > 0
    assert result.profit_factor is None or result.profit_factor >= 0.0
    assert result.expectancy != 0.0 or result.n_trades == 0
    assert result.avg_r_multiple is not None
    assert result.annualized_return is not None
    assert result.sortino_ratio is None or isinstance(result.sortino_ratio, float)
    assert result.brier_score is None or result.brier_score >= 0.0
    assert result.long_trades + result.short_trades == result.n_trades

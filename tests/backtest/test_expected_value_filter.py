from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_price_frame(rows: int = 240) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100, 104, rows)
    noise = np.sin(np.linspace(0, 10 * np.pi, rows)) * 3.0
    close = base + noise
    data = {
        "open": close - 0.1,
        "high": close + 0.6,
        "low": close - 0.6,
        "close": close,
        "adj_close": close,
        "volume": np.linspace(1_000, 1_800, rows),
    }
    return pd.DataFrame(data, index=index)


class _ConstantProbabilityModel:
    def __init__(self, probability: float) -> None:
        self._probability = probability

    def fit(self, *_args, **_kwargs):
        return None, None

    def predict_proba(self, feature_df: pd.DataFrame) -> np.ndarray:
        return np.full(len(feature_df), self._probability, dtype=float)


def _patch_probability_model(monkeypatch: pytest.MonkeyPatch, probability: float) -> None:
    def _create(cls, **kwargs):  # pylint: disable=unused-argument
        return _ConstantProbabilityModel(probability)

    monkeypatch.setattr(runner.ProbabilityModel, "create", classmethod(_create))


def test_expected_value_filter_skips_negative_ev_trades(monkeypatch):
    prices = _synthetic_price_frame()
    monkeypatch.setattr(
        runner.MarketDataFetcher,
        "get_prices",
        lambda self, **kwargs: prices.copy(),
    )
    risk_cfg = RiskConfig(
        balance=10_000.0,
        risk_pct=0.08,
        atr_multiplier=1.0,
        reward_multiple=0.4,  # intentionally small so EV requires very high confidence
        min_position_size=1,
    )

    # High-confidence regime -> EV positive -> trades occur
    _patch_probability_model(monkeypatch, probability=0.9)
    cfg_positive = runner.BacktestConfig(
        ticker="TEST",
        period="3mo",
        initial_balance=10_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        risk=risk_cfg,
        random_state=123,
    )
    result_positive = runner.run_backtest(cfg_positive)
    assert result_positive.n_trades > 0

    # Moderate confidence -> EV negative with reward_multiple=0.4 -> trades skipped
    _patch_probability_model(monkeypatch, probability=0.6)
    cfg_negative = runner.BacktestConfig(
        ticker="TEST",
        period="3mo",
        initial_balance=10_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        risk=risk_cfg,
        random_state=123,
    )
    result_negative = runner.run_backtest(cfg_negative)
    assert result_negative.n_trades < result_positive.n_trades


def test_min_ev_multiple_blocks_low_quality_trades(monkeypatch):
    prices = _synthetic_price_frame()
    monkeypatch.setattr(
        runner.MarketDataFetcher,
        "get_prices",
        lambda self, **kwargs: prices.copy(),
    )
    risk_cfg = RiskConfig(
        balance=10_000.0,
        risk_pct=0.05,
        atr_multiplier=1.0,
        reward_multiple=1.5,
        min_position_size=1,
    )
    _patch_probability_model(monkeypatch, probability=0.62)
    cfg_no_gate = runner.BacktestConfig(
        ticker="TEST",
        period="3mo",
        initial_balance=10_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        risk=risk_cfg,
        random_state=123,
    )
    trades_without_gate = runner.run_backtest(cfg_no_gate).n_trades

    cfg_gate = runner.BacktestConfig(
        ticker="TEST",
        period="3mo",
        initial_balance=10_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        risk=risk_cfg,
        random_state=123,
        min_ev_multiple=0.6,
    )
    trades_with_gate = runner.run_backtest(cfg_gate).n_trades

    assert trades_with_gate < trades_without_gate
    assert trades_with_gate >= 0

from __future__ import annotations

from itertools import cycle

import numpy as np
import pandas as pd

from hyprl.adaptive.engine import AdaptiveConfig, AdaptiveRegime
from hyprl.backtest import runner
from hyprl.risk.manager import RiskConfig


def _synthetic_prices(index: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(100, 105, len(index))
    data = {
        "open": base - 0.2,
        "high": base + 0.5,
        "low": base - 0.5,
        "close": base,
        "adj_close": base,
        "volume": np.full(len(index), 1_000),
    }
    return pd.DataFrame(data, index=index)


def test_adaptive_mode_records_regime_changes(monkeypatch):
    feature_index = pd.date_range("2024-01-01", periods=60, freq="h", tz="UTC")
    price_df = _synthetic_prices(feature_index)
    feature_frame = pd.DataFrame(
        {
            "sma_ratio": np.linspace(0.9, 1.1, len(feature_index)),
            "ema_ratio": np.linspace(0.95, 1.05, len(feature_index)),
            "rsi_normalized": np.linspace(0.4, 0.6, len(feature_index)),
            "volatility": np.linspace(0.01, 0.02, len(feature_index)),
            "atr_normalized": np.linspace(0.02, 0.03, len(feature_index)),
            "range_pct": np.linspace(0.01, 0.02, len(feature_index)),
            "rolling_return": np.linspace(-0.01, 0.01, len(feature_index)),
            "sentiment_score": np.zeros(len(feature_index)),
            "returns_next": np.tile([0.01, -0.015], len(feature_index) // 2),
            "atr_14": np.full(len(feature_index), 1.5),
        },
        index=feature_index,
    )

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", lambda *args, **kwargs: price_df.copy())
    monkeypatch.setattr(runner, "prepare_feature_frame", lambda *args, **kwargs: feature_frame.copy())

    prob_values = cycle([0.85, 0.82, 0.2, 0.18, 0.8, 0.15])

    class StubModel:
        def fit(self, feature_df, target):
            return None

        def predict_proba(self, feature_df):
            return np.array([next(prob_values)])

    monkeypatch.setattr(runner.ProbabilityModel, "create", lambda *args, **kwargs: StubModel())

    outcomes = cycle(["win", "loss", "loss", "win"])

    def _fake_simulate(prices, start_pos, risk):
        outcome = next(outcomes)
        exit_price = risk.take_profit_price if outcome == "win" else risk.stop_price
        return exit_price, min(len(prices) - 1, start_pos + 1), "signal"

    monkeypatch.setattr(runner, "_simulate_trade", _fake_simulate)

    adaptive_cfg = AdaptiveConfig(
        enable=True,
        lookback_trades=2,
        default_regime="normal",
        regimes={
            "normal": AdaptiveRegime(
                name="normal",
                min_equity_drawdown=-0.01,
                max_equity_drawdown=0.05,
                min_profit_factor=1.2,
                min_sharpe=0.0,
            ),
            "safe": AdaptiveRegime(
                name="safe",
                min_equity_drawdown=-0.01,
                max_equity_drawdown=1.0,
                threshold_overrides={"long_shift": 0.05, "short_shift": -0.05},
                risk_overrides={"risk_pct": 0.01},
            ),
        },
    )

    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=20_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=20_000.0, risk_pct=0.02, atr_multiplier=1.0, reward_multiple=1.5, min_position_size=5),
        adaptive=adaptive_cfg,
        risk_profile="normal",
        risk_profiles={
            "normal": {"risk_pct": 0.02, "atr_multiplier": 1.0, "reward_multiple": 1.6, "min_position_size": 5},
            "safe": {"risk_pct": 0.01, "atr_multiplier": 1.2, "reward_multiple": 1.3, "min_position_size": 5},
        },
    )

    result = runner.run_backtest(cfg)

    assert result.n_trades > 0
    assert result.adaptive_profile_changes >= 1
    assert "safe" in result.regime_usage
    assert any(trade.regime_name == "safe" for trade in result.trades)

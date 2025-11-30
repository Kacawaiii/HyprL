from __future__ import annotations

from dataclasses import replace
import math

import numpy as np
import pandas as pd
import pytest

from hyprl.backtest import runner
from hyprl.native.supercalc import native_available as wrapper_native_available, run_backtest_native
from hyprl.risk.manager import RiskConfig
from hyprl.supercalc import evaluate_candidates, native_available as legacy_native_available, prepare_supercalc_dataset


def _synthetic_price_frame(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100.0, 110.0, rows)
    noise = np.sin(np.linspace(0.0, 8 * np.pi, rows)) * 0.5
    close = base + noise
    open_ = close - 0.1
    high = close + 0.7
    low = close - 0.7
    volume = np.linspace(1_000, 2_000, rows)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "adj_close": close, "volume": volume},
        index=index,
    )


def _stub_prices(monkeypatch: pytest.MonkeyPatch, frame: pd.DataFrame) -> None:
    def _fake_get_prices(self, **kwargs):  # noqa: ANN001
        return frame.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)


def test_prepare_dataset_matches_run_backtest(monkeypatch: pytest.MonkeyPatch) -> None:
    price_df = _synthetic_price_frame()
    _stub_prices(monkeypatch, price_df)
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=25_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=25_000.0, risk_pct=0.02, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=5),
    )
    dataset = prepare_supercalc_dataset(cfg)
    stats = runner.simulate_from_dataset(dataset, cfg)
    reference = runner.run_backtest(cfg)

    assert stats.n_trades == reference.n_trades
    assert math.isclose(stats.final_balance, reference.final_balance, rel_tol=1e-6)
    assert math.isclose(stats.expectancy, reference.expectancy, rel_tol=1e-6)


def test_evaluate_candidates_python_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    price_df = _synthetic_price_frame()
    _stub_prices(monkeypatch, price_df)
    base_cfg = runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=30_000.0,
        long_threshold=0.58,
        short_threshold=0.42,
        risk=RiskConfig(balance=30_000.0, risk_pct=0.03, atr_multiplier=1.0, reward_multiple=1.5, min_position_size=5),
    )
    dataset = prepare_supercalc_dataset(base_cfg)
    configs = [
        base_cfg,
        replace(
            base_cfg,
            long_threshold=0.62,
            short_threshold=0.38,
            risk=RiskConfig(balance=30_000.0, risk_pct=0.025, atr_multiplier=1.0, reward_multiple=1.5, min_position_size=5),
        ),
    ]
    stats_list = evaluate_candidates(dataset, configs, engine="python")
    assert len(stats_list) == 2
    assert stats_list[0].n_trades >= 0


def test_native_engine_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    price_df = _synthetic_price_frame()
    _stub_prices(monkeypatch, price_df)
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="6mo",
        initial_balance=20_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        risk=RiskConfig(balance=20_000.0, risk_pct=0.02, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=5),
    )
    dataset = prepare_supercalc_dataset(cfg)
    if legacy_native_available():
        stats = evaluate_candidates(dataset, [cfg], engine="native")
        assert stats and stats[0].n_trades >= 0
    else:
        with pytest.raises(RuntimeError):
            evaluate_candidates(dataset, [cfg], engine="native")


def test_run_backtest_native_wrapper_smoke() -> None:
    pytest.importorskip("hyprl_supercalc")
    pytest.importorskip("polars")
    if not wrapper_native_available():
        pytest.skip("hyprl_supercalc not built for native wrapper")

    df = _synthetic_price_frame(rows=8)
    signal = [0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0]
    cfg = runner.BacktestConfig(
        ticker="TEST",
        period="1mo",
        initial_balance=15_000.0,
        long_threshold=0.55,
        short_threshold=0.45,
        risk=RiskConfig(balance=15_000.0, risk_pct=0.02, atr_multiplier=1.0, reward_multiple=2.0, min_position_size=10),
    )

    result = run_backtest_native(df, signal, cfg)
    assert isinstance(result, runner.BacktestResult)
    assert result.final_balance > 0.0
    assert len(result.equity_curve) == len(signal)

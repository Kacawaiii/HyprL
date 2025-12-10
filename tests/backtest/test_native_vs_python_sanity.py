from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hyprl.backtest import runner
from hyprl.backtest.runner import BacktestConfig
from hyprl.native.supercalc import native_available, run_backtest_native
from hyprl.supercalc import _build_signal_series, prepare_supercalc_dataset


def _synthetic_price_frame(rows: int = 320) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    base = np.linspace(100.0, 110.0, rows)
    noise = np.sin(np.linspace(0.0, 8 * np.pi, rows)) * 0.5
    close = base + noise
    open_ = close - 0.1
    high = close + 0.7
    low = close - 0.7
    volume = np.linspace(1_000.0, 2_000.0, rows)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=index,
    )


@pytest.mark.skipif(not native_available(), reason="native engine not built")
def test_native_vs_python_sanity(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("hyprl_supercalc")
    pytest.importorskip("polars")

    price_df = _synthetic_price_frame()

    def _fake_get_prices(self, **kwargs):  # noqa: ANN001
        return price_df.copy()

    monkeypatch.setattr(runner.MarketDataFetcher, "get_prices", _fake_get_prices)

    cfg = BacktestConfig(
        ticker="TEST",
        period="6mo",
        interval="1h",
        initial_balance=20_000.0,
        long_threshold=0.6,
        short_threshold=0.4,
        commission_pct=0.0005,
        slippage_pct=0.0005,
    )

    dataset = prepare_supercalc_dataset(cfg)
    py_stats = runner.simulate_from_dataset(dataset, cfg)
    signal, _, _ = _build_signal_series(dataset, cfg)
    native_result = run_backtest_native(dataset.prices, signal, cfg)
    native_metrics = native_result.native_metrics or {}

    assert native_result.final_balance == pytest.approx(py_stats.final_balance, rel=0.05, abs=5.0)
    if py_stats.profit_factor is not None and native_result.profit_factor is not None:
        assert native_result.profit_factor == pytest.approx(py_stats.profit_factor, rel=0.2, abs=0.2)

    assert native_metrics.get("risk_of_ruin") is not None
    assert native_metrics.get("pnl_p95") is not None
    assert native_metrics.get("robustness_score") is not None

#*"""Smoke tests for the hyprl_supercalc PyO3 bridge."""

from __future__ import annotations

import pytest

polars = pytest.importorskip("polars")
hyprl_supercalc = pytest.importorskip("hyprl_supercalc")


def _sample_dataframe() -> "polars.DataFrame":
    return polars.DataFrame(
        {
            "ts": [1, 2, 3, 4],
            "open": [100.0, 101.0, 102.0, 103.0],
            "high": [101.0, 102.0, 103.0, 104.0],
            "low": [99.0, 100.0, 101.0, 102.0],
            "close": [100.0, 101.0, 102.0, 103.0],
            "volume": [1_000.0, 1_100.0, 1_200.0, 1_300.0],
        }
    )


def _default_config() -> dict:
    return {
        "risk_pct": 0.1,
        "commission_pct": 0.0005,
        "slippage_pct": 0.0005,
        "max_leverage": 1.0,
        "params": [],
        "allow_short": True,
        "label": "smoke",
    }


def test_run_batch_backtest_smoke() -> None:
    df = _sample_dataframe()
    signal = [0.0, 0.5, 0.5, 0.0]
    reports = hyprl_supercalc.run_batch_backtest_py(df, signal, [_default_config()])

    assert len(reports) == 1
    report = reports[0]
    assert "metrics" in report
    assert "total_return" in report["metrics"]
    assert report["n_trades"] >= 0


def test_run_batch_backtest_length_mismatch() -> None:
    df = _sample_dataframe()
    signal = [0.0, 0.5, 0.5, 0.0]

    with pytest.raises(ValueError):
        hyprl_supercalc.run_batch_backtest_py(df, signal[:-1], [_default_config()])

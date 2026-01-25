from __future__ import annotations

from types import SimpleNamespace

from scripts import run_backtest


def test_legacy_threshold_fallback():
    args = SimpleNamespace(
        short_threshold=None,
        long_threshold=None,
        ticker="SPY",
        interval="1h",
    )
    settings = {"threshold": 0.4}

    long_thr, short_thr = run_backtest._resolve_thresholds(args, settings)
    assert long_thr == 0.4  # legacy fallback
    assert short_thr == 0.4  # legacy fallback for shorts


def test_long_threshold_defaults_to_point_six():
    args = SimpleNamespace(
        short_threshold=None,
        long_threshold=None,
        ticker="SPY",
        interval="1h",
    )
    settings = {}

    long_thr, short_thr = run_backtest._resolve_thresholds(args, settings)
    assert long_thr == 0.6
    assert short_thr == 0.4

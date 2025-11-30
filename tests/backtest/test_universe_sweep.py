from __future__ import annotations

import importlib.util
from pathlib import Path
import types

import pytest

import numpy as np

from hyprl.backtest.runner import ThresholdSummary

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_universe_sweep.py"
SPEC = importlib.util.spec_from_file_location("run_universe_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)

parse_tickers = MODULE.parse_tickers
parse_thresholds = MODULE.parse_thresholds
select_best_summary = MODULE.select_best_summary
compute_score = MODULE.compute_score


def _summary(
    threshold: float,
    total_return: float,
    benchmark_return: float,
    alpha: float,
    sharpe: float | None,
) -> ThresholdSummary:
    return ThresholdSummary(
        threshold=threshold,
        final_balance=10_000.0,
        total_return=total_return,
        n_trades=15,
        win_rate=0.55,
        max_drawdown=0.08,
        sharpe_ratio=sharpe,
        benchmark_return=benchmark_return,
        alpha_return=alpha,
        annualized_return=total_return / 100.0,
        profit_factor=1.5,
        expectancy=1.0,
    )


def test_parse_helpers():
    assert parse_tickers("AAPL, msft , spy") == ["AAPL", "MSFT", "SPY"]
    assert np.allclose(parse_thresholds(None), [0.55, 0.6, 0.65, 0.7, 0.75])
    assert parse_thresholds("0.6,0.7") == [0.6, 0.7]


def test_select_best_summary_prefers_positive_alpha():
    summaries = [
        _summary(0.55, total_return=-5.0, benchmark_return=-2.0, alpha=-3.0, sharpe=0.2),
        _summary(0.65, total_return=8.0, benchmark_return=3.0, alpha=5.0, sharpe=1.1),
        _summary(0.75, total_return=6.0, benchmark_return=4.0, alpha=2.0, sharpe=0.9),
    ]
    best = select_best_summary(summaries)
    assert best.threshold == 0.65


def test_select_best_summary_handles_all_negative():
    summaries = [
        _summary(0.55, total_return=-4.0, benchmark_return=-1.0, alpha=-3.0, sharpe=None),
        _summary(0.65, total_return=-2.0, benchmark_return=1.0, alpha=-3.0, sharpe=0.1),
        _summary(0.75, total_return=-1.0, benchmark_return=-2.0, alpha=1.0, sharpe=None),
    ]
    best = select_best_summary(summaries)
    assert best.threshold == 0.75


def test_compute_score_accounts_for_tradable_flag():
    summary = _summary(0.6, total_return=5.0, benchmark_return=2.0, alpha=3.0, sharpe=1.0)
    assert compute_score(summary, True) > compute_score(summary, False)


def _make_args(**overrides):
    defaults = dict(
        period="1y",
        start=None,
        end=None,
        interval="1h",
        initial_balance=10_000.0,
        seed=42,
        short_threshold=None,
        thresholds_list=[0.55],
        model_type=None,
        calibration=None,
        risk_profile=None,
        adaptive=False,
        adaptive_lookback=None,
        adaptive_default_regime=None,
        force_non_adaptive=False,
    )
    defaults.update(overrides)
    return types.SimpleNamespace(**defaults)


def test_summarize_ticker_force_non_adaptive(monkeypatch: pytest.MonkeyPatch):
    args = _make_args(force_non_adaptive=True)
    monkeypatch.setattr(MODULE, "load_ticker_settings", lambda ticker, interval: {"tradable": True})
    monkeypatch.setattr(MODULE, "load_short_threshold", lambda settings: 0.4)
    monkeypatch.setattr(MODULE, "get_risk_settings", lambda settings, profile: {"risk_pct": 0.01})

    captured: dict[str, bool | None] = {"enable": None}

    def fake_build_config(ticker, args_obj, short_threshold, adaptive_overrides):
        captured["enable"] = adaptive_overrides.get("enable")

        class DummyCfg:
            def __init__(self, enabled: bool):
                self.adaptive = types.SimpleNamespace(enable=enabled)

        return DummyCfg(bool(adaptive_overrides.get("enable")))

    monkeypatch.setattr(MODULE, "build_config", fake_build_config)
    monkeypatch.setattr(MODULE, "sweep_thresholds", lambda cfg, thresholds: [_summary(0.55, 5.0, 2.0, 3.0, 1.0)])

    row = MODULE.summarize_ticker("TEST", args, [0.55])
    assert captured["enable"] is False
    assert row["adaptive"] is False


def test_summarize_ticker_adaptive_flag(monkeypatch: pytest.MonkeyPatch):
    args = _make_args(adaptive=True)
    monkeypatch.setattr(MODULE, "load_ticker_settings", lambda ticker, interval: {"tradable": True})
    monkeypatch.setattr(MODULE, "load_short_threshold", lambda settings: 0.4)
    monkeypatch.setattr(MODULE, "get_risk_settings", lambda settings, profile: {"risk_pct": 0.01})

    monkeypatch.setattr(MODULE, "build_config", lambda *a, **k: types.SimpleNamespace(adaptive=types.SimpleNamespace(enable=True)))
    monkeypatch.setattr(MODULE, "sweep_thresholds", lambda cfg, thresholds: [_summary(0.55, 5.0, 2.0, 3.0, 1.0)])

    row = MODULE.summarize_ticker("TEST", args, [0.55])
    assert row["adaptive"] is True

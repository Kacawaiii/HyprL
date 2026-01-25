from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from hyprl.search.optimizer import CandidateConfig, SearchResult


def _load_run_supersearch_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_supersearch.py"
    spec = importlib.util.spec_from_file_location("run_supersearch_module", script_path)
    if spec is None or spec.loader is None:
        pytest.skip("run_supersearch.py introuvable")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("run_supersearch_module", module)
    spec.loader.exec_module(module)
    return module


def _sample_result() -> SearchResult:
    candidate = CandidateConfig(
        long_threshold=0.55,
        short_threshold=0.4,
        risk_pct=0.02,
        min_ev_multiple=0.0,
        trend_filter=False,
        sentiment_min=-1.0,
        sentiment_max=1.0,
        sentiment_regime="off",
    )
    return SearchResult(
        config=candidate,
        strategy_return_pct=12.0,
        benchmark_return_pct=5.0,
        alpha_pct=7.0,
        profit_factor=1.2,
        sharpe=0.9,
        max_drawdown_pct=15.0,
        expectancy=1.0,
        n_trades=80,
        win_rate_pct=55.0,
        portfolio_profit_factor=1.2,
        portfolio_sharpe=0.9,
        portfolio_max_drawdown_pct=14.0,
    )


def test_supersearch_cli_binary_smoke(tmp_path, monkeypatch):
    module = _load_run_supersearch_module()
    monkeypatch.setattr(module, "run_search", lambda cfg: [_sample_result()])
    out_path = tmp_path / "binary.csv"
    argv = [
        "--ticker",
        "TEST",
        "--period",
        "90d",
        "--interval",
        "1h",
        "--min-trades",
        "1",
        "--min-pf",
        "0.0",
        "--min-sharpe",
        "-5",
        "--max-dd",
        "1.0",
        "--max-ror",
        "1.0",
        "--output",
        str(out_path),
    ]
    exit_code = module.main(argv)
    assert exit_code == 0
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 1
    required = {"long_threshold", "short_threshold", "risk_pct", "profit_factor", "primary_ticker"}
    assert required.issubset(df.columns)


def test_supersearch_cli_amplitude_allows_empty(tmp_path, monkeypatch):
    module = _load_run_supersearch_module()
    monkeypatch.setattr(module, "run_search", lambda cfg: [])
    out_path = tmp_path / "amplitude.csv"
    argv = [
        "--ticker",
        "TEST",
        "--period",
        "90d",
        "--interval",
        "1h",
        "--label-mode",
        "amplitude",
        "--output",
        str(out_path),
    ]
    exit_code = module.main(argv)
    assert exit_code == 0
    assert out_path.exists()
    df = pd.read_csv(out_path)
    required = {"long_threshold", "short_threshold", "risk_pct"}
    assert required.issubset(df.columns)


def test_supersearch_cli_relaxed_sharpe(tmp_path, monkeypatch):
    """Validate that --min-sharpe 0.0 fully disables Sharpe filtering."""
    module = _load_run_supersearch_module()

    captured_config = {}

    def capture_run_search(cfg):
        captured_config["cfg"] = cfg
        return [_sample_result()]

    monkeypatch.setattr(module, "run_search", capture_run_search)
    out_path = tmp_path / "relaxed.csv"
    argv = [
        "--ticker",
        "TEST",
        "--period",
        "90d",
        "--interval",
        "1h",
        "--min-sharpe",
        "0.0",
        "--output",
        str(out_path),
    ]
    exit_code = module.main(argv)
    assert exit_code == 0
    assert captured_config["cfg"].min_sharpe == 0.0
    df = pd.read_csv(out_path)
    assert len(df) == 1


def test_supersearch_cli_default_sharpe(tmp_path, monkeypatch):
    module = _load_run_supersearch_module()

    captured_config = {}

    def capture_run_search(cfg):
        captured_config["cfg"] = cfg
        return [_sample_result()]

    monkeypatch.setattr(module, "run_search", capture_run_search)
    out_path = tmp_path / "default_sharpe.csv"
    argv = [
        "--ticker",
        "TEST",
        "--period",
        "90d",
        "--interval",
        "1h",
        "--output",
        str(out_path),
    ]
    exit_code = module.main(argv)
    assert exit_code == 0
    cfg = captured_config["cfg"]
    expected_default = cfg.__class__.__dataclass_fields__["min_sharpe"].default
    assert cfg.min_sharpe == expected_default

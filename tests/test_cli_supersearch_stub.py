from __future__ import annotations

import runpy
import sys
from pathlib import Path

import pandas as pd


def test_supersearch_cli_drysim(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_supersearch.py"
    output_csv = tmp_path / "dry.csv"
    argv = [
        "run_supersearch.py",
        "--ticker",
        "AAPL",
        "--period",
        "90d",
        "--interval",
        "1h",
        "--output",
        str(output_csv),
        "--drysim",
        "ok",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.syspath_prepend(str(repo_root))

    runpy.run_path(str(script), run_name="__main__")

    assert output_csv.exists()
    df = pd.read_csv(output_csv)
    assert "sentiment_min" in df.columns
    assert "portfolio_weights" in df.columns
    assert len(df) == 1


def test_supersearch_cli_invokes_run_search(tmp_path: Path, monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    script = repo_root / "scripts" / "run_supersearch.py"
    output_csv = tmp_path / "live.csv"

    captured = {}

    from hyprl.search.optimizer import CandidateConfig, SearchResult

    def _fake_run_search(config):  # noqa: ANN001
        captured["ticker"] = config.ticker
        captured["weighting_scheme"] = config.weighting_scheme
        candidate = CandidateConfig(
            long_threshold=0.6,
            short_threshold=0.4,
            risk_pct=0.02,
            min_ev_multiple=0.1,
            trend_filter=False,
            sentiment_min=-0.5,
            sentiment_max=0.5,
            sentiment_regime="off",
        )
        return [
            SearchResult(
                config=candidate,
                strategy_return_pct=12.0,
                benchmark_return_pct=8.0,
                alpha_pct=4.0,
                profit_factor=1.3,
                sharpe=0.9,
                max_drawdown_pct=10.0,
                expectancy=35.0,
                n_trades=25,
                win_rate_pct=56.0,
                sentiment_stats={"trades_in_fear": 2, "trades_in_greed": 3},
            )
        ]

    monkeypatch.setattr("hyprl.search.optimizer.run_search", _fake_run_search)

    argv = [
        "run_supersearch.py",
        "--ticker",
        "MSFT",
        "--period",
        "120d",
        "--interval",
        "1h",
        "--long-thresholds",
        "0.55,0.6",
        "--short-thresholds",
        "0.35,0.4",
        "--risk-pcts",
        "0.01,0.02",
        "--min-ev-multiples",
        "0.0,0.2",
        "--trend-filter-flags",
        "true,false",
        "--sentiment-min-values=-0.4,-0.2",
        "--sentiment-max-values=0.4,0.6",
        "--sentiment-regimes",
        "off,neutral_only",
        "--output",
        str(output_csv),
        "--weighting-scheme",
        "inv_vol",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.syspath_prepend(str(repo_root))

    runpy.run_path(str(script), run_name="__main__")

    assert captured["ticker"] == "MSFT"
    assert captured["weighting_scheme"] == "inv_vol"
    df = pd.read_csv(output_csv)
    assert {"sentiment_min", "sentiment_max", "sentiment_regime"}.issubset(df.columns)

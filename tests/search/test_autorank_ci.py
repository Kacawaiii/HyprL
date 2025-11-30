from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hyprl.meta.model import MetaRobustnessModel


def _train_meta_model(path: Path) -> Path:
    rows = 20
    df = pd.DataFrame(
        {
            "long_threshold": np.linspace(0.55, 0.65, rows),
            "short_threshold": np.linspace(0.35, 0.45, rows),
            "risk_pct": 0.02,
            "min_ev_multiple": 0.1,
            "trend_filter": ["true"] * rows,
            "sentiment_min": -0.4,
            "sentiment_max": 0.4,
            "sentiment_regime": "off",
            "weighting_scheme": "equal",
            "pf_backtest": np.linspace(1.0, 1.5, rows),
            "sharpe_backtest": np.linspace(0.5, 1.0, rows),
            "maxdd_backtest": np.linspace(10, 25, rows),
            "winrate_backtest": 0.55,
            "equity_vol_backtest": 0.1,
            "trades_backtest": 50,
            "correlation_mean": 0.2,
            "correlation_max": 0.4,
            "robustness_score": np.linspace(0.3, 0.9, rows),
        }
    )
    from hyprl.meta.features import select_meta_features

    X, _ = select_meta_features(df)
    y = df["robustness_score"].to_numpy()
    model = MetaRobustnessModel(model_type="rf", random_state=7)
    model.fit(X, y)
    outdir = path / "artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    model_path = outdir / "model.joblib"
    model.save(model_path)
    meta = {
        "model_type": "rf",
        "dataset_hash": "deadbeef",
        "timestamp": "2025-11-11T00:00:00Z",
    }
    with (outdir / "meta.json").open("w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    return model_path


def _build_supersearch_csv(path: Path) -> Path:
    df = pd.DataFrame(
        [
            {
                "strat_id": "S1",
                "config_index": 0,
                "long_threshold": 0.6,
                "short_threshold": 0.4,
                "risk_pct": 0.02,
                "min_ev_multiple": 0.1,
                "trend_filter": True,
                "sentiment_min": -0.4,
                "sentiment_max": 0.4,
                "sentiment_regime": "off",
                "portfolio_profit_factor": 1.2,
                "portfolio_sharpe": 0.8,
                "portfolio_max_drawdown_pct": 20.0,
                "win_rate_pct": 55.0,
                "portfolio_equity_vol": 0.12,
                "n_trades": 60,
                "correlation_mean": 0.2,
                "correlation_max": 0.4,
                "tickers": "AAA,BBB",
                "base_score": 0.5,
            },
            {
                "strat_id": "S2",
                "config_index": 1,
                "long_threshold": 0.62,
                "short_threshold": 0.38,
                "risk_pct": 0.018,
                "min_ev_multiple": 0.05,
                "trend_filter": False,
                "sentiment_min": -0.3,
                "sentiment_max": 0.5,
                "sentiment_regime": "off",
                "portfolio_profit_factor": 1.1,
                "portfolio_sharpe": 0.7,
                "portfolio_max_drawdown_pct": 22.0,
                "win_rate_pct": 53.0,
                "portfolio_equity_vol": 0.15,
                "n_trades": 55,
                "correlation_mean": 0.25,
                "correlation_max": 0.45,
                "tickers": "CCC",
                "base_score": 0.5,
            },
        ]
    )
    csv_path = path / "supersearch.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def test_autorank_script(tmp_path: Path, monkeypatch) -> None:
    model_path = _train_meta_model(tmp_path)
    csv_path = _build_supersearch_csv(tmp_path)
    output_path = tmp_path / "ranked.csv"
    eval_path = tmp_path / "summary.txt"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "autorank_supersearch.py"
    argv = [
        "autorank_supersearch.py",
        "--csv",
        str(csv_path),
        "--meta-robustness",
        str(model_path),
        "--meta-weight",
        "0.6",
        "--topk",
        "2",
        "--out",
        str(output_path),
        "--seed",
        "7",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(script), run_name="__main__")

    assert output_path.exists()
    ranked = pd.read_csv(output_path)
    assert {"meta_pred", "final_score", "meta_model_type"}.issubset(ranked.columns)
    assert ranked["base_score_normalized"].between(0, 1).all()
    assert ranked.iloc[0]["tickers"] == "AAA,BBB"
    summary_path = output_path.with_suffix(".SUMMARY.txt")
    assert summary_path.exists()
    content = summary_path.read_text()
    assert "meta_model_type" in content or "Meta info" in content

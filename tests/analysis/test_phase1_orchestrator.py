from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pandas as pd

from scripts.run_phase1_from_autorank import run_workflow


def test_phase1_orchestrator_creates_artifacts(tmp_path, monkeypatch):
    source_csv = tmp_path / "supersearch.csv"
    supersearch_df = pd.DataFrame(
        {
            "long_threshold": [0.55],
            "short_threshold": [0.45],
            "risk_pct": [0.01],
            "min_ev_multiple": [0.0],
            "trend_filter": [False],
            "sentiment_min": [-0.4],
            "sentiment_max": [0.4],
            "sentiment_regime": ["off"],
        }
    )
    supersearch_df.to_csv(source_csv, index=False)
    autoranked = tmp_path / "autoranked.csv"
    autorank_df = pd.DataFrame(
        {
            "tickers": ["AAA"],
            "config_index": [0],
            "source_csv": [str(source_csv)],
            "long_threshold": [0.55],
            "short_threshold": [0.45],
            "risk_pct": [0.01],
            "min_ev_multiple": [0.0],
            "trend_filter": [False],
            "sentiment_min": [-0.4],
            "sentiment_max": [0.4],
            "sentiment_regime": ["off"],
            "portfolio_pf": [1.5],
            "portfolio_sharpe": [1.1],
            "portfolio_dd": [10.0],
            "trades_backtest": [80],
            "corr_max": [0.4],
            "final_score": [0.9],
            "base_score_normalized": [0.8],
            "interval": ["1h"],
            "period": ["1y"],
        }
    )
    autorank_df.to_csv(autoranked, index=False)

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    def fake_sessions(panel_df, config):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "session_id": "sess_1",
                    "strat_id": panel_df.loc[0, "strat_id"],
                    "source_csv": panel_df.loc[0, "source_csv"],
                    "config_index": int(panel_df.loc[0, "config_index"]),
                    "tickers": panel_df.loc[0, "tickers"],
                    "interval": panel_df.loc[0, "interval"],
                    "period": panel_df.loc[0, "period"],
                    "initial_balance": config.initial_balance,
                    "engine": config.engine,
                    "log_dir": str(logs_dir),
                }
            ]
        )

    def fake_results(sessions_df):  # noqa: ANN001
        return pd.DataFrame(
            [
                {
                    "session_id": sessions_df.loc[0, "session_id"],
                    "pf_live": 1.1,
                    "robustness_score": 0.95,
                }
            ]
        )

    monkeypatch.setattr("hyprl.phase1.paper.run_phase1_sessions", fake_sessions)
    monkeypatch.setattr("hyprl.phase1.results.build_phase1_results", fake_results)
    monkeypatch.setattr("scripts.run_phase1_from_autorank._git_hash", lambda: "deadbeef")

    args = Namespace(
        autoranked=str(autoranked),
        csv=None,
        meta_robustness=None,
        meta_registry=None,
        meta_calibration=None,
        meta_calibration_registry=None,
        meta_weight=0.4,
        min_portfolio_pf=0.0,
        min_portfolio_sharpe=-10.0,
        max_portfolio_dd=1.0,
        max_corr=1.0,
        min_trades=0,
        min_weight_per_ticker=None,
        max_weight_per_ticker=None,
        max_strategies=1,
        initial_balance=10_000.0,
        period="1y",
        interval="1h",
        start=None,
        end=None,
        engine="auto",
        model_type="logistic",
        calibration="none",
        default_long_threshold=0.6,
        default_short_threshold=0.4,
        session_prefix="phase1",
        outdir=str(tmp_path / "run"),
    )

    outputs = run_workflow(args)
    for key in ("panel", "sessions", "results"):
        csv_path = outputs[key]
        assert csv_path.exists()
        meta_path = csv_path.with_suffix(csv_path.suffix + ".meta.json")
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "sha256" in meta
    assert outputs["summary"].exists()
    provenance = json.loads(outputs["provenance"].read_text())
    assert provenance["git_hash"] == "deadbeef"
    assert str(outputs["panel"]) in provenance["dataset_hashes"]

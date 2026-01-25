from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hyprl.meta.features import select_meta_features
from hyprl.meta.model import MetaRobustnessModel


def _build_dataset(path: Path, rows: int = 40) -> None:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "long_threshold": rng.uniform(0.52, 0.65, size=rows),
            "short_threshold": rng.uniform(0.35, 0.48, size=rows),
            "risk_pct": rng.uniform(0.01, 0.03, size=rows),
            "min_ev_multiple": rng.uniform(0.0, 0.3, size=rows),
            "trend_filter": rng.choice(["true", "false"], size=rows),
            "sentiment_min": rng.uniform(-0.6, -0.1, size=rows),
            "sentiment_max": rng.uniform(0.1, 0.6, size=rows),
            "sentiment_regime": rng.choice(["off", "neutral_only"], size=rows),
            "weighting_scheme": rng.choice(["equal", "inv_vol"], size=rows),
            "pf_backtest": rng.uniform(1.0, 1.6, size=rows),
            "sharpe_backtest": rng.uniform(0.3, 1.2, size=rows),
            "maxdd_backtest": rng.uniform(10.0, 35.0, size=rows),
            "winrate_backtest": rng.uniform(0.4, 0.7, size=rows),
            "equity_vol_backtest": rng.uniform(0.05, 0.2, size=rows),
            "trades_backtest": rng.integers(30, 120, size=rows),
            "correlation_mean": rng.uniform(0.1, 0.5, size=rows),
            "correlation_max": rng.uniform(0.3, 0.8, size=rows),
            "pf_ratio": rng.uniform(0.7, 1.2, size=rows),
            "sharpe_ratio": rng.uniform(0.6, 1.3, size=rows),
            "dd_ratio": rng.uniform(0.7, 1.4, size=rows),
            "winrate_delta": rng.uniform(-0.1, 0.1, size=rows),
            "equity_vol_ratio": rng.uniform(0.8, 1.3, size=rows),
        }
    )
    df["robustness_score"] = (
        0.4 * (df["pf_ratio"] / 2.0)
        + 0.3 * (df["sharpe_ratio"] / 2.0)
        + 0.2 * (2.0 - np.clip(df["dd_ratio"], 0.0, 2.0)) / 2.0
        + 0.1 * (2.0 - np.clip(df["equity_vol_ratio"], 0.0, 2.0)) / 2.0
    )
    df["robustness_score"] = df["robustness_score"].clip(0.0, 1.0)
    df.to_csv(path, index=False)


def test_train_meta_robustness_cli(tmp_path: Path, monkeypatch) -> None:
    dataset = tmp_path / "meta.csv"
    _build_dataset(dataset)
    outdir = tmp_path / "artifacts"
    eval_path = tmp_path / "eval.json"
    repo_root = Path(__file__).resolve().parents[2]
    script = repo_root / "scripts" / "train_meta_robustness.py"
    argv = [
        "train_meta_robustness.py",
        "--dataset",
        str(dataset),
        "--model",
        "rf",
        "--kfold",
        "3",
        "--outdir",
        str(outdir),
        "--eval-output",
        str(eval_path),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(script), run_name="__main__")

    model_path = outdir / "model.joblib"
    meta_path = outdir / "meta.json"
    assert model_path.exists()
    assert meta_path.exists()
    meta_model = MetaRobustnessModel.load(model_path)
    df = pd.read_csv(dataset)
    X, _ = select_meta_features(df)
    preds = meta_model.predict(X)
    assert preds.shape[0] == len(df)
    with meta_path.open() as fh:
        meta = json.load(fh)
    assert meta["model_type"] == "rf"

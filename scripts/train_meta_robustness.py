#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hyprl.meta.features import select_meta_features
from hyprl.meta.model import MetaRobustnessModel
from hyprl.meta.scoring import EvalReport, evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Meta-ML model to predict Phase1 robustness.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experiments/meta_robustness_dataset.csv"),
        help="Meta dataset (features + robustness_score).",
    )
    parser.add_argument("--model", choices=["elasticnet", "rf", "xgb"], default="rf")
    parser.add_argument("--kfold", type=int, default=5, help="Number of CV splits.")
    parser.add_argument("--outdir", type=Path, default=Path("artifacts/meta_ml/robustness_v0"))
    parser.add_argument(
        "--eval-output",
        type=Path,
        default=Path("data/experiments/meta_ml_eval.json"),
        help="Path to export evaluation metrics summary.",
    )
    return parser.parse_args()


def _hash_dataframe(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return hasher.hexdigest()


def _aggregate_reports(reports: list[EvalReport]) -> dict[str, float]:
    if not reports:
        return {"r2": 0.0, "mae": 0.0, "spearman": float("nan"), "kendall": float("nan"), "n": 0}
    r2 = float(np.mean([rep.r2 for rep in reports]))
    mae = float(np.mean([rep.mae for rep in reports]))
    spearman = float(np.nanmean([rep.spearman for rep in reports]))
    kendall = float(np.nanmean([rep.kendall for rep in reports]))
    total_n = int(np.sum([rep.n for rep in reports]))
    return {"r2": r2, "mae": mae, "spearman": spearman, "kendall": kendall, "n": total_n}


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset introuvable: {args.dataset}")
    df = pd.read_csv(args.dataset)
    if df.empty:
        raise SystemExit("Dataset vide, impossible d'entraîner le modèle Meta-ML.")
    if "robustness_score" not in df.columns:
        raise SystemExit("Dataset sans colonne robustness_score.")

    X_raw, feature_names = select_meta_features(df)
    y = df["robustness_score"].clip(0.0, 1.0).to_numpy()

    if len(df) < args.kfold:
        print(f"[WARN] Dataset trop petit pour kfold={args.kfold}, réduction à n={len(df)} splits.")
        args.kfold = len(df)

    kf = KFold(n_splits=args.kfold, shuffle=True, random_state=42)
    reports: list[EvalReport] = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_raw), start=1):
        model = MetaRobustnessModel(model_type=args.model, random_state=42)
        model.fit(X_raw.iloc[train_idx], y[train_idx])
        preds = model.predict(X_raw.iloc[test_idx])
        report = evaluate_predictions(y[test_idx], preds)
        reports.append(report)
        print(
            f"[CV] Fold {fold}/{args.kfold} – R2={report.r2:.3f} "
            f"MAE={report.mae:.4f} Spearman={report.spearman:.3f} Kendall={report.kendall:.3f}"
        )

    summary = _aggregate_reports(reports)
    print(
        f"[CV] Moyenne – R2={summary['r2']:.3f} MAE={summary['mae']:.4f} "
        f"Spearman={summary['spearman']:.3f} Kendall={summary['kendall']:.3f}"
    )

    final_model = MetaRobustnessModel(model_type=args.model, random_state=42)
    final_model.fit(X_raw, y)
    args.outdir.mkdir(parents=True, exist_ok=True)
    model_path = args.outdir / "model.joblib"
    final_model.save(model_path)

    dataset_hash = _hash_dataframe(df)
    timestamp = datetime.now(timezone.utc).isoformat()
    meta = {
        "model_type": args.model,
        "kfold": args.kfold,
        "feature_names": feature_names,
        "cv_metrics": summary,
        "n_rows": len(df),
        "dataset": str(args.dataset),
        "dataset_hash": dataset_hash,
        "model_path": str(model_path),
        "timestamp": timestamp,
    }

    meta_path = args.outdir / "meta.json"
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    args.eval_output.parent.mkdir(parents=True, exist_ok=True)
    with args.eval_output.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[OK] Modèle sauvegardé → {model_path}")
    print(f"[OK] Rapport écrit → {meta_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from hyprl.meta.calibration import build_calibrator
from hyprl.meta.features import select_meta_features
from hyprl.meta.model import MetaRobustnessModel, apply_calibration
from hyprl.meta.registry import resolve_model
from hyprl.meta.scoring import evaluate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate Meta-ML predictions via isotonic or quantile mapping.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/experiments/meta_robustness_dataset.csv"),
        help="Meta dataset CSV (must include robustness_score).",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path or registry key@stage referencing the Meta-ML model joblib.",
    )
    parser.add_argument(
        "--method",
        choices=["isotonic", "quantile"],
        default="isotonic",
        help="Calibration method (default isotonic).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/meta_ml/robustness_v0/calibrator.joblib"),
        help="Output path for the calibrator artifact.",
    )
    parser.add_argument("--kfold", type=int, default=5, help="Number of CV splits for OOF predictions.")
    return parser.parse_args()


def _hash_dataframe(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return hasher.hexdigest()


def _resolve_model_path(payload: str) -> Path:
    candidate = Path(payload)
    if candidate.exists():
        return candidate
    if "@" in payload:
        return resolve_model(payload)
    raise FileNotFoundError(f"Model artifact not found: {payload}")


def _compute_oof_predictions(
    features: pd.DataFrame,
    targets: np.ndarray,
    base_model: MetaRobustnessModel,
    n_splits: int,
) -> np.ndarray:
    splits = min(max(2, n_splits), len(features))
    kf = KFold(n_splits=splits, shuffle=True, random_state=42)
    preds = np.zeros(len(features), dtype=float)
    for train_idx, test_idx in kf.split(features):
        fold_model = MetaRobustnessModel(model_type=base_model.model_type, random_state=base_model.random_state)
        fold_model.fit(features.iloc[train_idx], targets[train_idx])
        preds[test_idx] = fold_model.predict(features.iloc[test_idx])
    return preds


def main() -> None:
    args = parse_args()
    if not args.dataset.exists():
        raise SystemExit(f"Dataset introuvable: {args.dataset}")
    df = pd.read_csv(args.dataset)
    if df.empty or "robustness_score" not in df.columns:
        raise SystemExit("Dataset invalide: colonne robustness_score manquante ou vide.")

    model_path = _resolve_model_path(args.model)
    base_model = MetaRobustnessModel.load(model_path)
    X_raw, _ = select_meta_features(df)
    y = df["robustness_score"].clip(0.0, 1.0).to_numpy()
    if len(df) < 2:
        raise SystemExit("Dataset trop petit pour calibrer les prédictions.")

    oof_preds = _compute_oof_predictions(X_raw, y, base_model, args.kfold)
    baseline_report = evaluate_predictions(y, oof_preds)
    calibrator = build_calibrator(args.method, oof_preds, y)
    calibrated = apply_calibration(oof_preds, calibrator)
    calibrated_report = evaluate_predictions(y, calibrated)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrator, args.out)
    meta = {
        "method": args.method,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset),
        "dataset_hash": _hash_dataframe(df),
        "model": str(model_path),
        "kfold": min(max(2, args.kfold), len(df)),
        "metrics_before": {
            "r2": baseline_report.r2,
            "mae": baseline_report.mae,
            "spearman": baseline_report.spearman,
            "kendall": baseline_report.kendall,
        },
        "metrics_after": {
            "r2": calibrated_report.r2,
            "mae": calibrated_report.mae,
            "spearman": calibrated_report.spearman,
            "kendall": calibrated_report.kendall,
        },
        "calibrator_path": str(args.out),
    }
    meta_path = args.out.with_name("calibration_meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(
        "[OK] Calibrator saved → {path} (method={method}, Spearman {before:.3f}→{after:.3f})".format(
            path=args.out,
            method=args.method,
            before=baseline_report.spearman,
            after=calibrated_report.spearman,
        )
    )
    print(f"[OK] Calibration meta → {meta_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train NVDA 1h v3 model with symmetric labels (long/short, neutral dropped)."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from hyprl.labels.amplitude import compute_symmetric_binary_labels
from hyprl.model.probability import ProbabilityCalibrator, ProbabilityModel

FEATURES: Sequence[str] = (
    # Momentum horizons
    "ret_1h",
    "ret_3h",
    "ret_6h",
    "ret_24h",
    # ATR stack
    "atr_14",
    "atr_72",
    "atr_14_norm",
    "atr_72_norm",
    # RSI stack
    "rsi_7",
    "rsi_14",
    "rsi_21",
    # Volatility regime
    "vol_ratio_10_30",
    "vol_regime_high",
    # Volume signals
    "volume_zscore_24",
    "volume_surge",
    # Range/volatility
    "range_pct",
    "true_range",
    # Higher moments
    "ret_skew_20",
    "ret_kurt_20",
)

DEFAULT_DATASET = pathlib.Path("data/cache/nvda_1h_features_v3.parquet")
DEFAULT_ARTIFACT = pathlib.Path("models/nvda_1h_xgb_v3.joblib")
DEFAULT_FEATURE_LIST = pathlib.Path("models/nvda_1h_xgb_v3_features.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NVDA 1h v3 (symmetric labels, 5y window).")
    parser.add_argument("--dataset", type=pathlib.Path, default=DEFAULT_DATASET, help="Parquet with NVDA features.")
    parser.add_argument("--artifact", type=pathlib.Path, default=DEFAULT_ARTIFACT, help="Output artifact path.")
    parser.add_argument(
        "--feature-list",
        type=pathlib.Path,
        default=DEFAULT_FEATURE_LIST,
        help="Path to write selected feature names.",
    )
    parser.add_argument("--horizon", type=int, default=6, help="Forward horizon (bars) for label computation.")
    parser.add_argument(
        "--threshold-pct",
        type=float,
        default=0.003,
        help="Decimal threshold for symmetric labels (e.g., 0.003 = 0.3%).",
    )
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction (chronological split).")
    parser.add_argument("--n-estimators", type=int, default=800, help="XGBoost n_estimators.")
    parser.add_argument("--max-depth", type=int, default=6, help="XGBoost max_depth.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="XGBoost learning_rate.")
    parser.add_argument("--subsample", type=float, default=0.8, help="XGBoost subsample.")
    parser.add_argument("--colsample", type=float, default=0.8, help="XGBoost colsample_bytree.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel threads for XGBoost.")
    return parser.parse_args()


def _resolve_close(df: pd.DataFrame) -> pd.Series:
    for col in ("close", "price", "adj_close"):
        if col in df.columns:
            return df[col].astype(float)
    raise SystemExit("Dataset must contain a close/price column for label computation.")


def load_dataset(path: pathlib.Path, horizon: int, threshold_pct: float) -> tuple[pd.DataFrame, list[str]]:
    if not path.exists():
        raise SystemExit(f"Dataset not found: {path}")
    df = pd.read_parquet(path)
    close = _resolve_close(df)
    labels = compute_symmetric_binary_labels(close, horizon=horizon, threshold_pct=threshold_pct)
    df = df.copy()
    df["target"] = labels
    feat_cols = [col for col in FEATURES if col in df.columns]
    missing = set(FEATURES) - set(feat_cols)
    if missing:
        raise SystemExit(f"Missing feature columns: {sorted(missing)}")
    df = df[feat_cols + ["target"]].dropna()
    if df.empty:
        raise SystemExit("Dataset is empty after dropping NaNs/labels.")
    return df, feat_cols


def split_time(df: pd.DataFrame, val_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.copy()
    split = int(len(df_sorted) * (1 - val_frac))
    split = max(1, min(split, len(df_sorted) - 1))
    return df_sorted.iloc[:split], df_sorted.iloc[split:]


def train_model(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feat_cols: list[str],
    *,
    n_estimators: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample: float,
    random_state: int,
    n_jobs: int,
) -> tuple[ProbabilityModel, StandardScaler, tuple[np.ndarray, np.ndarray]]:
    X_train = df_train[feat_cols].to_numpy()
    y_train = df_train["target"].to_numpy()
    X_val = df_val[feat_cols].to_numpy()
    y_val = df_val["target"].to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

    clf = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=random_state,
        n_jobs=n_jobs,
        eval_metric="logloss",
    )
    clf.fit(X_train_s, y_train, sample_weight=sample_weights, eval_set=[(X_val_s, y_val)], verbose=False)

    calibrator = ProbabilityCalibrator(method="none")
    model = ProbabilityModel(scaler=scaler, classifier=clf, calibrator=calibrator, model_type="xgboost")
    return model, scaler, (X_val_s, y_val)


def save_artifact(model: ProbabilityModel, feat_cols: list[str], artifact: pathlib.Path, feature_list: pathlib.Path) -> None:
    artifact.parent.mkdir(parents=True, exist_ok=True)
    model.dump(artifact)
    feature_list.write_text(json.dumps(feat_cols, indent=2))


def main() -> None:
    args = parse_args()
    t0 = time.time()
    df, feat_cols = load_dataset(args.dataset, args.horizon, args.threshold_pct)
    train_df, val_df = split_time(df, args.val_frac)
    model, scaler, val_data = train_model(
        train_df,
        val_df,
        feat_cols,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample=args.colsample,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )
    save_artifact(model, feat_cols, args.artifact, args.feature_list)
    t1 = time.time()

    X_val_s, y_val = val_data
    proba = model.predict_proba(X_val_s)
    preds = (proba >= 0.5).astype(int)
    acc = float((preds == y_val).mean())
    q = np.quantile(proba, [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1])

    pos = float((train_df["target"] == 1).sum())
    neg = float((train_df["target"] == 0).sum())
    print(
        f"[DATA] rows={len(df)} train={len(train_df)} val={len(val_df)} "
        f"class_balance train pos={pos:.0f} neg={neg:.0f}"
    )
    print(
        f"[VAL] acc={acc:.3f} proba_range=({proba.min():.3f},{proba.max():.3f}) "
        f"quantiles={q}"
    )
    print(f"[OUT] artifact={args.artifact} | features={args.feature_list} | time={t1 - t0:.2f}s")


if __name__ == "__main__":
    main()

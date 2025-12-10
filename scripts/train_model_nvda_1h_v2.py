#!/usr/bin/env python
"""Train NVDA 1h v2 model (balanced long/short) on 21-feature cache."""

from __future__ import annotations

import json
import pathlib
import time
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from hyprl.model.probability import ProbabilityCalibrator, ProbabilityModel

FEATURES: Sequence[str] = (
    "sma_short",
    "sma_long",
    "sma_ratio",
    "trend_ratio",
    "ema_short",
    "ema_long",
    "ema_ratio",
    "rsi_raw",
    "rsi_normalized",
    "volatility",
    "atr_14",
    "atr_normalized",
    "range_pct",
    "rolling_return",
    "sentiment_score",
    "sentiment_zscore",
    "sentiment_volume",
    "extreme_fear_flag",
    "extreme_greed_flag",
)

TARGET_COL = "returns_next"  # >0 → long, <=0 → short
DATA_PATH = pathlib.Path("data/cache/features_b0f4663cf8b0b464.parquet")
ARTIFACT_PATH = pathlib.Path("models/nvda_1h_xgb_v2.joblib")
FEATURE_LIST_PATH = pathlib.Path("models/nvda_1h_xgb_v2_features.json")


def load_dataset(path: pathlib.Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if TARGET_COL not in df:
        raise SystemExit(f"Target column {TARGET_COL} missing in {path}")
    df = df.dropna(subset=[TARGET_COL])
    # Label: 1 for up, 0 for down/short
    df["target"] = (df[TARGET_COL] > 0).astype(int)
    feat_cols = [c for c in FEATURES if c in df.columns]
    missing = set(FEATURES) - set(feat_cols)
    if missing:
        raise SystemExit(f"Missing feature columns: {sorted(missing)}")
    df = df[feat_cols + ["target"]].dropna()
    return df, feat_cols


def split_time(df: pd.DataFrame, val_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_sorted = df.copy()
    split = int(len(df_sorted) * (1 - val_frac))
    return df_sorted.iloc[:split], df_sorted.iloc[split:]


def train_model(df_train: pd.DataFrame, df_val: pd.DataFrame, feat_cols: list[str]):
    X_train = df_train[feat_cols].to_numpy()
    y_train = df_train["target"].to_numpy()
    X_val = df_val[feat_cols].to_numpy()
    y_val = df_val["target"].to_numpy()

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    clf = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        random_state=42,
        n_jobs=4,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    clf.fit(X_train_s, y_train, eval_set=[(X_val_s, y_val)], verbose=False)

    calibrator = ProbabilityCalibrator(method="none")
    model = ProbabilityModel(scaler=scaler, classifier=clf, calibrator=calibrator, model_type="xgboost")
    return model, scaler, (X_val_s, y_val)


def save_artifact(model: ProbabilityModel, feat_cols: list[str]) -> None:
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.dump(ARTIFACT_PATH)
    FEATURE_LIST_PATH.write_text(json.dumps(feat_cols, indent=2))


def main() -> None:
    t0 = time.time()
    df, feat_cols = load_dataset(DATA_PATH)
    train_df, val_df = split_time(df, val_frac=0.2)
    model, scaler, val_data = train_model(train_df, val_df, feat_cols)
    save_artifact(model, feat_cols)
    t1 = time.time()

    X_val_s, y_val = val_data
    proba = model.predict_proba(X_val_s)
    preds = (proba >= 0.5).astype(int)
    acc = float((preds == y_val).mean())
    q = np.quantile(proba, [0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1])

    print(f"[TRAIN] done in {t1 - t0:.2f}s | rows={len(df)} train={len(train_df)} val={len(val_df)}")
    print(f"[VAL] acc={acc:.3f} | proba_range=({proba.min():.3f},{proba.max():.3f}) | quantiles={q}")
    print(f"[OUT] artifact={ARTIFACT_PATH} | features={FEATURE_LIST_PATH}")


if __name__ == "__main__":
    main()

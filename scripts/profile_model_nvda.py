#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from joblib import load


def main() -> None:
    # adapt to your cached feature file hash
    features_path = Path("data/cache/features_b0f4663cf8b0b464.parquet")
    model_path = Path("artifacts/xgb_prob_nvda_1h_v1.pkl")

    print(f"[PROFILE] Loading features from {features_path}...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    df = pd.read_parquet(features_path)
    t1 = time.perf_counter()
    print(f"[PROFILE] Features loaded in {t1 - t0:.3f}s, shape={df.shape}", file=sys.stderr, flush=True)

    feature_cols = [
        "sma_ratio",
        "ema_ratio",
        "rsi_normalized",
        "volatility",
        "atr_normalized",
        "range_pct",
        "rolling_return",
        "atr_14",
    ]
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"[PROFILE] Missing columns in features: {missing}", file=sys.stderr, flush=True)
        return
    numeric_df = df[feature_cols]
    print(f"[PROFILE] Selected features shape={numeric_df.shape}", file=sys.stderr, flush=True)
    X = numeric_df.to_numpy(dtype=float, copy=False)

    print(f"[PROFILE] Loading model from {model_path}...", file=sys.stderr, flush=True)
    t2 = time.perf_counter()
    model = load(model_path)
    t3 = time.perf_counter()
    print(f"[PROFILE] Model loaded in {t3 - t2:.3f}s, type={type(model)}", file=sys.stderr, flush=True)
    print(f"[PROFILE] Model repr: {repr(model)[:200]}...", file=sys.stderr, flush=True)

    if hasattr(model, "n_jobs"):
        try:
            model.n_jobs = 1
            print("[PROFILE] Set model.n_jobs = 1", file=sys.stderr, flush=True)
        except Exception as exc:  # pragma: no cover - best effort
            print(f"[PROFILE] Failed to set n_jobs: {exc}", file=sys.stderr, flush=True)

    print("[PROFILE] Calling predict_proba on full matrix (DataFrame)...", file=sys.stderr, flush=True)
    t4 = time.perf_counter()
    proba = model.predict_proba(numeric_df)
    t5 = time.perf_counter()
    print(f"[PROFILE] predict_proba done in {t5 - t4:.3f}s, proba shape={proba.shape}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

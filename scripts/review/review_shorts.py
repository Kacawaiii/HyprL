#!/usr/bin/env python3
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACT = Path("models/nvda_1h_xgb_v2.joblib")
FEATURES = json.loads(Path("models/nvda_1h_xgb_v2_features.json").read_text())
DATA = Path("data/cache/nvda_1h_features_v2.parquet")


def main() -> None:
    df = pd.read_parquet(DATA).dropna()
    X = df[FEATURES].to_numpy()

    payload = joblib.load(ARTIFACT)
    scaler = payload["scaler"]
    model = payload["classifier"]

    Xs = scaler.transform(X)
    proba = model.predict_proba(Xs)[:, 1]  # proba_up

    print("\n=== PROBABILITY DISTRIBUTION ===")
    print("min/max:", proba.min(), proba.max())
    print(
        "quantiles:",
        np.quantile(proba, [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]),
    )

    shorts_count = (proba < 0.45).sum()
    longs_count = (proba > 0.55).sum()

    print(f"\nPotential longs (proba>0.55): {longs_count}")
    print(f"Potential shorts (proba<0.45): {shorts_count}")

    print("\n=== LABEL BALANCE ===")
    if "label_long_short" in df.columns:
        print(df["label_long_short"].value_counts())

    print("\nReview complete.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Generate drift detection baseline from historical data.

Computes RT engine features over the training period and saves as
models/drift_baseline.npz for DriftDetector to load at engine startup.

Usage:
    python scripts/generate_drift_baseline.py
    python scripts/generate_drift_baseline.py --symbols NVDA,MSFT --period 1y
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.rt.features import compute_features


def _compute_rolling_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute RT features for every bar using a rolling window."""
    records = []
    for i in range(60, len(prices)):
        window = prices.iloc[:i + 1]
        feats = compute_features(window)
        if feats:
            records.append(feats)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Generate drift baseline")
    parser.add_argument("--symbols", type=str, default="NVDA,MSFT")
    parser.add_argument("--period", type=str, default="1y")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--out", type=str, default="models/drift_baseline.npz")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    all_features: dict[str, list] = {}

    for symbol in symbols:
        print(f"Loading {symbol} ({args.period}, {args.interval})...")
        try:
            ticker = yf.Ticker(symbol)
            prices = ticker.history(period=args.period, interval=args.interval)
        except Exception as e:
            print(f"  Error loading {symbol}: {e}")
            continue

        if prices is None or len(prices) < 100:
            print(f"  Not enough data for {symbol} ({len(prices) if prices is not None else 0} bars)")
            continue

        # Normalize column names to lowercase
        prices.columns = [c.lower() for c in prices.columns]
        if "adj close" in prices.columns:
            prices = prices.drop(columns=["adj close"], errors="ignore")
        prices = prices[["open", "high", "low", "close", "volume"]]

        print(f"  Computing RT features over {len(prices)} bars...")
        features_df = _compute_rolling_features(prices)

        # Exclude non-feature columns
        skip_cols = {"close", "open", "volume"}
        feature_cols = [c for c in features_df.columns if c not in skip_cols]

        for col in feature_cols:
            vals = features_df[col].dropna().values
            if len(vals) > 50:
                if col not in all_features:
                    all_features[col] = []
                all_features[col].extend(vals.tolist())

        print(f"  {len(feature_cols)} features extracted, {len(features_df)} samples")

    if not all_features:
        print("No features extracted. Check data availability.")
        sys.exit(1)

    baseline = {k: np.array(v) for k, v in all_features.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **baseline)

    print(f"\nBaseline saved: {out_path}")
    print(f"Features: {len(baseline)}")
    for name, arr in sorted(baseline.items()):
        print(f"  {name}: n={len(arr)}, mean={np.mean(arr):.4f}, std={np.std(arr):.4f}")


if __name__ == "__main__":
    main()

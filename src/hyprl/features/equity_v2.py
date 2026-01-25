from __future__ import annotations

import numpy as np
import pandas as pd

from hyprl.features.nvda_v2 import compute_nvda_v2_features


def compute_equity_v2_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Shared equity v2 feature builder (reuses NVDA v2 stack).
    """
    feats = compute_nvda_v2_features(prices)
    # Backward-compat columns expected by strategy FEATURE_COLUMNS
    if "sma_ratio" not in feats.columns:
        feats["sma_ratio"] = feats.get("ret_6h", pd.Series(0.0, index=feats.index))
    if "ema_ratio" not in feats.columns:
        feats["ema_ratio"] = feats.get("ret_3h", pd.Series(0.0, index=feats.index))
    if "rsi_normalized" not in feats.columns and "rsi_14" in feats.columns:
        feats["rsi_normalized"] = feats["rsi_14"]
    if "volatility" not in feats.columns:
        feats["volatility"] = feats["ret_1h"].rolling(window=14, min_periods=14).std()
    if "atr_normalized" not in feats.columns and "atr_14_norm" in feats.columns:
        feats["atr_normalized"] = feats["atr_14_norm"]
    if "range_pct" not in feats.columns and "range_pct" in feats.columns:
        feats["range_pct"] = feats["range_pct"]
    if "rolling_return" not in feats.columns and "ret_24h" in feats.columns:
        feats["rolling_return"] = feats["ret_24h"]
    if "sentiment_score" not in feats.columns:
        feats["sentiment_score"] = 0.0
    feats = feats.replace([np.inf, -np.inf], np.nan).dropna()
    return feats

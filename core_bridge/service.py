"""Prediction bridge handling stub/real implementations."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List

logger = logging.getLogger(__name__)

STUB_RESPONSES: Dict[str, Dict[str, float]] = {
    "AAPL": {"prob_up": 0.512, "tp": 123.45, "sl": 120.12},
    "MSFT": {"prob_up": 0.431, "tp": 234.56, "sl": 230.01},
}


def predict_batch(
    symbols: List[str],
    interval: str,
    features: List[str],
    threshold: float,
    risk_pct: float,
) -> List[Dict[str, float | str]]:
    """Return predictions using the configured implementation."""
    impl = os.environ.get("HYPRL_PREDICT_IMPL", "stub").lower()
    if impl != "real":
        return _predict_stub(symbols, threshold, risk_pct)
    try:
        return _predict_real(symbols, interval, features, threshold, risk_pct)
    except Exception as exc:  # pragma: no cover - resilience path
        logger.warning("[PREDICT] fallback stub due to real bridge failure: %s", exc)
        return _predict_stub(symbols, threshold, risk_pct)


def _predict_stub(
    symbols: Iterable[str], threshold: float, risk_pct: float
) -> List[Dict[str, float | str]]:
    """Deterministic stub aligned with Slice 1 contract."""
    results: List[Dict[str, float | str]] = []
    for symbol in symbols:
        lookup = STUB_RESPONSES.get(symbol.upper(), {"prob_up": 0.5, "tp": 0.0, "sl": 0.0})
        prob_up = lookup["prob_up"]
        direction = "UP" if prob_up >= 0.5 else "DOWN"
        results.append(
            {
                "symbol": symbol.upper(),
                "prob_up": prob_up,
                "direction": direction,
                "threshold": threshold,
                "tp": lookup["tp"],
                "sl": lookup["sl"],
                "risk_pct": risk_pct,
            }
        )
    return results


def _predict_real(
    symbols: Iterable[str],
    interval: str,
    features: Iterable[str],
    threshold: float,
    risk_pct: float,
) -> List[Dict[str, float | str]]:
    """Lightweight real bridge using ProbabilityModel with synthetic features."""
    import numpy as np
    import pandas as pd
    from src.hyprl.model.probability import ProbabilityModel

    model = ProbabilityModel.create(model_type="logistic")
    synthetic_features = pd.DataFrame(
        [[0.0, 0.0, 0.1], [1.0, 1.0, 0.9]],
        columns=["feat_len", "feat_ascii", "feat_bias"],
    )
    synthetic_target = pd.Series([0, 1])
    model.fit(synthetic_features, synthetic_target)

    results: List[Dict[str, float | str]] = []
    for symbol in symbols:
        feature_row = pd.DataFrame(
            [
                [
                    float(len(symbol)),
                    float(sum(ord(ch) for ch in symbol) % 101) / 100.0,
                    min(0.99, max(0.01, risk_pct + threshold / 10.0)),
                ]
            ],
            columns=["feat_len", "feat_ascii", "feat_bias"],
        )
        prob_up = float(model.predict_proba(feature_row)[0])
        direction = "UP" if prob_up >= threshold else "DOWN"
        base_price = 100.0 + len(symbol) * 1.5
        tp = round(base_price * (1.0 + risk_pct), 4)
        sl = round(base_price * (1.0 - risk_pct), 4)
        results.append(
            {
                "symbol": symbol.upper(),
                "prob_up": round(prob_up, 3),
                "direction": direction,
                "threshold": threshold,
                "tp": tp,
                "sl": sl,
                "risk_pct": risk_pct,
            }
        )
    return results

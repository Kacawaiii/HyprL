from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

from hyprl.meta.features import build_feature_frame_from_records, select_meta_features
from hyprl.meta.model import MetaRobustnessModel


def _normalize_scores(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_val = float(numeric.min())
    max_val = float(numeric.max())
    if max_val - min_val <= 1e-12:
        return pd.Series(1.0, index=numeric.index)
    normalized = (numeric - min_val) / (max_val - min_val)
    return normalized.clip(0.0, 1.0)


def _ensure_base_score(df: pd.DataFrame) -> pd.Series:
    if "base_score_normalized" in df.columns:
        return df["base_score_normalized"].clip(0.0, 1.0)
    if "base_score" in df.columns:
        return _normalize_scores(df["base_score"])
    if "final_score" in df.columns:
        return _normalize_scores(df["final_score"])
    if "portfolio_profit_factor" in df.columns:
        return _normalize_scores(df["portfolio_profit_factor"])
    return pd.Series(0.0, index=df.index)


def _prepare_meta_predictions(
    df: pd.DataFrame,
    base_scores: pd.Series,
    model: Optional[MetaRobustnessModel],
    calibrator: Any | None,
) -> pd.Series:
    if model is None:
        if "meta_prediction" in df.columns:
            return pd.to_numeric(df["meta_prediction"], errors="coerce").fillna(base_scores).clip(0.0, 1.0)
        if "meta_pred" in df.columns:
            return pd.to_numeric(df["meta_pred"], errors="coerce").fillna(base_scores).clip(0.0, 1.0)
        return base_scores
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "long_threshold": row.get("long_threshold"),
                "short_threshold": row.get("short_threshold"),
                "risk_pct": row.get("risk_pct"),
                "min_ev_multiple": row.get("min_ev_multiple", row.get("min_ev", 0.0)),
                "trend_filter": row.get("trend_filter"),
                "sentiment_min": row.get("sentiment_min"),
                "sentiment_max": row.get("sentiment_max"),
                "sentiment_regime": row.get("sentiment_regime", row.get("sentiment_regimes", "off")),
                "weighting_scheme": row.get("search_weighting_scheme", row.get("weighting_scheme", "equal")),
                "pf_backtest": row.get("pf_backtest", row.get("portfolio_profit_factor")),
                "sharpe_backtest": row.get("sharpe_backtest", row.get("portfolio_sharpe")),
                "maxdd_backtest": row.get("maxdd_backtest", row.get("portfolio_max_drawdown_pct")),
                "winrate_backtest": row.get("winrate_backtest", (row.get("win_rate_pct", 0.0) or 0.0) / 100.0),
                "equity_vol_backtest": row.get("equity_vol_backtest", row.get("portfolio_equity_vol", 0.0)),
                "trades_backtest": row.get("trades_backtest", row.get("n_trades", 0)),
                "correlation_mean": row.get("correlation_mean", 0.0),
                "correlation_max": row.get("correlation_max", 0.0),
            }
        )
    feature_df = build_feature_frame_from_records(records)
    features, _ = select_meta_features(feature_df)
    # Some MetaRobustnessModel subclasses (in tests) override predict(X) without
    # accepting a 'calibrator' kwarg. Be robust by attempting the new signature
    # and falling back to predict(X) if the kwarg is not supported.
    try:
        preds = model.predict(features, calibrator=calibrator)
    except TypeError:
        preds = model.predict(features)
    return pd.Series(preds, index=df.index)


def build_meta_diag_frame(
    supersearch_df: pd.DataFrame,
    meta_weight: float = 0.4,
    model: Optional[MetaRobustnessModel] = None,
    calibrator: Any | None = None,
) -> pd.DataFrame:
    """
    Build a diagnostics dataframe combining base scores, meta predictions, and filters.
    """

    if supersearch_df.empty:
        return supersearch_df.copy()
    diag = supersearch_df.copy()
    weighting_col = diag.get("search_weighting_scheme")
    if weighting_col is None:
        weighting_col = diag.get("weighting_scheme")
    if weighting_col is None:
        diag["weighting_scheme"] = "equal"
    else:
        diag["weighting_scheme"] = (
            pd.Series(weighting_col)
            .fillna("equal")
            .astype(str)
            .str.lower()
        )
    base_scores = _ensure_base_score(diag)
    diag["base_score_normalized"] = base_scores
    meta_preds = _prepare_meta_predictions(diag, base_scores, model, calibrator)
    diag["meta_pred"] = meta_preds.clip(0.0, 1.0)
    weight = float(np.clip(meta_weight, 0.0, 1.0))
    diag["final_score"] = (1.0 - weight) * diag["base_score_normalized"] + weight * diag["meta_pred"]
    diag["score_delta"] = diag["final_score"] - diag["base_score_normalized"]
    diag["delta_flag"] = np.where(diag["score_delta"].abs() >= 0.2, "⚠️", "")
    diag["trades_backtest"] = diag.get("n_trades", diag.get("trades_backtest", 0))
    diag["portfolio_pf"] = diag.get("pf_backtest", diag.get("portfolio_profit_factor"))
    diag["portfolio_sharpe"] = diag.get("sharpe_backtest", diag.get("portfolio_sharpe"))
    diag["portfolio_dd"] = diag.get("maxdd_backtest", diag.get("portfolio_max_drawdown_pct"))
    diag["corr_mean"] = diag.get("correlation_mean", 0.0)
    diag["corr_max"] = diag.get("correlation_max", 0.0)
    diag["timestamp_evaluated"] = datetime.now(timezone.utc).isoformat()
    return diag

from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd


NUMERIC_FEATURES: list[str] = [
    "long_threshold",
    "short_threshold",
    "risk_pct",
    "min_ev_multiple",
    "sentiment_min",
    "sentiment_max",
    "pf_backtest",
    "sharpe_backtest",
    "maxdd_backtest",
    "winrate_backtest",
    "equity_vol_backtest",
    "trades_backtest",
    "correlation_mean",
    "correlation_max",
]

CATEGORICAL_FEATURES: list[str] = [
    "trend_filter",
    "sentiment_regime",
    "weighting_scheme",
]

DEFAULTS: dict[str, object] = {
    "long_threshold": 0.55,
    "short_threshold": 0.45,
    "risk_pct": 0.015,
    "min_ev_multiple": 0.0,
    "sentiment_min": -0.4,
    "sentiment_max": 0.4,
    "pf_backtest": 1.0,
    "sharpe_backtest": 0.5,
    "maxdd_backtest": 25.0,
    "winrate_backtest": 0.5,
    "equity_vol_backtest": 0.1,
    "trades_backtest": 40.0,
    "correlation_mean": 0.0,
    "correlation_max": 0.0,
    "trend_filter": "false",
    "sentiment_regime": "off",
    "weighting_scheme": "equal",
}


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    for column in columns:
        if column not in df.columns:
            df[column] = DEFAULTS.get(column, 0.0)
    return df


def _normalize_categorical(series: pd.Series, column: str) -> pd.Series:
    base = series.fillna(DEFAULTS.get(column, "unknown"))
    return base.astype(str).str.lower()


def select_meta_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, list[str]]:
    """
    Prepare the feature matrix for meta-robustness modeling.

    Returns
    -------
    X : pd.DataFrame
        Feature dataframe with numeric + categorical columns ready for preprocessing.
    feature_names : list[str]
        Ordered list of feature columns for model persistence.
    """

    work = df.copy()
    work = _ensure_columns(work, NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    numeric = work[NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    categorical = pd.DataFrame(index=work.index)
    for column in CATEGORICAL_FEATURES:
        categorical[column] = _normalize_categorical(work[column], column)
    features = pd.concat([numeric, categorical], axis=1)
    return features, list(features.columns)


def build_feature_frame_from_records(records: list[dict[str, object]]) -> pd.DataFrame:
    """
    Build a feature frame from generic mapping records (e.g., SearchResult or CSV rows).
    """

    rows: list[dict[str, object]] = []
    for record in records:
        rows.append(
            {
                "long_threshold": float(record.get("long_threshold", DEFAULTS["long_threshold"])),
                "short_threshold": float(record.get("short_threshold", DEFAULTS["short_threshold"])),
                "risk_pct": float(record.get("risk_pct", DEFAULTS["risk_pct"])),
                "min_ev_multiple": float(record.get("min_ev_multiple", DEFAULTS["min_ev_multiple"])),
                "trend_filter": str(record.get("trend_filter", DEFAULTS["trend_filter"])).lower(),
                "sentiment_min": float(record.get("sentiment_min", DEFAULTS["sentiment_min"])),
                "sentiment_max": float(record.get("sentiment_max", DEFAULTS["sentiment_max"])),
                "sentiment_regime": str(record.get("sentiment_regime", DEFAULTS["sentiment_regime"])).lower(),
                "weighting_scheme": str(
                    record.get("weighting_scheme", record.get("search_weighting_scheme", DEFAULTS["weighting_scheme"]))
                ).lower(),
                "pf_backtest": float(record.get("pf_backtest", record.get("portfolio_profit_factor", DEFAULTS["pf_backtest"]))),
                "sharpe_backtest": float(record.get("sharpe_backtest", record.get("portfolio_sharpe", DEFAULTS["sharpe_backtest"]))),
                "maxdd_backtest": float(
                    record.get("maxdd_backtest", record.get("portfolio_max_drawdown_pct", DEFAULTS["maxdd_backtest"]))
                ),
                "winrate_backtest": float(
                    record.get("winrate_backtest", record.get("win_rate_pct", DEFAULTS["winrate_backtest"])) or 0.0
                )
                / (100.0 if "win_rate_pct" in record else 1.0),
                "equity_vol_backtest": float(
                    record.get("equity_vol_backtest", record.get("portfolio_equity_vol", DEFAULTS["equity_vol_backtest"]))
                ),
                "trades_backtest": float(record.get("trades_backtest", record.get("n_trades", DEFAULTS["trades_backtest"]))),
                "correlation_mean": float(record.get("correlation_mean", DEFAULTS["correlation_mean"])),
                "correlation_max": float(record.get("correlation_max", DEFAULTS["correlation_max"])),
            }
        )
    return pd.DataFrame(rows)

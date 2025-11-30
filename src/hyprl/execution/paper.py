from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from hyprl.backtest.runner import BacktestConfig
from hyprl.risk.manager import RiskConfig


@dataclass(slots=True)
class PaperBuildParams:
    """Paramètres invariants pour reconstruire un BacktestConfig depuis un CSV Supersearch."""

    period: str | None
    start: str | None
    end: str | None
    interval: str
    model_type: str
    calibration: str
    default_long_threshold: float
    default_short_threshold: float


def _coerce_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def build_backtest_config_from_row(
    row: pd.Series | Mapping[str, object],
    ticker: str,
    capital_share: float,
    params: PaperBuildParams,
) -> BacktestConfig:
    """Construit un BacktestConfig réutilisable par le moteur de backtest/paper trading."""

    payload = row if isinstance(row, Mapping) else row.to_dict()
    risk_pct = float(payload.get("risk_pct", 0.01))
    atr_multiplier = float(payload.get("atr_multiplier", 1.0))
    reward_multiple = float(payload.get("reward_multiple", 2.0))
    min_position_size = int(payload.get("min_position_size", 1))
    risk = RiskConfig(
        balance=capital_share,
        risk_pct=risk_pct,
        atr_multiplier=atr_multiplier,
        reward_multiple=reward_multiple,
        min_position_size=min_position_size,
    )
    trend_flag = _coerce_bool(payload.get("trend_filter", False))
    sentiment_min = float(payload.get("sentiment_min", payload.get("sentiment_min_value", -1.0)))
    sentiment_max = float(payload.get("sentiment_max", payload.get("sentiment_max_value", 1.0)))
    sentiment_regime = str(payload.get("sentiment_regime", "off"))
    commission_pct = float(payload.get("commission_pct", 0.0005))
    slippage_pct = float(payload.get("slippage_pct", 0.0005))
    cfg = BacktestConfig(
        ticker=ticker,
        period=params.period,
        start=params.start,
        end=params.end,
        interval=params.interval,
        initial_balance=capital_share,
        long_threshold=float(payload.get("long_threshold", params.default_long_threshold)),
        short_threshold=float(payload.get("short_threshold", params.default_short_threshold)),
        risk=risk,
        model_type=params.model_type,
        calibration=params.calibration,
        min_ev_multiple=float(payload.get("min_ev_multiple", 0.0)),
        enable_trend_filter=trend_flag,
        sentiment_min=sentiment_min,
        sentiment_max=sentiment_max,
        sentiment_regime=sentiment_regime,
        commission_pct=commission_pct,
        slippage_pct=slippage_pct,
    )
    return cfg

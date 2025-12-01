"""Shared strategy decision helpers for backtest and live engines."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
import pandas as pd

from hyprl.adaptive.engine import AdaptiveConfig, AdaptiveState
from hyprl.features.sentiment import enrich_sentiment_features
from hyprl.indicators.technical import compute_feature_frame
from hyprl.features.equity_v2 import compute_equity_v2_features
from hyprl.risk.sizing import clamp_position_size
from hyprl.labels.amplitude import (
    TRAINABLE_LABELS,
    LabelConfig,
    attach_amplitude_labels,
    encode_amplitude_target,
    validate_label_support,
)
from hyprl.model.probability import ProbabilityModel
from hyprl.risk.manager import RiskConfig, RiskOutcome, plan_trade

if TYPE_CHECKING:  # pragma: no cover
    from hyprl.backtest.runner import BacktestConfig


FEATURE_COLUMNS = [
    "sma_ratio",
    "ema_ratio",
    "rsi_normalized",
    "volatility",
    "atr_normalized",
    "range_pct",
    "rolling_return",
    "sentiment_score",
]

_ARTIFACT_MODELS: dict[str, ProbabilityModel] = {}


def _load_artifact_model(path: str) -> ProbabilityModel:
    cached = _ARTIFACT_MODELS.get(path)
    if cached is not None:
        return cached
    model = ProbabilityModel.load_artifact(path)
    _ARTIFACT_MODELS[path] = model
    return model


def _feature_signature(feature_row: pd.Series) -> str:
    parts: list[str] = []
    for col in FEATURE_COLUMNS:
        value = feature_row.get(col)
        try:
            formatted = f"{float(value):.6f}" if value is not None else "nan"
        except (TypeError, ValueError):
            formatted = "nan"
        parts.append(f"{col}={formatted}")
    return "|".join(parts)


@dataclass(slots=True)
class StrategyDecision:
    """Represents a trade decision produced for a given bar."""

    timestamp: pd.Timestamp
    direction: str
    probability_up: float
    threshold: float
    entry_price: float
    risk_plan: RiskOutcome
    expected_pnl: float
    long_threshold: float
    short_threshold: float
    regime_name: Optional[str]
    profile_name: Optional[str]
    fear_flag: bool
    greed_flag: bool
    multi_probabilities: Optional[dict[str, float]] = None


def initial_regime_name(adaptive_cfg: AdaptiveConfig, fallback: str | None) -> str:
    regime = adaptive_cfg.get_regime(
        adaptive_cfg.default_regime if adaptive_cfg.default_regime else None
    )
    if regime is not None:
        return regime.name
    return fallback or "normal"


def prepare_design_and_target(
    feature_slice: pd.DataFrame,
    label_cfg: LabelConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = FEATURE_COLUMNS
    design = feature_slice[feature_cols]
    finite_mask = np.isfinite(design).all(axis=1)
    design = design.loc[finite_mask]
    if label_cfg.mode == "amplitude":
        if "label_amplitude" not in feature_slice.columns:
            raise RuntimeError("Amplitude labels missing from feature slice.")
        target = feature_slice.loc[design.index, "label_amplitude"].copy()
        if label_cfg.neutral_strategy == "ignore":
            mask = target.isin(TRAINABLE_LABELS)
            design = design.loc[mask]
            target = target.loc[mask]
        target = encode_amplitude_target(target)
    else:
        returns_forward = feature_slice.loc[design.index, "returns_next"]
        target = (returns_forward > 0).astype(int)
    valid_mask = target.notna()
    design = design.loc[valid_mask]
    target = target.loc[valid_mask].astype(int)
    return design, target


def _clamp_probability(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def effective_thresholds(
    base_long: float,
    base_short: float,
    regime,
) -> tuple[float, float]:
    if regime is None:
        long_value = base_long
        short_value = base_short
    else:
        overrides = regime.threshold_overrides or {}
        if "long" in overrides:
            long_value = float(overrides["long"])
        else:
            long_value = base_long + float(overrides.get("long_shift", 0.0))
        if "short" in overrides:
            short_value = float(overrides["short"])
        else:
            short_value = base_short + float(overrides.get("short_shift", 0.0))
    long_value = _clamp_probability(long_value)
    short_value = _clamp_probability(short_value)
    if short_value > long_value:
        short_value = long_value
    return long_value, short_value


def _effective_model(config: "BacktestConfig", regime) -> tuple[str, str]:
    model_type = config.model_type
    calibration = config.calibration
    if regime is not None and regime.model_overrides:
        overrides = regime.model_overrides
        model_type = overrides.get("model_type", model_type)
        calibration = overrides.get("calibration", calibration)
    return model_type, calibration


def _trend_permits_trade(
    feature_row: pd.Series,
    direction: str,
    config: "BacktestConfig",
) -> bool:
    if not config.enable_trend_filter:
        return True
    rolling_return = float(feature_row.get("rolling_return", 0.0))
    if direction == "long":
        return rolling_return >= config.trend_long_min
    threshold = config.trend_short_min
    if threshold <= 0.0:
        return rolling_return <= 0.0
    return rolling_return <= -threshold


def _sentiment_permits_trade(feature_row: pd.Series, config: "BacktestConfig") -> bool:
    score = float(feature_row.get("sentiment_score", 0.0))
    if score < config.sentiment_min or score > config.sentiment_max:
        return False
    regime = (config.sentiment_regime or "off").lower()
    fear_active = bool(int(feature_row.get("extreme_fear_flag", 0)))
    greed_active = bool(int(feature_row.get("extreme_greed_flag", 0)))
    if regime == "off":
        return True
    if regime == "fear_only":
        return fear_active
    if regime == "greed_only":
        return greed_active
    if regime == "neutral_only":
        return not fear_active and not greed_active
    return True


def _build_risk_config(
    base: RiskConfig,
    profiles: dict[str, dict[str, float]],
    profile_name: str,
    balance: float,
    overrides: Optional[dict[str, float]] = None,
) -> RiskConfig:
    params = {
        "risk_pct": base.risk_pct,
        "atr_multiplier": base.atr_multiplier,
        "reward_multiple": base.reward_multiple,
        "min_position_size": base.min_position_size,
        "max_leverage": base.max_leverage,
        "trailing_stop_activation": base.trailing_stop_activation,
        "trailing_stop_distance": base.trailing_stop_distance,
    }
    if profile_name in profiles:
        for key, value in profiles[profile_name].items():
            params[key] = value
    if overrides:
        for key, value in overrides.items():
            params[key] = value

    return RiskConfig(
        balance=balance,
        risk_pct=float(params["risk_pct"]),
        atr_multiplier=float(params["atr_multiplier"]),
        reward_multiple=float(params["reward_multiple"]),
        min_position_size=float(params["min_position_size"]),
        max_leverage=float(params.get("max_leverage", 5.0)),
        trailing_stop_activation=params.get("trailing_stop_activation"),
        trailing_stop_distance=params.get("trailing_stop_distance"),
    )


def expected_trade_pnl(risk: RiskOutcome, probability_up: float) -> float:
    prob = float(probability_up)
    if not math.isfinite(prob):
        return 0.0
    prob = min(max(prob, 0.0), 1.0)
    p_win = prob if risk.direction == "long" else 1.0 - prob
    p_win = min(max(p_win, 0.0), 1.0)
    loss_amount = -risk.risk_amount
    win_amount = risk.risk_amount * risk.rr_multiple
    return p_win * win_amount + (1.0 - p_win) * loss_amount


def prepare_feature_frame(prices: pd.DataFrame, config: "BacktestConfig") -> pd.DataFrame:
    """Utility used by live engine to mirror backtest feature construction."""

    preset = (config.feature_preset or "").lower()
    if preset in {
        "nvda_v2",
        "equity_v2",
        "meta_v2",
        "msft_v2",
        "amd_v2",
        "qqq_v2",
        "spy_v2",
    }:
        features = compute_equity_v2_features(prices)
    else:
        features = compute_feature_frame(
            prices,
            sma_short_window=config.sma_short_window,
            sma_long_window=config.sma_long_window,
            rsi_window=config.rsi_window,
            atr_window=config.atr_window,
        )
    if features.empty:
        return features
    if "sentiment_score" not in features.columns:
        features["sentiment_score"] = 0.0
    features = enrich_sentiment_features(features)
    features = features.replace([np.inf, -np.inf], np.nan)
    required_cols = [
        "sma_ratio",
        "ema_ratio",
        "rsi_normalized",
        "volatility",
        "atr_normalized",
        "range_pct",
        "rolling_return",
        "sentiment_score",
    ]
    atr_col = f"atr_{config.atr_window}"
    if atr_col not in required_cols:
        required_cols.append(atr_col)
    existing = [col for col in required_cols if col in features.columns]
    if existing:
        features = features.dropna(subset=existing)
    features = attach_amplitude_labels(features, prices, config.label)
    validate_label_support(features, config.label)
    return features


def _normalize_interval(interval: str) -> str:
    """Normalize pandas offset aliases for minute granularity."""
    if interval.endswith("m") and not interval.endswith("min"):
        return f"{interval[:-1]}min"
    return interval


def _resample_prices(prices: pd.DataFrame, interval: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    normalized_interval = _normalize_interval(interval)
    resampled = prices.resample(normalized_interval).agg(agg)
    resampled = resampled.dropna(subset=["open", "high", "low", "close"])
    return resampled


def build_multiframe_feature_set(
    prices: pd.DataFrame,
    config: "BacktestConfig",
    intervals: list[str],
) -> dict[str, pd.DataFrame]:
    """
    Compute feature frames for multiple timeframes and align them to the base index.

    Returns a mapping keyed by timeframe string; "base" always present.
    """

    base_features = prepare_feature_frame(prices, config)
    result: dict[str, pd.DataFrame] = {"base": base_features}
    if not intervals:
        return result
    base_index = base_features.index
    for interval in intervals:
        try:
            tf_prices = _resample_prices(prices, interval)
        except Exception:
            continue
        if tf_prices.empty:
            continue
        try:
            tf_features = prepare_feature_frame(tf_prices, config)
        except Exception:
            continue
        if tf_features.empty:
            continue
        aligned = tf_features.reindex(base_index, method="ffill")
        result[interval] = aligned
    return result


def decide_signals_on_bar(
    *,
    config: "BacktestConfig",
    features: pd.DataFrame,
    prices: pd.DataFrame,
    current_idx: int,
    equity: float,
    adaptive_cfg: AdaptiveConfig,
    adaptive_state: AdaptiveState,
    trace_log: Callable[[dict[str, object]], None] | None = None,
    multiframe_features: dict[str, pd.DataFrame] | None = None,
    fusion_method: str = "mean",
    fusion_weights: dict[str, float] | None = None,
    risk_pct_override: float | None = None,
) -> StrategyDecision | None:
    trace_state: dict[str, object] = {
        "timestamp": None,
        "probability_up": None,
        "long_threshold": None,
        "short_threshold": None,
        "direction": None,
        "expected_pnl": None,
        "min_ev": None,
        "trend_ok": None,
        "sentiment_ok": None,
        "atr_value": None,
        "position_size": None,
        "risk_amount": None,
        "equity": float(equity),
        "feature_idx": None,
        "design_rows": None,
        "feature_signature": None,
    }

    def emit(reason: str, *, decision_state: str = "reject", **updates: object) -> None:
        if trace_log is None:
            return
        payload = trace_state.copy()
        payload.update(updates)
        payload["reason"] = reason
        payload["decision"] = decision_state
        trace_log(**payload)

    if current_idx < 0 or current_idx >= len(features):
        emit("index_out_of_range")
        return None

    timestamp = features.index[current_idx]
    trace_state["timestamp"] = timestamp

    if current_idx <= 0:
        emit("insufficient_history_index")
        return None

    trace_state["feature_idx"] = current_idx
    train_slice = features.iloc[:current_idx]
    design, target = prepare_design_and_target(train_slice, config.label)
    trace_state["design_rows"] = int(len(design))
    if design.empty or target.nunique() < 2:
        emit("insufficient_training_data")
        return None

    active_regime = adaptive_cfg.get_regime(adaptive_state.regime_name) if adaptive_cfg.enable else None
    model_type, calibration = _effective_model(config, active_regime)
    artifact_path = config.model_artifact_path or os.environ.get("HYPRL_MODEL_ARTIFACT")
    model_artifact_used = False
    if artifact_path:
        artifact_path = str(artifact_path)
        model = _load_artifact_model(artifact_path)
        model_artifact_used = True
    else:
        model = ProbabilityModel.create(
            model_type=model_type,
            calibration=calibration,
            random_state=config.random_state,
        )
        model.fit(design, target)
    inference_design = features.iloc[[current_idx]][design.columns]
    base_probability = float(model.predict_proba(inference_design)[0])
    probabilities: dict[str, float] = {"base": base_probability}
    if multiframe_features:
        fusion_weights = fusion_weights or {}
        for tf_key, tf_frame in multiframe_features.items():
            if tf_key == "base":
                continue
            if tf_frame.empty:
                continue
            try:
                tf_idx = int(tf_frame.index.get_loc(timestamp, method="ffill"))
            except Exception:
                continue
            if tf_idx <= 0 or tf_idx >= len(tf_frame):
                continue
            try:
                tf_row = tf_frame.iloc[[tf_idx]][design.columns]
            except KeyError:
                continue
            try:
                tf_prob = float(model.predict_proba(tf_row)[0])
            except Exception:
                continue
            probabilities[tf_key] = tf_prob

    fused_probability = base_probability
    if probabilities:
        if fusion_method == "max":
            fused_probability = max(probabilities.values())
        elif fusion_method == "min":
            fused_probability = min(probabilities.values())
        else:
            weights = []
            probs = []
            for key, value in probabilities.items():
                weight = fusion_weights.get(key, 1.0) if fusion_weights else 1.0
                weights.append(weight)
                probs.append(value)
            try:
                fused_probability = float(np.average(probs, weights=weights))
            except Exception:
                fused_probability = float(np.mean(probs))

    trace_state["model_artifact"] = artifact_path if model_artifact_used else None
    trace_state["probability_up_base"] = base_probability
    trace_state["probability_up"] = fused_probability
    trace_state["multi_probabilities"] = probabilities if len(probabilities) > 1 else None
    long_threshold, short_threshold = effective_thresholds(
        config.long_threshold,
        config.short_threshold,
        active_regime if adaptive_cfg.enable else None,
    )
    trace_state["long_threshold"] = long_threshold
    trace_state["short_threshold"] = short_threshold

    feature_row = features.iloc[current_idx]
    trace_state["feature_signature"] = _feature_signature(feature_row)
    probability_up = fused_probability
    if probability_up >= long_threshold:
        direction = "long"
        decision_threshold = long_threshold
    elif probability_up <= short_threshold:
        direction = "short"
        decision_threshold = short_threshold
    else:
        trace_state["direction"] = None
        emit("threshold_not_met")
        return None

    trace_state["direction"] = direction
    if not _trend_permits_trade(feature_row, direction, config):
        trace_state["trend_ok"] = False
        emit("trend_filter_fail", trend_ok=False)
        return None
    trace_state["trend_ok"] = True
    if not _sentiment_permits_trade(feature_row, config):
        trace_state["sentiment_ok"] = False
        emit("sentiment_filter_fail", sentiment_ok=False)
        return None
    trace_state["sentiment_ok"] = True

    atr_column = f"atr_{config.atr_window}"
    atr_value = float(feature_row[atr_column])
    if atr_value <= 0 or not math.isfinite(atr_value):
        trace_state["atr_value"] = atr_value
        emit("invalid_atr", atr_value=atr_value)
        return None
    trace_state["atr_value"] = atr_value

    entry_price = float(prices.loc[timestamp, "close"])
    profile_candidate = (
        adaptive_state.regime_name if adaptive_state.regime_name in config.risk_profiles else config.risk_profile
    )
    risk_overrides = active_regime.risk_overrides if active_regime and adaptive_cfg.enable else None
    overrides = dict(risk_overrides) if risk_overrides else {}
    if risk_pct_override is not None:
        overrides["risk_pct"] = float(risk_pct_override)
    overrides = overrides or None
    risk_cfg = _build_risk_config(
        config.risk,
        config.risk_profiles,
        profile_candidate or config.risk_profile or "",
        balance=equity,
        overrides=overrides,
    )
    risk_plan = plan_trade(entry_price=entry_price, atr=atr_value, direction=direction, config=risk_cfg)
    per_unit_risk = abs(entry_price - risk_plan.stop_price)

    clamped_size, clamped_risk = clamp_position_size(
        entry_price=entry_price,
        stop_price=risk_plan.stop_price,
        position_size=risk_plan.position_size,
        equity=equity,
        max_notional=config.max_notional_per_trade,
        max_notional_pct=config.max_position_notional_pct,
    )
    risk_plan.position_size = clamped_size
    risk_plan.risk_amount = clamped_risk
    trace_state["position_size"] = risk_plan.position_size
    trace_state["risk_amount"] = risk_plan.risk_amount
    if risk_plan.position_size <= 0 or risk_plan.risk_amount <= 0:
        emit("position_size_zero", position_size=risk_plan.position_size)
        return None

    expected_pnl = expected_trade_pnl(risk_plan, probability_up)
    min_ev = float(risk_plan.risk_amount) * max(config.min_ev_multiple, 0.0)
    trace_state["expected_pnl"] = expected_pnl
    trace_state["min_ev"] = min_ev
    if expected_pnl <= max(0.0, min_ev):
        emit("expected_value_below_min")
        return None

    fear_flag = bool(int(feature_row.get("extreme_fear_flag", 0)) == 1)
    greed_flag = bool(int(feature_row.get("extreme_greed_flag", 0)) == 1)

    emit("signal_emitted", decision_state="emit")

    return StrategyDecision(
        timestamp=timestamp,
        direction=direction,
        probability_up=probability_up,
        threshold=decision_threshold,
        entry_price=entry_price,
        risk_plan=risk_plan,
        expected_pnl=float(expected_pnl),
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        regime_name=adaptive_state.regime_name,
        profile_name=profile_candidate,
        fear_flag=fear_flag,
        greed_flag=greed_flag,
    )

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Optional

import math
import numpy as np
import pandas as pd

from hyprl.adaptive.engine import AdaptiveConfig, AdaptiveState, evaluate_window, update_state
from hyprl.data.market import MarketDataFetcher
from hyprl.indicators.technical import compute_feature_frame
from hyprl.labels.amplitude import (
    LabelConfig,
    TRAINABLE_LABELS,
    attach_amplitude_labels,
    encode_amplitude_target,
    validate_label_support,
)
from hyprl.model.probability import ProbabilityModel
from hyprl.metrics.calibration import trade_calibration_metrics
from hyprl.risk.gates import cvar_from_pnl
from hyprl.risk.manager import RiskConfig, RiskOutcome, plan_trade
from hyprl.features.sentiment import enrich_sentiment_features

try:
    from hyprl_accel import simulate_trade_path as _simulate_trade_path_accel
except Exception:  # pragma: no cover - optional accelerator
    _simulate_trade_path_accel = None


@dataclass(slots=True)
class BacktestConfig:
    """Configuration for backtest execution.
    
    Cost Model:
    -----------
    - commission_pct: Percentage of notional value charged as commission per trade
                      (e.g., 0.0005 = 0.05% = 5 basis points)
    - slippage_pct: Percentage of notional value approximating execution slippage
                    (e.g., 0.0005 = 0.05% = 5 basis points)
    
    Both costs are applied to each trade side (entry and exit), so the total
    round-trip cost is approximately 2 * (commission_pct + slippage_pct).
    
    For example, with defaults:
    - commission_pct = 0.0005 (0.05%)
    - slippage_pct = 0.0005 (0.05%)
    - Total cost per side = 0.001 (0.10%)
    - Total round-trip cost ≈ 0.002 (0.20%)
    
    The cost model is applied in run_backtest() when computing net_pnl:
        notional = abs(position_size) * entry_price
        total_rate = commission_pct + slippage_pct
        cost = 2.0 * total_rate * notional  # Both entry and exit
        net_pnl = gross_pnl - cost
    """
    ticker: str
    period: Optional[str] = "6mo"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1h"
    initial_balance: float = 10_000.0
    long_threshold: float = 0.6
    short_threshold: float = 0.4
    sma_short_window: int = 5
    sma_long_window: int = 36
    rsi_window: int = 14
    atr_window: int = 14
    risk: RiskConfig = field(default_factory=RiskConfig)
    random_state: int = 42
    commission_pct: float = 0.0005  # 0.05% commission per trade side
    slippage_pct: float = 0.0005    # 0.05% slippage approximation per side
    model_type: str = "logistic"
    calibration: str = "none"
    risk_profile: str = "normal"
    risk_profiles: dict[str, dict[str, float]] = field(default_factory=dict)
    adaptive: AdaptiveConfig = field(default_factory=AdaptiveConfig)
    min_ev_multiple: float = 0.0
    enable_trend_filter: bool = False
    trend_long_min: float = 0.0
    trend_short_min: float = 0.0
    sentiment_min: float = -1.0
    sentiment_max: float = 1.0
    sentiment_regime: str = "off"
    label: LabelConfig = field(default_factory=LabelConfig)


@dataclass(slots=True)
class TradeRecord:
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    direction: str
    probability_up: float
    threshold: float
    entry_price: float
    exit_price: float
    position_size: int
    pnl: float
    return_pct: float
    equity_after: float
    risk_amount: float
    expected_pnl: float
    risk_profile: Optional[str] = None
    effective_long_threshold: Optional[float] = None
    effective_short_threshold: Optional[float] = None
    regime_name: Optional[str] = None


@dataclass(slots=True)
class BacktestResult:
    final_balance: float
    equity_curve: list[float]
    n_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    trades: list[TradeRecord]
    benchmark_final_balance: float
    benchmark_return: float
    annualized_return: Optional[float]
    annualized_benchmark_return: Optional[float]
    annualized_volatility: Optional[float]
    sortino_ratio: Optional[float]
    profit_factor: Optional[float]
    expectancy: float
    avg_r_multiple: float
    avg_expected_pnl: float
    long_trades: int
    short_trades: int
    long_win_rate: float
    short_win_rate: float
    long_total_pnl: float
    short_total_pnl: float
    brier_score: Optional[float]
    log_loss: Optional[float]
    final_risk_profile: str
    final_long_threshold: float
    final_short_threshold: float
    adaptive_profile_changes: int
    regime_usage: dict[str, int]
    regime_transitions: list[dict[str, int | str]]
    sentiment_stats: dict[str, int] = field(default_factory=dict)
    trade_returns: list[float] = field(default_factory=list)


@dataclass(slots=True)
class SupercalcRow:
    price_index: int
    timestamp: pd.Timestamp
    probability_up: float
    rolling_return: float
    atr_value: float
    sentiment_score: float
    extreme_fear_flag: int
    extreme_greed_flag: int


@dataclass(slots=True)
class SupercalcDataset:
    rows: list[SupercalcRow]
    prices: pd.DataFrame
    benchmark_return_pct: float
    initial_balance: float


@dataclass(slots=True)
class StrategyStats:
    final_balance: float
    profit_factor: float | None
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float
    expectancy: float
    n_trades: int
    win_rate: float
    trades_in_fear: int
    trades_in_greed: int
    trade_returns: list[float] = field(default_factory=list)
    equity_history: list[tuple[pd.Timestamp, float]] = field(default_factory=list)


@dataclass(slots=True)
class ThresholdSummary:
    threshold: float
    final_balance: float
    total_return: float
    n_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: Optional[float]
    benchmark_return: float
    alpha_return: float
    annualized_return: Optional[float]
    profit_factor: Optional[float]
    expectancy: float
    final_regime: Optional[str] = None
    adaptive_changes: int = 0
    regime_usage: dict[str, int] = field(default_factory=dict)


def _prepare_design_and_target(
    feature_slice: pd.DataFrame,
    label_cfg: LabelConfig,
) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [
        "sma_ratio",
        "ema_ratio",
        "rsi_normalized",
        "volatility",
        "atr_normalized",
        "range_pct",
        "rolling_return",
        "sentiment_score",
    ]
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


def _effective_thresholds(
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


def _effective_model(config: BacktestConfig, regime) -> tuple[str, str]:
    model_type = config.model_type
    calibration = config.calibration
    if regime is not None and regime.model_overrides:
        overrides = regime.model_overrides
        model_type = overrides.get("model_type", model_type)
        calibration = overrides.get("calibration", calibration)
    return model_type, calibration


def _initial_regime_name(adaptive_cfg: AdaptiveConfig, fallback: str | None) -> str:
    regime = adaptive_cfg.get_regime(adaptive_cfg.default_regime if adaptive_cfg.default_regime else None)
    if regime is not None:
        return regime.name
    return fallback or "normal"


def _trend_permits_trade(
    feature_row: pd.Series,
    direction: str,
    config: BacktestConfig,
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


def _sentiment_permits_trade(feature_row: pd.Series, config: BacktestConfig) -> bool:
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
        min_position_size=int(params["min_position_size"]),
    )


def _compute_trade_pnl(risk: RiskOutcome, entry_price: float, exit_price: float) -> float:
    delta = exit_price - entry_price if risk.direction == "long" else entry_price - exit_price
    return delta * risk.position_size


def _expected_trade_pnl(risk: RiskOutcome, probability_up: float) -> float:
    prob = float(probability_up)
    if not math.isfinite(prob):
        return 0.0
    prob = min(max(prob, 0.0), 1.0)
    p_win = prob if risk.direction == "long" else 1.0 - prob
    p_win = min(max(p_win, 0.0), 1.0)
    loss_amount = -risk.risk_amount
    win_amount = risk.risk_amount * risk.rr_multiple
    return p_win * win_amount + (1.0 - p_win) * loss_amount


def _simulate_trade_python(prices: pd.DataFrame, start_pos: int, risk: RiskOutcome) -> tuple[float, int]:
    price_values = prices[["high", "low", "close"]]
    for idx in range(start_pos + 1, len(price_values)):
        high = float(price_values.iloc[idx]["high"])
        low = float(price_values.iloc[idx]["low"])
        if risk.direction == "long":
            hit_stop = low <= risk.stop_price
            hit_take = high >= risk.take_profit_price
        else:
            hit_stop = high >= risk.stop_price
            hit_take = low <= risk.take_profit_price
        if hit_stop or hit_take:
            exit_price = risk.stop_price if hit_stop else risk.take_profit_price
            return exit_price, idx
    last_close = float(price_values.iloc[-1]["close"])
    return last_close, len(price_values) - 1


def _locate_exit_index(price_values: pd.DataFrame, start_pos: int, risk: RiskOutcome, exit_price: float) -> int:
    if math.isclose(exit_price, risk.stop_price, rel_tol=1e-9, abs_tol=1e-9):
        target = "stop"
    elif math.isclose(exit_price, risk.take_profit_price, rel_tol=1e-9, abs_tol=1e-9):
        target = "take"
    else:
        return len(price_values) - 1
    for idx in range(start_pos + 1, len(price_values)):
        high = float(price_values.iloc[idx]["high"])
        low = float(price_values.iloc[idx]["low"])
        if target == "stop":
            if risk.direction == "long" and low <= risk.stop_price:
                return idx
            if risk.direction == "short" and high >= risk.stop_price:
                return idx
        else:  # take profit
            if risk.direction == "long" and high >= risk.take_profit_price:
                return idx
            if risk.direction == "short" and low <= risk.take_profit_price:
                return idx
    return len(price_values) - 1


def _simulate_trade(prices: pd.DataFrame, start_pos: int, risk: RiskOutcome) -> tuple[float, int]:
    price_values = prices[["high", "low", "close"]]
    if start_pos >= len(price_values) - 1:
        last_close = float(price_values.iloc[-1]["close"])
        return last_close, len(price_values) - 1
    if _simulate_trade_path_accel is None:
        return _simulate_trade_python(prices, start_pos, risk)
    future_slice = price_values.iloc[start_pos + 1 :]
    exit_price = float(
        _simulate_trade_path_accel(
            future_slice["high"].astype(float).tolist(),
            future_slice["low"].astype(float).tolist(),
            future_slice["close"].astype(float).tolist(),
            risk.direction,
            risk.entry_price,
            risk.stop_price,
            risk.take_profit_price,
        )
    )
    exit_idx = _locate_exit_index(price_values, start_pos, risk, exit_price)
    return exit_price, exit_idx


def _max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak <= 0:
            continue
        drawdown = (peak - value) / peak
        max_dd = max(max_dd, drawdown)
    return max_dd


def prepare_supercalc_dataset(config: BacktestConfig) -> SupercalcDataset:
    fetcher = MarketDataFetcher(config.ticker)
    period = config.period if not (config.start or config.end) else None
    prices = fetcher.get_prices(
        interval=config.interval,
        period=period,
        start=config.start,
        end=config.end,
    )
    prices = prices.sort_index()
    if prices.empty:
        raise ValueError("No price data available for backtest.")

    first_close = float(prices["close"].iloc[0])
    last_close = float(prices["close"].iloc[-1])
    initial_balance = config.initial_balance
    if initial_balance > 0 and first_close > 0:
        buy_hold_shares = initial_balance / first_close
        benchmark_final_balance = buy_hold_shares * last_close
        benchmark_return = (benchmark_final_balance / initial_balance - 1.0) * 100.0
    else:
        benchmark_return = 0.0

    features = compute_feature_frame(
        prices,
        sma_short_window=config.sma_short_window,
        sma_long_window=config.sma_long_window,
        rsi_window=config.rsi_window,
        atr_window=config.atr_window,
    )
    if features.empty:
        raise RuntimeError("Feature frame is empty; cannot prepare dataset.")
    if "sentiment_score" not in features.columns:
        features["sentiment_score"] = 0.0
    features = enrich_sentiment_features(features)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    features = attach_amplitude_labels(features, prices, config.label)
    validate_label_support(features, config.label)
    if features.empty:
        raise RuntimeError("Feature frame dropped to zero rows after cleaning.")

    price_pos_lookup = {ts: idx for idx, ts in enumerate(prices.index)}
    feature_index = features.index
    atr_column = f"atr_{config.atr_window}"
    min_history = max(config.sma_long_window, config.rsi_window, config.atr_window) + 5
    rows: list[SupercalcRow] = []

    for current_idx in range(min_history, len(features)):
        train_slice = features.iloc[:current_idx]
        design, target = _prepare_design_and_target(train_slice, config.label)
        if design.empty or target.nunique() < 2:
            continue
        model = ProbabilityModel.create(
            model_type=config.model_type,
            calibration=config.calibration,
            random_state=config.random_state,
        )
        model.fit(design, target)
        inference_design = features.iloc[[current_idx]][design.columns]
        probability_up = float(model.predict_proba(inference_design)[0])
        timestamp = feature_index[current_idx]
        price_pos = price_pos_lookup.get(timestamp)
        if price_pos is None or price_pos >= len(prices) - 1:
            continue
        atr_value = float(features.iloc[current_idx][atr_column])
        if atr_value <= 0 or not math.isfinite(atr_value):
            continue
        rows.append(
            SupercalcRow(
                price_index=price_pos,
                timestamp=timestamp,
                probability_up=probability_up,
                rolling_return=float(features.iloc[current_idx].get("rolling_return", 0.0)),
                atr_value=atr_value,
                sentiment_score=float(features.iloc[current_idx].get("sentiment_score", 0.0)),
                extreme_fear_flag=int(features.iloc[current_idx].get("extreme_fear_flag", 0)),
                extreme_greed_flag=int(features.iloc[current_idx].get("extreme_greed_flag", 0)),
            )
        )

    if not rows:
        raise RuntimeError("Supercalc dataset is empty; insufficient history for modeling.")

    return SupercalcDataset(
        rows=rows,
        prices=prices,
        benchmark_return_pct=float(benchmark_return),
        initial_balance=float(initial_balance),
    )


def _row_trend_permits(row: SupercalcRow, direction: str, config: BacktestConfig) -> bool:
    if not config.enable_trend_filter:
        return True
    rolling_return = row.rolling_return
    if direction == "long":
        return rolling_return >= config.trend_long_min
    threshold = config.trend_short_min
    if threshold <= 0.0:
        return rolling_return <= 0.0
    return rolling_return <= -threshold


def _row_sentiment_permits(row: SupercalcRow, config: BacktestConfig) -> bool:
    score = row.sentiment_score
    if score < config.sentiment_min or score > config.sentiment_max:
        return False
    regime = (config.sentiment_regime or "off").lower()
    fear_active = bool(row.extreme_fear_flag)
    greed_active = bool(row.extreme_greed_flag)
    if regime == "off":
        return True
    if regime == "fear_only":
        return fear_active
    if regime == "greed_only":
        return greed_active
    if regime == "neutral_only":
        return not fear_active and not greed_active
    return True


def simulate_from_dataset(dataset: SupercalcDataset, config: BacktestConfig) -> StrategyStats:
    equity = float(dataset.initial_balance)
    equity_curve = [equity]
    trade_returns: list[float] = []
    positive_pnl = 0.0
    negative_pnl = 0.0
    wins = 0
    n_trades = 0
    trades_in_fear = 0
    trades_in_greed = 0
    prices = dataset.prices
    last_exit_index = -1
    equity_history: list[tuple[pd.Timestamp, float]] = []

    for row in dataset.rows:
        equity_history.append((row.timestamp, float(equity)))
        if row.price_index <= last_exit_index:
            continue
        probability_up = row.probability_up
        if probability_up >= config.long_threshold:
            direction = "long"
        elif probability_up <= config.short_threshold:
            direction = "short"
        else:
            continue

        if not _row_trend_permits(row, direction, config):
            continue
        if not _row_sentiment_permits(row, config):
            continue

        atr_value = row.atr_value
        if atr_value <= 0 or not math.isfinite(atr_value):
            continue
        price_index = row.price_index
        if price_index >= len(prices) - 1:
            continue
        entry_price = float(prices.iloc[price_index]["close"])

        risk_cfg = RiskConfig(
            balance=float(equity),
            risk_pct=float(config.risk.risk_pct),
            atr_multiplier=float(config.risk.atr_multiplier),
            reward_multiple=float(config.risk.reward_multiple),
            min_position_size=int(config.risk.min_position_size),
        )
        risk_plan = plan_trade(entry_price=entry_price, atr=atr_value, direction=direction, config=risk_cfg)
        if risk_plan.position_size <= 0:
            continue

        expected_pnl = _expected_trade_pnl(risk_plan, probability_up)
        min_ev = float(risk_plan.risk_amount) * max(config.min_ev_multiple, 0.0)
        if expected_pnl <= max(0.0, min_ev):
            continue

        exit_price, exit_pos = _simulate_trade(prices, price_index, risk_plan)
        last_exit_index = exit_pos
        balance_before = equity
        pnl = _compute_trade_pnl(risk_plan, entry_price, exit_price)
        notional = abs(risk_plan.position_size) * entry_price
        total_rate = config.commission_pct + config.slippage_pct
        cost = 2.0 * total_rate * notional if notional > 0 else 0.0
        net_pnl = pnl - cost
        equity += net_pnl
        n_trades += 1
        if net_pnl > 0:
            wins += 1
        if row.extreme_fear_flag:
            trades_in_fear += 1
        if row.extreme_greed_flag:
            trades_in_greed += 1

        return_pct = net_pnl / balance_before if balance_before > 0 else 0.0
        trade_returns.append(return_pct)
        equity_curve.append(float(equity))
        exit_timestamp = prices.index[exit_pos]
        equity_history.append((exit_timestamp, float(equity)))
        if net_pnl > 0:
            positive_pnl += net_pnl
        elif net_pnl < 0:
            negative_pnl += net_pnl

    win_rate = wins / n_trades if n_trades else 0.0
    max_dd = _max_drawdown(equity_curve)
    sharpe = _sharpe_ratio(trade_returns)
    expectancy = (positive_pnl + negative_pnl) / n_trades if n_trades else 0.0
    profit_factor = positive_pnl / abs(negative_pnl) if negative_pnl < 0 else None

    return StrategyStats(
        final_balance=float(equity),
        profit_factor=float(profit_factor) if profit_factor is not None else None,
        sharpe_ratio=sharpe,
        max_drawdown_pct=float(max_dd * 100.0),
        expectancy=float(expectancy),
        n_trades=n_trades,
        win_rate=win_rate,
        trades_in_fear=trades_in_fear,
        trades_in_greed=trades_in_greed,
        trade_returns=[float(x) for x in trade_returns],
        equity_history=equity_history,
    )


def _deflated_sharpe_ratio(sharpe: float, n_obs: int, n_trials: int = 1) -> float:
    if not math.isfinite(sharpe) or n_obs <= 1:
        return float("nan")
    sr_std = math.sqrt((1 + 0.5 * sharpe**2) / max(n_obs - 1, 1))
    sr_crit = sr_std * (n_trials - 1)
    if sr_std == 0:
        return float("nan")
    return (sharpe - sr_crit) / sr_std


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def summarize_oos_metrics(
    equity_oos: pd.Series,
    trades_oos: list[TradeRecord],
    bars_per_year: float | None = None,
) -> dict[str, float]:
    """
    Compute research metrics (PF, Sharpe, Calmar, MaxDD, CVaR95 monthly, DSR, PBoC) from OOS data.
    """
    equity = equity_oos.dropna()
    result = {
        "pf": float("nan"),
        "sharpe": float("nan"),
        "calmar": float("nan"),
        "maxdd": float("nan"),
        "cvar95_m": float("nan"),
        "dsr": float("nan"),
        "pboc": float("nan"),
    }
    if equity.empty:
        return result
    pnl = np.array([trade.pnl for trade in trades_oos], dtype=float)
    positive = pnl[pnl > 0].sum()
    negative = pnl[pnl < 0].sum()
    if negative < 0:
        pf = positive / abs(negative)
    elif positive > 0:
        pf = float("inf")
    else:
        pf = 0.0
    result["pf"] = float(min(max(pf, 0.0), 10.0))

    returns = equity.pct_change().dropna()
    if len(returns) >= 2:
        if bars_per_year is None:
            duration_days = max((equity.index[-1] - equity.index[0]).days + 1, 1)
            bars_per_year_local = max(len(equity) * (365.25 / duration_days), 1.0)
        else:
            bars_per_year_local = max(bars_per_year, 1.0)
        sharpe = float((returns.mean() / returns.std(ddof=1)) * math.sqrt(bars_per_year_local))
        result["sharpe"] = sharpe
    else:
        sharpe = float("nan")

    maxdd = _max_drawdown(equity.tolist())
    result["maxdd"] = float(maxdd)

    span_years = max(
        (equity.index[-1] - equity.index[0]).total_seconds() / (365.25 * 24 * 3600),
        1e-6,
    )
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / span_years) - 1 if equity.iloc[0] > 0 else 0.0
    result["calmar"] = float(cagr / max(maxdd, 1e-6))

    if trades_oos:
        exit_index = [
            getattr(trade, "exit_timestamp", getattr(trade, "exit_time", None)) for trade in trades_oos
        ]
        trade_series = pd.Series(data=[trade.pnl for trade in trades_oos], index=exit_index).dropna().sort_index()
        monthly = trade_series.resample("30D").sum()
        result["cvar95_m"] = cvar_from_pnl(monthly.values, alpha=0.95)

    dsr = _deflated_sharpe_ratio(result["sharpe"], len(trades_oos) or len(returns) or 0)
    result["dsr"] = dsr
    result["pboc"] = float(1.0 - _norm_cdf(dsr)) if math.isfinite(dsr) else float("nan")
    return result


def _sharpe_ratio(returns: list[float]) -> Optional[float]:
    if len(returns) < 2:
        return None
    arr = np.array(returns, dtype=float)
    std = arr.std(ddof=1)
    if std == 0:
        return None
    return float((arr.mean() / std) * math.sqrt(len(arr)))


def _sortino_ratio(returns: list[float], trades_per_year: float, risk_free: float = 0.0) -> Optional[float]:
    if len(returns) < 2 or trades_per_year <= 0:
        return None
    arr = np.array(returns, dtype=float)
    target = risk_free / trades_per_year
    excess = arr - target
    downside = excess[excess < 0]
    if downside.size < 1:
        return None
    mean_excess = excess.mean()
    downside_std = downside.std(ddof=1)
    if downside_std == 0:
        return None
    annualized_return = mean_excess * trades_per_year
    annualized_downside = downside_std * math.sqrt(trades_per_year)
    if annualized_downside == 0:
        return None
    return float(annualized_return / annualized_downside)


def run_backtest(config: BacktestConfig) -> BacktestResult:
    if not (0.0 < config.short_threshold <= config.long_threshold < 1.0):
        raise ValueError("short_threshold must be >0, long_threshold <1, and short_threshold <= long_threshold.")

    fetcher = MarketDataFetcher(config.ticker)
    period = config.period if not (config.start or config.end) else None
    prices = fetcher.get_prices(
        interval=config.interval,
        period=period,
        start=config.start,
        end=config.end,
    )
    prices = prices.sort_index()
    if prices.empty:
        raise ValueError("No price data available for backtest.")
    initial_balance = config.initial_balance
    first_close = float(prices["close"].iloc[0])
    last_close = float(prices["close"].iloc[-1])
    if initial_balance > 0 and first_close > 0:
        buy_hold_shares = initial_balance / first_close
        benchmark_final_balance = buy_hold_shares * last_close
        benchmark_return = (benchmark_final_balance / initial_balance - 1.0) * 100.0
    else:
        benchmark_final_balance = float(initial_balance)
        benchmark_return = 0.0

    features = compute_feature_frame(
        prices,
        sma_short_window=config.sma_short_window,
        sma_long_window=config.sma_long_window,
        rsi_window=config.rsi_window,
        atr_window=config.atr_window,
    )
    if features.empty:
        raise RuntimeError("Feature frame is empty; cannot run backtest.")
    if "sentiment_score" not in features.columns:
        features["sentiment_score"] = 0.0
    features = enrich_sentiment_features(features)
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    features = attach_amplitude_labels(features, prices, config.label)
    validate_label_support(features, config.label)
    if features.empty:
        raise RuntimeError("Feature frame dropped to zero rows after cleaning.")

    price_pos_lookup = {ts: idx for idx, ts in enumerate(prices.index)}
    equity = config.initial_balance
    equity_curve = [float(equity)]
    trade_returns: list[float] = []
    r_multiples: list[float] = []
    expected_values: list[float] = []
    wins = 0
    n_trades = 0
    long_trades = 0
    short_trades = 0
    long_wins = 0
    short_wins = 0
    long_pnl = 0.0
    short_pnl = 0.0
    sentiment_fear_trades = 0
    sentiment_greed_trades = 0
    feature_index = features.index
    atr_column = f"atr_{config.atr_window}"
    min_history = max(config.sma_long_window, config.rsi_window, config.atr_window) + 5
    current_idx = 0
    feature_count = len(features)
    trades: list[TradeRecord] = []
    adaptive_cfg = config.adaptive
    window_len = adaptive_cfg.window()
    recent_trades: deque[TradeRecord] = deque(maxlen=window_len)
    initial_regime = _initial_regime_name(adaptive_cfg, config.risk_profile)
    adaptive_state = AdaptiveState(regime_name=initial_regime)

    while current_idx < feature_count:
        if current_idx < min_history:
            current_idx += 1
            continue

        timestamp = feature_index[current_idx]
        price_pos = price_pos_lookup.get(timestamp)
        if price_pos is None or price_pos >= len(prices) - 1:
            current_idx += 1
            continue

        train_slice = features.iloc[:current_idx]
        design, target = _prepare_design_and_target(train_slice, config.label)
        if design.empty or target.nunique() < 2:
            current_idx += 1
            continue

        active_regime = adaptive_cfg.get_regime(adaptive_state.regime_name) if adaptive_cfg.enable else None
        model_type, calibration = _effective_model(config, active_regime)
        model = ProbabilityModel.create(
            model_type=model_type,
            calibration=calibration,
            random_state=config.random_state,
        )
        model.fit(design, target)
        inference_design = features.iloc[[current_idx]][design.columns]
        probability_up = float(model.predict_proba(inference_design)[0])
        regime_for_thresholds = active_regime if adaptive_cfg.enable else None
        long_threshold, short_threshold = _effective_thresholds(
            config.long_threshold,
            config.short_threshold,
            regime_for_thresholds,
        )

        if probability_up >= long_threshold:
            direction = "long"
            decision_threshold = long_threshold
        elif probability_up <= short_threshold:
            direction = "short"
            decision_threshold = short_threshold
        else:
            current_idx += 1
            continue
        feature_row = features.iloc[current_idx]
        if not _trend_permits_trade(feature_row, direction, config):
            current_idx += 1
            continue
        if not _sentiment_permits_trade(feature_row, config):
            current_idx += 1
            continue
        atr_value = float(feature_row[atr_column])
        if atr_value <= 0 or not math.isfinite(atr_value):
            current_idx += 1
            continue

        entry_price = float(prices.loc[timestamp, "close"])
        profile_candidate = adaptive_state.regime_name if adaptive_state.regime_name in config.risk_profiles else config.risk_profile
        risk_cfg = _build_risk_config(
            config.risk,
            config.risk_profiles,
            profile_candidate or config.risk_profile or "",
            equity,
            overrides=active_regime.risk_overrides if active_regime and active_regime.risk_overrides else None,
        )
        risk_plan = plan_trade(entry_price=entry_price, atr=atr_value, direction=direction, config=risk_cfg)
        if risk_plan.position_size <= 0:
            current_idx += 1
            continue

        expected_pnl = _expected_trade_pnl(risk_plan, probability_up)
        min_ev = float(risk_plan.risk_amount) * max(config.min_ev_multiple, 0.0)
        if expected_pnl <= max(0.0, min_ev):
            current_idx += 1
            continue

        exit_price, exit_pos = _simulate_trade(prices, price_pos, risk_plan)
        balance_before = equity
        pnl = _compute_trade_pnl(risk_plan, entry_price, exit_price)
        notional = abs(risk_plan.position_size) * entry_price
        total_rate = config.commission_pct + config.slippage_pct
        cost = 2.0 * total_rate * notional if notional > 0 else 0.0
        net_pnl = pnl - cost
        equity += net_pnl
        n_trades += 1
        if net_pnl > 0:
            wins += 1
            if direction == "long":
                long_wins += 1
            else:
                short_wins += 1
        return_pct = net_pnl / balance_before if balance_before > 0 else 0.0
        trade_returns.append(return_pct)
        r_multiple = net_pnl / risk_plan.risk_amount if risk_plan.risk_amount > 0 else 0.0
        r_multiples.append(r_multiple)
        expected_values.append(expected_pnl)
        equity_curve.append(float(equity))

        exit_timestamp = prices.index[exit_pos]
        if direction == "long":
            long_trades += 1
            long_pnl += net_pnl
        else:
            short_trades += 1
            short_pnl += net_pnl
        if int(feature_row.get("extreme_fear_flag", 0)) == 1:
            sentiment_fear_trades += 1
        if int(feature_row.get("extreme_greed_flag", 0)) == 1:
            sentiment_greed_trades += 1
        record = TradeRecord(
            entry_timestamp=timestamp,
            exit_timestamp=exit_timestamp,
            direction=direction,
            probability_up=probability_up,
            threshold=decision_threshold,
            entry_price=entry_price,
            exit_price=float(exit_price),
            position_size=risk_plan.position_size,
            pnl=float(net_pnl),
            return_pct=float(return_pct),
            equity_after=float(equity),
            risk_amount=float(risk_plan.risk_amount),
            expected_pnl=float(expected_pnl),
            risk_profile=profile_candidate,
            effective_long_threshold=long_threshold,
            effective_short_threshold=short_threshold,
            regime_name=adaptive_state.regime_name,
        )
        trades.append(record)
        recent_trades.append(record)
        adaptive_state.record_trade()
        if adaptive_cfg.enable and len(recent_trades) == recent_trades.maxlen:
            metrics = evaluate_window(list(recent_trades))
            adaptive_state = update_state(adaptive_cfg, adaptive_state, metrics, trade_index=n_trades)
        next_idx = feature_index.searchsorted(exit_timestamp, side="right")
        if next_idx >= feature_count:
            break
        current_idx = next_idx

    win_rate = (wins / n_trades) if n_trades else 0.0
    max_dd = _max_drawdown(equity_curve)
    sharpe = _sharpe_ratio(trade_returns)
    final_balance = equity_curve[-1] if equity_curve else config.initial_balance
    time_delta = prices.index[-1] - prices.index[0]
    years = max(time_delta.total_seconds() / (365.0 * 24 * 3600), 0.0)
    if years > 0 and config.initial_balance > 0:
        annualized_return = (final_balance / config.initial_balance) ** (1 / years) - 1
        annualized_benchmark = (
            benchmark_final_balance / config.initial_balance
        ) ** (1 / years) - 1
    else:
        annualized_return = None
        annualized_benchmark = None
    if trade_returns and years > 0:
        trades_per_year = len(trade_returns) / years
    else:
        trades_per_year = 0.0
    if len(trade_returns) > 1 and trades_per_year > 0:
        volatility = float(np.std(trade_returns, ddof=1) * math.sqrt(trades_per_year))
    else:
        volatility = None
    sortino = _sortino_ratio(trade_returns, trades_per_year)
    positive_pnl = sum(trade.pnl for trade in trades if trade.pnl > 0)
    negative_pnl = sum(trade.pnl for trade in trades if trade.pnl < 0)
    profit_factor = (
        positive_pnl / abs(negative_pnl) if negative_pnl < 0 else None
    )
    expectancy = (positive_pnl + negative_pnl) / n_trades if n_trades else 0.0
    avg_r_multiple = float(np.mean(r_multiples)) if r_multiples else 0.0
    avg_expected_pnl = float(np.mean(expected_values)) if expected_values else 0.0
    long_win_rate = (long_wins / long_trades) if long_trades else 0.0
    short_win_rate = (short_wins / short_trades) if short_trades else 0.0

    calibration_metrics = trade_calibration_metrics(trades)

    final_regime = adaptive_cfg.get_regime(adaptive_state.regime_name) if adaptive_cfg.enable else None
    final_long_threshold, final_short_threshold = _effective_thresholds(
        config.long_threshold,
        config.short_threshold,
        final_regime,
    )# src/hyprl/backtest/runner.py - Améliorations documentées

    from dataclasses import dataclass

    @dataclass
    class BacktestConfig:
        """Configuration for backtest execution with cost modeling.

        Cost Model:
        -----------
        - commission_pct: Percentage of notional value charged as commission per trade
                          (e.g., 0.0005 = 0.05% = 5 basis points)
        - slippage_pct: Percentage of notional value approximating execution slippage
                        (e.g., 0.0005 = 0.05% = 5 basis points)

        Both costs are applied to each trade side (entry and exit), so the total
        round-trip cost is approximately 2 * (commission_pct + slippage_pct).

        For example, with defaults:
        - commission_pct = 0.0005 (0.05%)
        - slippage_pct = 0.0005 (0.05%)
        - Total cost per side = 0.001 (0.10%)
        - Total round-trip cost ≈ 0.002 (0.20%)
        """
        # Existing fields...

        # Cost parameters with clear defaults
        commission_pct: float = 0.0005  # 0.05% commission per trade
        slippage_pct: float = 0.0005    # 0.05% slippage approximation

        def get_total_cost_rate(self) -> float:
            """Returns the combined commission + slippage rate per trade side.

            Returns:
                float: Total cost rate (e.g., 0.001 = 0.10%)
            """
            return self.commission_pct + self.slippage_pct


    class BacktestRunner:
        def __init__(self, config: BacktestConfig):
            self.config = config
            # Cache the total rate for performance
            self._total_cost_rate = config.get_total_cost_rate()

        def _execute_trade(self, side: str, price: float, qty: float) -> tuple[float, float]:
            """Execute a trade with costs applied.

            Args:
                side: 'BUY' or 'SELL'
                price: Execution price (close of signal bar)
                qty: Position size

            Returns:
                tuple: (fill_price, total_cost)
            """
            notional = price * qty

            # Apply combined commission + slippage as a percentage adjustment
            # total_rate ~ 0.001 => ~0.10% combined commission + slippage
            if side == 'BUY':
                # Pay more when buying (price increases)
                fill_price = price * (1 + self._total_cost_rate)
            else:  # SELL
                # Receive less when selling (price decreases)
                fill_price = price * (1 - self._total_cost_rate)

            # Total cost is the difference between ideal and actual fill
            total_cost = abs(fill_price - price) * qty

            return fill_price, total_cost

    return BacktestResult(
        final_balance=float(final_balance),
        equity_curve=[float(x) for x in equity_curve],
        n_trades=n_trades,
        win_rate=win_rate,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
        trades=trades,
        benchmark_final_balance=float(benchmark_final_balance),
        benchmark_return=float(benchmark_return),
        annualized_return=float(annualized_return) if annualized_return is not None else None,
        annualized_benchmark_return=float(annualized_benchmark) if annualized_benchmark is not None else None,
        annualized_volatility=volatility,
        sortino_ratio=sortino,
        profit_factor=profit_factor,
        expectancy=float(expectancy),
        avg_r_multiple=avg_r_multiple,
        avg_expected_pnl=avg_expected_pnl,
        long_trades=long_trades,
        short_trades=short_trades,
        long_win_rate=long_win_rate,
        short_win_rate=short_win_rate,
        long_total_pnl=float(long_pnl),
        short_total_pnl=float(short_pnl),
        brier_score=calibration_metrics["brier"],
        log_loss=calibration_metrics["log_loss"],
        final_risk_profile=adaptive_state.regime_name,
        final_long_threshold=final_long_threshold,
        final_short_threshold=final_short_threshold,
        adaptive_profile_changes=adaptive_state.regime_changes,
        regime_usage=dict(adaptive_state.regime_counts),
        regime_transitions=list(adaptive_state.transitions),
        sentiment_stats={
            "trades_in_fear": sentiment_fear_trades,
            "trades_in_greed": sentiment_greed_trades,
        },
        trade_returns=[float(x) for x in trade_returns],
    )


def sweep_thresholds(base_config: BacktestConfig, thresholds: list[float]) -> list[ThresholdSummary]:
    summaries: list[ThresholdSummary] = []
    initial_balance = base_config.initial_balance
    for threshold in thresholds:
        cfg = replace(base_config, long_threshold=threshold)
        result = run_backtest(cfg)
        if initial_balance > 0:
            total_return = (result.final_balance / initial_balance - 1.0) * 100.0
        else:
            total_return = 0.0
        benchmark_return = result.benchmark_return
        alpha_return = total_return - benchmark_return
        summaries.append(
            ThresholdSummary(
                threshold=threshold,
                final_balance=result.final_balance,
                total_return=total_return,
                n_trades=result.n_trades,
                win_rate=result.win_rate,
                max_drawdown=result.max_drawdown,
                sharpe_ratio=result.sharpe_ratio,
                benchmark_return=benchmark_return,
                alpha_return=alpha_return,
                annualized_return=result.annualized_return,
                profit_factor=result.profit_factor,
                expectancy=result.expectancy,
                final_regime=result.final_risk_profile,
                adaptive_changes=result.adaptive_profile_changes,
                regime_usage=dict(result.regime_usage),
            )
        )
    return summaries

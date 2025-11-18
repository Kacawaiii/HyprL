from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import math
import numpy as np
import pandas as pd

from hyprl.data.market import MarketDataFetcher
from hyprl.indicators.technical import compute_feature_frame
from hyprl.model.probability import ProbabilityModel
from hyprl.risk.manager import RiskConfig, RiskOutcome, plan_trade

try:
    from hyprl_accel import simulate_trade_path as _simulate_trade_path_accel
except Exception:  # pragma: no cover - optional accelerator
    _simulate_trade_path_accel = None


@dataclass(slots=True)
class BacktestConfig:
    ticker: str
    period: Optional[str] = "6mo"
    start: Optional[str] = None
    end: Optional[str] = None
    interval: str = "1h"
    initial_balance: float = 10_000.0
    threshold: float = 0.4
    sma_short_window: int = 5
    sma_long_window: int = 36
    rsi_window: int = 14
    atr_window: int = 14
    risk: RiskConfig = field(default_factory=RiskConfig)
    random_state: int = 42
    min_expected_value: Optional[float] = None
    require_trend_alignment: bool = False
    sentiment_threshold: Optional[float] = None


@dataclass(slots=True)
class BacktestResult:
    final_balance: float
    equity_curve: list[float]
    n_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: Optional[float]


@dataclass(slots=True)
class ThresholdSummary:
    threshold: float
    final_balance: float
    total_return: float
    n_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: Optional[float]


def _prepare_design_and_target(feature_slice: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = ["trend_ratio", "rsi_normalized", "volatility", "sentiment_score"]
    design = feature_slice[feature_cols]
    finite_mask = np.isfinite(design).all(axis=1)
    design = design.loc[finite_mask]
    target = (feature_slice.loc[design.index, "returns_next"] > 0).astype(int)
    return design, target


def _compute_trade_pnl(risk: RiskOutcome, entry_price: float, exit_price: float) -> float:
    delta = exit_price - entry_price if risk.direction == "long" else entry_price - exit_price
    return delta * risk.position_size


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


def _sharpe_ratio(returns: list[float]) -> Optional[float]:
    if len(returns) < 2:
        return None
    arr = np.array(returns, dtype=float)
    std = arr.std(ddof=1)
    if std == 0:
        return None
    return float((arr.mean() / std) * math.sqrt(len(arr)))


def _compute_expected_value(
    probability_up: float,
    direction: str,
    risk_plan: RiskOutcome,
) -> float:
    """Compute expected value of a trade based on probability and risk/reward."""
    if risk_plan.position_size <= 0:
        return 0.0
    
    per_unit_risk = abs(risk_plan.entry_price - risk_plan.stop_price)
    per_unit_reward = abs(risk_plan.take_profit_price - risk_plan.entry_price)
    
    if direction == "long":
        win_prob = probability_up
    else:
        win_prob = 1.0 - probability_up
    
    lose_prob = 1.0 - win_prob
    expected_value = (win_prob * per_unit_reward) - (lose_prob * per_unit_risk)
    return expected_value


def _check_trend_alignment(
    current_row: pd.Series,
    direction: str,
    sma_short_col: str = "sma_short",
    sma_long_col: str = "sma_long",
) -> bool:
    """Check if trade direction aligns with SMA trend."""
    sma_short = float(current_row.get(sma_short_col, 0.0))
    sma_long = float(current_row.get(sma_long_col, 0.0))
    
    if sma_short <= 0 or sma_long <= 0:
        return False
    
    if direction == "long":
        return sma_short > sma_long
    else:
        return sma_short < sma_long


def run_backtest(config: BacktestConfig) -> BacktestResult:
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

    features = compute_feature_frame(
        prices,
        sma_short_window=config.sma_short_window,
        sma_long_window=config.sma_long_window,
        rsi_window=config.rsi_window,
        atr_window=config.atr_window,
    )
    if features.empty:
        raise RuntimeError("Feature frame is empty; cannot run backtest.")
    features["sentiment_score"] = 0.0
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if features.empty:
        raise RuntimeError("Feature frame dropped to zero rows after cleaning.")

    price_pos_lookup = {ts: idx for idx, ts in enumerate(prices.index)}
    equity = config.initial_balance
    equity_curve = [float(equity)]
    trade_returns: list[float] = []
    wins = 0
    n_trades = 0
    feature_index = features.index
    atr_column = f"atr_{config.atr_window}"
    min_history = max(config.sma_long_window, config.rsi_window, config.atr_window) + 5
    current_idx = 0
    feature_count = len(features)

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
        design, target = _prepare_design_and_target(train_slice)
        if design.empty or target.nunique() < 2:
            current_idx += 1
            continue

        model = ProbabilityModel.create(random_state=config.random_state)
        model.fit(design, target)
        inference_design = features.iloc[[current_idx]][design.columns]
        probability_up = float(model.predict_proba(inference_design)[0])
        direction = "long" if probability_up >= config.threshold else "short"
        atr_value = float(features.iloc[current_idx][atr_column])
        if atr_value <= 0 or not math.isfinite(atr_value):
            current_idx += 1
            continue

        entry_price = float(prices.loc[timestamp, "close"])
        risk_cfg = replace(config.risk, balance=equity)
        risk_plan = plan_trade(entry_price=entry_price, atr=atr_value, direction=direction, config=risk_cfg)
        if risk_plan.position_size <= 0:
            current_idx += 1
            continue

        # Apply minimum expected value filter
        if config.min_expected_value is not None:
            ev = _compute_expected_value(probability_up, direction, risk_plan)
            if ev < config.min_expected_value:
                current_idx += 1
                continue

        # Apply trend alignment filter
        if config.require_trend_alignment:
            current_row = features.iloc[current_idx]
            if not _check_trend_alignment(current_row, direction):
                current_idx += 1
                continue

        # Apply sentiment threshold filter
        if config.sentiment_threshold is not None:
            sentiment_score = float(features.iloc[current_idx].get("sentiment_score", 0.0))
            if direction == "long" and sentiment_score < config.sentiment_threshold:
                current_idx += 1
                continue
            if direction == "short" and sentiment_score > -config.sentiment_threshold:
                current_idx += 1
                continue

        exit_price, exit_pos = _simulate_trade(prices, price_pos, risk_plan)
        balance_before = equity
        pnl = _compute_trade_pnl(risk_plan, entry_price, exit_price)
        equity += pnl
        n_trades += 1
        if pnl > 0:
            wins += 1
        trade_returns.append(pnl / balance_before if balance_before > 0 else 0.0)
        equity_curve.append(float(equity))

        exit_timestamp = prices.index[exit_pos]
        next_idx = feature_index.searchsorted(exit_timestamp, side="right")
        if next_idx >= feature_count:
            break
        current_idx = next_idx

    win_rate = (wins / n_trades) if n_trades else 0.0
    max_dd = _max_drawdown(equity_curve)
    sharpe = _sharpe_ratio(trade_returns)
    final_balance = equity_curve[-1] if equity_curve else config.initial_balance

    return BacktestResult(
        final_balance=float(final_balance),
        equity_curve=[float(x) for x in equity_curve],
        n_trades=n_trades,
        win_rate=win_rate,
        max_drawdown=max_dd,
        sharpe_ratio=sharpe,
    )


def sweep_thresholds(base_config: BacktestConfig, thresholds: list[float]) -> list[ThresholdSummary]:
    summaries: list[ThresholdSummary] = []
    initial_balance = base_config.initial_balance
    for threshold in thresholds:
        cfg = replace(base_config, threshold=threshold)
        result = run_backtest(cfg)
        if initial_balance > 0:
            total_return = (result.final_balance / initial_balance - 1.0) * 100.0
        else:
            total_return = 0.0
        summaries.append(
            ThresholdSummary(
                threshold=threshold,
                final_balance=result.final_balance,
                total_return=total_return,
                n_trades=result.n_trades,
                win_rate=result.win_rate,
                max_drawdown=result.max_drawdown,
                sharpe_ratio=result.sharpe_ratio,
            )
        )
    return summaries

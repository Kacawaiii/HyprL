from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

from hyprl.backtest.runner import (
    BacktestConfig,
    StrategyStats,
    SupercalcDataset,
    prepare_supercalc_dataset,
    simulate_from_dataset,
    _row_trend_permits,
    _row_sentiment_permits,
)

try:  # pragma: no cover - optional native module v2
    from hyprl.native.supercalc import run_backtest_native

    HAS_NATIVE_WRAPPER = True
except Exception:  # pragma: no cover
    HAS_NATIVE_WRAPPER = False
    run_backtest_native = None  # type: ignore

try:
    from hyprl_supercalc import BacktestParams as _LegacyParams, run_batch as _run_legacy_batch

    HAS_LEGACY_NATIVE = True
except ImportError:  # pragma: no cover - optional native module
    HAS_LEGACY_NATIVE = False
    _LegacyParams = None  # type: ignore
    _run_legacy_batch = None  # type: ignore


@dataclass(slots=True)
class EngineCandidate:
    config: BacktestConfig


def native_available() -> bool:
    return HAS_NATIVE_WRAPPER or HAS_LEGACY_NATIVE


def _build_native_params(config: BacktestConfig):
    if _LegacyParams is None:  # pragma: no cover - guarded upstream
        raise RuntimeError("hyprl_supercalc native extension not available.")
    regime = (config.sentiment_regime or "off").lower()
    return _LegacyParams(
        long_threshold=float(config.long_threshold),
        short_threshold=float(config.short_threshold),
        risk_pct=float(config.risk.risk_pct),
        atr_multiplier=float(config.risk.atr_multiplier),
        reward_multiple=float(config.risk.reward_multiple),
        min_position_size=int(config.risk.min_position_size),
        commission_pct=float(config.commission_pct),
        slippage_pct=float(config.slippage_pct),
        min_ev_multiple=float(config.min_ev_multiple),
        trend_filter=bool(config.enable_trend_filter),
        trend_long_min=float(config.trend_long_min),
        trend_short_min=float(config.trend_short_min),
        sentiment_min=float(config.sentiment_min),
        sentiment_max=float(config.sentiment_max),
        sentiment_regime=regime,
    )


def _dataset_arrays(dataset: SupercalcDataset) -> dict[str, List[float]]:
    prices = dataset.prices
    rows = dataset.rows
    return {
        "highs": prices["high"].astype(float).to_list(),
        "lows": prices["low"].astype(float).to_list(),
        "closes": prices["close"].astype(float).to_list(),
        "price_indices": [int(row.price_index) for row in rows],
        "probabilities": [float(row.probability_up) for row in rows],
        "rolling_returns": [float(row.rolling_return) for row in rows],
        "atr_values": [float(row.atr_value) for row in rows],
        "sentiment_scores": [float(row.sentiment_score) for row in rows],
        "fear_flags": [int(row.extreme_fear_flag) for row in rows],
        "greed_flags": [int(row.extreme_greed_flag) for row in rows],
    }


def _run_native_engine(dataset: SupercalcDataset, candidate_configs: list[BacktestConfig]) -> list[StrategyStats]:
    if not candidate_configs:
        return []
    if HAS_NATIVE_WRAPPER and run_backtest_native is not None:
        return _run_native_engine_v2(dataset, candidate_configs)
    if HAS_LEGACY_NATIVE:
        return _run_native_engine_legacy(dataset, candidate_configs)
    raise RuntimeError("hyprl_supercalc native module unavailable.")


def _run_native_engine_legacy(dataset: SupercalcDataset, candidate_configs: list[BacktestConfig]) -> list[StrategyStats]:
    arrays = _dataset_arrays(dataset)
    params = [_build_native_params(cfg) for cfg in candidate_configs]
    native_results = _run_legacy_batch(
        arrays["highs"],
        arrays["lows"],
        arrays["closes"],
        arrays["price_indices"],
        arrays["probabilities"],
        arrays["rolling_returns"],
        arrays["atr_values"],
        arrays["sentiment_scores"],
        arrays["fear_flags"],
        arrays["greed_flags"],
        float(dataset.initial_balance),
        params,
    )
    results: list[StrategyStats] = []
    for item in native_results:
        results.append(
            StrategyStats(
                final_balance=float(item.final_balance),
                profit_factor=float(item.profit_factor) if item.profit_factor is not None else None,
                sharpe_ratio=item.sharpe_ratio,
                max_drawdown_pct=float(item.max_drawdown_pct),
                expectancy=float(item.expectancy),
                n_trades=int(item.n_trades),
                win_rate=float(item.win_rate),
                trades_in_fear=int(item.trades_in_fear),
                trades_in_greed=int(item.trades_in_greed),
                trade_returns=[],
                equity_history=[],
            )
        )
    return results


def _run_native_engine_v2(dataset: SupercalcDataset, candidate_configs: list[BacktestConfig]) -> list[StrategyStats]:
    results: list[StrategyStats] = []
    prices = dataset.prices
    price_index = prices.index
    for cfg in candidate_configs:
        signal, fear_count, greed_count = _build_signal_series(dataset, cfg)
        native_result = run_backtest_native(prices, signal, cfg)
        results.append(
            _result_to_strategy_stats(
                native_result,
                fear_count,
                greed_count,
                price_index,
            )
        )
    return results


def _build_signal_series(
    dataset: SupercalcDataset,
    cfg: BacktestConfig,
) -> tuple[list[float], int, int]:
    prices_len = len(dataset.prices)
    signal = [0.0] * prices_len
    current = 0.0
    cursor = 0
    trades_in_fear = 0
    trades_in_greed = 0
    allow_short = cfg.short_threshold > 0.0

    for row in dataset.rows:
        idx = max(0, min(int(row.price_index), prices_len - 1))
        while cursor < idx:
            signal[cursor] = current
            cursor += 1

        desired = 0.0
        probability_down = 1.0 - row.probability_up
        force_short = (row.probability_up < 0.45) or (row.rolling_return < 0.0)

        long_candidate = row.probability_up >= cfg.long_threshold
        short_candidate = allow_short and (
            force_short or probability_down >= cfg.short_threshold or not long_candidate
        )

        long_ok = False
        if long_candidate and _row_trend_permits(row, "long", cfg) and _row_sentiment_permits(row, cfg):
            long_ok = True

        # Force short when the probability up is weak to counter long-bias models.
        if row.probability_up < 0.6:
            desired = -1.0
        elif long_ok:
            desired = 1.0
        elif short_candidate:
            # Force-rule short: bypass trend/sentiment
            desired = -1.0

        current = desired
        signal[idx] = current
        cursor = idx + 1

        if desired != 0.0:
            if row.extreme_fear_flag:
                trades_in_fear += 1
            if row.extreme_greed_flag:
                trades_in_greed += 1

    while cursor < prices_len:
        signal[cursor] = current
        cursor += 1

    return signal, trades_in_fear, trades_in_greed


def _result_to_strategy_stats(
    result,
    trades_in_fear: int,
    trades_in_greed: int,
    price_index: pd.Index,
) -> StrategyStats:
    equity_history: list[tuple[pd.Timestamp, float]] = []
    limit = min(len(result.equity_curve), len(price_index))
    if limit:
        for ts, equity in zip(price_index[:limit], result.equity_curve[:limit]):
            equity_history.append((pd.Timestamp(ts), float(equity)))

    return StrategyStats(
        final_balance=float(result.final_balance),
        profit_factor=float(result.profit_factor) if result.profit_factor is not None else None,
        sharpe_ratio=result.sharpe_ratio,
        max_drawdown_pct=float(result.max_drawdown * 100.0),
        expectancy=float(result.expectancy),
        n_trades=int(result.n_trades),
        win_rate=float(result.win_rate),
        trades_in_fear=trades_in_fear,
        trades_in_greed=trades_in_greed,
        trade_returns=[float(x) for x in result.trade_returns],
        equity_history=equity_history,
    )


def evaluate_candidates(
    dataset: SupercalcDataset,
    candidate_configs: list[BacktestConfig],
    engine: str = "auto",
    require_trade_returns: bool = False,
) -> list[StrategyStats]:
    mode = (engine or "auto").lower()
    native_ready = native_available()
    if mode == "native" and not native_ready:
        raise RuntimeError(
            "hyprl_supercalc native module is not available. Build it via scripts/build_supercalc.sh."
        )
    use_native = mode in ("auto", "native") and native_ready

    if use_native:
        try:
            stats = _run_native_engine(dataset, candidate_configs)
            if require_trade_returns:
                for idx, cfg in enumerate(candidate_configs):
                    extra = simulate_from_dataset(dataset, cfg)
                    stats[idx].trade_returns = extra.trade_returns
                    stats[idx].equity_history = extra.equity_history
            return stats
        except Exception:
            if mode == "native":
                raise
            # fall back silently for "auto"
    stats = [simulate_from_dataset(dataset, cfg) for cfg in candidate_configs]
    return stats


__all__ = [
    "EngineCandidate",
    "evaluate_candidates",
    "native_available",
    "prepare_supercalc_dataset",
]

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict

import numpy as np

from hyprl.backtest import runner
from hyprl.backtest.runner import BacktestConfig, SupercalcDataset
from hyprl.data.market import MarketDataFetcher
from hyprl.native.supercalc import native_available, run_backtest_native
from hyprl.risk.manager import RiskConfig
from hyprl.risk.metrics import bootstrap_equity_drawdowns, compute_risk_of_ruin
from hyprl.supercalc import _build_signal_series, prepare_supercalc_dataset


def _positive(value: float, fallback: float) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return fallback
    return value


def _safe_ratio(num: float, denom: float) -> float:
    denom = float(denom)
    if abs(denom) <= 1e-12 or not math.isfinite(denom):
        return float("nan")
    return float(num / denom)


def _ratio_component(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    return float(np.clip(value, 0.0, 2.0) / 2.0)


def _inverse_component(value: float) -> float:
    if not math.isfinite(value):
        return 0.5
    clipped = float(np.clip(value, 0.0, 2.0))
    return float(np.clip(2.0 - clipped, 0.0, 1.0))


def _win_component(delta: float) -> float:
    if not math.isfinite(delta):
        return 0.5
    return float(np.clip((delta + 0.2) / 0.4, 0.0, 1.0))


def _robustness_from_python_metrics(
    total_return: float,
    sharpe: float,
    max_drawdown_pct: float,
    expectancy: float,
    win_rate: float,
    boot_summary,
) -> float:
    pnl_base = 1.0 + float(boot_summary.pnl_p05)
    total_base = 1.0 + total_return
    pf_ratio = _safe_ratio(pnl_base, total_base)
    sharpe_ratio = _safe_ratio(abs(sharpe) if math.isfinite(sharpe) else float("nan"), abs(sharpe) if math.isfinite(sharpe) else float("nan"))
    dd_ratio = _safe_ratio(abs(boot_summary.maxdd_p95), abs(max_drawdown_pct / 100.0) or 1e-9)
    pnl_spread = abs(float(boot_summary.pnl_p95) - float(boot_summary.pnl_p05))
    vol_ratio = _safe_ratio(pnl_spread, abs(expectancy) if abs(expectancy) > 1e-9 else float("nan"))
    win_delta = float(win_rate) - 0.5

    return float(
        np.clip(
            0.3 * _ratio_component(pf_ratio)
            + 0.3 * _ratio_component(sharpe_ratio)
            + 0.2 * _inverse_component(dd_ratio)
            + 0.1 * _inverse_component(vol_ratio)
            + 0.1 * _win_component(np.clip(win_delta, -0.2, 0.2)),
            0.0,
            1.0,
        )
    )


def _build_config(args: argparse.Namespace) -> BacktestConfig:
    risk_cfg = RiskConfig(
        balance=args.initial_balance,
        risk_pct=args.risk_pct,
        atr_multiplier=args.atr_multiplier,
        reward_multiple=args.reward_multiple,
        min_position_size=args.min_position_size,
    )
    return BacktestConfig(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        initial_balance=args.initial_balance,
        long_threshold=args.long_threshold,
        short_threshold=args.short_threshold,
        risk=risk_cfg,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        enable_trend_filter=args.trend_filter,
        sentiment_min=args.sentiment_min,
        sentiment_max=args.sentiment_max,
        sentiment_regime=args.sentiment_regime,
    )


def _python_metrics(stats, cfg: BacktestConfig, bootstrap_runs: int) -> Dict[str, Any]:
    trade_returns = stats.trade_returns or []
    equity_vol = np.std([val for _, val in stats.equity_history], ddof=1) if stats.equity_history else 0.0
    risk_per_trade = _positive(cfg.initial_balance * cfg.risk.risk_pct, 1e-6)
    ror = compute_risk_of_ruin(trade_returns, cfg.initial_balance or 1.0, risk_per_trade)
    _, _, boot = bootstrap_equity_drawdowns(trade_returns, n_runs=bootstrap_runs, seed=cfg.random_state)
    total_return = (stats.final_balance / (cfg.initial_balance or 1.0)) - 1.0
    robustness_score = _robustness_from_python_metrics(
        total_return=total_return,
        sharpe=stats.sharpe_ratio or 0.0,
        max_drawdown_pct=stats.max_drawdown_pct,
        expectancy=stats.expectancy,
        win_rate=stats.win_rate,
        boot_summary=boot,
    )
    return {
        "final_balance": stats.final_balance,
        "profit_factor": stats.profit_factor,
        "sharpe": stats.sharpe_ratio,
        "max_drawdown_pct": stats.max_drawdown_pct,
        "expectancy": stats.expectancy,
        "win_rate": stats.win_rate,
        "risk_of_ruin": ror,
        "pnl_p05": boot.pnl_p05,
        "pnl_p95": boot.pnl_p95,
        "maxdd_p95": boot.maxdd_p95 * 100.0,
        "equity_vol": equity_vol,
        "robustness_score": robustness_score,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python runner vs native hyprl_supercalc metrics on a real ticker.",
    )
    parser.add_argument("ticker", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--period", default="1y", help="yfinance period")
    parser.add_argument("--interval", default="1h", help="chart interval (e.g., 1h)")
    parser.add_argument("--long-threshold", type=float, default=0.6)
    parser.add_argument("--short-threshold", type=float, default=0.4)
    parser.add_argument("--risk-pct", type=float, default=0.02)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--atr-multiplier", type=float, default=2.0)
    parser.add_argument("--reward-multiple", type=float, default=2.0)
    parser.add_argument("--min-position-size", type=int, default=1)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--trend-filter", action="store_true")
    parser.add_argument("--sentiment-min", type=float, default=-1.0)
    parser.add_argument("--sentiment-max", type=float, default=1.0)
    parser.add_argument("--sentiment-regime", default="off")
    parser.add_argument("--bootstrap-runs", type=int, default=256)
    return parser.parse_args()


def _build_dataset(cfg: BacktestConfig, args: argparse.Namespace) -> SupercalcDataset:
    try:
        return prepare_supercalc_dataset(cfg)
    except Exception as exc:
        print(
            f"[WARN] prepare_supercalc_dataset failed ({exc!r}); falling back to raw OHLCV loader",
            flush=True,
        )
        return _fallback_dataset(cfg, args, exc)


def _fallback_dataset(
    cfg: BacktestConfig,
    args: argparse.Namespace,
    original_exc: Exception,
) -> SupercalcDataset:
    try:
        fetcher = MarketDataFetcher(cfg.ticker)
        prices = fetcher.get_prices(interval=cfg.interval, period=args.period)
    except Exception as fallback_exc:
        raise SystemExit(
            "Failed to build dataset via supercalc and fallback price loader: "
            f"original={original_exc!r}, fallback={fallback_exc!r}"
        ) from fallback_exc

    prices = prices.sort_index()
    if prices.empty:
        raise SystemExit(
            "Fallback loader produced empty dataframe; cannot proceed with native smoke"
        )

    first_close = float(prices["close"].iloc[0])
    last_close = float(prices["close"].iloc[-1])
    benchmark_return = 0.0
    if cfg.initial_balance > 0 and first_close > 0:
        buy_hold = cfg.initial_balance * (last_close / first_close)
        benchmark_return = (buy_hold / cfg.initial_balance - 1.0) * 100.0

    return SupercalcDataset(
        rows=[],
        prices=prices,
        benchmark_return_pct=float(benchmark_return),
        initial_balance=float(cfg.initial_balance),
    )


def main() -> None:
    if not native_available():
        raise SystemExit("hyprl_supercalc native module is not available. Build it via scripts/build_supercalc.sh first.")

    args = _parse_args()
    cfg = _build_config(args)
    dataset = _build_dataset(cfg, args)
    python_stats = runner.simulate_from_dataset(dataset, cfg)
    signal, _, _ = _build_signal_series(dataset, cfg)
    native_result = run_backtest_native(dataset.prices, signal, cfg)

    py_metrics = _python_metrics(python_stats, cfg, args.bootstrap_runs)
    native_metrics = native_result.native_metrics or {}

    print("=== Python StrategyStats vs Native BacktestResult ===")
    print(json.dumps({
        "python": py_metrics,
        "native": {
            "final_balance": native_result.final_balance,
            "profit_factor": native_result.profit_factor,
            "sharpe": native_result.sharpe_ratio,
            "max_drawdown_pct": native_result.max_drawdown * 100.0,
            "expectancy": native_result.expectancy,
            "win_rate": native_result.win_rate,
            "risk_of_ruin": native_result.risk_of_ruin,
            "maxdd_p95": native_metrics.get("maxdd_p95"),
            "pnl_p05": native_metrics.get("pnl_p05"),
            "pnl_p95": native_metrics.get("pnl_p95"),
            "robustness_score": native_metrics.get("robustness_score"),
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

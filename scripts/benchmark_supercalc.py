#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import time
from typing import Sequence

import numpy as np
import polars as pl

from hyprl.backtest.runner import BacktestConfig
from hyprl.native.supercalc import native_available, run_backtest_native
from hyprl.risk.manager import RiskConfig


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def compute_atr(df: pl.DataFrame, window: int) -> list[float]:
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    closes = df["close"].to_list()
    n = len(highs)
    if n == 0:
        return []
    window = max(int(window or 1), 1)
    tr = []
    for i in range(n):
        if i == 0:
            tr.append(max(highs[i] - lows[i], 0.0))
        else:
            tr.append(
                max(
                    highs[i] - lows[i],
                    abs(highs[i] - closes[i - 1]),
                    abs(lows[i] - closes[i - 1]),
                )
            )
    atr = [0.0] * n
    init = min(window, n)
    running = 0.0
    for i in range(init):
        running += tr[i]
        atr[i] = running / (i + 1)
    prev = atr[init - 1]
    for i in range(init, n):
        prev = ((prev * (window - 1)) + tr[i]) / window
        atr[i] = prev
    return atr


def transaction_cost(equity: float, delta: float, cfg: dict, atr_mode: bool) -> float:
    if delta <= 1e-12:
        return 0.0
    roundtrip = max(cfg["commission_pct"] + cfg["slippage_pct"], 0.0)
    if roundtrip <= 0.0:
        return 0.0
    if atr_mode:
        notional = equity * delta
    else:
        notional = equity * cfg["risk_pct"] * delta
    return notional * roundtrip


def compute_py_atr_sizing(equity, entry_price, atr_value, cfg, leverage_cap):
    if entry_price <= 0.0 or atr_value <= 0.0:
        return None
    stop_dist = cfg["atr_mult_stop"] * atr_value
    tp_dist = cfg["atr_mult_tp"] * atr_value
    if stop_dist <= 0.0 or tp_dist <= 0.0:
        return None
    risk_amount = equity * cfg["risk_pct"]
    if risk_amount <= 0.0:
        return None
    units = risk_amount / stop_dist
    notional = units * entry_price
    if notional <= 0.0:
        return None
    leverage = min(notional / equity, leverage_cap)
    if leverage <= 0.0:
        return None
    return leverage, stop_dist, tp_dist


def py_equity_curve_from_df(df: pl.DataFrame, signal: Sequence[float], cfg: dict):
    closes = df["close"].to_list()
    highs = df["high"].to_list()
    lows = df["low"].to_list()
    ts = df["ts"].to_list()
    if len(closes) != len(signal):
        raise ValueError("Signal length must match dataframe rows")

    equity = 1.0
    position = 0.0
    leverage_cap = max(cfg.get("max_leverage", 1.0), 1.0)
    use_atr = cfg.get("use_atr_position_sizing", False)
    atr_vals = compute_atr(df, cfg.get("atr_window", 14)) if use_atr else None
    stop_price = None
    take_profit_price = None

    equity_curve = [{"ts": ts[0], "equity": equity}]
    n_trades = 0

    for i in range(1, len(closes)):
        prev_close = closes[i - 1]
        close = closes[i]
        high = highs[i]
        low = lows[i]
        closed_this_bar = False

        if use_atr and position != 0.0 and stop_price is not None and take_profit_price is not None:
            exit_price = None
            if position > 0.0:
                if low <= stop_price:
                    exit_price = stop_price
                elif high >= take_profit_price:
                    exit_price = take_profit_price
            else:
                if high >= stop_price:
                    exit_price = stop_price
                elif low <= take_profit_price:
                    exit_price = take_profit_price
            if exit_price is not None:
                ret = exit_price / prev_close - 1.0 if prev_close > 0.0 else 0.0
                equity *= 1.0 + position * ret
                equity = max(equity, 1e-12)
                cost = transaction_cost(equity, abs(position), cfg, use_atr)
                equity = max(equity - cost, 1e-12)
                position = 0.0
                stop_price = None
                take_profit_price = None
                n_trades += 1
                closed_this_bar = True

        desired_pos = clamp(signal[i], -1.0, 1.0)
        if not cfg.get("allow_short", False):
            desired_pos = max(0.0, desired_pos)

        target_position = position
        if use_atr:
            if closed_this_bar:
                desired_pos = 0.0
            if desired_pos > 1e-9:
                desired_sign = 1.0
            elif desired_pos < -1e-9:
                desired_sign = -1.0
            else:
                desired_sign = 0.0
            if desired_sign == 0.0:
                target_position = 0.0
            elif position == 0.0 or math.copysign(1.0, position) != desired_sign:
                atr_value = atr_vals[i] if atr_vals else 0.0
                sizing = compute_py_atr_sizing(equity, prev_close, atr_value, cfg, leverage_cap)
                if sizing:
                    leverage, stop_dist, tp_dist = sizing
                    target_position = desired_sign * leverage
                    if target_position > 0.0:
                        stop_price = prev_close - stop_dist
                        take_profit_price = prev_close + tp_dist
                    elif target_position < 0.0:
                        stop_price = prev_close + stop_dist
                        take_profit_price = prev_close - tp_dist
                    else:
                        stop_price = None
                        take_profit_price = None
                else:
                    target_position = desired_sign * leverage_cap
                    stop_price = None
                    take_profit_price = None
        else:
            desired_pos = clamp(desired_pos, -leverage_cap, leverage_cap)
            target_position = desired_pos

        delta_pos = target_position - position
        if abs(delta_pos) > 1e-9:
            cost = transaction_cost(equity, abs(delta_pos), cfg, use_atr)
            equity = max(equity - cost, 1e-12)
            position = target_position
            n_trades += 1
            if use_atr and abs(position) <= 1e-9:
                stop_price = None
                take_profit_price = None

        if prev_close > 0.0:
            r = close / prev_close - 1.0
            equity *= 1.0 + position * r
        equity = max(equity, 1e-12)

        equity_curve.append({"ts": ts[i], "equity": equity})

    return equity_curve, n_trades


def equity_to_returns(equity_curve):
    if len(equity_curve) < 2:
        return []
    out = []
    for i in range(1, len(equity_curve)):
        prev = equity_curve[i - 1]["equity"]
        curr = equity_curve[i]["equity"]
        out.append(curr / prev - 1.0 if prev > 0.0 else 0.0)
    return out


def total_return(ec):
    if len(ec) < 2:
        return 0.0
    first = ec[0]["equity"]
    last = ec[-1]["equity"]
    return last / first - 1.0 if first > 0.0 else 0.0


def cagr(ec):
    if len(ec) < 2:
        return 0.0
    first = ec[0]
    last = ec[-1]
    dt_ms = float(last["ts"] - first["ts"])
    if dt_ms <= 0.0:
        return 0.0
    years = dt_ms / (365.25 * 24 * 3600 * 1000.0)
    if years <= 0.0:
        return 0.0
    tr = total_return(ec)
    if tr <= -1.0:
        return -1.0
    return (1.0 + tr) ** (1.0 / years) - 1.0


def max_drawdown(ec):
    if not ec:
        return 0.0, 0
    peak = ec[0]["equity"]
    max_dd = 0.0
    max_dd_dur = 0
    peak_idx = 0
    for i, pt in enumerate(ec):
        if pt["equity"] > peak:
            peak = pt["equity"]
            peak_idx = i
        if peak > 0.0:
            dd = pt["equity"] / peak - 1.0
            if dd < max_dd:
                max_dd = dd
                max_dd_dur = i - peak_idx
    return max_dd, max_dd_dur


def sharpe_ratio(returns, risk_free=0.0):
    if not returns:
        return 0.0
    sum_ex = 0.0
    sum_sq = 0.0
    for r in returns:
        ex = r - risk_free
        sum_ex += ex
        sum_sq += ex * ex
    n = float(len(returns))
    mean = sum_ex / n
    var = sum_sq / n - mean * mean
    std = math.sqrt(var) if var > 0.0 else 0.0
    return 0.0 if std == 0.0 else mean / std


def profit_factor(returns):
    gp = 0.0
    gl = 0.0
    for r in returns:
        if r > 0.0:
            gp += r
        elif r < 0.0:
            gl += r
    if gl == 0.0:
        return float("inf") if gp > 0.0 else 0.0
    return gp / abs(gl)


def win_rate_and_expectancy(returns):
    if not returns:
        return 0.0, 0.0
    wins = 0.0
    total = 0.0
    for r in returns:
        total += r
        if r > 0.0:
            wins += 1.0
    n = float(len(returns))
    win_rate = wins / n if n > 0.0 else 0.0
    expectancy = total / n if n > 0.0 else 0.0
    return win_rate, expectancy


def py_metrics_from_equity(ec):
    tr = total_return(ec)
    cagr_val = cagr(ec)
    max_dd, _ = max_drawdown(ec)
    rets = equity_to_returns(ec)
    sharpe = sharpe_ratio(rets, 0.0)
    pf = profit_factor(rets)
    win_rate, expectancy = win_rate_and_expectancy(rets)
    return {
        "total_return": tr,
        "cagr": cagr_val,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "profit_factor": pf,
        "win_rate": win_rate,
        "expectancy": expectancy,
    }


def _synthetic_ohlcv(rows: int, seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    base = np.cumsum(rng.normal(0.0, 0.6, size=rows)) + 100.0
    noise = rng.normal(0.0, 0.2, size=rows)
    open_ = base + noise
    close = base - noise
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.3, size=rows))
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.3, size=rows))
    volume = np.linspace(1_000.0, 1_000.0 + rows, rows)
    ts = (np.arange(rows, dtype=np.int64) * 3_600_000).tolist()
    return pl.DataFrame(
        {
            "ts": ts,
            "open": open_.tolist(),
            "high": high.tolist(),
            "low": low.tolist(),
            "close": close.tolist(),
            "volume": volume.tolist(),
        }
    )


def _synthetic_signal(length: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    signal: list[float] = []
    for i in range(length):
        value = math.sin(i / 5.0 + seed * 0.01) + rng.normal(0.0, 0.15)
        if value > 0.35:
            signal.append(1.0)
        elif value < -0.35:
            signal.append(-1.0)
        else:
            signal.append(0.0)
    return signal


def _config_dicts(count: int) -> list[dict]:
    configs: list[dict] = []
    for idx in range(count):
        risk_pct = 0.005 + 0.002 * (idx % 3)
        atr_stop = 1.25 + 0.25 * (idx % 4)
        reward_multiple = 1.5 + 0.5 * (idx % 3)
        configs.append(
            {
                "risk_pct": risk_pct,
                "commission_pct": 0.0005,
                "slippage_pct": 0.0005,
                "max_leverage": 2.0,
                "params": [],
                "allow_short": bool(idx % 2 == 0),
                "label": f"cfg_{idx}",
                "atr_window": 14,
                "atr_mult_stop": atr_stop,
                "atr_mult_tp": atr_stop * reward_multiple,
                "use_atr_position_sizing": True,
            }
        )
    return configs


def _build_bt_config(cfg: dict, args: argparse.Namespace) -> BacktestConfig:
    stop_mult = cfg["atr_mult_stop"]
    tp_mult = cfg["atr_mult_tp"]
    reward_multiple = tp_mult / stop_mult if stop_mult > 0 else 2.0
    risk_cfg = RiskConfig(
        balance=args.initial_balance,
        risk_pct=cfg["risk_pct"],
        atr_multiplier=stop_mult,
        reward_multiple=reward_multiple,
        min_position_size=1,
    )
    short_threshold = 0.4 if cfg["allow_short"] else 0.0
    return BacktestConfig(
        ticker=args.ticker,
        period=args.period,
        interval=args.interval,
        initial_balance=args.initial_balance,
        long_threshold=0.6,
        short_threshold=short_threshold,
        risk=risk_cfg,
        commission_pct=cfg["commission_pct"],
        slippage_pct=cfg["slippage_pct"],
        atr_window=cfg["atr_window"],
    )


def _run_python_reference(df: pl.DataFrame, signal: Sequence[float], cfg_dicts: list[dict]):
    metrics: list[dict] = []
    start = time.perf_counter()
    for cfg in cfg_dicts:
        equity_curve, _ = py_equity_curve_from_df(df, signal, cfg)
        metrics.append(py_metrics_from_equity(equity_curve))
    duration = time.perf_counter() - start
    return metrics, duration


def _run_native(df: pl.DataFrame, signal: Sequence[float], bt_cfgs: list[BacktestConfig]):
    metrics: list[dict] = []
    start = time.perf_counter()
    for cfg in bt_cfgs:
        result = run_backtest_native(df, signal, cfg)
        total_return = result.final_balance / cfg.initial_balance - 1.0 if cfg.initial_balance else 0.0
        metrics.append(
            {
                "total_return": total_return,
                "cagr": result.annualized_return or 0.0,
                "sharpe": result.sharpe_ratio or 0.0,
                "max_drawdown": -result.max_drawdown,
                "profit_factor": result.profit_factor or 0.0,
                "win_rate": result.win_rate,
                "expectancy": result.expectancy,
            }
        )
    duration = time.perf_counter() - start
    return metrics, duration


def _summarize_deltas(py_metrics: list[dict], native_metrics: list[dict]) -> dict[str, tuple[float, float]]:
    keys = ["total_return", "cagr", "sharpe", "max_drawdown", "profit_factor", "win_rate", "expectancy"]
    summary: dict[str, tuple[float, float]] = {}
    for key in keys:
        diffs = [abs(py[key] - nv[key]) for py, nv in zip(py_metrics, native_metrics)]
        summary[key] = (
            float(sum(diffs) / len(diffs)) if diffs else 0.0,
            float(max(diffs)) if diffs else 0.0,
        )
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Python vs native hyprl_supercalc engines.")
    parser.add_argument("--rows", type=int, default=2048, help="Number of synthetic OHLCV rows to generate.")
    parser.add_argument("--configs", type=int, default=12, help="Number of config points to benchmark.")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the synthetic generators.")
    parser.add_argument("--ticker", default="BENCH", help="Synthetic ticker label for BacktestConfig.")
    parser.add_argument("--period", default="1y", help="Dummy period assigned to BacktestConfig.")
    parser.add_argument("--interval", default="1h", help="Interval label for BacktestConfig.")
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=100_000.0,
        help="Initial balance forwarded to BacktestConfig / RiskConfig.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not native_available():
        raise SystemExit("hyprl_supercalc is not available. Build it via scripts/build_supercalc.sh before benchmarking.")

    df = _synthetic_ohlcv(args.rows, args.seed)
    signal = _synthetic_signal(df.height, args.seed)
    cfg_dicts = _config_dicts(args.configs)
    bt_cfgs = [_build_bt_config(cfg, args) for cfg in cfg_dicts]

    py_metrics, py_time = _run_python_reference(df, signal, cfg_dicts)
    native_metrics, native_time = _run_native(df, signal, bt_cfgs)

    print(f"Python reference: {py_time:.4f}s for {len(cfg_dicts)} configs")
    print(f"Native engine:   {native_time:.4f}s for {len(cfg_dicts)} configs")
    if native_time > 0:
        print(f"Speedup:        {py_time / native_time:.2f}x")

    summary = _summarize_deltas(py_metrics, native_metrics)
    print("\nMetric deltas (mean abs / max abs):")
    for key, (mean_delta, max_delta) in summary.items():
        print(f"  {key:>13}: mean={mean_delta:.6f}  max={max_delta:.6f}")

    return 0


if __name__ == "__main__":  # pragma: no cover - manual entry point
    raise SystemExit(main())

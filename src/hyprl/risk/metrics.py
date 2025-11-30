from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def compute_basic_stats(trade_returns: Sequence[float]) -> dict[str, float]:
    """
    Compute expectancy, variance, win rate, and gain/loss ratios from trade returns.
    """

    arr = np.asarray(list(trade_returns), dtype=float)
    if arr.size == 0:
        return {
            "expectancy": 0.0,
            "variance": 0.0,
            "std": 0.0,
            "win_rate": 0.0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "avg_win_loss_ratio": 0.0,
        }

    expectancy = float(arr.mean())
    variance = float(arr.var(ddof=0))
    std = float(arr.std(ddof=0))
    winners = arr[arr > 0.0]
    losers = arr[arr < 0.0]
    win_rate = float((arr > 0.0).mean())
    avg_win = float(winners.mean()) if winners.size else 0.0
    avg_loss = float(losers.mean()) if losers.size else 0.0
    if avg_loss >= 0.0:
        win_loss_ratio = float("inf") if avg_win > 0 else 0.0
    elif avg_win <= 0.0:
        win_loss_ratio = 0.0
    else:
        win_loss_ratio = avg_win / abs(avg_loss)
    return {
        "expectancy": expectancy,
        "variance": variance,
        "std": std,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "avg_win_loss_ratio": win_loss_ratio,
    }


def compute_risk_of_ruin(
    trade_returns: Sequence[float],
    initial_capital: float,
    risk_per_trade: float,
) -> float:
    """
    Approximate the probability of ruin using a simplified gambler's ruin model.

    The formula assumes IID trade outcomes and uses win rate plus win/loss ratio
    to estimate the edge. It is a conservative approximation; values are clipped
    to [0, 1].
    """

    stats = compute_basic_stats(trade_returns)
    p = stats["win_rate"]
    q = 1.0 - p
    ratio = stats["avg_win_loss_ratio"]
    if initial_capital <= 0.0 or risk_per_trade <= 0.0:
        return 1.0
    if ratio <= 0.0 or p <= 0.0:
        return 1.0
    if not np.isfinite(ratio):
        if stats["avg_win"] > 0.0 and stats["avg_loss"] == 0.0:
            return 0.0
        ratio = 1e6
    if stats["avg_loss"] == 0.0 and stats["avg_win"] > 0.0:
        return 0.0
    edge = p * ratio - q
    if edge <= 0.0:
        return 1.0
    capital_units = max(initial_capital / risk_per_trade, 1.0)
    base = q / (p * ratio)
    base = float(np.clip(base, 0.0, 0.999999))
    return float(np.clip(base**capital_units, 0.0, 1.0))


@dataclass(slots=True)
class BootstrapSummary:
    maxdd_p95: float
    maxdd_p99: float
    pnl_p05: float
    pnl_p50: float
    pnl_p95: float


def bootstrap_equity_drawdowns(
    trade_returns: Sequence[float],
    n_runs: int = 512,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, BootstrapSummary]:
    """
    Run bootstrap simulations on the trade return series.

    Returns
    -------
    maxdd_distribution:
        Array of maximum drawdown ratios per simulation.
    pnl_distribution:
        Array of ending PnL (in return units) per simulation.
    summary:
        Dataclass containing selected quantiles.
    """

    arr = np.asarray(list(trade_returns), dtype=float)
    if arr.size == 0 or n_runs <= 0:
        empty = np.zeros(0, dtype=float)
        summary = BootstrapSummary(0.0, 0.0, 0.0, 0.0, 0.0)
        return empty, empty, summary

    rng = np.random.default_rng(seed)
    maxdds = np.zeros(n_runs, dtype=float)
    pnl = np.zeros(n_runs, dtype=float)
    for i in range(n_runs):
        sample = rng.choice(arr, size=arr.size, replace=True)
        equity = 1.0 + np.concatenate(([0.0], np.cumsum(sample)))
        running_max = np.maximum.accumulate(equity)
        drawdowns = np.divide(
            running_max - equity,
            running_max,
            out=np.zeros_like(equity),
            where=running_max > 0,
        )
        maxdds[i] = float(np.max(drawdowns))
        pnl[i] = float(equity[-1] - 1.0)

    summary = BootstrapSummary(
        maxdd_p95=float(np.quantile(maxdds, 0.95)),
        maxdd_p99=float(np.quantile(maxdds, 0.99)),
        pnl_p05=float(np.quantile(pnl, 0.05)),
        pnl_p50=float(np.quantile(pnl, 0.5)),
        pnl_p95=float(np.quantile(pnl, 0.95)),
    )
    return maxdds, pnl, summary

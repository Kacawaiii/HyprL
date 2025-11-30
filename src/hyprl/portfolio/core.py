from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from hyprl.risk.metrics import (
    bootstrap_equity_drawdowns,
    compute_risk_of_ruin,
)


def compute_portfolio_weights(
    equity_curves: Mapping[str, pd.Series],
    scheme: str = "equal",
    vol_window: int = 20,
) -> dict[str, float]:
    """
    Derive portfolio weights per ticker.

    Parameters
    ----------
    equity_curves:
        Mapping ticker -> equity curve (must be positive, aligned index).
    scheme:
        "equal" (default) or "inv_vol".
    vol_window:
        Lookback window for volatility in periods (inv_vol only).
    """

    if not equity_curves:
        return {}

    cleaned: dict[str, pd.Series] = {}
    for ticker, series in equity_curves.items():
        if series is None:
            continue
        trimmed = series.dropna().sort_index()
        if trimmed.empty:
            continue
        cleaned[ticker] = trimmed

    if not cleaned:
        return {}

    tickers = list(cleaned.keys())
    if len(tickers) == 1:
        return {tickers[0]: 1.0}

    scheme = (scheme or "equal").lower()
    if scheme != "inv_vol":
        weight = 1.0 / len(tickers)
        return {ticker: weight for ticker in tickers}

    vol_window = max(int(vol_window or 0), 1)
    raw_weights: dict[str, float] = {}
    for ticker, series in cleaned.items():
        returns = series.pct_change().dropna()
        if returns.empty:
            raw_weights[ticker] = 0.0
            continue
        if len(returns) > vol_window:
            returns = returns.tail(vol_window)
        sigma = float(returns.std(ddof=1))
        if not np.isfinite(sigma) or sigma <= 0.0:
            raw_weights[ticker] = 0.0
        else:
            raw_weights[ticker] = 1.0 / sigma

    total = float(sum(raw_weights.values()))
    if total <= 0.0 or not np.isfinite(total):
        weight = 1.0 / len(tickers)
        return {ticker: weight for ticker in tickers}

    return {ticker: raw_weights[ticker] / total for ticker in tickers}


def build_portfolio_equity(
    per_ticker_equities: Mapping[str, pd.Series],
    total_capital: float,
    weights: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """
    Combine equity curves into a single weighted portfolio series.
    All series must be positive and share comparable timestamps.
    """

    if not per_ticker_equities:
        return pd.Series(dtype=float)

    valid_entries: list[tuple[str, pd.Series]] = []
    for ticker, series in per_ticker_equities.items():
        if series is None:
            continue
        cleaned = series.dropna().sort_index()
        if cleaned.empty:
            continue
        initial = float(cleaned.iloc[0])
        if initial <= 0:
            continue
        valid_entries.append((ticker, cleaned))

    if not valid_entries:
        return pd.Series(dtype=float)

    tickers = [ticker for ticker, _ in valid_entries]
    n = len(tickers)
    if n == 1:
        ticker, single_series = valid_entries[0]
        capital_share = total_capital
        normalized = single_series / float(single_series.iloc[0]) * capital_share
        normalized.name = ticker
        return normalized

    if weights is None:
        weights_array = np.full(n, 1.0 / n, dtype=float)
    else:
        weights_array = np.array([float(weights.get(ticker, 0.0)) for ticker in tickers], dtype=float)
        total_weights = weights_array.sum()
        if total_weights <= 0 or not np.isfinite(total_weights):
            weights_array = np.full(n, 1.0 / n, dtype=float)
        else:
            weights_array = weights_array / total_weights

    aligned_frames = []
    for idx, (ticker, series) in enumerate(valid_entries):
        capital_share = total_capital * weights_array[idx]
        normalized = series / float(series.iloc[0]) * capital_share
        aligned_frames.append(normalized)

    combined = pd.concat(aligned_frames, axis=1).ffill().bfill()
    combined.columns = tickers
    portfolio = combined.sum(axis=1)
    return portfolio


def compute_portfolio_stats(
    portfolio_equity: pd.Series,
    initial_balance: float,
    seed: int | None = None,
    bootstrap_runs: int = 512,
) -> dict[str, float]:
    """
    Compute aggregate metrics (PF, Sharpe, maxDD, ROR, etc.) from a portfolio equity series.
    """

    if portfolio_equity.empty:
        return {
            "final_balance": initial_balance,
            "return_pct": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "max_drawdown_pct": 0.0,
            "risk_of_ruin": 1.0,
            "maxdd_p95": 0.0,
            "pnl_p05": 0.0,
        }

    equity = portfolio_equity.sort_index()
    initial = float(equity.iloc[0])
    final_balance = float(equity.iloc[-1])
    diffs = equity.diff().dropna()
    returns = equity.pct_change().dropna()
    positive = diffs[diffs > 0].sum()
    negative = diffs[diffs < 0].sum()
    if negative < 0:
        profit_factor = float(positive / abs(negative))
    elif positive > 0:
        profit_factor = float("inf")
    else:
        profit_factor = 0.0
    sharpe = (
        float((returns.mean() / returns.std(ddof=1)) * np.sqrt(len(returns)))
        if len(returns) > 1 and returns.std(ddof=1) > 0
        else 0.0
    )
    drawdowns = (equity.cummax() - equity) / equity.cummax().replace(0.0, np.nan)
    max_dd_pct = float(drawdowns.max() * 100.0)
    ruin_series = returns.tolist() if not returns.empty else [0.0]
    risk_of_ruin = compute_risk_of_ruin(
        ruin_series,
        initial_capital=1.0,
        risk_per_trade=max(0.01, 1e-6),
    )
    _, _, bootstrap_summary = bootstrap_equity_drawdowns(
        ruin_series,
        n_runs=bootstrap_runs,
        seed=seed,
    )
    return {
        "final_balance": final_balance,
        "return_pct": ((final_balance / initial_balance) - 1.0) * 100.0 if initial_balance > 0 else 0.0,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd_pct,
        "risk_of_ruin": risk_of_ruin,
        "maxdd_p95": float(bootstrap_summary.maxdd_p95 * 100.0),
        "pnl_p05": float(bootstrap_summary.pnl_p05),
    }


def compute_correlation_matrix(per_ticker_returns: Mapping[str, pd.Series]) -> pd.DataFrame:
    """
    Compute correlation matrix between tickers given return series aligned by timestamp.
    """

    if not per_ticker_returns:
        return pd.DataFrame()
    df = pd.concat(
        [series.sort_index() for series in per_ticker_returns.values()],
        axis=1,
        join="outer",
    ).ffill().bfill()
    df.columns = list(per_ticker_returns.keys())
    if df.shape[0] < 2:
        return pd.DataFrame(np.eye(df.shape[1]), index=df.columns, columns=df.columns)
    return df.corr().fillna(0.0)

"""Trade aggregation and metric computation for HyprL gate checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


MANDATORY_COLUMNS = {
    "entry_ts",
    "exit_ts",
    "symbol",
    "side",
    "qty",
    "entry_price",
    "exit_price",
    "pnl",
    "pnl_pct",
    "exit_reason",
    "source_type",
    "session_id",
    "strategy_label",
    "strategy_id",
}


_ALIAS_COLUMNS = {
    "entry_timestamp": "entry_ts",
    "exit_timestamp": "exit_ts",
    "return_pct": "pnl_pct",
    "position_size": "qty",
    "direction": "side",
}


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    converted = pd.to_datetime(series, utc=True, errors="coerce")
    return converted


def _coerce_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _normalize_aliases(df: pd.DataFrame) -> pd.DataFrame:
    renamed = {col: _ALIAS_COLUMNS[col] for col in df.columns if col in _ALIAS_COLUMNS}
    if renamed:
        df = df.rename(columns=renamed)
    return df


def _ensure_columns(
    df: pd.DataFrame,
    default_source: str | None,
    default_symbol: str | None,
    default_session: str | None,
) -> pd.DataFrame:
    df = _normalize_aliases(df)
    missing = MANDATORY_COLUMNS - set(df.columns)
    # Fill reasonable defaults for some missing fields
    df = df.copy()
    if "symbol" not in df.columns and default_symbol is not None:
        df["symbol"] = default_symbol
        missing.discard("symbol")
    if "session_id" not in df.columns and default_session is not None:
        df["session_id"] = default_session
        missing.discard("session_id")
    if missing:
        raise ValueError(f"Missing mandatory columns: {sorted(missing)}")

    df["entry_ts"] = _coerce_timestamp(df["entry_ts"])
    df["exit_ts"] = _coerce_timestamp(df["exit_ts"])
    for col in ("qty", "entry_price", "exit_price", "pnl", "pnl_pct"):
        df[col] = _coerce_float(df[col])

    if default_source:
        df["source_type"] = df["source_type"].fillna(default_source)
    df["source_type"] = df["source_type"].fillna("unknown").astype(str)
    df["strategy_id"] = df["strategy_id"].astype(str)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["side"] = df["side"].astype(str).str.upper()
    if "side" in df.columns:
        df.loc[df["side"] == "LONG", "side"] = "LONG"
        df.loc[df["side"] == "SHORT", "side"] = "SHORT"
    return df


def load_trades(
    paths: Iterable[str | Path],
    default_source: str | None = None,
    default_symbol: str | None = None,
    default_session: str | None = None,
) -> pd.DataFrame:
    """Load and normalize multiple trade CSV files into a single DataFrame."""
    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df = _ensure_columns(df, default_source, default_symbol, default_session)
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=sorted(MANDATORY_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values(by=["exit_ts", "entry_ts"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


def group_trades_by_strategy(trades: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return trades grouped by strategy_id, sorted by exit_ts then entry_ts."""
    if trades.empty:
        return {}
    grouped: dict[str, pd.DataFrame] = {}
    for strategy_id, sub in trades.groupby("strategy_id"):
        ordered = sub.sort_values(by=["exit_ts", "entry_ts"]).reset_index(drop=True)
        grouped[strategy_id] = ordered
    return grouped


def _profit_factor(pnl: np.ndarray) -> float:
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _max_drawdown(pnl: np.ndarray) -> float:
    if pnl.size == 0:
        return 0.0
    equity = np.cumsum(pnl)
    peaks = np.maximum.accumulate(equity)
    drawdowns = peaks - equity
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peaks != 0, drawdowns / peaks, 0.0)
    return float(np.nanmax(dd_pct) if dd_pct.size else 0.0)


def _rolling_metrics(pnl: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    if pnl.size < window:
        return np.array([]), np.array([])
    pf_values = []
    dd_values = []
    for i in range(window, pnl.size + 1):
        window_slice = pnl[i - window : i]
        pf_values.append(_profit_factor(window_slice))
        dd_values.append(_max_drawdown(window_slice))
    return np.array(pf_values), np.array(dd_values)


def _source_mix(trades: pd.DataFrame) -> str:
    counts = trades["source_type"].value_counts(normalize=True)
    parts = [f"{pct*100:.1f}% {src}" for src, pct in counts.items()]
    return " / ".join(parts)


@dataclass
class StrategyMetrics:
    strategy_id: str
    n_trades: int
    pf: float
    sharpe: float
    maxdd: float
    expectancy: float
    winrate: float
    source_mix: str
    last_exit_ts: pd.Timestamp | None


@dataclass
class RollingMetrics:
    pf_min_100: float | None
    maxdd_max_100: float | None
    pf_series_100: np.ndarray
    maxdd_series_100: np.ndarray


def _equity_curve(pnl: np.ndarray, initial_equity: float) -> np.ndarray:
    equity = initial_equity + np.cumsum(pnl, dtype=np.float64)
    return equity


def _equity_from_returns(returns: np.ndarray, initial_equity: float) -> np.ndarray:
    equity = [float(initial_equity)]
    eq = float(initial_equity)
    for r in returns:
        eq *= 1.0 + float(r)
        equity.append(eq)
    return np.array(equity, dtype=np.float64)


def _pnl_from_returns(returns: np.ndarray, equity: np.ndarray) -> np.ndarray:
    # equity[i] is equity before trade i; returns and pnl have same length
    if returns.size == 0:
        return returns
    equity_prev = equity[:-1]
    return equity_prev * returns


def _drawdown_from_equity(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peaks = np.maximum.accumulate(equity)
    dd = np.where(peaks != 0, (peaks - equity) / peaks, 0.0)
    return float(np.nanmax(dd) if dd.size else 0.0)


def compute_metrics(trades: pd.DataFrame, initial_equity: float = 10_000.0) -> StrategyMetrics:
    """Compute core metrics for a strategy."""
    if trades.empty:
        raise ValueError("Cannot compute metrics on empty trades")
    n_trades = len(trades)

    returns = None
    if "pnl_pct" in trades.columns:
        pnl_pct = pd.to_numeric(trades["pnl_pct"], errors="coerce")
        if not pnl_pct.isna().all():
            returns = pnl_pct.to_numpy(np.float64)

    if returns is not None:
        equity_path = _equity_from_returns(returns, initial_equity)
        pnl_series = _pnl_from_returns(returns, equity_path)
    else:
        pnl_series = trades["pnl"].to_numpy(np.float64)
        equity_path = _equity_curve(pnl_series, initial_equity)
        equity_prev = initial_equity + np.cumsum(np.insert(pnl_series[:-1], 0, 0.0))
        returns = np.divide(
            pnl_series, equity_prev, out=np.full_like(pnl_series, np.nan), where=equity_prev != 0
        )

    pf = _profit_factor(pnl_series)
    if returns.size == 0:
        sharpe = float("nan")
        expectancy = float("nan")
    else:
        mean_ret = float(np.nanmean(returns))
        std_ret = float(np.nanstd(returns, ddof=1)) if n_trades > 1 else 0.0
        sharpe = mean_ret / std_ret if std_ret > 0 else float("nan")
        expectancy = mean_ret

    maxdd = _drawdown_from_equity(equity_path)
    winrate = float(np.nanmean(pnl_series > 0)) if n_trades else float("nan")
    mix = _source_mix(trades)
    last_exit_ts = trades["exit_ts"].max() if "exit_ts" in trades.columns else None

    return StrategyMetrics(
        strategy_id=str(trades["strategy_id"].iloc[0]),
        n_trades=n_trades,
        pf=pf,
        sharpe=sharpe,
        maxdd=maxdd,
        expectancy=expectancy,
        winrate=winrate,
        source_mix=mix,
        last_exit_ts=last_exit_ts,
    )


def compute_rolling(trades: pd.DataFrame, window: int = 100, initial_equity: float = 10_000.0) -> RollingMetrics:
    """Compute rolling PF and MaxDD for a strategy."""
    returns = None
    if "pnl_pct" in trades.columns:
        pnl_pct = pd.to_numeric(trades["pnl_pct"], errors="coerce")
        if not pnl_pct.isna().all():
            returns = pnl_pct.to_numpy(np.float64)

    if returns is not None:
        equity_path = _equity_from_returns(returns, initial_equity)
        if returns.size < window:
            pf_series = np.array([])
            dd_series = np.array([])
        else:
            pf_series = []
            dd_series = []
            for i in range(window, returns.size + 1):
                window_returns = returns[i - window : i]
                start_equity = equity_path[i - window]
                pnl_window = _pnl_from_returns(window_returns, equity_path[i - window : i + 1])
                equity_window = _equity_from_returns(window_returns, start_equity)
                pf_series.append(_profit_factor(pnl_window))
                dd_series.append(_drawdown_from_equity(equity_window))
            pf_series = np.array(pf_series)
            dd_series = np.array(dd_series)
    else:
        pnl = trades["pnl"].to_numpy(np.float64)
        if pnl.size < window:
            pf_series = np.array([])
            dd_series = np.array([])
        else:
            pf_series = []
            dd_series = []
            for i in range(window, pnl.size + 1):
                window_slice = pnl[i - window : i]
                start_equity = initial_equity + pnl[: i - window].sum()
                equity_window = _equity_curve(window_slice, start_equity)
                pf_series.append(_profit_factor(window_slice))
                dd_series.append(_drawdown_from_equity(equity_window))
            pf_series = np.array(pf_series)
            dd_series = np.array(dd_series)
    pf_min = float(np.nanmin(pf_series)) if pf_series.size else None
    dd_max = float(np.nanmax(dd_series)) if dd_series.size else None
    return RollingMetrics(
        pf_min_100=pf_min,
        maxdd_max_100=dd_max,
        pf_series_100=pf_series,
        maxdd_series_100=dd_series,
    )

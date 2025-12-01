#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_equity_csv(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "equity" in df.columns and "timestamp" in df.columns:
        series = pd.Series(df["equity"].astype(float).values, index=pd.to_datetime(df["timestamp"]))
    elif "equity_after" in df.columns and "exit_timestamp" in df.columns:
        series = pd.Series(df["equity_after"].astype(float).values, index=pd.to_datetime(df["exit_timestamp"]))
    elif "equity" in df.columns:
        series = pd.Series(df["equity"].astype(float).values)
    else:
        raise SystemExit(f"Cannot find equity column in {path}")
    series = series.sort_index()
    return series.ffill()


def compute_returns(equity: pd.Series) -> pd.Series:
    equity = equity.replace([np.inf, -np.inf], np.nan).dropna()
    returns = equity.pct_change().dropna()
    return returns


def _annualized_sharpe(returns: pd.Series, bars_per_year: float) -> float:
    if returns.std(ddof=1) == 0 or returns.empty:
        return float("nan")
    return float((returns.mean() / returns.std(ddof=1)) * np.sqrt(bars_per_year))


def _max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return float("nan")
    peaks = equity_curve.cummax()
    dd = (peaks - equity_curve) / peaks
    return float(dd.max()) if not dd.empty else 0.0


def portfolio_stats(weighted_returns: pd.Series, bars_per_year: float, trades_count: int | None = None) -> dict:
    if weighted_returns.empty:
        return {
            "pf": float("nan"),
            "sharpe": float("nan"),
            "maxdd": float("nan"),
            "trades": trades_count or 0,
        }
    pnl = weighted_returns
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    pf = gains / losses if losses > 0 else float("inf")
    sharpe = _annualized_sharpe(pnl, bars_per_year)
    equity_curve = (1.0 + pnl).cumprod()
    maxdd = _max_drawdown(equity_curve)
    return {
        "pf": float(pf),
        "sharpe": sharpe,
        "maxdd": maxdd,
        "trades": trades_count or len(pnl),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute portfolio metrics from per-ticker equity logs.")
    parser.add_argument("--equity-logs", nargs="+", required=True, help="CSV files with equity per ticker")
    parser.add_argument("--tickers", nargs="+", required=True, help="Ticker symbols in same order as equity logs")
    parser.add_argument("--weights", nargs="+", required=True, type=float, help="Weights aligned with tickers")
    parser.add_argument(
        "--annualization",
        type=float,
        default=1638.0,
        help="Annualization factor for Sharpe (default 1638 ≈ 252 * 6.5h for 1h equity trading)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    parser.add_argument("--scan", action="store_true", help="Enable random weight scan")
    parser.add_argument("--scan-samples", type=int, default=200, help="Number of random weight samples when scan enabled")
    parser.add_argument("--max-weight", type=float, default=0.4, help="Max weight per ticker in scan")
    parser.add_argument("--max-maxdd", type=float, default=0.2, help="Max drawdown filter in scan (0.2=20%)")
    args = parser.parse_args()

    logs = [Path(p) for p in args.equity_logs]
    if len(logs) != len(args.tickers) or len(logs) != len(args.weights):
        raise SystemExit("equity-logs, tickers, and weights must have the same length.")

    series_list: List[pd.Series] = []
    for path in logs:
        series_list.append(load_equity_csv(path))

    aligned = pd.concat(series_list, axis=1, keys=args.tickers).ffill().dropna()
    returns = aligned.pct_change().dropna()

    def compute_for_weights(weights: np.ndarray) -> Tuple[dict, np.ndarray]:
        weighted_returns = (returns * weights).sum(axis=1)
        stats = portfolio_stats(weighted_returns, bars_per_year=args.annualization)
        stats["bars"] = len(weighted_returns)
        return stats, weights

    # Primary weights
    weights = np.array(args.weights, dtype=float)
    weights = weights / weights.sum()
    stats, weights = compute_for_weights(weights)
    payload = {
        "tickers": args.tickers,
        "weights": weights.tolist(),
        "pf": stats["pf"],
        "sharpe": stats["sharpe"],
        "maxdd": stats["maxdd"],
        "trades": stats["trades"],
        "bars": stats["bars"],
        "annualization": args.annualization,
    }

    scan_results = []
    if args.scan:
        rng = np.random.default_rng(42)
        for _ in range(int(args.scan_samples)):
            w = rng.random(len(args.tickers))
            w = w / w.sum()
            if (w > args.max_weight).any():
                continue
            stats_scan, w_norm = compute_for_weights(w)
            if stats_scan["maxdd"] <= args.max_maxdd:
                scan_results.append(
                    {
                        "weights": w_norm.tolist(),
                        "pf": stats_scan["pf"],
                        "sharpe": stats_scan["sharpe"],
                        "maxdd": stats_scan["maxdd"],
                    }
                )
        scan_results = sorted(scan_results, key=lambda x: x["sharpe"] if np.isfinite(x["sharpe"]) else -np.inf, reverse=True)[:10]
        payload["scan_top"] = scan_results

    print(json.dumps(payload, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

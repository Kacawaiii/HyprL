#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_weights(weights_arg: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for item in weights_arg.split(","):
        if not item:
            continue
        if "=" not in item:
            raise SystemExit(f"Invalid weight entry: {item}")
        ticker, val = item.split("=", 1)
        weights[ticker.strip().upper()] = float(val)
    if not weights:
        raise SystemExit("No weights parsed.")
    return weights


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor portfolio health from trade logs or equity CSV.")
    parser.add_argument("--trade-logs", nargs="+", help="Per-ticker trade CSVs (replay-orch style).")
    parser.add_argument("--weights", help="Comma-separated mapping TICKER=WEIGHT if trade logs are provided.")
    parser.add_argument("--equity-csv", type=Path, help="Aggregated equity CSV (uses column equity_portfolio or equity).")
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--start", help="Optional start date to filter trades/equity (YYYY-MM-DD).")
    parser.add_argument("--end", help="Optional end date to filter trades/equity (YYYY-MM-DD).")
    parser.add_argument("--annualization", type=float, default=1638.0)
    parser.add_argument("--pf-alert", type=float, default=1.3)
    parser.add_argument("--dd-alert", type=float, default=20.0)
    parser.add_argument("--summary-out", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


@dataclass
class SeriesMetrics:
    pf: float
    maxdd: float
    trades: int
    win_rate: float
    sharpe: Optional[float] = None
    equity_end: Optional[float] = None


def load_trade_log(path: Path, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "exit_timestamp" not in df.columns or "pnl" not in df.columns:
        raise SystemExit(f"Trade log {path} missing required columns.")
    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])
    if start:
        df = df[df["exit_timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["exit_timestamp"] <= pd.to_datetime(end)]
    return df.sort_values("exit_timestamp").reset_index(drop=True)


def portfolio_from_trades(
    trade_logs: List[Path],
    weights: Dict[str, float],
    initial_equity: float,
    start: Optional[str],
    end: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    per_ticker: Dict[str, pd.DataFrame] = {}
    for path in trade_logs:
        stem = path.stem
        parts = stem.split("_")
        ticker = parts[1].upper() if len(parts) >= 2 else stem.split("-")[0].upper()
        if ticker not in weights:
            raise SystemExit(f"No weight provided for ticker {ticker} (file {path}).")
        w = weights[ticker]
        df = load_trade_log(path, start, end)
        scaled_pnl = df["pnl"].astype(float) * w
        equity0 = initial_equity * w
        equity = equity0 + scaled_pnl.cumsum()
        per_ticker[ticker] = pd.DataFrame(
            {"timestamp": df["exit_timestamp"], "pnl": scaled_pnl, "equity": equity}
        )
    all_ts = sorted({ts for curve in per_ticker.values() for ts in curve["timestamp"]})
    merged = pd.DataFrame({"timestamp": all_ts}).set_index("timestamp")
    for ticker, curve in per_ticker.items():
        merged = merged.join(
            curve.set_index("timestamp")[["equity", "pnl"]].rename(
                columns={"equity": f"equity_{ticker}", "pnl": f"pnl_{ticker}"}
            ),
            how="left",
        )
    merged = merged.sort_index().ffill().bfill()
    equity_cols = [c for c in merged.columns if c.startswith("equity_")]
    pnl_cols = [c for c in merged.columns if c.startswith("pnl_")]
    merged["equity_portfolio"] = merged[equity_cols].sum(axis=1)
    merged["pnl_portfolio"] = merged[pnl_cols].sum(axis=1)
    merged = merged.reset_index()
    return merged, per_ticker


def equity_from_csv(path: Path, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    col = "equity_portfolio" if "equity_portfolio" in df.columns else "equity"
    df = df[["timestamp", col]].rename(columns={col: "equity_portfolio"})
    if start:
        df = df[df["timestamp"] >= pd.to_datetime(start)]
    if end:
        df = df[df["timestamp"] <= pd.to_datetime(end)]
    return df.sort_values("timestamp").reset_index(drop=True)


def series_metrics_from_pnl(pnl: pd.Series, equity: pd.Series, annualization: float) -> SeriesMetrics:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = -pnl[pnl < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    dd = (equity.cummax() - equity) / equity.cummax()
    maxdd = float(dd.max() * 100) if not dd.empty else 0.0
    trades = len(pnl)
    win_rate = float((pnl > 0).mean() * 100) if trades else 0.0
    returns = equity.pct_change().dropna()
    if returns.std(ddof=0) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * (annualization**0.5))
    else:
        sharpe = 0.0
    equity_end = float(equity.iloc[-1]) if not equity.empty else 0.0
    return SeriesMetrics(pf=float(pf), maxdd=maxdd, trades=trades, win_rate=win_rate, sharpe=sharpe, equity_end=equity_end)


def per_ticker_metrics(per_ticker: Dict[str, pd.DataFrame], annualization: float, initial_equity: float) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for ticker, curve in per_ticker.items():
        pnl = curve["pnl"]
        equity0 = initial_equity * 0  # unused, equity curve already scaled
        equity = curve["equity"]
        metrics = series_metrics_from_pnl(pnl, equity, annualization)
        out[ticker] = {
            "pf": metrics.pf,
            "maxdd": metrics.maxdd,
            "trades": metrics.trades,
            "win_rate": metrics.win_rate,
        }
    return out


def main() -> None:
    args = parse_args()
    portfolio_equity: pd.DataFrame
    per_ticker: Dict[str, pd.DataFrame] = {}

    if args.trade_logs:
        if not args.weights:
            raise SystemExit("Provide --weights when using --trade-logs.")
        weights = parse_weights(args.weights)
        portfolio_equity, per_ticker = portfolio_from_trades(
            [Path(p) for p in args.trade_logs], weights, args.initial_equity, args.start, args.end
        )
    elif args.equity_csv:
        portfolio_equity = equity_from_csv(args.equity_csv, args.start, args.end)
    else:
        raise SystemExit("Provide either --trade-logs (with --weights) or --equity-csv.")

    if portfolio_equity.empty:
        raise SystemExit("No data to monitor after filtering.")

    pnl_port = portfolio_equity["pnl_portfolio"] if "pnl_portfolio" in portfolio_equity else portfolio_equity["equity_portfolio"].diff().fillna(0.0)
    equity_port = portfolio_equity["equity_portfolio"]
    port_metrics = series_metrics_from_pnl(pnl_port, equity_port, args.annualization)

    per_ticker_stats = per_ticker_metrics(per_ticker, args.annualization, args.initial_equity) if per_ticker else {}

    status = "ALERT" if (port_metrics.pf < args.pf_alert or port_metrics.maxdd > args.dd_alert) else "OK"
    print(
        f"STATUS: {status} | PF={port_metrics.pf:.3f} | MaxDD={port_metrics.maxdd:.2f}% | Sharpe={port_metrics.sharpe:.2f} | Trades={port_metrics.trades}"
    )

    summary = {
        "status": status,
        "pf": port_metrics.pf,
        "maxdd": port_metrics.maxdd,
        "sharpe": port_metrics.sharpe,
        "trades": port_metrics.trades,
        "win_rate": port_metrics.win_rate,
        "equity_end": port_metrics.equity_end,
        "pf_alert": args.pf_alert,
        "dd_alert": args.dd_alert,
        "per_ticker": per_ticker_stats,
    }
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

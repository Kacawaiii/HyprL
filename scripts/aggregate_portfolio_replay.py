#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


def load_trade_log(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "exit_timestamp" not in df.columns or "pnl" not in df.columns:
        raise SystemExit(f"Trade log {path} missing required columns.")
    df["exit_timestamp"] = pd.to_datetime(df["exit_timestamp"])
    df = df.sort_values("exit_timestamp").reset_index(drop=True)
    return df


def equity_curve_from_trades(trades: pd.DataFrame, initial_equity: float) -> pd.DataFrame:
    pnl = trades["pnl"].astype(float)
    equity = initial_equity + pnl.cumsum()
    return pd.DataFrame({"timestamp": trades["exit_timestamp"], "pnl": pnl, "equity": equity})


def aggregate_portfolio(
    trade_logs: List[Path],
    weights: Dict[str, float],
    initial_equity: float,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    curves: Dict[str, pd.DataFrame] = {}
    for path in trade_logs:
        # Infer ticker from filename (trades_<TICKER>_orch.csv)
        stem = path.stem
        parts = stem.split("_")
        ticker = parts[1].upper() if len(parts) >= 2 else stem.split("-")[0].upper()
        if ticker not in weights:
            raise SystemExit(f"No weight provided for ticker {ticker} (file {path}).")
        w = weights[ticker]
        df = load_trade_log(path)
        # Logs are produced with a 10k initial balance per ticker; scale pnl by portfolio weight.
        scaled_pnl = df["pnl"].astype(float) * w
        sub_equity0 = initial_equity * w
        equity = sub_equity0 + scaled_pnl.cumsum()
        curve = pd.DataFrame({"timestamp": df["exit_timestamp"], "pnl": scaled_pnl, "equity": equity})
        curves[ticker] = curve

    # Merge on timestamp by stacking all events and forward-filling per ticker equity.
    all_timestamps = sorted({ts for curve in curves.values() for ts in curve["timestamp"]})
    merged = pd.DataFrame({"timestamp": all_timestamps})
    merged = merged.set_index("timestamp")
    for ticker, curve in curves.items():
        merged = merged.join(curve.set_index("timestamp")[["equity", "pnl"]].rename(columns={
            "equity": f"equity_{ticker}",
            "pnl": f"pnl_{ticker}",
        }), how="left")
    merged = merged.sort_index().ffill().fillna(method="bfill")

    # Compute portfolio equity as sum of per-ticker equities.
    equity_cols = [col for col in merged.columns if col.startswith("equity_")]
    merged["equity_portfolio"] = merged[equity_cols].sum(axis=1)
    pnl_cols = [col for col in merged.columns if col.startswith("pnl_")]
    merged["pnl_portfolio"] = merged[pnl_cols].sum(axis=1)

    # Portfolio metrics
    pnl_port = merged["pnl_portfolio"]
    gross_profit = pnl_port[pnl_port > 0].sum()
    gross_loss = -pnl_port[pnl_port < 0].sum()
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    equity_port = merged["equity_portfolio"]
    dd = (equity_port.cummax() - equity_port) / equity_port.cummax()
    maxdd = float(dd.max() * 100) if not dd.empty else 0.0
    total_trades = sum(len(curves[t]) for t in curves)
    stats = {
        "pf": float(pf),
        "maxdd": maxdd,
        "trades": int(total_trades),
        "equity_end": float(equity_port.iloc[-1]) if not equity_port.empty else 0.0,
    }
    merged = merged.reset_index()
    return stats, merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-ticker replay trade logs into portfolio metrics.")
    parser.add_argument("--trade-logs", nargs="+", required=True, help="Paths to per-ticker trade CSVs.")
    parser.add_argument(
        "--weights",
        required=True,
        help="Comma-separated mapping TICKER=WEIGHT, e.g. NVDA=0.2313,META=0.0968,...",
    )
    parser.add_argument("--initial-equity", type=float, default=10000.0)
    parser.add_argument("--summary-out", type=Path, help="Optional JSON output for stats.")
    parser.add_argument("--equity-out", type=Path, help="Optional CSV for portfolio equity curve.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = parse_weights(args.weights)
    trade_logs = [Path(p) for p in args.trade_logs]
    stats, merged = aggregate_portfolio(trade_logs, weights, args.initial_equity)
    print(json.dumps(stats, indent=2))
    if args.summary_out:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    if args.equity_out:
        args.equity_out.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(args.equity_out, index=False)


if __name__ == "__main__":
    main()

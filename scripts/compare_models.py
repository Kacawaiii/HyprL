#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from hyprl.backtest.runner import BacktestConfig, run_backtest
from hyprl.metrics.calibration import trade_calibration_metrics
from hyprl.risk.manager import RiskConfig

RISK_PROFILES = {
    "safe": RiskConfig(risk_pct=0.01, atr_multiplier=1.0, reward_multiple=1.0),
    "normal": RiskConfig(risk_pct=0.02, atr_multiplier=1.0, reward_multiple=1.5),
    "aggressive": RiskConfig(risk_pct=0.05, atr_multiplier=1.2, reward_multiple=2.0),
}

MODEL_PRESETS = [
    ("logistic", "none"),
    ("logistic", "platt"),
    ("random_forest", "platt"),
    ("xgboost", "platt"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare model/calibration presets on a ticker.")
    parser.add_argument("--ticker", default="NVDA")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--long-threshold", type=float, default=0.55)
    parser.add_argument("--short-threshold", type=float, default=0.40)
    parser.add_argument("--risk-profile", choices=list(RISK_PROFILES.keys()), default="normal")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to store comparison results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    risk_cfg = RISK_PROFILES[args.risk_profile]
    rows: list[dict[str, float | str]] = []
    for model_type, calibration in MODEL_PRESETS:
        config = BacktestConfig(
            ticker=args.ticker,
            period=args.period,
            interval=args.interval,
            initial_balance=args.initial_balance,
            long_threshold=args.long_threshold,
            short_threshold=args.short_threshold,
            model_type=model_type,
            calibration=calibration,
            risk=risk_cfg,
            random_state=args.seed,
        )
        try:
            result = run_backtest(config)
        except Exception as exc:  # pragma: no cover
            print(f"[WARN] Failed combination {model_type}/{calibration}: {exc}")
            continue
        trades_df = pd.DataFrame([asdict(trade) for trade in result.trades])
        export_path = Path(f"data/trades_{args.ticker}_{model_type}_{calibration}.csv")
        export_path.parent.mkdir(parents=True, exist_ok=True)
        trades_df.to_csv(export_path, index=False)
        cal_metrics = trade_calibration_metrics(result.trades)
        rows.append(
            {
                "model_type": model_type,
                "calibration": calibration,
                "strategy_return_pct": (result.final_balance / args.initial_balance - 1.0) * 100.0,
                "alpha_pct": (result.final_balance / args.initial_balance - 1.0) * 100.0 - result.benchmark_return,
                "sharpe": result.sharpe_ratio or 0.0,
                "sortino": result.sortino_ratio or 0.0,
                "profit_factor": result.profit_factor or 0.0,
                "max_dd_pct": result.max_drawdown * 100.0,
                "brier": cal_metrics["brier"] or 0.0,
                "log_loss": cal_metrics["log_loss"] or 0.0,
                "trades": result.n_trades,
            }
        )
    if not rows:
        print("No successful runs.")
        return
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

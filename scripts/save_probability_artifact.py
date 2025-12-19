#!/usr/bin/env python3
from __future__ import annotations

"""Train and persist a ProbabilityModel artifact for reuse in BT/Replay parity runs."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from hyprl.backtest.runner import BacktestConfig
from hyprl.configs import get_risk_settings, load_long_threshold, load_short_threshold, load_ticker_settings
from hyprl.data.market import MarketDataFetcher
from hyprl.labels.amplitude import LabelConfig
from hyprl.model.probability import ProbabilityModel
from hyprl.risk.manager import RiskConfig
from hyprl.strategy import prepare_design_and_target, prepare_feature_frame


def build_config(args: argparse.Namespace) -> BacktestConfig:
    settings = load_ticker_settings(args.ticker, args.interval)
    long_threshold = args.long_threshold if args.long_threshold is not None else settings.get("long_threshold")
    short_threshold = args.short_threshold if args.short_threshold is not None else settings.get("short_threshold")
    if long_threshold is None:
        long_threshold = load_long_threshold(settings, default=0.6)
    if short_threshold is None:
        short_threshold = load_short_threshold(settings)
    if short_threshold is None:
        short_threshold = 0.4
    risk_profile = settings.get("default_risk_profile") or "normal"
    risk_params = get_risk_settings(settings, risk_profile)
    risk_cfg = RiskConfig(balance=args.initial_balance, **risk_params)
    return BacktestConfig(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        period=args.period,
        interval=args.interval,
        initial_balance=args.initial_balance,
        long_threshold=float(long_threshold),
        short_threshold=float(short_threshold),
        risk=risk_cfg,
        risk_profile=risk_profile,
        risk_profiles=settings.get("risk_profiles", {}),
        label=LabelConfig(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save a ProbabilityModel artifact.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--period", default="1y")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--long-threshold", type=float)
    parser.add_argument("--short-threshold", type=float)
    parser.add_argument("--model-type", default="logistic", choices=["logistic", "random_forest", "xgboost"])
    parser.add_argument("--calibration", default="none", choices=["none", "platt", "isotonic"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True, help="Destination .pkl path")
    # XGBoost hyper-parameters (ignored for other models)
    parser.add_argument("--xgb-max-depth", type=int, default=4)
    parser.add_argument("--xgb-estimators", type=int, default=400)
    parser.add_argument("--xgb-eta", type=float, default=0.05)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)
    args = parser.parse_args()

    config = build_config(args)
    fetcher = MarketDataFetcher(args.ticker)
    prices = fetcher.get_prices(interval=args.interval, period=args.period, start=args.start, end=args.end)
    features = prepare_feature_frame(prices, config)
    if features.empty:
        raise SystemExit("Feature frame empty; cannot train artifact")
    feature_cols = config.model_feature_columns or None
    design, target = prepare_design_and_target(features, config.label, feature_cols)
    if design.empty or target.nunique() < 2:
        raise SystemExit("Insufficient training data for artifact")

    model = ProbabilityModel.create(
        model_type=args.model_type,
        calibration=args.calibration,
        random_state=args.seed,
        xgb_max_depth=args.xgb_max_depth,
        xgb_estimators=args.xgb_estimators,
        xgb_eta=args.xgb_eta,
        xgb_subsample=args.xgb_subsample,
        xgb_colsample=args.xgb_colsample,
    )
    model.fit(design, target)
    probs = model.predict_proba(design)
    preds = (probs >= 0.5).astype(int)
    hit_rate = accuracy_score(target, preds)
    try:
        auc = roc_auc_score(target, probs)
    except ValueError:
        auc = float("nan")
    model.dump(args.output)
    metrics = {
        "rows": len(design),
        "hit_rate": hit_rate,
        "roc_auc": auc,
        "model_type": args.model_type,
        "calibration": args.calibration,
        "seed": args.seed,
        "xgb_max_depth": args.xgb_max_depth,
        "xgb_estimators": args.xgb_estimators,
        "xgb_eta": args.xgb_eta,
        "xgb_subsample": args.xgb_subsample,
        "xgb_colsample": args.xgb_colsample,
    }
    metrics_path = args.output.with_suffix(args.output.suffix + ".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(
        f"Saved ProbabilityModel artifact to {args.output} "
        f"(rows={len(design)}, hit_rate={hit_rate:.4f}, auc={auc:.4f})"
    )
    print(f"Metrics written to {metrics_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Sweep thresholds and symbols to find optimal Sharpe configuration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.backtest.runner import BacktestConfig, run_backtest
from hyprl.risk.manager import RiskConfig
from hyprl.adaptive.engine import AdaptiveConfig
import json

SYMBOLS = ["NVDA", "MSFT", "QQQ", "AAPL", "META", "AMZN", "SPY"]

THRESHOLDS = {
    "default":     {"long": 0.55, "short": 0.45},
    "moderate":    {"long": 0.58, "short": 0.42},
    "strict":      {"long": 0.60, "short": 0.40},
    "very_strict": {"long": 0.62, "short": 0.38},
}

def run_one(symbol, long_th, short_th):
    """Run a single backtest, return key metrics."""
    # Feature list
    feat_path = Path(f"models/{symbol.lower()}_1h_xgb_v3_features.json")
    if feat_path.exists():
        feature_cols = json.loads(feat_path.read_text())
    else:
        feature_cols = [
            "ret_1h", "ret_3h", "ret_6h", "ret_24h",
            "atr_14", "atr_72", "atr_14_norm", "atr_72_norm",
            "rsi_7", "rsi_14", "rsi_21",
            "vol_ratio_10_30", "vol_regime_high", "volume_zscore_24", "volume_surge",
            "range_pct", "true_range", "ret_skew_20", "ret_kurt_20",
        ]

    # Model path
    artifact = f"models/{symbol.lower()}_1h_xgb_v3.joblib"
    if not Path(artifact).exists():
        artifact = None

    risk = RiskConfig(
        risk_pct=0.015,
        atr_multiplier=1.5,
        reward_multiple=2.0,
        trailing_stop_activation=1.5,
        trailing_stop_distance=1.0,
    )

    adaptive = AdaptiveConfig(enable=True, lookback_trades=50)

    config = BacktestConfig(
        ticker=symbol,
        period="1y",
        interval="1h",
        initial_balance=100000,
        model_type="xgboost",
        model_artifact_path=artifact,
        model_feature_columns=feature_cols,
        feature_preset=f"{symbol.lower()}_v2",
        long_threshold=long_th,
        short_threshold=short_th,
        risk=risk,
        commission_pct=0.0,
        slippage_pct=0.0005,
        enable_trend_filter=True,
        sentiment_regime="cautious",
        min_ev_multiple=0.0,
        multi_timeframes=["15m"],
        fusion_method="mean",
        fusion_weights={"base": 0.7, "15m": 0.3},
        adaptive=adaptive,
        guards_config={
            "enable_guards_v2": True,
            "earnings_blackout_days": 3,
            "max_correlated_notional": 15000,
            "vix_threshold_reduce": 25,
            "vix_threshold_block": 35,
        },
        dynamic_sizing_mode="kelly",
        dynamic_sizing_base_pct=0.015,
        dynamic_sizing_lookback=50,
        dynamic_sizing_min_trades=10,
        dynamic_sizing_max_multiplier=2.0,
        dynamic_sizing_min_multiplier=0.25,
    )

    try:
        result = run_backtest(config)
        ret = (result.final_balance / 100000 - 1) * 100
        return {
            "return_pct": round(ret, 2),
            "sharpe": round(result.sharpe_ratio or 0, 2),
            "win_rate": round(result.win_rate * 100, 1),
            "trades": result.n_trades,
            "max_dd": round(result.max_drawdown or 0, 2),
            "pf": round(result.profit_factor or 0, 2),
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    results = {}

    for sym in SYMBOLS:
        results[sym] = {}
        for name, th in THRESHOLDS.items():
            print(f"Testing {sym} @ {name} (L={th['long']}, S={th['short']})...", end=" ", flush=True)
            r = run_one(sym, th["long"], th["short"])
            results[sym][name] = r
            if "error" in r:
                print(f"ERROR: {r['error']}")
            else:
                print(f"Ret={r['return_pct']:+.1f}% Sharpe={r['sharpe']:.2f} WR={r['win_rate']:.0f}% Trades={r['trades']} DD={r['max_dd']:.1f}% PF={r['pf']:.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Best threshold per symbol (by Sharpe)")
    print("=" * 80)
    print(f"{'Symbol':<8} {'Best Th':<14} {'Return':>8} {'Sharpe':>8} {'WR':>6} {'Trades':>7} {'MaxDD':>8} {'PF':>6}")
    print("-" * 80)

    good_symbols = []
    for sym in SYMBOLS:
        best_name = None
        best_sharpe = -999
        for name, r in results[sym].items():
            if "error" not in r and r["sharpe"] > best_sharpe:
                best_sharpe = r["sharpe"]
                best_name = name
        if best_name:
            r = results[sym][best_name]
            flag = " ***" if r["sharpe"] >= 0.8 else ""
            print(f"{sym:<8} {best_name:<14} {r['return_pct']:>+7.1f}% {r['sharpe']:>7.2f} {r['win_rate']:>5.0f}% {r['trades']:>7} {r['max_dd']:>7.1f}% {r['pf']:>5.2f}{flag}")
            if r["sharpe"] >= 0.5:
                good_symbols.append((sym, best_name, r))

    print("\n" + "=" * 80)
    print(f"RECOMMENDED PORTFOLIO: {[s[0] for s in good_symbols]}")
    if good_symbols:
        avg_sharpe = sum(s[2]["sharpe"] for s in good_symbols) / len(good_symbols)
        avg_ret = sum(s[2]["return_pct"] for s in good_symbols) / len(good_symbols)
        print(f"Avg Sharpe: {avg_sharpe:.2f}, Avg Return: {avg_ret:+.1f}%")
    print("=" * 80)

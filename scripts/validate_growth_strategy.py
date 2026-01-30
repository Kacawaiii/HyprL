#!/usr/bin/env python3
"""
Walk-Forward Validation pour Growth Strategy
=============================================
Vérifie que les +23% ne sont pas de l'overfitting.
"""

import sys
import json
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.simple_v5 import compute_features, generate_signal, calculate_position_size
from hyprl.monitoring import detect_regime, MarketRegime


def download_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """Download data for all symbols."""
    data = {}

    def fetch(sym):
        try:
            df = yf.Ticker(sym).history(period="2y", interval="1h")
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz:
                df.index = df.index.tz_localize(None)
            return sym, df
        except:
            return sym, None

    with ThreadPoolExecutor(max_workers=5) as ex:
        for sym, df in [f.result() for f in as_completed([ex.submit(fetch, s) for s in symbols])]:
            if df is not None and len(df) > 100:
                data[sym] = compute_features(df)

    return data


def backtest_period(
    all_data: Dict[str, pd.DataFrame],
    start_idx: int,
    end_idx: int,
    config: Dict,
    equity: float = 100000,
    slippage: float = 0.0005
) -> Dict[str, Any]:
    """Backtest a specific period."""
    rules = config.get("rules", {})
    capital = config.get("capital", {})
    risk = config.get("risk", {})
    symbols_cfg = config.get("symbols", {})

    base_risk = capital.get("base_risk_per_trade", 0.02)
    high_risk = capital.get("high_confidence_risk", 0.03)
    max_pos = capital.get("max_position_pct", 0.20)
    max_exp = capital.get("max_total_exposure", 1.5)
    stop_atr = risk.get("stop_loss_atr", 2.0)
    tp_atr = risk.get("take_profit_atr", 3.0)
    min_conf = config.get("guards", {}).get("min_confidence", 0.5)

    positions = {}
    trades = []
    eq = equity
    eq_curve = [eq]

    for i in range(start_idx, end_idx):
        exposure = sum(p['val'] for p in positions.values()) / eq if positions else 0

        # Check exits
        to_close = []
        for sym, pos in positions.items():
            if sym not in all_data or i >= len(all_data[sym]):
                continue
            price = all_data[sym].iloc[i]['close']
            exit_p, reason = None, None

            if pos['dir'] == 'long':
                if price <= pos['stop']:
                    exit_p, reason = pos['stop'] * (1-slippage), 'stop'
                elif price >= pos['tp']:
                    exit_p, reason = pos['tp'] * (1-slippage), 'tp'
            else:
                if price >= pos['stop']:
                    exit_p, reason = pos['stop'] * (1+slippage), 'stop'
                elif price <= pos['tp']:
                    exit_p, reason = pos['tp'] * (1+slippage), 'tp'

            if exit_p:
                pnl = (exit_p - pos['entry']) * pos['shares'] if pos['dir'] == 'long' else (pos['entry'] - exit_p) * pos['shares']
                eq += pnl
                trades.append({'pnl': pnl, 'sym': sym})
                to_close.append(sym)

        for s in to_close:
            del positions[s]

        # New signals
        for sym, df in all_data.items():
            if sym in positions or i >= len(df):
                continue
            cfg = symbols_cfg.get(sym, {})
            if not cfg.get('enabled', True):
                continue

            row = df.iloc[i]
            price, atr = row['close'], row['atr_14']

            direction, conf = generate_signal(
                row,
                long_rsi_below=rules.get('long', {}).get('rsi_below', 45),
                long_momentum_above=rules.get('long', {}).get('momentum_above', 0.004),
                short_rsi_above=rules.get('short', {}).get('rsi_above', 65),
                short_momentum_below=rules.get('short', {}).get('momentum_below', -0.004)
            )

            if direction == 'flat' or conf < min_conf or exposure >= max_exp:
                continue

            risk_pct = high_risk if conf > 0.7 else base_risk
            lev = 1.5 if cfg.get('leverage_allowed') and conf > 0.6 else 1.0

            shares, stop_d, val = calculate_position_size(eq, price, atr, risk_pct, max_pos, stop_atr)
            shares *= lev
            val *= lev

            if shares <= 0:
                continue

            if direction == 'long':
                entry = price * (1+slippage)
                stop = entry - stop_d
                tp = entry + stop_d * (tp_atr/stop_atr)
            else:
                entry = price * (1-slippage)
                stop = entry + stop_d
                tp = entry - stop_d * (tp_atr/stop_atr)

            positions[sym] = {'dir': direction, 'entry': entry, 'shares': shares, 'stop': stop, 'tp': tp, 'val': val}
            exposure = sum(p['val'] for p in positions.values()) / eq

        eq_curve.append(eq)

    if not trades:
        return {'return': 0, 'sharpe': 0, 'trades': 0, 'win_rate': 0}

    eq_s = pd.Series(eq_curve)
    rets = eq_s.pct_change().dropna()

    total_ret = (eq - equity) / equity
    sharpe = rets.mean() / rets.std() * np.sqrt(252*6.5) if rets.std() > 0 else 0
    wins = sum(1 for t in trades if t['pnl'] > 0)

    return {
        'return': total_ret,
        'sharpe': sharpe,
        'trades': len(trades),
        'win_rate': wins / len(trades) if trades else 0
    }


def walk_forward_validation(all_data: Dict[str, pd.DataFrame], config: Dict, n_splits: int = 5):
    """Run WF validation."""
    min_len = min(len(df) for df in all_data.values())
    window = min_len // n_splits

    print(f"\n{'='*60}")
    print(f"WALK-FORWARD VALIDATION GROWTH ({n_splits} splits)")
    print(f"{'='*60}")
    print(f"Bars: {min_len}, Window: {window}")

    is_results = []
    oos_results = []

    for i in range(n_splits - 1):
        train_start = i * window + 50
        train_end = train_start + int(window * 0.7)
        test_start = train_end
        test_end = min((i + 2) * window, min_len)

        is_res = backtest_period(all_data, train_start, train_end, config)
        oos_res = backtest_period(all_data, test_start, test_end, config)

        is_results.append(is_res)
        oos_results.append(oos_res)

        print(f"\nSplit {i+1}:")
        print(f"  Train: {train_end-train_start} bars | Ret: {is_res['return']:+.1%} | Sharpe: {is_res['sharpe']:.2f}")
        print(f"  Test:  {test_end-test_start} bars | Ret: {oos_res['return']:+.1%} | Sharpe: {oos_res['sharpe']:.2f}")

    avg_is_ret = np.mean([r['return'] for r in is_results])
    avg_is_sharpe = np.mean([r['sharpe'] for r in is_results])
    avg_oos_ret = np.mean([r['return'] for r in oos_results])
    avg_oos_sharpe = np.mean([r['sharpe'] for r in oos_results])

    overfit = avg_is_sharpe / avg_oos_sharpe if avg_oos_sharpe > 0 else float('inf')

    print(f"\n{'='*60}")
    print("RÉSULTATS AGRÉGÉS")
    print(f"{'='*60}")
    print(f"\nIN-SAMPLE:  Ret {avg_is_ret:+.1%} | Sharpe {avg_is_sharpe:.2f}")
    print(f"OOS:        Ret {avg_oos_ret:+.1%} | Sharpe {avg_oos_sharpe:.2f}")
    print(f"\nOVERFIT RATIO: {overfit:.2f}")

    if overfit < 1.5:
        print("✅ BON - Stratégie validée")
    elif overfit < 2.0:
        print("⚠️ ATTENTION - Légère dégradation OOS")
    else:
        print("❌ DANGER - Overfitting détecté")

    return {
        'is_return': avg_is_ret,
        'is_sharpe': avg_is_sharpe,
        'oos_return': avg_oos_ret,
        'oos_sharpe': avg_oos_sharpe,
        'overfit_ratio': overfit
    }


def main():
    config_path = "configs/runtime/strategy_growth.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    symbols = [s for s, c in config.get('symbols', {}).items() if c.get('enabled', True)]

    print("Téléchargement des données...")
    all_data = download_data(symbols)
    print(f"✓ {len(all_data)} symboles chargés")

    result = walk_forward_validation(all_data, config)

    # Save
    with open("live/logs/growth_wf_validation.json", 'w') as f:
        json.dump({'timestamp': datetime.now().isoformat(), 'results': result}, f, indent=2)

    print(f"\nSaved to live/logs/growth_wf_validation.json")


if __name__ == "__main__":
    main()

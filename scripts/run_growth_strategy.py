#!/usr/bin/env python3
"""
HYPRL Growth Strategy - Objectif +15%/an
=========================================
Multi-asset + Levier modéré + Crypto

Usage:
    python scripts/run_growth_strategy.py --config configs/runtime/strategy_growth.yaml
    python scripts/run_growth_strategy.py --backtest  # Backtest first
"""

import argparse
import json
import sys
import yaml
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.simple_v5 import compute_features, generate_signal, calculate_position_size
from hyprl.monitoring import detect_regime, MarketRegime


def download_multi_symbol(symbols: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Download data for multiple symbols in parallel."""
    data = {}

    def fetch_one(symbol):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1h")
            df.columns = [c.lower() for c in df.columns]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return symbol, df
        except Exception as e:
            print(f"  ⚠️ {symbol}: {e}")
            return symbol, None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_one, s) for s in symbols]
        for future in as_completed(futures):
            symbol, df = future.result()
            if df is not None and len(df) > 50:
                data[symbol] = df

    return data


def backtest_growth_strategy(
    all_data: Dict[str, pd.DataFrame],
    config: Dict[str, Any],
    initial_equity: float = 100000,
    slippage_pct: float = 0.0005
) -> Dict[str, Any]:
    """
    Backtest the growth strategy across all symbols.
    """
    rules = config.get("rules", {})
    capital_cfg = config.get("capital", {})
    risk_cfg = config.get("risk", {})
    symbols_cfg = config.get("symbols", {})

    # Extract params
    base_risk = capital_cfg.get("base_risk_per_trade", 0.02)
    high_conf_risk = capital_cfg.get("high_confidence_risk", 0.03)
    max_pos_pct = capital_cfg.get("max_position_pct", 0.20)
    max_exposure = capital_cfg.get("max_total_exposure", 1.5)
    stop_atr = risk_cfg.get("stop_loss_atr", 2.0)
    tp_atr = risk_cfg.get("take_profit_atr", 3.0)
    min_confidence = config.get("guards", {}).get("min_confidence", 0.5)

    # Prepare data with features
    for symbol, df in all_data.items():
        all_data[symbol] = compute_features(df)

    # Find common date range
    min_len = min(len(df) for df in all_data.values())
    start_idx = 50  # Need enough history for features

    equity = initial_equity
    peak_equity = equity
    positions = {}  # symbol -> position info
    all_trades = []
    equity_curve = [equity]
    daily_equity = {}

    # Iterate through time
    for i in range(start_idx, min_len):
        current_exposure = sum(
            p['value'] for p in positions.values()
        ) / equity if positions else 0

        # Check exits for all positions
        symbols_to_close = []
        for symbol, pos in positions.items():
            if symbol not in all_data:
                continue

            df = all_data[symbol]
            if i >= len(df):
                continue

            price = df.iloc[i]['close']
            exit_price = None
            exit_reason = None

            if pos['direction'] == 'long':
                if price <= pos['stop']:
                    exit_price = pos['stop'] * (1 - slippage_pct)
                    exit_reason = 'stop'
                elif price >= pos['tp']:
                    exit_price = pos['tp'] * (1 - slippage_pct)
                    exit_reason = 'tp'
                # Trailing stop
                elif pos.get('trailing_active'):
                    trail_stop = price * (1 - risk_cfg.get('trailing_distance_pct', 0.01))
                    if price <= pos.get('trail_stop', 0):
                        exit_price = pos['trail_stop'] * (1 - slippage_pct)
                        exit_reason = 'trailing'
                    elif trail_stop > pos.get('trail_stop', 0):
                        pos['trail_stop'] = trail_stop
                # Check if trailing should activate
                elif price >= pos['entry'] * (1 + risk_cfg.get('trailing_activation_pct', 0.02)):
                    pos['trailing_active'] = True
                    pos['trail_stop'] = price * (1 - risk_cfg.get('trailing_distance_pct', 0.01))

            else:  # short
                if price >= pos['stop']:
                    exit_price = pos['stop'] * (1 + slippage_pct)
                    exit_reason = 'stop'
                elif price <= pos['tp']:
                    exit_price = pos['tp'] * (1 + slippage_pct)
                    exit_reason = 'tp'

            if exit_price:
                if pos['direction'] == 'long':
                    pnl = (exit_price - pos['entry']) * pos['shares']
                else:
                    pnl = (pos['entry'] - exit_price) * pos['shares']

                equity += pnl
                all_trades.append({
                    'symbol': symbol,
                    'direction': pos['direction'],
                    'entry': pos['entry'],
                    'exit': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl / pos['value'],
                    'reason': exit_reason,
                    'leverage': pos.get('leverage', 1.0)
                })
                symbols_to_close.append(symbol)

        for s in symbols_to_close:
            del positions[s]

        # Generate new signals
        for symbol, df in all_data.items():
            if symbol in positions:
                continue
            if i >= len(df):
                continue

            sym_cfg = symbols_cfg.get(symbol, {})
            if not sym_cfg.get('enabled', True):
                continue

            row = df.iloc[i]
            price = row['close']
            atr = row['atr_14']

            # Check regime
            if i > 50:
                regime = detect_regime(df.iloc[i-50:i+1])
                regime_mult = 1.0
                if regime.regime == MarketRegime.VOLATILE:
                    regime_mult = 0.5
                elif regime.regime in [MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN]:
                    regime_mult = 1.25
            else:
                regime_mult = 1.0

            # Generate signal
            direction, confidence = generate_signal(
                row,
                long_rsi_below=rules.get('long', {}).get('rsi_below', 45),
                long_momentum_above=rules.get('long', {}).get('momentum_above', 0.004),
                short_rsi_above=rules.get('short', {}).get('rsi_above', 65),
                short_momentum_below=rules.get('short', {}).get('momentum_below', -0.004)
            )

            if direction == 'flat' or confidence < min_confidence:
                continue

            # Check exposure limit
            if current_exposure >= max_exposure:
                continue

            # Calculate position with leverage
            risk_pct = high_conf_risk if confidence > 0.7 else base_risk
            risk_pct *= regime_mult

            # Apply leverage if allowed
            leverage = 1.0
            if sym_cfg.get('leverage_allowed', False) and confidence > 0.6:
                leverage = min(1.5, max_exposure - current_exposure + 1)

            shares, stop_dist, pos_value = calculate_position_size(
                equity, price, atr, risk_pct,
                max_position_pct=max_pos_pct,
                atr_multiplier=stop_atr
            )

            # Apply leverage to position size
            shares *= leverage
            pos_value *= leverage

            if shares <= 0:
                continue

            # Entry with slippage
            if direction == 'long':
                entry_price = price * (1 + slippage_pct)
                stop_price = entry_price - stop_dist
                tp_price = entry_price + stop_dist * (tp_atr / stop_atr)
            else:
                entry_price = price * (1 - slippage_pct)
                stop_price = entry_price + stop_dist
                tp_price = entry_price - stop_dist * (tp_atr / stop_atr)

            positions[symbol] = {
                'direction': direction,
                'entry': entry_price,
                'shares': shares,
                'stop': stop_price,
                'tp': tp_price,
                'value': pos_value,
                'leverage': leverage,
                'trailing_active': False
            }

            current_exposure = sum(p['value'] for p in positions.values()) / equity

        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

        # Track daily equity
        if i < len(list(all_data.values())[0]):
            date = list(all_data.values())[0].index[i].date()
            daily_equity[date] = equity

    # Calculate metrics
    if not all_trades:
        return {'error': 'No trades'}

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    total_return = (equity - initial_equity) / initial_equity
    n_hours = min_len - start_idx
    years = n_hours / (6.5 * 252)
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5) if returns.std() > 0 else 0

    wins = [t for t in all_trades if t['pnl'] > 0]
    losses = [t for t in all_trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(all_trades)

    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min()

    # By symbol
    by_symbol = {}
    for t in all_trades:
        s = t['symbol']
        if s not in by_symbol:
            by_symbol[s] = {'trades': 0, 'pnl': 0, 'wins': 0}
        by_symbol[s]['trades'] += 1
        by_symbol[s]['pnl'] += t['pnl']
        if t['pnl'] > 0:
            by_symbol[s]['wins'] += 1

    # Leveraged trades stats
    leveraged_trades = [t for t in all_trades if t.get('leverage', 1) > 1]
    leveraged_pnl = sum(t['pnl'] for t in leveraged_trades)

    return {
        'total_return': total_return,
        'annual_return': ann_return,
        'sharpe': sharpe,
        'win_rate': win_rate,
        'trades': len(all_trades),
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'final_equity': equity,
        'by_symbol': by_symbol,
        'leveraged_trades': len(leveraged_trades),
        'leveraged_pnl': leveraged_pnl,
        'avg_leverage': np.mean([t.get('leverage', 1) for t in all_trades])
    }


def main():
    parser = argparse.ArgumentParser(description="Growth Strategy - Target +15%")
    parser.add_argument("--config", default="configs/runtime/strategy_growth.yaml")
    parser.add_argument("--backtest", action="store_true", help="Run backtest")
    parser.add_argument("--equity", type=float, default=100000)
    parser.add_argument("--output", default="live/logs/growth_backtest.json")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    symbols = [s for s, cfg in config.get('symbols', {}).items() if cfg.get('enabled', True)]

    print("\n" + "=" * 70)
    print("HYPRL GROWTH STRATEGY - OBJECTIF +15%/AN")
    print("=" * 70)
    print(f"Symboles: {len(symbols)} ({', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''})")
    print(f"Capital: ${args.equity:,.0f}")
    print(f"Max exposure: {config.get('capital', {}).get('max_total_exposure', 1.5):.1f}x")

    if args.backtest:
        print("\n" + "-" * 70)
        print("TÉLÉCHARGEMENT DES DONNÉES...")
        print("-" * 70)

        all_data = download_multi_symbol(symbols, period="2y")
        print(f"✓ {len(all_data)} symboles chargés")

        print("\n" + "-" * 70)
        print("BACKTEST EN COURS...")
        print("-" * 70)

        result = backtest_growth_strategy(
            all_data, config,
            initial_equity=args.equity,
            slippage_pct=0.0005
        )

        if 'error' in result:
            print(f"❌ Erreur: {result['error']}")
            return

        print(f"\n{'=' * 70}")
        print("RÉSULTATS")
        print(f"{'=' * 70}")
        print(f"\nPerformance globale:")
        print(f"  Return total: {result['total_return']:+.1%}")
        print(f"  Return annualisé: {result['annual_return']:+.1%}")
        print(f"  Sharpe ratio: {result['sharpe']:.2f}")
        print(f"  Win rate: {result['win_rate']:.1%}")
        print(f"  Trades: {result['trades']}")
        print(f"  Max drawdown: {result['max_dd']:.1%}")
        print(f"  Profit factor: {result['profit_factor']:.2f}")
        print(f"  Equity finale: ${result['final_equity']:,.0f}")

        print(f"\nLevier:")
        print(f"  Trades avec levier: {result['leveraged_trades']}")
        print(f"  P/L trades levier: ${result['leveraged_pnl']:+,.0f}")
        print(f"  Levier moyen: {result['avg_leverage']:.2f}x")

        print(f"\nPar symbole:")
        for sym, stats in sorted(result['by_symbol'].items(), key=lambda x: -x[1]['pnl']):
            wr = stats['wins'] / stats['trades'] if stats['trades'] > 0 else 0
            print(f"  {sym:6} | {stats['trades']:3} trades | ${stats['pnl']:+8,.0f} | WR {wr:.0%}")

        # Compare to MVP
        print(f"\n{'=' * 70}")
        print("COMPARAISON")
        print(f"{'=' * 70}")
        print(f"  MVP (3 symboles):    +3.4%/an  |  Sharpe 0.95")
        print(f"  GROWTH (10 symboles): {result['annual_return']:+.1%}/an  |  Sharpe {result['sharpe']:.2f}")

        target_met = result['annual_return'] >= 0.15
        print(f"\n  Objectif +15%: {'✅ ATTEINT' if target_met else '❌ NON ATTEINT'}")

        # Save results
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'config': args.config,
                'results': result
            }, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Backtest MVP avec Coûts Réalistes
=================================
Compare les résultats avec et sans slippage/commissions.

Coûts modélisés:
- Slippage: 0.05% par trade (entrée + sortie)
- Commission Alpaca: $0 (gratuit)
- Spread implicite: inclus dans slippage

Usage:
    python scripts/backtest_with_costs.py --symbols NVDA,MSFT,QQQ
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.simple_v5 import compute_features, generate_signal, calculate_position_size


def download_data(symbol: str, period: str = "2y") -> pd.DataFrame:
    """Download historical data."""
    print(f"  Downloading {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval="1h")
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def run_backtest(
    df: pd.DataFrame,
    params: Dict[str, float],
    slippage_pct: float = 0.0,
    initial_equity: float = 100000
) -> Dict[str, Any]:
    """
    Run backtest with configurable slippage.

    Args:
        df: OHLCV DataFrame
        params: Strategy parameters
        slippage_pct: Slippage as decimal (0.0005 = 0.05%)
        initial_equity: Starting capital

    Returns:
        Dict with performance metrics
    """
    df = compute_features(df.copy())

    equity = initial_equity
    peak_equity = equity
    position = None
    trades = []
    equity_curve = [equity]
    daily_returns = []

    for i in range(20, len(df)):
        row = df.iloc[i]
        price = row['close']
        atr = row['atr_14']

        # Check exit if in position
        if position is not None:
            exit_price = None
            exit_reason = None

            if position['direction'] == 'long':
                # Check stop (with slippage - worse fill)
                if price <= position['stop']:
                    exit_price = position['stop'] * (1 - slippage_pct)
                    exit_reason = 'stop'
                # Check TP (with slippage)
                elif price >= position['tp']:
                    exit_price = position['tp'] * (1 - slippage_pct)
                    exit_reason = 'tp'
            else:  # short
                if price >= position['stop']:
                    exit_price = position['stop'] * (1 + slippage_pct)
                    exit_reason = 'stop'
                elif price <= position['tp']:
                    exit_price = position['tp'] * (1 + slippage_pct)
                    exit_reason = 'tp'

            if exit_price:
                if position['direction'] == 'long':
                    pnl = (exit_price - position['entry']) * position['shares']
                else:
                    pnl = (position['entry'] - exit_price) * position['shares']

                equity += pnl
                trades.append({
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl / (position['entry'] * position['shares']),
                    'direction': position['direction'],
                    'reason': exit_reason,
                    'slippage_cost': slippage_pct * position['entry'] * position['shares'] * 2
                })
                position = None

        # Generate signal if flat
        if position is None:
            direction, confidence = generate_signal(
                row,
                long_rsi_below=params.get('long_rsi_below', 45),
                long_momentum_above=params.get('long_momentum_above', 0.004),
                short_rsi_above=params.get('short_rsi_above', 65),
                short_momentum_below=params.get('short_momentum_below', -0.004)
            )

            if direction != 'flat' and confidence > 0.3:
                risk_pct = params.get('risk_per_trade', 0.02)
                stop_atr = params.get('stop_loss_atr', 2.0)
                tp_atr = params.get('take_profit_atr', 2.5)

                shares, stop_dist, pos_value = calculate_position_size(
                    equity, price, atr, risk_pct,
                    max_position_pct=0.25,
                    atr_multiplier=stop_atr
                )

                if shares > 0:
                    # Entry with slippage (worse fill)
                    if direction == 'long':
                        entry_price = price * (1 + slippage_pct)
                        stop_price = entry_price - stop_dist
                        tp_price = entry_price + stop_dist * (tp_atr / stop_atr)
                    else:
                        entry_price = price * (1 - slippage_pct)
                        stop_price = entry_price + stop_dist
                        tp_price = entry_price - stop_dist * (tp_atr / stop_atr)

                    position = {
                        'direction': direction,
                        'entry': entry_price,
                        'shares': shares,
                        'stop': stop_price,
                        'tp': tp_price
                    }

        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

    # Calculate metrics
    if not trades:
        return {
            'return_pct': 0,
            'return_ann': 0,
            'sharpe': 0,
            'sortino': 0,
            'win_rate': 0,
            'trades': 0,
            'max_dd': 0,
            'profit_factor': 0,
            'avg_trade': 0,
            'total_slippage_cost': 0
        }

    equity_series = pd.Series(equity_curve)
    returns = equity_series.pct_change().dropna()

    total_return = (equity - initial_equity) / initial_equity

    # Annualized (assuming 6.5 hours/day, 252 days/year)
    n_hours = len(df) - 20
    years = n_hours / (6.5 * 252)
    ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe
    if returns.std() > 0:
        sharpe = returns.mean() / returns.std() * np.sqrt(252 * 6.5)
    else:
        sharpe = 0

    # Sortino
    neg_returns = returns[returns < 0]
    if len(neg_returns) > 0 and neg_returns.std() > 0:
        sortino = returns.mean() / neg_returns.std() * np.sqrt(252 * 6.5)
    else:
        sortino = sharpe

    # Win rate
    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]
    win_rate = len(wins) / len(trades) if trades else 0

    # Profit factor
    gross_profit = sum(t['pnl'] for t in wins) if wins else 0
    gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Max drawdown
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_dd = drawdown.min()

    # Average trade
    avg_trade = np.mean([t['pnl'] for t in trades])

    # Total slippage cost
    total_slippage = sum(t.get('slippage_cost', 0) for t in trades)

    return {
        'return_pct': total_return,
        'return_ann': ann_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'trades': len(trades),
        'max_dd': max_dd,
        'profit_factor': profit_factor,
        'avg_trade': avg_trade,
        'final_equity': equity,
        'total_slippage_cost': total_slippage
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Backtest with realistic costs")
    parser.add_argument("--symbols", default="NVDA,MSFT,QQQ")
    parser.add_argument("--equity", type=float, default=100000)
    parser.add_argument("--output", default="live/logs/backtest_costs.json")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    # MVP optimized params
    params = {
        'long_rsi_below': 45,
        'long_momentum_above': 0.004,
        'short_rsi_above': 65,
        'short_momentum_below': -0.004,
        'risk_per_trade': 0.02,
        'stop_loss_atr': 2.0,
        'take_profit_atr': 2.5
    }

    print("\n" + "=" * 70)
    print("BACKTEST AVEC COÛTS RÉALISTES")
    print("=" * 70)
    print(f"Capital: ${args.equity:,.0f}")
    print(f"Params: RSI<{params['long_rsi_below']} (long), RSI>{params['short_rsi_above']} (short)")
    print()

    # Test with different slippage levels
    slippage_levels = [0, 0.0003, 0.0005, 0.001]  # 0%, 0.03%, 0.05%, 0.1%

    all_results = {}

    for symbol in symbols:
        print(f"\n{'#' * 70}")
        print(f"# {symbol}")
        print(f"{'#' * 70}")

        try:
            df = download_data(symbol)
            print(f"  Loaded {len(df)} bars")

            symbol_results = {}

            for slip in slippage_levels:
                result = run_backtest(df, params, slippage_pct=slip, initial_equity=args.equity)
                symbol_results[f"slip_{slip*100:.2f}pct"] = result

            all_results[symbol] = symbol_results

            # Print comparison table
            print(f"\n  {'Slippage':<12} {'Return':<12} {'Sharpe':<10} {'Win Rate':<10} {'Trades':<8} {'Max DD':<10}")
            print(f"  {'-'*62}")

            for slip in slippage_levels:
                key = f"slip_{slip*100:.2f}pct"
                r = symbol_results[key]
                print(f"  {slip*100:>5.2f}%      {r['return_pct']:>+8.1%}    {r['sharpe']:>6.2f}    {r['win_rate']:>6.1%}    {r['trades']:>5}    {r['max_dd']:>7.1%}")

        except Exception as e:
            print(f"  Error: {e}")
            all_results[symbol] = {"error": str(e)}

    # Summary comparison
    print("\n\n" + "=" * 70)
    print("RÉSUMÉ - IMPACT DES COÛTS")
    print("=" * 70)

    valid_results = {k: v for k, v in all_results.items() if 'error' not in v}

    if valid_results:
        print(f"\n{'Symbol':<8} {'Sans coûts':>14} {'Avec 0.05%':>14} {'Différence':>12}")
        print("-" * 50)

        for sym, results in valid_results.items():
            no_cost = results.get('slip_0.00pct', {}).get('return_pct', 0)
            with_cost = results.get('slip_0.05pct', {}).get('return_pct', 0)
            diff = with_cost - no_cost

            print(f"{sym:<8} {no_cost:>+12.1%}   {with_cost:>+12.1%}   {diff:>+10.1%}")

        # Averages
        avg_no_cost = np.mean([r['slip_0.00pct']['return_pct'] for r in valid_results.values()])
        avg_with_cost = np.mean([r['slip_0.05pct']['return_pct'] for r in valid_results.values()])
        avg_diff = avg_with_cost - avg_no_cost

        print("-" * 50)
        print(f"{'MOYENNE':<8} {avg_no_cost:>+12.1%}   {avg_with_cost:>+12.1%}   {avg_diff:>+10.1%}")

        # Expected annual with realistic costs
        avg_sharpe_with_cost = np.mean([r['slip_0.05pct']['sharpe'] for r in valid_results.values()])
        avg_ann_with_cost = np.mean([r['slip_0.05pct']['return_ann'] for r in valid_results.values()])

        print(f"\n{'='*70}")
        print("PERFORMANCE ATTENDUE EN LIVE (avec 0.05% slippage)")
        print(f"{'='*70}")
        print(f"Return annualisé: {avg_ann_with_cost:+.1%}")
        print(f"Sharpe ratio: {avg_sharpe_with_cost:.2f}")
        print(f"Impact slippage: {avg_diff:.1%} sur période totale")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'params': params,
            'results': all_results
        }, f, indent=2, default=str)

    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

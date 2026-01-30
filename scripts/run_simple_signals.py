#!/usr/bin/env python3
"""
HYPRL v5 - Simple Signal Generator
===================================
Génère des signaux basés sur des règles simples.
Pas de ML complexe, juste RSI + Momentum.

Usage:
    python scripts/run_simple_signals.py --symbols NVDA,MSFT,QQQ
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.simple_v5 import compute_features, generate_signal, calculate_position_size


def get_latest_data(symbol: str, period: str = "5d", interval: str = "1h") -> pd.DataFrame:
    """Fetch latest OHLCV data."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)
    df.columns = [c.lower() for c in df.columns]
    return df


def main():
    parser = argparse.ArgumentParser(description="Simple signal generator")
    parser.add_argument("--symbols", default="NVDA,MSFT,QQQ", help="Comma-separated symbols")
    parser.add_argument("--equity", type=float, default=100000, help="Portfolio equity")
    parser.add_argument("--risk-pct", type=float, default=0.02, help="Risk per trade")
    parser.add_argument("--output", default="live/logs/simple_signals.jsonl", help="Output file")
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(",")]
    equity = args.equity
    risk_pct = args.risk_pct
    
    print("=" * 50)
    print("HYPRL v5 - SIMPLE SIGNALS")
    print("=" * 50)
    print(f"Symbols: {symbols}")
    print(f"Equity: ${equity:,.0f}")
    print(f"Risk: {risk_pct:.1%}")
    print()
    
    signals = []
    
    for symbol in symbols:
        print(f"\n{symbol}:")
        
        # Get data
        df = get_latest_data(symbol)
        if len(df) < 20:
            print(f"  ⚠️ Not enough data")
            continue
        
        # Compute features
        df = compute_features(df)
        latest = df.iloc[-1]
        
        # Generate signal
        direction, confidence = generate_signal(latest)
        
        # Get current values
        price = latest['close']
        atr = latest['atr_14']
        rsi = latest['rsi_14']
        momentum = latest['ret_3h']
        
        print(f"  Price: ${price:.2f}")
        print(f"  RSI: {rsi:.1f}")
        print(f"  Momentum 3h: {momentum*100:+.2f}%")
        print(f"  ATR: ${atr:.2f}")
        print(f"  → Signal: {direction.upper()} (conf: {confidence:.2f})")
        
        # Calculate position if signal
        if direction != 'flat':
            shares, stop_dist, pos_value = calculate_position_size(
                equity, price, atr, risk_pct
            )
            
            if direction == 'long':
                stop_price = price - stop_dist
                tp_price = price + stop_dist * 2
            else:
                stop_price = price + stop_dist
                tp_price = price - stop_dist * 2
            
            print(f"  → Shares: {shares:.1f} (${pos_value:,.0f})")
            print(f"  → Stop: ${stop_price:.2f}, TP: ${tp_price:.2f}")
            
            signal = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": direction,
                "confidence": confidence,
                "price": price,
                "shares": shares,
                "position_value": pos_value,
                "stop_price": stop_price,
                "take_profit": tp_price,
                "rsi": rsi,
                "momentum": momentum,
                "atr": atr
            }
            signals.append(signal)
        else:
            signals.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": symbol,
                "direction": "flat",
                "confidence": 0,
                "price": price,
                "rsi": rsi,
                "momentum": momentum
            })
    
    # Save signals
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "a") as f:
        for sig in signals:
            f.write(json.dumps(sig) + "\n")
    
    print(f"\n✓ Signals saved to {output_path}")
    
    # Summary
    actionable = [s for s in signals if s['direction'] != 'flat']
    print(f"\n📊 Summary: {len(actionable)}/{len(signals)} actionable signals")
    
    for sig in actionable:
        print(f"  → {sig['symbol']}: {sig['direction'].upper()} {sig['shares']:.0f} shares @ ${sig['price']:.2f}")


if __name__ == "__main__":
    main()

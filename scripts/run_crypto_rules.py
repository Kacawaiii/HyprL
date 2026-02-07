#!/usr/bin/env python3
"""
Generate crypto signals using pure rule-based strategy.
No ML, no overfitting - just trend following with pullback entries.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

# Alpaca imports
try:
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
except ImportError:
    print("ERROR: alpaca-py not installed")
    exit(1)

from hyprl.crypto.strategy_rules import get_rule_based_signal


def fetch_bars(symbol: str, days: int = 30) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fetch OHLCV bars from Alpaca."""

    client = CryptoHistoricalDataClient()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)

    request = CryptoBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Hour),
        start=start,
        end=end,
    )

    bars = client.get_crypto_bars(request)
    bar_data = bars.data if hasattr(bars, "data") else bars

    if symbol not in bar_data:
        return np.array([]), np.array([]), np.array([]), np.array([])

    bar_list = list(bar_data[symbol])

    opens = np.array([float(b.open) for b in bar_list])
    highs = np.array([float(b.high) for b in bar_list])
    lows = np.array([float(b.low) for b in bar_list])
    closes = np.array([float(b.close) for b in bar_list])

    return opens, highs, lows, closes


def generate_signal(symbol: str) -> dict:
    """Generate signal for a symbol."""

    opens, highs, lows, closes = fetch_bars(symbol, days=30)

    if len(closes) < 200:
        return {
            "symbol": symbol,
            "direction": "flat",
            "probability": 0.5,
            "confidence": 0.0,
            "entry_price": 0,
            "stop_loss": 0,
            "take_profit": 0,
            "reason": f"insufficient_data ({len(closes)} bars, need 200)",
        }

    signal = get_rule_based_signal(symbol, closes, highs, lows)
    signal["timestamp"] = datetime.now(timezone.utc).isoformat()

    return signal


def main():
    parser = argparse.ArgumentParser(description="Rule-based crypto signals")
    parser.add_argument("--symbols", default="BTC/USD,ETH/USD", help="Comma-separated symbols")
    parser.add_argument("--output", default="live/logs/crypto_signals.jsonl", help="Output file")
    parser.add_argument("--verbose", action="store_true", help="Print signals to stdout")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    signals = []
    for symbol in symbols:
        try:
            signal = generate_signal(symbol)
            signals.append(signal)

            if args.verbose:
                print(f"{symbol}: {signal['direction']} (conf={signal['confidence']:.2f}) - {signal['reason']}")

        except Exception as e:
            print(f"ERROR {symbol}: {e}")
            signals.append({
                "symbol": symbol,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "direction": "flat",
                "probability": 0.5,
                "confidence": 0.0,
                "entry_price": 0,
                "stop_loss": 0,
                "take_profit": 0,
                "reason": f"error: {e}",
            })

    # Append to output file
    with output_path.open("a", encoding="utf-8") as f:
        for signal in signals:
            f.write(json.dumps(signal) + "\n")

    print(f"Wrote {len(signals)} signals to {output_path}")


if __name__ == "__main__":
    main()

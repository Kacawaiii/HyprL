#!/usr/bin/env python3
"""Run Crypto Signal Generator.

Generates signals for BTC, ETH and other cryptocurrencies.
Can be run 24/7 as crypto markets never close.

Usage:
    python scripts/run_crypto_signals.py --symbols BTC/USD ETH/USD
    python scripts/run_crypto_signals.py --all
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.crypto.trader import CryptoConfig, format_crypto_signal
from hyprl.crypto.signals import CryptoSignalGenerator, format_crypto_scan

DEFAULT_ENV_FILES = (
    ".env.ops",
    ".env.broker.alpaca",
    ".env.bridge",
)


def load_env_file(path: Path, override: bool = False) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith(";"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not override and key in os.environ:
            continue
        os.environ[key] = value


def normalize_alpaca_env() -> None:
    if not os.environ.get("APCA_API_KEY_ID"):
        legacy = os.environ.get("ALPACA_API_KEY")
        if legacy:
            os.environ["APCA_API_KEY_ID"] = legacy
    if not os.environ.get("APCA_API_SECRET_KEY"):
        legacy = os.environ.get("ALPACA_SECRET_KEY")
        if legacy:
            os.environ["APCA_API_SECRET_KEY"] = legacy
    if not os.environ.get("APCA_API_BASE_URL"):
        legacy = os.environ.get("ALPACA_API_BASE_URL")
        if legacy:
            os.environ["APCA_API_BASE_URL"] = legacy

def main():
    parser = argparse.ArgumentParser(description="Crypto Signal Generator")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC/USD", "ETH/USD"],
        help="Crypto symbols to analyze"
    )
    parser.add_argument("--all", action="store_true", help="Scan all supported cryptos")
    parser.add_argument("--timeframe", default="1Hour", help="Timeframe (1Min, 15Min, 1Hour)")
    parser.add_argument("--threshold-long", type=float, default=0.58, help="Long threshold")
    parser.add_argument("--threshold-short", type=float, default=0.42, help="Short threshold")
    parser.add_argument("--max-position-pct", type=float, default=0.10, help="Max position %%")
    parser.add_argument("--stop-loss-pct", type=float, default=0.03, help="Stop loss %%")
    parser.add_argument("--take-profit-pct", type=float, default=0.06, help="Take profit %%")
    parser.add_argument(
        "--signal-mode",
        choices=["ml", "policy", "rule"],
        default="ml",
        help="Signal mode (ml, policy, rule).",
    )
    parser.add_argument(
        "--policy-symbols",
        nargs="+",
        default=[],
        help="Symbols that should use policy mode when signal-mode=ml.",
    )
    parser.add_argument(
        "--env-file",
        action="append",
        default=[],
        help="Optional env file to load (repeatable). Example: --env-file /path/to/.env.ops",
    )
    parser.add_argument("--json", action="store_true", help="Output JSON")
    parser.add_argument("--base-dir", default=".", help="Base directory")
    args = parser.parse_args()

    for env_file in DEFAULT_ENV_FILES:
        load_env_file(Path(env_file))
    for env_file in args.env_file:
        load_env_file(Path(env_file), override=True)
    normalize_alpaca_env()

    # All supported symbols
    all_symbols = [
        "BTC/USD", "ETH/USD", "LTC/USD", "LINK/USD",
        "UNI/USD", "AAVE/USD", "AVAX/USD", "SOL/USD"
    ]

    symbols = all_symbols if args.all else args.symbols

    config = CryptoConfig(
        symbols=symbols,
        timeframe=args.timeframe,
        threshold_long=args.threshold_long,
        threshold_short=args.threshold_short,
        max_position_pct=args.max_position_pct,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        signal_mode=args.signal_mode,
        policy_symbols=args.policy_symbols,
    )

    generator = CryptoSignalGenerator(config, base_dir=args.base_dir)
    signals = generator.scan_all()

    if args.json:
        import json
        output = [
            {
                "symbol": s.symbol,
                "direction": s.direction,
                "probability": s.probability,
                "confidence": s.confidence,
                "size_pct": s.size_pct,
                "entry_price": s.entry_price,
                "stop_loss": s.stop_loss,
                "take_profit": s.take_profit,
                "reason": s.reason,
                "timestamp": s.timestamp.isoformat(),
            }
            for s in signals
        ]
        print(json.dumps(output, indent=2))
    else:
        print(format_crypto_scan(signals))

    # Return non-zero if no actionable signals
    actionable = [s for s in signals if s.direction not in ("neutral", "flat")]
    return 0 if actionable else 1


if __name__ == "__main__":
    sys.exit(main())

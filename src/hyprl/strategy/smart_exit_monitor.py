"""
Smart Exit Monitor for HyprL
Monitors positions and sentiment to generate smart exit signals.
Runs as a background service alongside the trading bridge.
"""

import time
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.hyprl.sentiment.multi_source import MultiSourceSentiment
from src.hyprl.strategy.position_aware import PositionAwareTrading, ExitSignal


class SmartExitMonitor:
    """
    Monitors positions and sentiment to recommend exits.

    Flow:
    1. Load current positions from Alpaca
    2. Fetch sentiment for each symbol
    3. Evaluate exit rules
    4. Write recommendations to file for bridge to read
    """

    def __init__(
        self,
        check_interval: int = 300,  # 5 minutes
        output_file: Path = Path("live/logs/exit_signals.jsonl"),
        alert_file: Path = Path("live/logs/exit_alerts.json")
    ):
        self.check_interval = check_interval
        self.output_file = output_file
        self.alert_file = alert_file

        self.sentiment = MultiSourceSentiment()
        self.trading = PositionAwareTrading(
            state_file=Path("data/position_state.json")
        )

    def sync_positions_from_alpaca(self, env_file: Path):
        """Sync positions from Alpaca account."""
        try:
            import os
            from alpaca.trading.client import TradingClient

            # Load env
            env = {}
            with open(env_file) as f:
                for line in f:
                    if "=" in line and not line.startswith("#"):
                        k, v = line.strip().split("=", 1)
                        env[k] = v

            client = TradingClient(
                env["ALPACA_API_KEY"],
                env["ALPACA_SECRET_KEY"],
                paper=True
            )

            positions = client.get_all_positions()

            # Track which symbols we have
            current_symbols = set()

            for pos in positions:
                symbol = pos.symbol
                current_symbols.add(symbol)

                qty = float(pos.qty)
                side = "long" if qty > 0 else "short"
                qty = abs(qty)

                entry_price = float(pos.avg_entry_price)
                current_price = float(pos.current_price)

                # Check if we already track this position
                existing = self.trading.tracker.get_position(symbol)

                if existing:
                    # Update price
                    self.trading.update_market_data(
                        symbol=symbol,
                        current_price=current_price
                    )
                else:
                    # New position - get current sentiment
                    sent = self.sentiment.get_sentiment(symbol)

                    self.trading.on_trade_entry(
                        symbol=symbol,
                        side=side,
                        price=entry_price,
                        quantity=qty,
                        stop_loss=0,  # We don't know from Alpaca
                        take_profit=0,
                        sentiment_score=sent.final_score,
                        sentiment_confidence=sent.final_confidence
                    )
                    print(f"[+] Tracking new position: {symbol} {side} @ ${entry_price:.2f}")

            # Remove closed positions
            tracked_symbols = set(p.symbol for p in self.trading.tracker.get_all_positions())
            for sym in tracked_symbols - current_symbols:
                self.trading.on_trade_exit(sym)
                print(f"[-] Position closed: {sym}")

            return len(current_symbols)

        except Exception as e:
            print(f"Error syncing from Alpaca: {e}")
            return 0

    def update_sentiment(self):
        """Update sentiment for all tracked positions."""
        for pos in self.trading.tracker.get_all_positions():
            try:
                sent = self.sentiment.get_sentiment(pos.symbol)
                self.trading.update_market_data(
                    symbol=pos.symbol,
                    current_price=pos.current_price,  # Keep current
                    sentiment_score=sent.final_score,
                    sentiment_confidence=sent.final_confidence
                )
                print(f"[~] Updated sentiment for {pos.symbol}: {sent.final_score:+.2f}")
            except Exception as e:
                print(f"Error updating sentiment for {pos.symbol}: {e}")

    def evaluate_and_alert(self) -> list[ExitSignal]:
        """Evaluate all positions and return urgent signals."""
        signals = self.trading.evaluate_all_positions()

        urgent_signals = [s for s in signals if s.urgency in ("immediate", "soon")]

        if urgent_signals:
            # Write alerts
            alerts = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alerts": [
                    {
                        "symbol": s.symbol,
                        "action": s.action,
                        "reason": s.reason,
                        "urgency": s.urgency,
                        "confidence": s.confidence,
                        "message": s.details.get("message", "")
                    }
                    for s in urgent_signals
                ]
            }

            self.alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alert_file, "w") as f:
                json.dump(alerts, f, indent=2)

            # Also append to log
            self.output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "a") as f:
                for s in urgent_signals:
                    record = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": s.symbol,
                        "action": s.action,
                        "reason": s.reason,
                        "urgency": s.urgency,
                        "confidence": s.confidence,
                        "details": s.details
                    }
                    f.write(json.dumps(record) + "\n")

        return urgent_signals

    def run_once(self, env_file: Path) -> str:
        """Run one check cycle and return status report."""
        print(f"\n{'='*50}")
        print(f"Smart Exit Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*50}\n")

        # Sync positions
        n_positions = self.sync_positions_from_alpaca(env_file)
        print(f"Positions actives: {n_positions}")

        if n_positions == 0:
            return "Aucune position à monitorer."

        # Update sentiment
        print("\nMise à jour sentiment...")
        self.update_sentiment()

        # Evaluate
        print("\nÉvaluation des positions...")
        urgent_signals = self.evaluate_and_alert()

        # Report
        report = self.trading.get_status_report()
        print("\n" + report)

        if urgent_signals:
            print("\n⚠️  ALERTES URGENTES:")
            for s in urgent_signals:
                print(f"  [{s.urgency.upper()}] {s.symbol}: {s.action} - {s.reason}")
                if s.details.get("message"):
                    print(f"    → {s.details['message']}")

        return report

    def run_loop(self, env_file: Path):
        """Run continuous monitoring loop."""
        print(f"Smart Exit Monitor starting...")
        print(f"Check interval: {self.check_interval}s")
        print(f"Alert file: {self.alert_file}")

        while True:
            try:
                self.run_once(env_file)
            except Exception as e:
                print(f"Error in monitoring loop: {e}")

            print(f"\nNext check in {self.check_interval}s...")
            time.sleep(self.check_interval)


def main():
    parser = argparse.ArgumentParser(description="Smart Exit Monitor")
    parser.add_argument("--env", type=Path, required=True, help="Alpaca env file")
    parser.add_argument("--interval", type=int, default=300, help="Check interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    monitor = SmartExitMonitor(check_interval=args.interval)

    if args.once:
        monitor.run_once(args.env)
    else:
        monitor.run_loop(args.env)


if __name__ == "__main__":
    main()

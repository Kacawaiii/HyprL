#!/usr/bin/env python3
"""
HYPRL Live Trading with Full Monitoring
========================================
Script complet pour le trading live avec:
- Health monitoring (heartbeat)
- Telegram notifications
- Trade journal
- Regime detection
- Risk monitoring

Usage:
    python scripts/run_live_monitored.py --config configs/runtime/strategy_mix.yaml
    python scripts/run_live_monitored.py --config configs/runtime/strategy_aggressive.yaml
"""

import argparse
import json
import sys
import time
import yaml
import signal
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import yfinance as yf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.simple_v5 import compute_features, generate_signal, calculate_position_size
from hyprl.monitoring import (
    # Health
    get_health_checker,
    get_risk_monitor,
    # Telegram
    get_notifier,
    # Journal
    get_journal,
    # Regime
    detect_regime,
    get_regime_detector,
    MarketRegime,
)


class LiveTradingBot:
    """Live trading bot with full monitoring."""

    def __init__(self, config_path: str, dry_run: bool = True):
        self.config = self._load_config(config_path)
        self.dry_run = dry_run

        # Initialize monitoring
        self.health = get_health_checker()
        self.risk = get_risk_monitor()
        self.journal = get_journal()
        self.notifier = get_notifier()
        self.regime_detector = get_regime_detector()

        # State
        self.equity = 100000  # Will be updated from broker
        self.open_positions: Dict[str, Dict] = {}
        self.running = False
        self.trade_count_today = 0
        self.daily_pnl = 0.0

        # Load config values
        self._apply_config()

    def _load_config(self, path: str) -> Dict[str, Any]:
        """Load strategy config from YAML."""
        with open(path) as f:
            return yaml.safe_load(f)

    def _apply_config(self) -> None:
        """Apply config values."""
        # Strategy rules
        self.rules = self.config.get("rules", {})

        # Capital management
        cap = self.config.get("capital", {})
        self.risk_per_trade = cap.get("risk_per_trade", 0.02)
        self.max_position_pct = cap.get("max_position_pct", 0.25)

        # Risk limits
        risk = self.config.get("risk", {})
        self.stop_loss_atr = risk.get("stop_loss_atr", 2.0)
        self.take_profit_atr = risk.get("take_profit_atr", 2.5)
        self.max_daily_loss = risk.get("max_daily_loss_pct", 0.05)
        self.max_drawdown = risk.get("max_drawdown_pct", 0.12)

        # Update risk monitor
        self.risk.max_daily_loss_pct = self.max_daily_loss
        self.risk.max_drawdown_pct = self.max_drawdown

        # Symbols
        self.symbols = []
        for sym, cfg in self.config.get("symbols", {}).items():
            if cfg.get("enabled", True):
                self.symbols.append(sym)

        # Execution
        exec_cfg = self.config.get("execution", {})
        self.max_orders_per_day = exec_cfg.get("max_orders_per_day", 15)
        self.cooldown_seconds = exec_cfg.get("cooldown_seconds", 300)
        self.close_eod = exec_cfg.get("close_positions_eod", False)

    def start(self) -> None:
        """Start the trading bot."""
        self.running = True

        # Start health monitoring
        self.health.start()
        self.risk.set_initial_equity(self.equity)

        # Notify startup
        strategy_name = self.config.get("strategy", {}).get("name", "unknown")
        self.notifier.notify_startup(strategy_name, self.symbols)

        print("\n" + "=" * 60)
        print(f"HYPRL LIVE TRADING - {strategy_name.upper()}")
        print("=" * 60)
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Equity: ${self.equity:,.0f}")
        print(f"Risk/trade: {self.risk_per_trade:.1%}")
        print(f"Dry run: {self.dry_run}")
        print("=" * 60 + "\n")

    def stop(self, reason: str = "manual") -> None:
        """Stop the trading bot."""
        self.running = False
        self.health.stop()
        self.notifier.notify_shutdown(reason)

        # Print journal summary
        self.journal.print_summary()

    def get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch latest market data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1h")
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            print(f"[ERROR] Failed to get data for {symbol}: {e}")
            return None

    def analyze_symbol(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze a symbol and generate signal."""
        # Compute features
        df = compute_features(df)
        latest = df.iloc[-1]

        # Detect regime
        regime = self.regime_detector.detect(df)
        adjustments = self.regime_detector.get_strategy_adjustment(regime)

        # Generate signal
        direction, confidence = generate_signal(
            latest,
            long_rsi_below=self.rules.get("long", {}).get("rsi_below", 45),
            long_momentum_above=self.rules.get("long", {}).get("momentum_above", 0.004),
            short_rsi_above=self.rules.get("short", {}).get("rsi_above", 65),
            short_momentum_below=self.rules.get("short", {}).get("momentum_below", -0.004)
        )

        # Apply regime adjustment
        if adjustments.get("favor_direction"):
            if direction != adjustments["favor_direction"] and direction != "flat":
                confidence *= 0.5  # Reduce confidence against trend

        price = float(latest['close'])
        atr = float(latest['atr_14'])
        rsi = float(latest['rsi_14'])
        momentum = float(latest['ret_3h'])

        result = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "atr": atr,
            "rsi": rsi,
            "momentum": momentum,
            "direction": direction,
            "confidence": confidence,
            "regime": regime.regime.value,
            "regime_confidence": regime.confidence,
            "regime_recommendation": regime.recommendation
        }

        # Calculate position if signal
        if direction != "flat" and confidence > 0.5:
            # Apply regime sizing
            size_mult = adjustments.get("position_size_multiplier", 1.0)
            adjusted_risk = self.risk_per_trade * size_mult

            shares, stop_dist, pos_value = calculate_position_size(
                self.equity, price, atr, adjusted_risk,
                max_position_pct=self.max_position_pct,
                atr_multiplier=self.stop_loss_atr
            )

            if direction == "long":
                stop_price = price - stop_dist
                tp_price = price + stop_dist * (self.take_profit_atr / self.stop_loss_atr)
            else:
                stop_price = price + stop_dist
                tp_price = price - stop_dist * (self.take_profit_atr / self.stop_loss_atr)

            result.update({
                "shares": shares,
                "position_value": pos_value,
                "stop_price": stop_price,
                "take_profit": tp_price,
                "risk_amount": shares * stop_dist
            })

        return result

    def execute_signal(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute a trading signal. Returns trade_id if executed."""
        symbol = signal["symbol"]
        direction = signal["direction"]

        # Check if we can trade
        if self.trade_count_today >= self.max_orders_per_day:
            print(f"[SKIP] Max orders reached for today")
            return None

        if symbol in self.open_positions:
            print(f"[SKIP] Already have position in {symbol}")
            return None

        shares = signal.get("shares", 0)
        if shares <= 0:
            print(f"[SKIP] Invalid position size")
            return None

        # Execute trade (dry run or real)
        if self.dry_run:
            print(f"[DRY RUN] Would {direction} {shares:.0f} shares of {symbol}")
        else:
            # TODO: Connect to real broker
            pass

        # Log to journal
        trade_id = self.journal.open_trade(
            symbol=symbol,
            direction=direction,
            shares=shares,
            entry_price=signal["price"],
            stop_price=signal["stop_price"],
            tp_price=signal["take_profit"],
            strategy=self.config.get("strategy", {}).get("name", "mvp"),
            rsi=signal["rsi"],
            momentum=signal["momentum"],
            notes=f"Regime: {signal['regime']}"
        )

        # Send Telegram notification
        self.notifier.notify_trade_entry(
            symbol=symbol,
            direction=direction,
            shares=shares,
            price=signal["price"],
            stop_price=signal["stop_price"],
            tp_price=signal["take_profit"],
            risk_amount=signal.get("risk_amount", 0)
        )

        # Track position
        self.open_positions[symbol] = {
            "trade_id": trade_id,
            "direction": direction,
            "shares": shares,
            "entry_price": signal["price"],
            "stop_price": signal["stop_price"],
            "take_profit": signal["take_profit"]
        }

        self.trade_count_today += 1

        # Update health stats
        self.health.update_stats(
            open_positions=len(self.open_positions),
            trades_today=self.trade_count_today,
            equity=self.equity
        )

        return trade_id

    def check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be exited."""
        if symbol not in self.open_positions:
            return None

        pos = self.open_positions[symbol]
        direction = pos["direction"]
        stop = pos["stop_price"]
        tp = pos["take_profit"]

        exit_reason = None

        if direction == "long":
            if current_price <= stop:
                exit_reason = "stop"
            elif current_price >= tp:
                exit_reason = "tp"
        else:  # short
            if current_price >= stop:
                exit_reason = "stop"
            elif current_price <= tp:
                exit_reason = "tp"

        if exit_reason:
            self._close_position(symbol, current_price, exit_reason)

        return exit_reason

    def _close_position(self, symbol: str, exit_price: float, reason: str) -> None:
        """Close a position."""
        if symbol not in self.open_positions:
            return

        pos = self.open_positions.pop(symbol)

        # Close in journal
        record = self.journal.close_trade(
            trade_id=pos["trade_id"],
            exit_price=exit_price,
            exit_reason=reason
        )

        if record:
            pnl = record.pnl
            pnl_pct = record.pnl_pct

            # Update daily PnL
            self.daily_pnl += pnl

            # Update risk monitor
            risk_status = self.risk.update(
                equity=self.equity + pnl,
                last_trade_pnl=pnl
            )

            # Send notification
            self.notifier.notify_trade_exit(
                symbol=symbol,
                direction=pos["direction"],
                pnl=pnl,
                pnl_pct=pnl_pct,
                exit_reason=reason
            )

            # Check if we should stop trading
            if risk_status.get("should_stop"):
                print("[RISK] Critical alert - stopping trading!")
                self.stop("risk_limit_hit")

    def run_cycle(self) -> None:
        """Run one trading cycle."""
        print(f"\n[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] Running cycle...")

        for symbol in self.symbols:
            # Get data
            df = self.get_market_data(symbol)
            if df is None or len(df) < 20:
                continue

            current_price = float(df['close'].iloc[-1])

            # Check exits first
            exit_reason = self.check_exits(symbol, current_price)
            if exit_reason:
                print(f"  {symbol}: Exited ({exit_reason})")
                continue

            # Analyze for new signals
            signal = self.analyze_symbol(symbol, df)

            # Print status
            print(f"  {symbol}: ${signal['price']:.2f} | RSI={signal['rsi']:.0f} | "
                  f"Mom={signal['momentum']*100:+.2f}% | "
                  f"Regime={signal['regime']} | "
                  f"Signal={signal['direction'].upper()}")

            # Execute if actionable
            if signal["direction"] != "flat" and signal.get("shares", 0) > 0:
                if signal["confidence"] > 0.5:
                    self.execute_signal(signal)

        # Update health
        self.health.update_stats(
            last_cycle=datetime.now(timezone.utc).isoformat(),
            daily_pnl=self.daily_pnl,
            open_positions=len(self.open_positions)
        )

    def run(self, interval_seconds: int = 300) -> None:
        """Main run loop."""
        self.start()

        # Handle graceful shutdown
        def signal_handler(sig, frame):
            print("\n[SIGNAL] Stopping...")
            self.stop("signal")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self.running:
                self.run_cycle()

                # Wait for next cycle
                print(f"\n  Next cycle in {interval_seconds}s...")
                for _ in range(interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)

        except Exception as e:
            print(f"[ERROR] {e}")
            self.stop(f"error: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="HYPRL Live Trading with Monitoring")
    parser.add_argument("--config", required=True, help="Path to strategy config YAML")
    parser.add_argument("--equity", type=float, default=100000, help="Initial equity")
    parser.add_argument("--interval", type=int, default=300, help="Cycle interval in seconds")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Dry run mode")
    parser.add_argument("--live", action="store_true", help="Live trading (disable dry run)")
    parser.add_argument("--once", action="store_true", help="Run only one cycle")
    args = parser.parse_args()

    dry_run = not args.live

    bot = LiveTradingBot(args.config, dry_run=dry_run)
    bot.equity = args.equity

    if args.once:
        bot.start()
        bot.run_cycle()
        bot.stop("single_run")
    else:
        bot.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════
HYPRL PRODUCTION TRADING BOT
════════════════════════════════════════════════════════════════════════

Script de trading production avec TOUTES les sécurités:
- Position sizing validé (min/max)
- Stop loss OBLIGATOIRE
- Fermeture EOD automatique
- Circuit breakers multi-niveaux
- Alertes inactivité
- Kill switch externe
- Monitoring complet

Usage:
    python scripts/run_production.py --config configs/runtime/strategy_production.yaml

════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import json
import time
import yaml
import signal
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyprl.strategy.simple_v5 import compute_features, generate_signal, calculate_position_size
from hyprl.monitoring import (
    get_health_checker,
    get_risk_monitor,
    get_notifier,
    get_journal,
    detect_regime,
    MarketRegime,
    get_reporter,
)


# ════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ════════════════════════════════════════════════════════════════════

class TradingState(Enum):
    RUNNING = "running"
    PAUSED = "paused"
    EOD_CLOSING = "eod_closing"
    STOPPED = "stopped"
    CIRCUIT_BREAKER = "circuit_breaker"


@dataclass
class Position:
    symbol: str
    direction: str
    shares: float
    entry_price: float
    stop_price: float
    tp_price: float
    entry_time: datetime
    trade_id: str
    trailing_active: bool = False
    trail_stop: float = 0.0


@dataclass
class TradingStats:
    trades_today: int = 0
    wins_today: int = 0
    losses_today: int = 0
    pnl_today: float = 0.0
    consecutive_losses: int = 0
    signals_generated: int = 0
    signals_skipped: int = 0
    last_signal_time: Optional[datetime] = None
    last_trade_time: Optional[datetime] = None


# ════════════════════════════════════════════════════════════════════
# PRODUCTION TRADING BOT
# ════════════════════════════════════════════════════════════════════

class ProductionBot:
    """Production trading bot with all safety features."""

    def __init__(self, config_path: str, dry_run: bool = True):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.dry_run = dry_run
        self.state = TradingState.STOPPED

        # Initialize components
        self.health = get_health_checker()
        self.risk_monitor = get_risk_monitor()
        self.notifier = get_notifier()
        self.journal = get_journal()
        self.reporter = get_reporter()

        # State
        self.equity = 0.0
        self.initial_equity = 0.0
        self.daily_start_equity = 0.0
        self.positions: Dict[str, Position] = {}
        self.stats = TradingStats()
        self.cooldowns: Dict[str, datetime] = {}

        # Extract config sections
        self._load_config()

        # Threading
        self._running = False
        self._lock = threading.Lock()

    def _load_config(self):
        """Load all config values."""
        # Capital
        cap = self.config.get("capital", {})
        self.base_risk = cap.get("base_risk_per_trade", 0.015)
        self.high_conf_risk = cap.get("high_confidence_risk", 0.025)
        self.min_position_usd = cap.get("min_position_usd", 500)
        self.max_position_usd = cap.get("max_position_usd", 25000)
        self.max_position_pct = cap.get("max_position_pct", 0.15)
        self.max_exposure = cap.get("max_total_exposure", 1.0)
        self.min_cash_reserve = cap.get("min_cash_reserve_pct", 0.10)

        # Risk
        risk = self.config.get("risk", {})
        self.stop_atr = risk.get("stop_loss_atr", 2.0)
        self.stop_max_pct = risk.get("stop_loss_max_pct", 0.05)
        self.tp_atr = risk.get("take_profit_atr", 3.0)
        self.use_trailing = risk.get("use_trailing_stop", True)
        self.trailing_activation = risk.get("trailing_activation_pct", 0.025)
        self.trailing_distance = risk.get("trailing_distance_pct", 0.012)
        self.max_daily_loss = risk.get("max_daily_loss_pct", 0.04)
        self.max_drawdown = risk.get("max_drawdown_pct", 0.12)

        # Guards
        guards = self.config.get("guards", {})
        self.min_confidence = guards.get("min_confidence", 0.5)
        self.max_consecutive_losses = guards.get("max_consecutive_losses", 4)
        self.cooldown_after_trade = guards.get("cooldown_after_trade_seconds", 180)
        self.cooldown_after_loss = guards.get("cooldown_after_loss_seconds", 300)
        self.kill_switch_file = guards.get("kill_switch_file", ".kill_switch")
        self.circuit_breakers = guards.get("circuit_breaker_levels", [])

        # EOD
        eod = self.config.get("eod", {})
        self.close_eod = eod.get("close_all_positions", True)
        self.eod_start_minutes = eod.get("close_start_minutes_before", 15)
        self.eod_deadline_minutes = eod.get("close_deadline_minutes_before", 5)
        self.no_new_trades_minutes = eod.get("no_new_trades_minutes_before", 30)

        # Alerts
        alerts = self.config.get("alerts", {})
        self.no_signal_alert_hours = alerts.get("inactivity_alerts", {}).get("no_signal_alert_hours", 4)
        self.no_trade_alert_hours = alerts.get("inactivity_alerts", {}).get("no_trade_alert_hours", 8)

        # Rules
        rules = self.config.get("rules", {})
        self.long_rsi_below = rules.get("long", {}).get("rsi_below", 45)
        self.long_momentum_above = rules.get("long", {}).get("momentum_above", 0.004)
        self.short_rsi_above = rules.get("short", {}).get("rsi_above", 65)
        self.short_momentum_below = rules.get("short", {}).get("momentum_below", -0.004)

        # Symbols
        self.symbols = []
        for sym, cfg in self.config.get("symbols", {}).items():
            if cfg.get("enabled", True):
                self.symbols.append(sym)

        # Schedule
        schedule = self.config.get("schedule", {})
        self.trading_start = schedule.get("trading_start", "09:45")
        self.trading_end = schedule.get("trading_end", "15:30")
        self.market_close = schedule.get("market_close", "16:00")

    # ════════════════════════════════════════════════════════════════
    # SAFETY CHECKS
    # ════════════════════════════════════════════════════════════════

    def _check_kill_switch(self) -> bool:
        """Check if kill switch is active."""
        if Path(self.kill_switch_file).exists():
            return True
        return False

    def _check_circuit_breakers(self) -> Optional[Dict]:
        """Check circuit breaker levels."""
        if not self.daily_start_equity:
            return None

        daily_pnl_pct = (self.equity - self.daily_start_equity) / self.daily_start_equity

        for level in sorted(self.circuit_breakers, key=lambda x: x.get("loss_pct", 0)):
            if daily_pnl_pct <= -level.get("loss_pct", 1):
                return level
        return None

    def _check_trading_hours(self) -> Dict[str, Any]:
        """Check if within trading hours."""
        now = datetime.now()  # Local time (should be NY)

        # Parse times
        start_h, start_m = map(int, self.trading_start.split(":"))
        end_h, end_m = map(int, self.trading_end.split(":"))
        close_h, close_m = map(int, self.market_close.split(":"))

        trading_start = now.replace(hour=start_h, minute=start_m, second=0)
        trading_end = now.replace(hour=end_h, minute=end_m, second=0)
        market_close = now.replace(hour=close_h, minute=close_m, second=0)

        # Check EOD
        minutes_to_close = (market_close - now).total_seconds() / 60
        is_eod_closing = minutes_to_close <= self.eod_start_minutes
        no_new_trades = minutes_to_close <= self.no_new_trades_minutes

        return {
            "in_hours": trading_start <= now <= trading_end,
            "is_eod": is_eod_closing,
            "no_new_trades": no_new_trades,
            "minutes_to_close": minutes_to_close
        }

    def _validate_position_size(self, shares: float, price: float, symbol: str) -> Dict[str, Any]:
        """Validate position size against all limits."""
        position_value = shares * price

        errors = []
        warnings = []

        # Check minimum
        if position_value < self.min_position_usd:
            errors.append(f"Position ${position_value:.0f} < min ${self.min_position_usd}")

        # Check maximum
        if position_value > self.max_position_usd:
            warnings.append(f"Position ${position_value:.0f} > max ${self.max_position_usd}, capping")
            shares = self.max_position_usd / price
            position_value = shares * price

        # Check % of portfolio
        if self.equity > 0:
            pct = position_value / self.equity
            if pct > self.max_position_pct:
                warnings.append(f"Position {pct:.1%} > max {self.max_position_pct:.1%}, capping")
                position_value = self.equity * self.max_position_pct
                shares = position_value / price

        # Check total exposure
        current_exposure = sum(p.shares * p.entry_price for p in self.positions.values())
        new_exposure = (current_exposure + position_value) / self.equity if self.equity > 0 else 0
        if new_exposure > self.max_exposure:
            errors.append(f"Total exposure {new_exposure:.1%} > max {self.max_exposure:.1%}")

        # Check cash reserve
        available_cash = self.equity - current_exposure
        if position_value > available_cash * (1 - self.min_cash_reserve):
            errors.append(f"Insufficient cash (need ${position_value:.0f}, have ${available_cash:.0f})")

        return {
            "valid": len(errors) == 0,
            "shares": shares,
            "value": position_value,
            "errors": errors,
            "warnings": warnings
        }

    def _check_cooldown(self, symbol: str) -> bool:
        """Check if symbol is in cooldown."""
        if symbol in self.cooldowns:
            if datetime.now(timezone.utc) < self.cooldowns[symbol]:
                return True
        return False

    def _check_inactivity(self):
        """Check for inactivity and send alerts."""
        now = datetime.now(timezone.utc)

        # No signals alert
        if self.stats.last_signal_time:
            hours_since_signal = (now - self.stats.last_signal_time).total_seconds() / 3600
            if hours_since_signal > self.no_signal_alert_hours:
                self._alert(f"⚠️ INACTIVITÉ: Aucun signal depuis {hours_since_signal:.1f}h")
                self.stats.last_signal_time = now  # Reset to avoid spam

        # No trades alert (if signals exist)
        if self.stats.signals_generated > 0 and self.stats.trades_today == 0:
            if self.stats.last_signal_time:
                hours_since_signal = (now - self.stats.last_signal_time).total_seconds() / 3600
                if hours_since_signal > self.no_trade_alert_hours:
                    self._alert(f"⚠️ INACTIVITÉ: {self.stats.signals_generated} signaux mais 0 trades")

    # ════════════════════════════════════════════════════════════════
    # DATA & SIGNALS
    # ════════════════════════════════════════════════════════════════

    def _get_market_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch latest market data."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="5d", interval="1h")
            df.columns = [c.lower() for c in df.columns]
            return df
        except Exception as e:
            self._log(f"[ERROR] Data fetch failed for {symbol}: {e}")
            return None

    def _generate_signal(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate trading signal with all validations."""
        df = compute_features(df)
        row = df.iloc[-1]

        price = float(row['close'])
        atr = float(row['atr_14'])
        rsi = float(row['rsi_14'])
        momentum = float(row['ret_3h'])
        volume_ratio = float(row.get('vol_ratio', 1.0))

        # Generate base signal
        direction, confidence = generate_signal(
            row,
            long_rsi_below=self.long_rsi_below,
            long_momentum_above=self.long_momentum_above,
            short_rsi_above=self.short_rsi_above,
            short_momentum_below=self.short_momentum_below
        )

        signal = {
            "symbol": symbol,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": price,
            "atr": atr,
            "rsi": rsi,
            "momentum": momentum,
            "volume_ratio": volume_ratio,
            "direction": direction,
            "confidence": confidence,
            "valid": False,
            "skip_reason": None
        }

        # Update stats
        if direction != "flat":
            self.stats.signals_generated += 1
            self.stats.last_signal_time = datetime.now(timezone.utc)

        # Validate signal
        if direction == "flat":
            signal["skip_reason"] = "no_signal"
            return signal

        if confidence < self.min_confidence:
            signal["skip_reason"] = f"low_confidence ({confidence:.2f} < {self.min_confidence})"
            self.stats.signals_skipped += 1
            return signal

        if volume_ratio < 0.8:
            signal["skip_reason"] = f"low_volume ({volume_ratio:.2f})"
            self.stats.signals_skipped += 1
            return signal

        if self._check_cooldown(symbol):
            signal["skip_reason"] = "cooldown"
            self.stats.signals_skipped += 1
            return signal

        if symbol in self.positions:
            signal["skip_reason"] = "already_in_position"
            self.stats.signals_skipped += 1
            return signal

        # Calculate position size
        risk_pct = self.high_conf_risk if confidence > 0.7 else self.base_risk
        shares, stop_dist, pos_value = calculate_position_size(
            self.equity, price, atr, risk_pct,
            max_position_pct=self.max_position_pct,
            atr_multiplier=self.stop_atr
        )

        # Validate position size
        validation = self._validate_position_size(shares, price, symbol)
        if not validation["valid"]:
            signal["skip_reason"] = f"position_invalid: {validation['errors']}"
            self.stats.signals_skipped += 1
            return signal

        shares = validation["shares"]
        pos_value = validation["value"]

        # Calculate stops
        if direction == "long":
            stop_price = price - stop_dist
            tp_price = price + stop_dist * (self.tp_atr / self.stop_atr)
        else:
            stop_price = price + stop_dist
            tp_price = price - stop_dist * (self.tp_atr / self.stop_atr)

        # Validate stop loss %
        stop_pct = abs(price - stop_price) / price
        if stop_pct > self.stop_max_pct:
            stop_dist = price * self.stop_max_pct
            if direction == "long":
                stop_price = price - stop_dist
            else:
                stop_price = price + stop_dist

        signal.update({
            "valid": True,
            "shares": shares,
            "position_value": pos_value,
            "stop_price": stop_price,
            "tp_price": tp_price,
            "risk_amount": shares * abs(price - stop_price)
        })

        return signal

    # ════════════════════════════════════════════════════════════════
    # TRADE EXECUTION
    # ════════════════════════════════════════════════════════════════

    def _execute_entry(self, signal: Dict[str, Any]) -> Optional[str]:
        """Execute trade entry."""
        symbol = signal["symbol"]

        # Final checks
        hours = self._check_trading_hours()
        if hours["no_new_trades"]:
            self._log(f"[SKIP] {symbol}: No new trades - EOD approaching")
            return None

        if self.state != TradingState.RUNNING:
            self._log(f"[SKIP] {symbol}: Bot not in RUNNING state ({self.state.value})")
            return None

        # Execute
        if self.dry_run:
            self._log(f"[DRY RUN] Would BUY {signal['shares']:.1f} {symbol} @ ${signal['price']:.2f}")
        else:
            # TODO: Real broker execution
            pass

        # Record in journal
        trade_id = self.journal.open_trade(
            symbol=symbol,
            direction=signal["direction"],
            shares=signal["shares"],
            entry_price=signal["price"],
            stop_price=signal["stop_price"],
            tp_price=signal["tp_price"],
            strategy="production_v1",
            rsi=signal["rsi"],
            momentum=signal["momentum"]
        )

        # Create position
        position = Position(
            symbol=symbol,
            direction=signal["direction"],
            shares=signal["shares"],
            entry_price=signal["price"],
            stop_price=signal["stop_price"],
            tp_price=signal["tp_price"],
            entry_time=datetime.now(timezone.utc),
            trade_id=trade_id
        )
        self.positions[symbol] = position

        # Update stats
        self.stats.trades_today += 1
        self.stats.last_trade_time = datetime.now(timezone.utc)

        # Set cooldown
        self.cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(seconds=self.cooldown_after_trade)

        # Send alert
        self.notifier.notify_trade_entry(
            symbol=symbol,
            direction=signal["direction"],
            shares=signal["shares"],
            price=signal["price"],
            stop_price=signal["stop_price"],
            tp_price=signal["tp_price"],
            risk_amount=signal["risk_amount"]
        )

        # Record in reporter
        self.reporter.record_trade_entry(
            symbol=symbol,
            direction=signal["direction"],
            shares=signal["shares"],
            price=signal["price"],
            stop_price=signal["stop_price"],
            tp_price=signal["tp_price"]
        )

        self._log(f"[ENTRY] {signal['direction'].upper()} {signal['shares']:.1f} {symbol} @ ${signal['price']:.2f} | Stop: ${signal['stop_price']:.2f} | TP: ${signal['tp_price']:.2f}")

        return trade_id

    def _check_exits(self, symbol: str, current_price: float) -> Optional[str]:
        """Check if position should be exited."""
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        exit_reason = None
        exit_price = current_price

        # Check stop loss
        if pos.direction == "long":
            if current_price <= pos.stop_price:
                exit_reason = "stop_loss"
                exit_price = pos.stop_price
            elif current_price >= pos.tp_price:
                exit_reason = "take_profit"
                exit_price = pos.tp_price
            # Trailing stop
            elif self.use_trailing and pos.trailing_active:
                if current_price <= pos.trail_stop:
                    exit_reason = "trailing_stop"
                    exit_price = pos.trail_stop
                elif current_price > pos.entry_price * (1 + self.trailing_activation):
                    new_trail = current_price * (1 - self.trailing_distance)
                    if new_trail > pos.trail_stop:
                        pos.trail_stop = new_trail
            # Activate trailing
            elif self.use_trailing and current_price >= pos.entry_price * (1 + self.trailing_activation):
                pos.trailing_active = True
                pos.trail_stop = current_price * (1 - self.trailing_distance)
        else:  # Short
            if current_price >= pos.stop_price:
                exit_reason = "stop_loss"
                exit_price = pos.stop_price
            elif current_price <= pos.tp_price:
                exit_reason = "take_profit"
                exit_price = pos.tp_price

        if exit_reason:
            self._execute_exit(symbol, exit_price, exit_reason)

        return exit_reason

    def _execute_exit(self, symbol: str, exit_price: float, reason: str):
        """Execute trade exit."""
        if symbol not in self.positions:
            return

        pos = self.positions.pop(symbol)

        # Calculate PnL
        if pos.direction == "long":
            pnl = (exit_price - pos.entry_price) * pos.shares
        else:
            pnl = (pos.entry_price - exit_price) * pos.shares

        pnl_pct = pnl / (pos.entry_price * pos.shares)

        # Update equity
        self.equity += pnl
        self.stats.pnl_today += pnl

        # Update stats
        if pnl > 0:
            self.stats.wins_today += 1
            self.stats.consecutive_losses = 0
        else:
            self.stats.losses_today += 1
            self.stats.consecutive_losses += 1

            # Extended cooldown after loss
            self.cooldowns[symbol] = datetime.now(timezone.utc) + timedelta(seconds=self.cooldown_after_loss)

        # Close in journal
        self.journal.close_trade(
            trade_id=pos.trade_id,
            exit_price=exit_price,
            exit_reason=reason
        )

        # Send alert
        self.notifier.notify_trade_exit(
            symbol=symbol,
            direction=pos.direction,
            pnl=pnl,
            pnl_pct=pnl_pct,
            exit_reason=reason
        )

        # Record in reporter
        self.reporter.record_trade_exit(
            symbol=symbol,
            exit_price=exit_price,
            pnl=pnl,
            exit_reason=reason
        )
        self.reporter.update_equity(self.equity)

        # Alert if stop hit
        if reason == "stop_loss":
            self._alert(f"🛑 STOP HIT: {symbol} | PnL: ${pnl:+,.0f} ({pnl_pct:+.1%})")

        self._log(f"[EXIT] {symbol} | {reason} | PnL: ${pnl:+,.0f} ({pnl_pct:+.1%})")

        # Check consecutive losses
        if self.stats.consecutive_losses >= self.max_consecutive_losses:
            self._alert(f"⚠️ {self.stats.consecutive_losses} pertes consécutives - PAUSE")
            self.state = TradingState.PAUSED
            # Schedule unpause
            threading.Timer(60 * 60, self._unpause).start()  # 1 hour

    def _close_all_positions(self, reason: str = "eod"):
        """Close all open positions."""
        self._log(f"[EOD] Closing all positions ({reason})")

        for symbol in list(self.positions.keys()):
            df = self._get_market_data(symbol)
            if df is not None:
                current_price = float(df['close'].iloc[-1])
                self._execute_exit(symbol, current_price, reason)

    # ════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ════════════════════════════════════════════════════════════════

    def _run_cycle(self):
        """Run one trading cycle."""
        # Check kill switch
        if self._check_kill_switch():
            self._alert("🛑 KILL SWITCH ACTIF - Arrêt immédiat")
            self._close_all_positions("kill_switch")
            self.state = TradingState.STOPPED
            return

        # Check circuit breakers
        cb = self._check_circuit_breakers()
        if cb:
            action = cb.get("action", "")
            self._alert(f"🚨 CIRCUIT BREAKER: {cb.get('message', '')}")

            if action == "shutdown":
                self._close_all_positions("circuit_breaker")
                self.state = TradingState.STOPPED
                return
            elif action == "close_all":
                self._close_all_positions("circuit_breaker")
                self.state = TradingState.CIRCUIT_BREAKER
                return
            elif action == "stop_new_trades":
                self.state = TradingState.CIRCUIT_BREAKER

        # Check trading hours
        hours = self._check_trading_hours()

        if not hours["in_hours"]:
            self._log("[SKIP] Outside trading hours")
            return

        # EOD closing
        if hours["is_eod"] and self.close_eod:
            self.state = TradingState.EOD_CLOSING
            self._close_all_positions("eod")
            return

        # Check inactivity
        self._check_inactivity()

        # Process each symbol
        for symbol in self.symbols:
            try:
                # Get data
                df = self._get_market_data(symbol)
                if df is None or len(df) < 20:
                    continue

                current_price = float(df['close'].iloc[-1])

                # Check exits first
                self._check_exits(symbol, current_price)

                # Skip if not running
                if self.state != TradingState.RUNNING:
                    continue

                # Skip new trades near EOD
                if hours["no_new_trades"]:
                    continue

                # Generate and validate signal
                signal = self._generate_signal(symbol, df)

                # Log signal
                if signal["direction"] != "flat":
                    status = "✓" if signal["valid"] else f"✗ {signal['skip_reason']}"
                    self._log(f"[SIGNAL] {symbol}: {signal['direction'].upper()} conf={signal['confidence']:.2f} | {status}")

                # Execute if valid
                if signal["valid"]:
                    self._execute_entry(signal)

            except Exception as e:
                self._log(f"[ERROR] {symbol}: {e}")

        # Update health stats
        self.health.update_stats(
            equity=self.equity,
            pnl_today=self.stats.pnl_today,
            trades_today=self.stats.trades_today,
            open_positions=len(self.positions),
            state=self.state.value
        )

    def start(self, equity: float):
        """Start the bot."""
        self.equity = equity
        self.initial_equity = equity
        self.daily_start_equity = equity
        self.state = TradingState.RUNNING
        self._running = True

        # Start health monitoring
        self.health.start()
        self.risk_monitor.set_initial_equity(equity)

        # Initialize reporter
        self.reporter.init_portfolio(equity, self.symbols)
        self.reporter.start_auto_reporting()

        # Notify startup
        self.notifier.notify_startup(
            strategy=self.config.get("strategy", {}).get("name", "production"),
            symbols=self.symbols
        )

        self._log("=" * 60)
        self._log(f"HYPRL PRODUCTION BOT STARTED")
        self._log("=" * 60)
        self._log(f"Equity: ${equity:,.0f}")
        self._log(f"Symbols: {len(self.symbols)}")
        self._log(f"Dry run: {self.dry_run}")
        self._log(f"Max daily loss: {self.max_daily_loss:.1%}")
        self._log(f"EOD close: {self.close_eod}")
        self._log("=" * 60)

    def stop(self, reason: str = "manual"):
        """Stop the bot."""
        self._running = False
        self.state = TradingState.STOPPED

        if self.positions:
            self._close_all_positions(reason)

        self.health.stop()
        self.notifier.notify_shutdown(reason)

        # Stop reporter and send final report
        self.reporter.stop_auto_reporting()
        self.reporter.send_full_report()

        # Print summary
        self._print_daily_summary()

    def run(self, interval_seconds: int = 300):
        """Main run loop."""
        # Handle signals
        def signal_handler(sig, frame):
            self._log("\n[SIGNAL] Graceful shutdown...")
            self.stop("signal")
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            while self._running:
                self._run_cycle()

                # Wait
                for _ in range(interval_seconds):
                    if not self._running:
                        break
                    time.sleep(1)

        except Exception as e:
            self._alert(f"🚨 ERREUR CRITIQUE: {e}")
            self.stop(f"error: {e}")
            raise

    # ════════════════════════════════════════════════════════════════
    # UTILITIES
    # ════════════════════════════════════════════════════════════════

    def _log(self, message: str):
        """Log message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _alert(self, message: str):
        """Send alert via Telegram."""
        self._log(f"[ALERT] {message}")
        self.notifier._send_sync(message)

    def _unpause(self):
        """Unpause after timeout."""
        if self.state == TradingState.PAUSED:
            self.state = TradingState.RUNNING
            self._log("[INFO] Bot unpaused after timeout")

    def _print_daily_summary(self):
        """Print daily summary."""
        print("\n" + "=" * 60)
        print("DAILY SUMMARY")
        print("=" * 60)
        print(f"Trades: {self.stats.trades_today}")
        print(f"Wins/Losses: {self.stats.wins_today}/{self.stats.losses_today}")
        print(f"Win Rate: {self.stats.wins_today/self.stats.trades_today:.1%}" if self.stats.trades_today > 0 else "N/A")
        print(f"PnL: ${self.stats.pnl_today:+,.0f}")
        print(f"Signals Generated: {self.stats.signals_generated}")
        print(f"Signals Skipped: {self.stats.signals_skipped}")
        print(f"Final Equity: ${self.equity:,.0f}")
        print("=" * 60)

        # Send Telegram summary
        self.notifier.notify_daily_summary(
            trades_today=self.stats.trades_today,
            pnl_today=self.stats.pnl_today,
            win_rate=self.stats.wins_today/self.stats.trades_today if self.stats.trades_today > 0 else 0,
            equity=self.equity,
            open_positions=len(self.positions)
        )


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

def main():
    import argparse

    parser = argparse.ArgumentParser(description="HyprL Production Trading Bot")
    parser.add_argument("--config", default="configs/runtime/strategy_production.yaml")
    parser.add_argument("--equity", type=float, default=10000)
    parser.add_argument("--interval", type=int, default=300, help="Cycle interval in seconds")
    parser.add_argument("--dry-run", action="store_true", default=True)
    parser.add_argument("--live", action="store_true", help="Enable live trading")
    parser.add_argument("--once", action="store_true", help="Run one cycle only")
    args = parser.parse_args()

    dry_run = not args.live

    bot = ProductionBot(args.config, dry_run=dry_run)
    bot.start(args.equity)

    if args.once:
        bot._run_cycle()
        bot.stop("single_run")
    else:
        bot.run(interval_seconds=args.interval)


if __name__ == "__main__":
    main()

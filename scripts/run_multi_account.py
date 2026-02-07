#!/usr/bin/env python3
"""
HyprL Multi-Account Production Runner v1.0
==========================================
Gère les 3 comptes paper trading en parallèle avec notifications Discord.

Comptes:
- NORMAL: Stratégie équilibrée, tous les guards activés
- AGGRESSIVE: High risk, filtres réduits
- MIX: 70% Normal + 30% Aggressive

Usage:
    python scripts/run_multi_account.py

Requires:
    - .env.discord avec les webhooks
    - .env.normal, .env.aggressive, .env.mix avec les API keys
"""

import os
import sys
import time
import asyncio
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import signal

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Load all env files
from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).parent.parent / "configs" / "runtime"
load_dotenv(CONFIG_DIR / ".env.discord")
load_dotenv(Path(__file__).parent.parent / ".env.ops")

import yaml
import numpy as np
import pandas as pd

try:
    import alpaca_trade_api as tradeapi
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False
    print("⚠️  alpaca-trade-api non installé")

from hyprl.monitoring.discord_notifier import (
    HyprLDiscordNotifier,
    PaperAccount,
    get_discord_notifier,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("HyprL.MultiAccount")


@dataclass
class AccountConfig:
    """Configuration d'un compte paper."""
    name: str
    paper_account: PaperAccount
    api_key: str
    api_secret: str
    base_url: str = "https://paper-api.alpaca.markets"

    # Strategy settings
    risk_per_trade: float = 0.015
    max_position_pct: float = 0.15
    max_daily_loss_pct: float = 0.04
    use_all_guards: bool = True


@dataclass
class AccountState:
    """État d'un compte en temps réel."""
    equity: float = 0
    cash: float = 0
    buying_power: float = 0
    daily_pnl: float = 0
    daily_pnl_pct: float = 0
    start_equity: float = 0
    positions: Dict[str, Any] = field(default_factory=dict)
    trades_today: List[Dict] = field(default_factory=list)
    consecutive_losses: int = 0
    is_paused: bool = False
    pause_until: Optional[datetime] = None


class AccountManager:
    """Gère un compte paper trading individuel."""

    def __init__(self, config: AccountConfig):
        self.config = config
        self.state = AccountState()
        self.api: Optional[tradeapi.REST] = None
        self.discord = get_discord_notifier()
        self._lock = threading.Lock()

    def connect(self) -> bool:
        """Connecte au compte Alpaca."""
        if not ALPACA_AVAILABLE:
            logger.error(f"{self.config.name}: alpaca-trade-api non disponible")
            return False

        try:
            self.api = tradeapi.REST(
                self.config.api_key,
                self.config.api_secret,
                self.config.base_url,
            )
            account = self.api.get_account()
            self.state.equity = float(account.equity)
            self.state.cash = float(account.cash)
            self.state.buying_power = float(account.buying_power)
            self.state.start_equity = self.state.equity

            logger.info(f"{self.config.name}: Connecté - Equity ${self.state.equity:,.2f}")
            return True

        except Exception as e:
            logger.error(f"{self.config.name}: Erreur connexion - {e}")
            return False

    def update_state(self):
        """Met à jour l'état du compte."""
        if not self.api:
            return

        try:
            account = self.api.get_account()
            self.state.equity = float(account.equity)
            self.state.cash = float(account.cash)
            self.state.buying_power = float(account.buying_power)

            # Calcul P/L journalier
            self.state.daily_pnl = self.state.equity - self.state.start_equity
            if self.state.start_equity > 0:
                self.state.daily_pnl_pct = self.state.daily_pnl / self.state.start_equity

            # Positions
            positions = self.api.list_positions()
            self.state.positions = {
                p.symbol: {
                    "qty": int(p.qty),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
                for p in positions
            }

        except Exception as e:
            logger.error(f"{self.config.name}: Erreur update - {e}")

    def check_circuit_breaker(self) -> bool:
        """Vérifie les circuit breakers. Retourne True si trading autorisé."""
        if not self.config.use_all_guards:
            return True

        # Check pause
        if self.state.is_paused:
            if self.state.pause_until and datetime.now() < self.state.pause_until:
                return False
            self.state.is_paused = False

        # Check daily loss
        if self.state.daily_pnl_pct < -self.config.max_daily_loss_pct:
            logger.warning(f"{self.config.name}: Circuit breaker - Daily loss {self.state.daily_pnl_pct:.2%}")
            asyncio.run(self.discord.send_alert(
                alert_type="Circuit Breaker",
                message=f"Perte journalière {self.state.daily_pnl_pct:.2%} - Trading stoppé",
                severity="error",
                account=self.config.paper_account,
            ))
            return False

        return True

    async def execute_signal(
        self,
        symbol: str,
        direction: str,
        confidence: float,
        stop_price: float,
        tp_price: float,
    ) -> bool:
        """Exécute un signal de trading."""
        if not self.api:
            return False

        if not self.check_circuit_breaker():
            return False

        with self._lock:
            try:
                # Calcul position size
                quote = self.api.get_latest_quote(symbol)
                price = float(quote.ask_price)

                risk_amount = self.state.equity * self.config.risk_per_trade
                risk_per_share = abs(price - stop_price)

                if risk_per_share <= 0:
                    return False

                shares = int(risk_amount / risk_per_share)

                # Limites
                position_value = shares * price
                max_position = self.state.equity * self.config.max_position_pct

                if position_value > max_position:
                    shares = int(max_position / price)

                if shares < 1:
                    return False

                # Submit order
                side = "buy" if direction == "long" else "sell"

                order = self.api.submit_order(
                    symbol=symbol,
                    qty=shares,
                    side=side,
                    type="market",
                    time_in_force="day",
                    order_class="bracket",
                    stop_loss={"stop_price": str(round(stop_price, 2))},
                    take_profit={"limit_price": str(round(tp_price, 2))},
                )

                logger.info(f"{self.config.name}: {direction.upper()} {symbol} x{shares} @ ${price:.2f}")

                # Discord notification
                await self.discord.send_trade_entry(
                    account=self.config.paper_account,
                    symbol=symbol,
                    direction=direction,
                    shares=shares,
                    price=price,
                    stop_price=stop_price,
                    tp_price=tp_price,
                    confidence=confidence,
                )

                # Track trade
                self.state.trades_today.append({
                    "symbol": symbol,
                    "direction": direction,
                    "shares": shares,
                    "entry_price": price,
                    "time": datetime.now(),
                    "order_id": order.id,
                })

                return True

            except Exception as e:
                logger.error(f"{self.config.name}: Erreur ordre - {e}")
                return False

    def get_summary_data(self) -> Dict:
        """Retourne les données pour le résumé."""
        winning = sum(1 for t in self.state.trades_today if t.get("pnl", 0) > 0)
        symbols = list(set(t["symbol"] for t in self.state.trades_today))

        return {
            "equity": self.state.equity,
            "daily_pnl": self.state.daily_pnl,
            "daily_pnl_pct": self.state.daily_pnl_pct,
            "total_trades": len(self.state.trades_today),
            "winning_trades": winning,
            "symbols_traded": symbols,
            "positions": self.state.positions,
        }


class MultiAccountRunner:
    """
    Runner multi-comptes pour HyprL.

    Gère les 3 comptes paper en parallèle avec notifications Discord unifiées.
    """

    def __init__(self):
        self.accounts: Dict[PaperAccount, AccountManager] = {}
        self.discord = get_discord_notifier()
        self.running = False
        self._stop_event = threading.Event()

        # Load account configs
        self._load_accounts()

    def _load_accounts(self):
        """Charge les configurations des comptes."""
        configs = [
            AccountConfig(
                name="Normal",
                paper_account=PaperAccount.NORMAL,
                api_key=self._get_key(".env.normal", "APCA_API_KEY_ID"),
                api_secret=self._get_key(".env.normal", "APCA_API_SECRET_KEY"),
                risk_per_trade=0.015,
                max_daily_loss_pct=0.04,
                use_all_guards=True,
            ),
            AccountConfig(
                name="Aggressive",
                paper_account=PaperAccount.AGGRESSIVE,
                api_key=self._get_key(".env.aggressive", "APCA_API_KEY_ID"),
                api_secret=self._get_key(".env.aggressive", "APCA_API_SECRET_KEY"),
                risk_per_trade=0.025,
                max_daily_loss_pct=0.06,
                use_all_guards=False,  # Reduced guards
            ),
            AccountConfig(
                name="Mix",
                paper_account=PaperAccount.MIX,
                api_key=self._get_key(".env.mix", "APCA_API_KEY_ID"),
                api_secret=self._get_key(".env.mix", "APCA_API_SECRET_KEY"),
                risk_per_trade=0.018,
                max_daily_loss_pct=0.05,
                use_all_guards=True,
            ),
        ]

        for config in configs:
            if config.api_key and config.api_secret:
                self.accounts[config.paper_account] = AccountManager(config)
                logger.info(f"Compte {config.name} chargé")
            else:
                logger.warning(f"Compte {config.name}: clés manquantes")

    def _get_key(self, env_file: str, key: str) -> str:
        """Récupère une clé depuis un fichier .env."""
        env_path = CONFIG_DIR / env_file
        if not env_path.exists():
            return ""

        for line in env_path.read_text().split("\n"):
            if line.startswith(key + "="):
                return line.split("=", 1)[1].strip()
        return ""

    def connect_all(self) -> bool:
        """Connecte tous les comptes."""
        print("\n" + "=" * 60)
        print("CONNEXION AUX COMPTES PAPER")
        print("=" * 60)

        success = 0
        for account_type, manager in self.accounts.items():
            if manager.connect():
                success += 1

        print(f"\n✅ {success}/{len(self.accounts)} comptes connectés")
        return success > 0

    async def send_global_summary(self):
        """Envoie le résumé global de tous les comptes."""
        accounts_data = {}

        for account_type, manager in self.accounts.items():
            manager.update_state()
            data = manager.get_summary_data()
            accounts_data[account_type] = data

        await self.discord.send_global_summary(accounts_data)

        # Aussi envoyer les résumés individuels
        for account_type, manager in self.accounts.items():
            data = manager.get_summary_data()
            await self.discord.send_summary(
                account=account_type,
                equity=data["equity"],
                daily_pnl=data["daily_pnl"],
                daily_pnl_pct=data["daily_pnl_pct"],
                total_trades=data["total_trades"],
                winning_trades=data["winning_trades"],
                symbols_traded=data["symbols_traded"],
            )

    def print_status(self):
        """Affiche le status de tous les comptes."""
        print("\n" + "=" * 60)
        print(f"STATUS - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 60)

        total_equity = 0
        total_pnl = 0

        for account_type, manager in self.accounts.items():
            manager.update_state()
            state = manager.state

            total_equity += state.equity
            total_pnl += state.daily_pnl

            emoji = "🟢" if state.daily_pnl >= 0 else "🔴"
            print(f"\n{emoji} {manager.config.name.upper()}")
            print(f"   Equity: ${state.equity:,.2f}")
            print(f"   P/L:    ${state.daily_pnl:+,.2f} ({state.daily_pnl_pct:+.2%})")
            print(f"   Trades: {len(state.trades_today)}")
            if state.positions:
                print(f"   Positions: {', '.join(state.positions.keys())}")

        print(f"\n{'─' * 40}")
        emoji = "🟢" if total_pnl >= 0 else "🔴"
        print(f"{emoji} TOTAL")
        print(f"   Equity: ${total_equity:,.2f}")
        print(f"   P/L:    ${total_pnl:+,.2f}")

    async def run_loop(self, symbols: List[str]):
        """Boucle principale de trading."""
        logger.info("Démarrage de la boucle de trading...")

        last_summary = datetime.now()
        summary_interval = timedelta(hours=1)

        while not self._stop_event.is_set():
            try:
                # Update tous les comptes
                for manager in self.accounts.values():
                    manager.update_state()

                # Envoyer résumé horaire
                if datetime.now() - last_summary > summary_interval:
                    await self.send_global_summary()
                    last_summary = datetime.now()

                # Print status
                self.print_status()

                # Wait
                await asyncio.sleep(60)

            except Exception as e:
                logger.error(f"Erreur boucle: {e}")
                await asyncio.sleep(10)

    def start(self, symbols: List[str]):
        """Démarre le runner."""
        self.running = True
        self._stop_event.clear()

        # Handle signals
        def signal_handler(signum, frame):
            logger.info("Signal d'arrêt reçu...")
            self.stop()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run
        asyncio.run(self.run_loop(symbols))

    def stop(self):
        """Arrête le runner."""
        logger.info("Arrêt en cours...")
        self._stop_event.set()
        self.running = False


def main():
    """Point d'entrée principal."""
    print("=" * 60)
    print("HYPRL MULTI-ACCOUNT PRODUCTION RUNNER v1.0")
    print("=" * 60)
    print(f"Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Symbols à trader
    symbols = [
        "NVDA", "META", "MSFT", "GOOGL", "NFLX",
        "AMD", "GS", "V", "QQQ", "TSLA",
        "LLY", "PLTR", "SHOP", "SMCI", "UBER",
    ]

    runner = MultiAccountRunner()

    if not runner.connect_all():
        print("\n❌ Aucun compte connecté - Arrêt")
        sys.exit(1)

    print(f"\n🎯 Symboles: {', '.join(symbols)}")
    print("\n" + "=" * 60)
    print("Ctrl+C pour arrêter")
    print("=" * 60)

    # Envoyer notification de démarrage
    asyncio.run(runner.discord.send_alert(
        alert_type="Système Démarré",
        message=f"HyprL Multi-Account Runner actif\n{len(runner.accounts)} comptes connectés",
        severity="info",
        account=None,
    ))

    runner.start(symbols)

    # Cleanup
    asyncio.run(runner.discord.send_alert(
        alert_type="Système Arrêté",
        message="HyprL Multi-Account Runner arrêté",
        severity="warning",
        account=None,
    ))


if __name__ == "__main__":
    main()

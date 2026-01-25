#!/usr/bin/env python3
"""
HyprL Discord Notifier v1.0
===========================
Notifications Discord par compte paper trading.

Structure des salons:
- HyprL Normal: trades, alerts, summary
- HyprL Aggressive: trades, alerts, summary
- HyprL Mix: trades, alerts, summary
- HyprL Global: resume global, risk alerts
"""

import os
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum

try:
    import discord
    from discord import Webhook
    import aiohttp
    DISCORD_AVAILABLE = True
except ImportError:
    DISCORD_AVAILABLE = False

logger = logging.getLogger(__name__)


class PaperAccount(Enum):
    """Comptes paper trading disponibles."""
    NORMAL = "normal"
    AGGRESSIVE = "aggressive"
    MIX = "mix"


@dataclass
class ChannelConfig:
    """Configuration d'un salon Discord."""
    webhook_url: str
    name: str
    account: Optional[PaperAccount] = None


@dataclass
class DiscordConfig:
    """Configuration complète Discord."""
    # Webhooks par compte et type
    webhooks: Dict[str, str] = field(default_factory=dict)

    # Noms des catégories
    category_normal: str = "HyprL Normal"
    category_aggressive: str = "HyprL Aggressive"
    category_mix: str = "HyprL Mix"
    category_global: str = "HyprL Global"

    @classmethod
    def from_env(cls) -> "DiscordConfig":
        """Charge la config depuis les variables d'environnement."""
        webhooks = {}

        # Format: DISCORD_WEBHOOK_NORMAL_TRADES=https://discord.com/api/webhooks/...
        for key, value in os.environ.items():
            if key.startswith("DISCORD_WEBHOOK_"):
                webhook_name = key.replace("DISCORD_WEBHOOK_", "").lower()
                webhooks[webhook_name] = value

        return cls(webhooks=webhooks)


class HyprLDiscordNotifier:
    """
    Notificateur Discord pour HyprL.

    Envoie des notifications formatées vers les salons Discord
    correspondant à chaque compte paper trading.
    """

    # Couleurs par type de message
    COLORS = {
        "success": 0x00FF00,  # Vert
        "error": 0xFF0000,    # Rouge
        "warning": 0xFFAA00,  # Orange
        "info": 0x0099FF,     # Bleu
        "trade_long": 0x00FF00,
        "trade_short": 0xFF6600,
        "trade_exit_profit": 0x00FF00,
        "trade_exit_loss": 0xFF0000,
        "summary": 0x9933FF,  # Violet
    }

    # Emojis
    EMOJIS = {
        "long": ":chart_with_upwards_trend:",
        "short": ":chart_with_downwards_trend:",
        "profit": ":white_check_mark:",
        "loss": ":x:",
        "warning": ":warning:",
        "money": ":moneybag:",
        "clock": ":clock3:",
        "fire": ":fire:",
        "robot": ":robot:",
        "chart": ":bar_chart:",
    }

    def __init__(self, config: Optional[DiscordConfig] = None):
        """
        Initialise le notificateur Discord.

        Args:
            config: Configuration Discord (ou charge depuis env)
        """
        if not DISCORD_AVAILABLE:
            logger.warning("discord.py non installé - notifications désactivées")
            self.enabled = False
            return

        self.config = config or DiscordConfig.from_env()
        self.enabled = len(self.config.webhooks) > 0

        if not self.enabled:
            logger.warning("Aucun webhook Discord configuré")
        else:
            logger.info(f"Discord activé avec {len(self.config.webhooks)} webhooks")

    async def _send_webhook(self, webhook_url: str, embed: "discord.Embed") -> bool:
        """Envoie un message via webhook."""
        if not self.enabled:
            return False

        try:
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url(webhook_url, session=session)
                await webhook.send(embed=embed, username="HyprL Bot")
            return True
        except Exception as e:
            logger.error(f"Erreur envoi webhook: {e}")
            return False

    def _get_webhook(self, account: PaperAccount, channel_type: str) -> Optional[str]:
        """Récupère l'URL du webhook pour un compte et type de salon."""
        key = f"{account.value}_{channel_type}"
        return self.config.webhooks.get(key)

    def _create_trade_entry_embed(
        self,
        symbol: str,
        direction: str,
        shares: int,
        price: float,
        stop_price: float,
        tp_price: float,
        confidence: float,
        account: PaperAccount,
    ) -> "discord.Embed":
        """Crée un embed pour une entrée de trade."""
        is_long = direction.lower() == "long"
        color = self.COLORS["trade_long"] if is_long else self.COLORS["trade_short"]
        emoji = self.EMOJIS["long"] if is_long else self.EMOJIS["short"]

        embed = discord.Embed(
            title=f"{emoji} {direction.upper()} {symbol}",
            color=color,
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="Shares", value=f"{shares}", inline=True)
        embed.add_field(name="Entry", value=f"${price:.2f}", inline=True)
        embed.add_field(name="Confidence", value=f"{confidence:.0%}", inline=True)

        embed.add_field(name="Stop Loss", value=f"${stop_price:.2f}", inline=True)
        embed.add_field(name="Take Profit", value=f"${tp_price:.2f}", inline=True)

        risk = abs(price - stop_price) * shares
        reward = abs(tp_price - price) * shares
        rr = reward / risk if risk > 0 else 0
        embed.add_field(name="R:R", value=f"{rr:.1f}", inline=True)

        embed.set_footer(text=f"HyprL {account.value.capitalize()} | Paper Trading")

        return embed

    def _create_trade_exit_embed(
        self,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
        account: PaperAccount,
    ) -> "discord.Embed":
        """Crée un embed pour une sortie de trade."""
        is_profit = pnl >= 0
        color = self.COLORS["trade_exit_profit"] if is_profit else self.COLORS["trade_exit_loss"]
        emoji = self.EMOJIS["profit"] if is_profit else self.EMOJIS["loss"]

        embed = discord.Embed(
            title=f"{emoji} EXIT {symbol} | {'+' if pnl >= 0 else ''}{pnl_pct:.1%}",
            color=color,
            timestamp=datetime.utcnow()
        )

        embed.add_field(name="Direction", value=direction.upper(), inline=True)
        embed.add_field(name="Shares", value=f"{shares}", inline=True)
        embed.add_field(name="Reason", value=exit_reason.upper(), inline=True)

        embed.add_field(name="Entry", value=f"${entry_price:.2f}", inline=True)
        embed.add_field(name="Exit", value=f"${exit_price:.2f}", inline=True)
        embed.add_field(name="P/L", value=f"${pnl:+,.2f}", inline=True)

        embed.set_footer(text=f"HyprL {account.value.capitalize()} | Paper Trading")

        return embed

    def _create_summary_embed(
        self,
        account: PaperAccount,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_trades: int,
        winning_trades: int,
        symbols_traded: List[str],
    ) -> "discord.Embed":
        """Crée un embed pour le résumé journalier."""
        is_profit = daily_pnl >= 0
        color = self.COLORS["success"] if is_profit else self.COLORS["error"]

        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        embed = discord.Embed(
            title=f"{self.EMOJIS['chart']} Résumé {account.value.capitalize()}",
            description=f"**Equity: ${equity:,.2f}**",
            color=color,
            timestamp=datetime.utcnow()
        )

        # P/L du jour
        pnl_emoji = self.EMOJIS["profit"] if is_profit else self.EMOJIS["loss"]
        embed.add_field(
            name=f"{pnl_emoji} P/L du Jour",
            value=f"${daily_pnl:+,.2f} ({daily_pnl_pct:+.2%})",
            inline=True
        )

        # Stats trades
        embed.add_field(
            name=f"{self.EMOJIS['fire']} Trades",
            value=f"{total_trades} ({win_rate:.0f}% win)",
            inline=True
        )

        # Symboles tradés
        if symbols_traded:
            symbols_str = ", ".join(symbols_traded[:8])
            if len(symbols_traded) > 8:
                symbols_str += f" +{len(symbols_traded)-8}"
            embed.add_field(
                name="Symboles",
                value=symbols_str,
                inline=False
            )

        embed.set_footer(text="HyprL Paper Trading")

        return embed

    def _create_alert_embed(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        account: Optional[PaperAccount] = None,
    ) -> "discord.Embed":
        """Crée un embed pour une alerte."""
        color = self.COLORS.get(severity, self.COLORS["warning"])
        emoji = self.EMOJIS.get(severity, self.EMOJIS["warning"])

        embed = discord.Embed(
            title=f"{emoji} {alert_type}",
            description=message,
            color=color,
            timestamp=datetime.utcnow()
        )

        if account:
            embed.set_footer(text=f"HyprL {account.value.capitalize()}")
        else:
            embed.set_footer(text="HyprL Global")

        return embed

    def _create_global_summary_embed(
        self,
        accounts_data: Dict[PaperAccount, dict],
    ) -> "discord.Embed":
        """Crée un embed pour le résumé global de tous les comptes."""
        total_equity = sum(d.get("equity", 0) for d in accounts_data.values())
        total_pnl = sum(d.get("daily_pnl", 0) for d in accounts_data.values())
        total_trades = sum(d.get("total_trades", 0) for d in accounts_data.values())

        is_profit = total_pnl >= 0
        color = self.COLORS["success"] if is_profit else self.COLORS["error"]

        embed = discord.Embed(
            title=f"{self.EMOJIS['robot']} HyprL - Résumé Global",
            description=f"**Total Equity: ${total_equity:,.2f}**\n**P/L Jour: ${total_pnl:+,.2f}**",
            color=color,
            timestamp=datetime.utcnow()
        )

        # Détail par compte
        for account, data in accounts_data.items():
            equity = data.get("equity", 0)
            pnl = data.get("daily_pnl", 0)
            trades = data.get("total_trades", 0)

            pnl_emoji = ":green_circle:" if pnl >= 0 else ":red_circle:"
            embed.add_field(
                name=f"{account.value.capitalize()}",
                value=f"{pnl_emoji} ${equity:,.0f}\nP/L: ${pnl:+,.0f}\nTrades: {trades}",
                inline=True
            )

        embed.set_footer(text="HyprL Paper Trading | All Accounts")

        return embed

    # ========== Méthodes publiques d'envoi ==========

    async def send_trade_entry(
        self,
        account: PaperAccount,
        symbol: str,
        direction: str,
        shares: int,
        price: float,
        stop_price: float,
        tp_price: float,
        confidence: float = 0.5,
    ) -> bool:
        """Envoie une notification d'entrée de trade."""
        webhook_url = self._get_webhook(account, "trades")
        if not webhook_url:
            logger.debug(f"Pas de webhook trades pour {account.value}")
            return False

        embed = self._create_trade_entry_embed(
            symbol, direction, shares, price, stop_price, tp_price, confidence, account
        )
        return await self._send_webhook(webhook_url, embed)

    async def send_trade_exit(
        self,
        account: PaperAccount,
        symbol: str,
        direction: str,
        shares: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        pnl_pct: float,
        exit_reason: str,
    ) -> bool:
        """Envoie une notification de sortie de trade."""
        webhook_url = self._get_webhook(account, "trades")
        if not webhook_url:
            return False

        embed = self._create_trade_exit_embed(
            symbol, direction, shares, entry_price, exit_price, pnl, pnl_pct, exit_reason, account
        )
        return await self._send_webhook(webhook_url, embed)

    async def send_summary(
        self,
        account: PaperAccount,
        equity: float,
        daily_pnl: float,
        daily_pnl_pct: float,
        total_trades: int,
        winning_trades: int,
        symbols_traded: List[str],
    ) -> bool:
        """Envoie le résumé journalier pour un compte."""
        webhook_url = self._get_webhook(account, "summary")
        if not webhook_url:
            return False

        embed = self._create_summary_embed(
            account, equity, daily_pnl, daily_pnl_pct, total_trades, winning_trades, symbols_traded
        )
        return await self._send_webhook(webhook_url, embed)

    async def send_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
        account: Optional[PaperAccount] = None,
    ) -> bool:
        """Envoie une alerte."""
        # Détermine le webhook
        if account:
            webhook_url = self._get_webhook(account, "alerts")
        else:
            webhook_url = self.config.webhooks.get("global_alerts")

        if not webhook_url:
            return False

        embed = self._create_alert_embed(alert_type, message, severity, account)
        return await self._send_webhook(webhook_url, embed)

    async def send_global_summary(
        self,
        accounts_data: Dict[PaperAccount, dict],
    ) -> bool:
        """Envoie le résumé global de tous les comptes."""
        webhook_url = self.config.webhooks.get("global_summary")
        if not webhook_url:
            return False

        embed = self._create_global_summary_embed(accounts_data)
        return await self._send_webhook(webhook_url, embed)

    # ========== Méthodes synchrones (wrapper) ==========

    def send_trade_entry_sync(self, *args, **kwargs) -> bool:
        """Version synchrone de send_trade_entry."""
        return asyncio.run(self.send_trade_entry(*args, **kwargs))

    def send_trade_exit_sync(self, *args, **kwargs) -> bool:
        """Version synchrone de send_trade_exit."""
        return asyncio.run(self.send_trade_exit(*args, **kwargs))

    def send_summary_sync(self, *args, **kwargs) -> bool:
        """Version synchrone de send_summary."""
        return asyncio.run(self.send_summary(*args, **kwargs))

    def send_alert_sync(self, *args, **kwargs) -> bool:
        """Version synchrone de send_alert."""
        return asyncio.run(self.send_alert(*args, **kwargs))

    def send_global_summary_sync(self, *args, **kwargs) -> bool:
        """Version synchrone de send_global_summary."""
        return asyncio.run(self.send_global_summary(*args, **kwargs))


# ========== Singleton et helpers ==========

_discord_notifier: Optional[HyprLDiscordNotifier] = None


def get_discord_notifier() -> HyprLDiscordNotifier:
    """Récupère l'instance singleton du notificateur Discord."""
    global _discord_notifier
    if _discord_notifier is None:
        _discord_notifier = HyprLDiscordNotifier()
    return _discord_notifier


def setup_discord_channels_guide() -> str:
    """Retourne un guide pour configurer les salons Discord."""
    return """
═══════════════════════════════════════════════════════════════════
GUIDE: Configuration Discord pour HyprL
═══════════════════════════════════════════════════════════════════

1. CRÉER LES CATÉGORIES ET SALONS
─────────────────────────────────
Sur ton serveur Discord, crée:

📁 HyprL Normal
   ├── #trades-normal
   ├── #alerts-normal
   └── #summary-normal

📁 HyprL Aggressive
   ├── #trades-aggressive
   ├── #alerts-aggressive
   └── #summary-aggressive

📁 HyprL Mix
   ├── #trades-mix
   ├── #alerts-mix
   └── #summary-mix

📁 HyprL Global
   ├── #global-summary
   └── #risk-alerts

2. CRÉER LES WEBHOOKS
─────────────────────
Pour chaque salon:
- Clic droit > Modifier le salon > Intégrations > Webhooks
- "Nouveau Webhook" > Copier l'URL

3. CONFIGURER .env
──────────────────
Ajoute dans ton fichier .env:

# Normal Account
DISCORD_WEBHOOK_NORMAL_TRADES=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_NORMAL_ALERTS=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_NORMAL_SUMMARY=https://discord.com/api/webhooks/...

# Aggressive Account
DISCORD_WEBHOOK_AGGRESSIVE_TRADES=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_AGGRESSIVE_ALERTS=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_AGGRESSIVE_SUMMARY=https://discord.com/api/webhooks/...

# Mix Account
DISCORD_WEBHOOK_MIX_TRADES=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_MIX_ALERTS=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_MIX_SUMMARY=https://discord.com/api/webhooks/...

# Global
DISCORD_WEBHOOK_GLOBAL_SUMMARY=https://discord.com/api/webhooks/...
DISCORD_WEBHOOK_GLOBAL_ALERTS=https://discord.com/api/webhooks/...

4. TESTER
─────────
python scripts/test_discord.py

═══════════════════════════════════════════════════════════════════
"""


if __name__ == "__main__":
    print(setup_discord_channels_guide())

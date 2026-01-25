"""
Telegram Alert System for HyprL
================================
Envoie des notifications pour:
- Nouveaux trades (entry/exit)
- Alertes de risque (drawdown, pertes consécutives)
- Health checks (bot alive)
- Résumé journalier

Setup:
1. Créer un bot via @BotFather sur Telegram
2. Obtenir le token
3. Obtenir ton chat_id via @userinfobot
4. Configurer dans .env:
   TELEGRAM_BOT_TOKEN=xxx
   TELEGRAM_CHAT_ID=xxx
"""

import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


class TelegramNotifier:
    """Send notifications via Telegram bot."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        enabled: bool = True
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = enabled and self.bot_token and self.chat_id
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""

        if not HTTPX_AVAILABLE:
            self.enabled = False

    def _send_sync(self, message: str, parse_mode: str = "HTML") -> bool:
        """Synchronous send (for simple use cases)."""
        if not self.enabled:
            return False

        try:
            import urllib.request
            import urllib.parse

            url = f"{self.base_url}/sendMessage"
            data = urllib.parse.urlencode({
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode
            }).encode()

            req = urllib.request.Request(url, data=data)
            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 200
        except Exception as e:
            print(f"[Telegram] Error: {e}")
            return False

    async def send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Async send message."""
        if not self.enabled or not HTTPX_AVAILABLE:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/sendMessage",
                    json={
                        "chat_id": self.chat_id,
                        "text": message,
                        "parse_mode": parse_mode
                    },
                    timeout=10.0
                )
                return response.status_code == 200
        except Exception as e:
            print(f"[Telegram] Error: {e}")
            return False

    # ════════════════════════════════════════════════════════════
    # MESSAGE TEMPLATES
    # ════════════════════════════════════════════════════════════

    def notify_trade_entry(
        self,
        symbol: str,
        direction: str,
        shares: float,
        price: float,
        stop_price: float,
        tp_price: float,
        risk_amount: float
    ) -> bool:
        """Notification pour nouvelle entrée."""
        emoji = "🟢" if direction.lower() == "long" else "🔴"

        message = f"""
{emoji} <b>NOUVEAU TRADE</b>

<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction.upper()}
<b>Shares:</b> {shares:.1f}
<b>Entry:</b> ${price:.2f}
<b>Stop:</b> ${stop_price:.2f}
<b>Target:</b> ${tp_price:.2f}
<b>Risk:</b> ${risk_amount:.0f}

<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>
"""
        return self._send_sync(message)

    def notify_trade_exit(
        self,
        symbol: str,
        direction: str,
        pnl: float,
        pnl_pct: float,
        exit_reason: str
    ) -> bool:
        """Notification pour sortie de trade."""
        emoji = "✅" if pnl > 0 else "❌"

        message = f"""
{emoji} <b>TRADE FERMÉ</b>

<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction.upper()}
<b>PnL:</b> ${pnl:+,.0f} ({pnl_pct:+.1%})
<b>Raison:</b> {exit_reason}

<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>
"""
        return self._send_sync(message)

    def notify_risk_alert(
        self,
        alert_type: str,
        message_detail: str,
        severity: str = "warning"
    ) -> bool:
        """Notification pour alerte de risque."""
        emoji = "⚠️" if severity == "warning" else "🚨"

        message = f"""
{emoji} <b>ALERTE RISQUE</b>

<b>Type:</b> {alert_type}
<b>Détail:</b> {message_detail}

<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>
"""
        return self._send_sync(message)

    def notify_daily_summary(
        self,
        trades_today: int,
        pnl_today: float,
        win_rate: float,
        equity: float,
        open_positions: int
    ) -> bool:
        """Résumé journalier."""
        emoji = "📈" if pnl_today >= 0 else "📉"

        message = f"""
{emoji} <b>RÉSUMÉ JOURNALIER</b>

<b>Trades:</b> {trades_today}
<b>PnL:</b> ${pnl_today:+,.0f}
<b>Win Rate:</b> {win_rate:.0%}
<b>Equity:</b> ${equity:,.0f}
<b>Positions ouvertes:</b> {open_positions}

<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</i>
"""
        return self._send_sync(message)

    def notify_heartbeat(self, status: str = "alive") -> bool:
        """Heartbeat - bot is alive."""
        message = f"💓 Bot {status} - {datetime.now(timezone.utc).strftime('%H:%M UTC')}"
        return self._send_sync(message)

    def notify_startup(self, strategy: str, symbols: list) -> bool:
        """Notification au démarrage."""
        message = f"""
🚀 <b>BOT DÉMARRÉ</b>

<b>Strategy:</b> {strategy}
<b>Symbols:</b> {', '.join(symbols)}

<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>
"""
        return self._send_sync(message)

    def notify_shutdown(self, reason: str = "manual") -> bool:
        """Notification à l'arrêt."""
        message = f"""
🛑 <b>BOT ARRÊTÉ</b>

<b>Raison:</b> {reason}

<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>
"""
        return self._send_sync(message)


# Singleton instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance."""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


# Convenience functions
def send_trade_entry(**kwargs) -> bool:
    return get_notifier().notify_trade_entry(**kwargs)

def send_trade_exit(**kwargs) -> bool:
    return get_notifier().notify_trade_exit(**kwargs)

def send_risk_alert(**kwargs) -> bool:
    return get_notifier().notify_risk_alert(**kwargs)

def send_daily_summary(**kwargs) -> bool:
    return get_notifier().notify_daily_summary(**kwargs)

def send_heartbeat(**kwargs) -> bool:
    return get_notifier().notify_heartbeat(**kwargs)

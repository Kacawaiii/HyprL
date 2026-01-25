"""
════════════════════════════════════════════════════════════════════════
HYPRL v1.1 - TELEGRAM REPORTING SYSTEM
════════════════════════════════════════════════════════════════════════

Envoie des rapports détaillés sur Telegram:
- Stats par symbole (return/h, trades, investissement)
- Résumé global (capital, P/L, performance)
- Alertes et notifications

Structure des messages:
📊 HYPRL v1.1 - RÉSUMÉ GLOBAL
📈 NVDA - Stats détaillées
📈 META - Stats détaillées
...

════════════════════════════════════════════════════════════════════════
"""

import os
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
import threading
import time

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False


@dataclass
class SymbolStats:
    """Stats pour un symbole."""
    symbol: str
    trades_total: int = 0
    trades_today: int = 0
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    pnl_today: float = 0.0
    pnl_this_hour: float = 0.0
    invested_total: float = 0.0
    invested_current: float = 0.0
    avg_hold_time_minutes: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    last_trade_time: Optional[str] = None
    current_position: Optional[Dict] = None


@dataclass
class PortfolioStats:
    """Stats globales du portfolio."""
    initial_equity: float = 0.0
    current_equity: float = 0.0
    peak_equity: float = 0.0
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    today_pnl: float = 0.0
    today_pnl_pct: float = 0.0
    this_hour_pnl: float = 0.0
    total_trades: int = 0
    today_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    open_positions: int = 0
    total_invested: float = 0.0
    sharpe_estimate: float = 0.0


class HyprLReporter:
    """
    Système de reporting Telegram pour HyprL v1.1
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        report_interval_minutes: int = 60,
        state_file: str = "live/state/reporter_state.json"
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.bot_token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else ""

        self.report_interval = report_interval_minutes * 60
        self.state_file = Path(state_file)
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        # Stats
        self.symbol_stats: Dict[str, SymbolStats] = {}
        self.portfolio = PortfolioStats()
        self.hourly_snapshots: List[Dict] = []

        # Threading
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Load state
        self._load_state()

    # ════════════════════════════════════════════════════════════════
    # TELEGRAM SENDING
    # ════════════════════════════════════════════════════════════════

    def _send(self, message: str, parse_mode: str = "HTML") -> bool:
        """Send message to Telegram."""
        if not self.enabled:
            print(f"[Reporter] Telegram disabled, would send:\n{message}")
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
            print(f"[Reporter] Telegram error: {e}")
            return False

    # ════════════════════════════════════════════════════════════════
    # STATE MANAGEMENT
    # ════════════════════════════════════════════════════════════════

    def _load_state(self):
        """Load state from file."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)

                # Load portfolio
                if "portfolio" in data:
                    for k, v in data["portfolio"].items():
                        if hasattr(self.portfolio, k):
                            setattr(self.portfolio, k, v)

                # Load symbol stats
                for sym, stats in data.get("symbols", {}).items():
                    self.symbol_stats[sym] = SymbolStats(symbol=sym, **stats)

                # Load hourly snapshots
                self.hourly_snapshots = data.get("hourly_snapshots", [])

            except Exception as e:
                print(f"[Reporter] Failed to load state: {e}")

    def _save_state(self):
        """Save state to file."""
        try:
            data = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "portfolio": asdict(self.portfolio),
                "symbols": {sym: asdict(stats) for sym, stats in self.symbol_stats.items()},
                "hourly_snapshots": self.hourly_snapshots[-168:]  # Keep 1 week
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"[Reporter] Failed to save state: {e}")

    # ════════════════════════════════════════════════════════════════
    # STATS UPDATES
    # ════════════════════════════════════════════════════════════════

    def init_portfolio(self, equity: float, symbols: List[str]):
        """Initialize portfolio tracking."""
        self.portfolio.initial_equity = equity
        self.portfolio.current_equity = equity
        self.portfolio.peak_equity = equity

        for sym in symbols:
            if sym not in self.symbol_stats:
                self.symbol_stats[sym] = SymbolStats(symbol=sym)

        self._save_state()

    def record_trade_entry(
        self,
        symbol: str,
        direction: str,
        shares: float,
        price: float,
        stop_price: float,
        tp_price: float
    ):
        """Record a trade entry."""
        if symbol not in self.symbol_stats:
            self.symbol_stats[symbol] = SymbolStats(symbol=symbol)

        stats = self.symbol_stats[symbol]
        position_value = shares * price

        stats.invested_current = position_value
        stats.invested_total += position_value
        stats.current_position = {
            "direction": direction,
            "shares": shares,
            "entry_price": price,
            "entry_time": datetime.now(timezone.utc).isoformat(),
            "stop": stop_price,
            "tp": tp_price
        }

        self.portfolio.open_positions += 1
        self.portfolio.total_invested += position_value

        self._save_state()

    def record_trade_exit(
        self,
        symbol: str,
        exit_price: float,
        pnl: float,
        exit_reason: str
    ):
        """Record a trade exit."""
        if symbol not in self.symbol_stats:
            return

        stats = self.symbol_stats[symbol]

        # Update stats
        stats.trades_total += 1
        stats.trades_today += 1
        stats.pnl_total += pnl
        stats.pnl_today += pnl
        stats.pnl_this_hour += pnl

        if pnl > 0:
            stats.wins += 1
            if pnl > stats.best_trade:
                stats.best_trade = pnl
        else:
            stats.losses += 1
            if pnl < stats.worst_trade:
                stats.worst_trade = pnl

        # Calculate hold time
        if stats.current_position:
            entry_time = datetime.fromisoformat(stats.current_position["entry_time"].replace("Z", "+00:00"))
            hold_minutes = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            # Running average
            if stats.avg_hold_time_minutes == 0:
                stats.avg_hold_time_minutes = hold_minutes
            else:
                stats.avg_hold_time_minutes = (stats.avg_hold_time_minutes + hold_minutes) / 2

        stats.invested_current = 0
        stats.current_position = None
        stats.last_trade_time = datetime.now(timezone.utc).isoformat()

        # Update portfolio
        self.portfolio.total_pnl += pnl
        self.portfolio.today_pnl += pnl
        self.portfolio.this_hour_pnl += pnl
        self.portfolio.current_equity += pnl
        self.portfolio.total_trades += 1
        self.portfolio.today_trades += 1
        self.portfolio.open_positions -= 1

        if self.portfolio.current_equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = self.portfolio.current_equity

        # Calculate derived stats
        total_wins = sum(s.wins for s in self.symbol_stats.values())
        total_trades = sum(s.trades_total for s in self.symbol_stats.values())
        self.portfolio.win_rate = total_wins / total_trades if total_trades > 0 else 0

        if self.portfolio.initial_equity > 0:
            self.portfolio.total_pnl_pct = self.portfolio.total_pnl / self.portfolio.initial_equity
            self.portfolio.current_drawdown = (self.portfolio.peak_equity - self.portfolio.current_equity) / self.portfolio.peak_equity
            if self.portfolio.current_drawdown > self.portfolio.max_drawdown:
                self.portfolio.max_drawdown = self.portfolio.current_drawdown

        self._save_state()

    def update_equity(self, equity: float):
        """Update current equity."""
        self.portfolio.current_equity = equity
        if equity > self.portfolio.peak_equity:
            self.portfolio.peak_equity = equity
        self._save_state()

    # ════════════════════════════════════════════════════════════════
    # REPORT GENERATION
    # ════════════════════════════════════════════════════════════════

    def generate_global_summary(self) -> str:
        """Generate global portfolio summary."""
        p = self.portfolio

        # Calculate return per hour (based on today's data)
        hours_trading = 6.5  # Approximate trading hours
        return_per_hour = p.today_pnl / hours_trading if hours_trading > 0 else 0

        msg = f"""
<b>📊 HYPRL v1.1 - RÉSUMÉ GLOBAL</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>💰 CAPITAL</b>
Initial:     <code>${p.initial_equity:>12,.0f}</code>
Actuel:      <code>${p.current_equity:>12,.0f}</code>
Peak:        <code>${p.peak_equity:>12,.0f}</code>

<b>📈 PERFORMANCE</b>
P/L Total:   <code>${p.total_pnl:>+12,.0f}</code> ({p.total_pnl_pct:+.1%})
P/L Jour:    <code>${p.today_pnl:>+12,.0f}</code>
P/L Heure:   <code>${p.this_hour_pnl:>+12,.0f}</code>
Return/h:    <code>${return_per_hour:>+12,.0f}</code>

<b>📊 STATISTIQUES</b>
Trades Total:   <code>{p.total_trades:>8}</code>
Trades Jour:    <code>{p.today_trades:>8}</code>
Win Rate:       <code>{p.win_rate:>7.1%}</code>
Max Drawdown:   <code>{p.max_drawdown:>7.1%}</code>

<b>📍 POSITIONS</b>
Ouvertes:    <code>{p.open_positions:>8}</code>
Investi:     <code>${p.total_invested:>12,.0f}</code>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>
"""
        return msg.strip()

    def generate_symbol_report(self, symbol: str) -> str:
        """Generate report for a single symbol."""
        if symbol not in self.symbol_stats:
            return f"No data for {symbol}"

        s = self.symbol_stats[symbol]
        win_rate = s.wins / s.trades_total if s.trades_total > 0 else 0

        # Position status
        if s.current_position:
            pos = s.current_position
            pos_status = f"🟢 {pos['direction'].upper()} {pos['shares']:.1f} @ ${pos['entry_price']:.2f}"
        else:
            pos_status = "⚪ Pas de position"

        msg = f"""
<b>📈 {symbol}</b>
━━━━━━━━━━━━━━━━━━━━

<b>Performance</b>
P/L Total:   <code>${s.pnl_total:>+10,.0f}</code>
P/L Jour:    <code>${s.pnl_today:>+10,.0f}</code>
P/L Heure:   <code>${s.pnl_this_hour:>+10,.0f}</code>

<b>Trades</b>
Total:       <code>{s.trades_total:>6}</code>
Aujourd'hui: <code>{s.trades_today:>6}</code>
Win Rate:    <code>{win_rate:>5.0%}</code>
Wins/Losses: <code>{s.wins}/{s.losses}</code>

<b>Records</b>
Meilleur:    <code>${s.best_trade:>+10,.0f}</code>
Pire:        <code>${s.worst_trade:>+10,.0f}</code>
Hold moy:    <code>{s.avg_hold_time_minutes:>6.0f} min</code>

<b>Position</b>
{pos_status}
Investi:     <code>${s.invested_current:>10,.0f}</code>

━━━━━━━━━━━━━━━━━━━━
"""
        return msg.strip()

    def generate_all_symbols_summary(self) -> str:
        """Generate compact summary of all symbols."""
        lines = ["<b>📊 HYPRL v1.1 - PAR SYMBOLE</b>", "━━━━━━━━━━━━━━━━━━━━━━━━━━━━", ""]

        # Sort by PnL
        sorted_symbols = sorted(
            self.symbol_stats.items(),
            key=lambda x: x[1].pnl_total,
            reverse=True
        )

        lines.append("<code>")
        lines.append(f"{'Symbol':<6} {'P/L Tot':>10} {'P/L Jour':>10} {'Trades':>6} {'WR':>5}")
        lines.append("-" * 45)

        for sym, stats in sorted_symbols:
            wr = stats.wins / stats.trades_total if stats.trades_total > 0 else 0
            emoji = "🟢" if stats.pnl_total > 0 else "🔴" if stats.pnl_total < 0 else "⚪"
            lines.append(f"{emoji}{sym:<5} ${stats.pnl_total:>+9,.0f} ${stats.pnl_today:>+9,.0f} {stats.trades_total:>6} {wr:>4.0%}")

        lines.append("-" * 45)

        # Totals
        total_pnl = sum(s.pnl_total for s in self.symbol_stats.values())
        total_today = sum(s.pnl_today for s in self.symbol_stats.values())
        total_trades = sum(s.trades_total for s in self.symbol_stats.values())

        lines.append(f"{'TOTAL':<6} ${total_pnl:>+9,.0f} ${total_today:>+9,.0f} {total_trades:>6}")
        lines.append("</code>")

        lines.append("")
        lines.append(f"<i>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</i>")

        return "\n".join(lines)

    def generate_hourly_report(self) -> str:
        """Generate hourly performance report."""
        p = self.portfolio

        # Take snapshot
        snapshot = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "equity": p.current_equity,
            "pnl_hour": p.this_hour_pnl,
            "trades_hour": p.today_trades,  # Would need hourly tracking
            "open_positions": p.open_positions
        }
        self.hourly_snapshots.append(snapshot)

        # Reset hourly stats
        hour_pnl = p.this_hour_pnl
        p.this_hour_pnl = 0
        for s in self.symbol_stats.values():
            s.pnl_this_hour = 0

        self._save_state()

        emoji = "📈" if hour_pnl > 0 else "📉" if hour_pnl < 0 else "➡️"

        msg = f"""
{emoji} <b>RAPPORT HORAIRE</b>

P/L cette heure: <code>${hour_pnl:>+,.0f}</code>
Equity actuel:   <code>${p.current_equity:>,.0f}</code>
Positions:       <code>{p.open_positions}</code>

<i>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</i>
"""
        return msg.strip()

    # ════════════════════════════════════════════════════════════════
    # SENDING REPORTS
    # ════════════════════════════════════════════════════════════════

    def send_global_summary(self):
        """Send global summary to Telegram."""
        msg = self.generate_global_summary()
        return self._send(msg)

    def send_symbol_report(self, symbol: str):
        """Send symbol report to Telegram."""
        msg = self.generate_symbol_report(symbol)
        return self._send(msg)

    def send_all_symbols_summary(self):
        """Send all symbols summary to Telegram."""
        msg = self.generate_all_symbols_summary()
        return self._send(msg)

    def send_hourly_report(self):
        """Send hourly report to Telegram."""
        msg = self.generate_hourly_report()
        return self._send(msg)

    def send_full_report(self):
        """Send complete report (global + all symbols)."""
        # Send global summary
        self._send(self.generate_global_summary())
        time.sleep(1)  # Avoid rate limiting

        # Send symbols summary
        self._send(self.generate_all_symbols_summary())

    # ════════════════════════════════════════════════════════════════
    # DAILY RESET
    # ════════════════════════════════════════════════════════════════

    def reset_daily_stats(self):
        """Reset daily statistics."""
        self.portfolio.today_pnl = 0
        self.portfolio.today_trades = 0

        for stats in self.symbol_stats.values():
            stats.pnl_today = 0
            stats.trades_today = 0

        self._save_state()

    # ════════════════════════════════════════════════════════════════
    # AUTO REPORTING THREAD
    # ════════════════════════════════════════════════════════════════

    def start_auto_reporting(self):
        """Start automatic hourly reporting."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._reporting_loop, daemon=True)
        self._thread.start()
        print("[Reporter] Auto reporting started")

    def stop_auto_reporting(self):
        """Stop automatic reporting."""
        self._running = False
        print("[Reporter] Auto reporting stopped")

    def _reporting_loop(self):
        """Background loop for automatic reports."""
        while self._running:
            try:
                # Wait for next hour
                now = datetime.now()
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                sleep_seconds = (next_hour - now).total_seconds()

                # Sleep in chunks to allow stopping
                for _ in range(int(sleep_seconds)):
                    if not self._running:
                        return
                    time.sleep(1)

                # Send hourly report
                if self._running:
                    self.send_hourly_report()

                    # Full report at market close (16:00 ET)
                    if now.hour == 16:
                        time.sleep(2)
                        self.send_full_report()

                    # Reset daily stats at midnight
                    if now.hour == 0:
                        self.reset_daily_stats()

            except Exception as e:
                print(f"[Reporter] Error in reporting loop: {e}")
                time.sleep(60)


# ════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCE
# ════════════════════════════════════════════════════════════════════

_reporter: Optional[HyprLReporter] = None


def get_reporter() -> HyprLReporter:
    """Get or create global reporter instance."""
    global _reporter
    if _reporter is None:
        _reporter = HyprLReporter()
    return _reporter

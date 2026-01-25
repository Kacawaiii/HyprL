"""
Health Check System for HyprL
==============================
- Vérifie que le bot tourne correctement
- Heartbeat toutes les X minutes
- Détecte les crashes silencieux
- Auto-restart si nécessaire
"""

import os
import json
import time
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

from hyprl.monitoring.telegram import get_notifier


class HealthChecker:
    """Monitor bot health and send heartbeats."""

    def __init__(
        self,
        heartbeat_file: str = "live/state/heartbeat.json",
        heartbeat_interval_minutes: int = 15,
        max_staleness_minutes: int = 30,
        enable_telegram: bool = True
    ):
        self.heartbeat_file = Path(heartbeat_file)
        self.heartbeat_interval = heartbeat_interval_minutes * 60
        self.max_staleness = max_staleness_minutes * 60
        self.enable_telegram = enable_telegram

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_heartbeat: Optional[datetime] = None
        self._error_count = 0
        self._stats: Dict[str, Any] = {}

    def start(self) -> None:
        """Start the health check background thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._thread.start()
        self._write_heartbeat("started")

        if self.enable_telegram:
            get_notifier().notify_heartbeat("started")

    def stop(self) -> None:
        """Stop the health check thread."""
        self._running = False
        self._write_heartbeat("stopped")

        if self.enable_telegram:
            get_notifier().notify_heartbeat("stopped")

    def _heartbeat_loop(self) -> None:
        """Background loop that writes heartbeat periodically."""
        while self._running:
            try:
                self._write_heartbeat("alive")
                self._last_heartbeat = datetime.now(timezone.utc)
            except Exception as e:
                self._error_count += 1
                print(f"[Health] Heartbeat error: {e}")

            for _ in range(self.heartbeat_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _write_heartbeat(self, status: str) -> None:
        """Write heartbeat to file."""
        self.heartbeat_file.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "pid": os.getpid(),
            "error_count": self._error_count,
            "stats": self._stats
        }

        with open(self.heartbeat_file, "w") as f:
            json.dump(data, f, indent=2)

    def update_stats(self, **kwargs) -> None:
        """Update stats that are included in heartbeat."""
        self._stats.update(kwargs)

    @staticmethod
    def check_bot_health(heartbeat_file: str = "live/state/heartbeat.json") -> Dict[str, Any]:
        """Static method to check bot health from outside."""
        path = Path(heartbeat_file)

        if not path.exists():
            return {"healthy": False, "reason": "heartbeat file not found"}

        try:
            with open(path) as f:
                data = json.load(f)

            timestamp = datetime.fromisoformat(data["timestamp"])
            age = (datetime.now(timezone.utc) - timestamp).total_seconds()

            return {
                "healthy": age < 1800 and data.get("status") != "stopped",
                "last_heartbeat": data["timestamp"],
                "age_seconds": age,
                "status": data.get("status"),
                "pid": data.get("pid"),
                "stats": data.get("stats", {})
            }
        except Exception as e:
            return {"healthy": False, "reason": str(e)}


class RiskMonitor:
    """Monitor risk metrics and alert on thresholds."""

    def __init__(
        self,
        max_daily_loss_pct: float = 0.05,
        max_drawdown_pct: float = 0.15,
        max_consecutive_losses: int = 4,
        enable_telegram: bool = True
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.enable_telegram = enable_telegram

        self._initial_equity: Optional[float] = None
        self._peak_equity: Optional[float] = None
        self._daily_start_equity: Optional[float] = None
        self._consecutive_losses = 0
        self._alerts_sent: set = set()

    def set_initial_equity(self, equity: float) -> None:
        self._initial_equity = equity
        self._peak_equity = equity
        self._daily_start_equity = equity

    def reset_daily(self, equity: float) -> None:
        self._daily_start_equity = equity
        self._alerts_sent.clear()

    def update(self, equity: float, last_trade_pnl: Optional[float] = None) -> Dict[str, Any]:
        """Update with current equity and check for alerts."""
        alerts = []

        if self._initial_equity is None:
            self.set_initial_equity(equity)

        if equity > self._peak_equity:
            self._peak_equity = equity

        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity else 0
        daily_pnl = (equity - self._daily_start_equity) / self._daily_start_equity if self._daily_start_equity else 0

        if last_trade_pnl is not None:
            if last_trade_pnl < 0:
                self._consecutive_losses += 1
            else:
                self._consecutive_losses = 0

        if drawdown > self.max_drawdown_pct and "drawdown" not in self._alerts_sent:
            alerts.append({
                "type": "drawdown",
                "severity": "critical",
                "message": f"Drawdown {drawdown:.1%} > max {self.max_drawdown_pct:.1%}"
            })
            self._alerts_sent.add("drawdown")

        if daily_pnl < -self.max_daily_loss_pct and "daily_loss" not in self._alerts_sent:
            alerts.append({
                "type": "daily_loss",
                "severity": "critical",
                "message": f"Perte jour {daily_pnl:.1%} > max {self.max_daily_loss_pct:.1%}"
            })
            self._alerts_sent.add("daily_loss")

        if self._consecutive_losses >= self.max_consecutive_losses and "consecutive" not in self._alerts_sent:
            alerts.append({
                "type": "consecutive_losses",
                "severity": "warning",
                "message": f"{self._consecutive_losses} pertes d'affilée"
            })
            self._alerts_sent.add("consecutive")

        if self.enable_telegram:
            for alert in alerts:
                get_notifier().notify_risk_alert(
                    alert_type=alert["type"],
                    message_detail=alert["message"],
                    severity=alert["severity"]
                )

        return {
            "equity": equity,
            "drawdown": drawdown,
            "daily_pnl": daily_pnl,
            "consecutive_losses": self._consecutive_losses,
            "alerts": alerts,
            "should_stop": any(a["severity"] == "critical" for a in alerts)
        }


_health_checker: Optional[HealthChecker] = None
_risk_monitor: Optional[RiskMonitor] = None

def get_health_checker() -> HealthChecker:
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker()
    return _health_checker

def get_risk_monitor() -> RiskMonitor:
    global _risk_monitor
    if _risk_monitor is None:
        _risk_monitor = RiskMonitor()
    return _risk_monitor

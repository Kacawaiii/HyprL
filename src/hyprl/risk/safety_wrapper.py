"""
Safety Wrapper for HyprL Trading.
Wraps all trading operations with safety checks.

MUST be called before any trade execution.
"""

from datetime import datetime
from pathlib import Path
from typing import Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)


class SafetyWrapper:
    """
    Central safety controller for all trading operations.

    Checks:
    1. Kill switch (file-based, external control)
    2. Market hours (no trading outside regular hours)
    3. Circuit breakers (loss limits)
    4. Daily trade limits
    """

    KILL_SWITCH_PATH = Path("/home/kyo/HyprL/.kill_switch")

    def __init__(
        self,
        initial_equity: float = 100000,
        max_daily_loss_pct: float = 0.05,
        max_total_loss_pct: float = 0.15,
        max_trades_per_day: int = 20,
        max_consecutive_losses: int = 5,
        require_market_hours: bool = True
    ):
        self.initial_equity = initial_equity
        self.session_start_equity = initial_equity
        self.current_equity = initial_equity
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_total_loss_pct = max_total_loss_pct
        self.max_trades_per_day = max_trades_per_day
        self.max_consecutive_losses = max_consecutive_losses
        self.require_market_hours = require_market_hours

        # Counters
        self.trades_today = 0
        self.consecutive_losses = 0
        self._halted = False
        self._halt_reason = ""

    def check_kill_switch(self) -> Tuple[bool, str]:
        """Check if kill switch is active."""
        try:
            if not self.KILL_SWITCH_PATH.exists():
                return True, "OK"

            content = self.KILL_SWITCH_PATH.read_text().strip()
            first_line = content.split('\n')[0].lower()

            if first_line in ('1', 'true', 'kill', 'stop', 'yes'):
                return False, "KILL SWITCH ACTIVE"

            return True, "OK"
        except Exception as e:
            logger.error(f"Kill switch check error: {e}")
            # Fail safe - don't trade if we can't read kill switch
            return False, f"Kill switch check failed: {e}"

    def check_market_hours(self) -> Tuple[bool, str]:
        """Check if market is open."""
        if not self.require_market_hours:
            return True, "Market hours check disabled"

        try:
            from zoneinfo import ZoneInfo
            from datetime import time

            ET = ZoneInfo("America/New_York")
            now = datetime.now(ET)

            # Weekend check
            if now.weekday() >= 5:
                return False, "Market closed (weekend)"

            # Hours check (9:30 AM - 4:00 PM ET)
            current_time = now.time()
            market_open = time(9, 30)
            market_close = time(16, 0)

            if current_time < market_open:
                return False, f"Market not open yet (opens 9:30 AM ET)"
            if current_time > market_close:
                return False, f"Market closed (closes 4:00 PM ET)"

            return True, "Market open"

        except Exception as e:
            logger.error(f"Market hours check error: {e}")
            # Fail safe - don't trade if we can't check hours
            return False, f"Market hours check failed: {e}"

    def check_circuit_breakers(self) -> Tuple[bool, str]:
        """Check circuit breaker conditions."""
        if self._halted:
            return False, self._halt_reason

        # Check daily loss
        if self.session_start_equity > 0:
            daily_loss = (self.current_equity - self.session_start_equity) / self.session_start_equity
            if daily_loss <= -self.max_daily_loss_pct:
                self._halted = True
                self._halt_reason = f"Daily loss limit hit: {daily_loss:.1%}"
                return False, self._halt_reason

        # Check total loss
        if self.initial_equity > 0:
            total_loss = (self.current_equity - self.initial_equity) / self.initial_equity
            if total_loss <= -self.max_total_loss_pct:
                self._halted = True
                self._halt_reason = f"Total loss limit hit: {total_loss:.1%}"
                return False, self._halt_reason

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            self._halted = True
            self._halt_reason = f"Max consecutive losses: {self.consecutive_losses}"
            return False, self._halt_reason

        # Check daily trade limit
        if self.trades_today >= self.max_trades_per_day:
            return False, f"Daily trade limit reached: {self.trades_today}"

        return True, "OK"

    def can_trade(self) -> Tuple[bool, str]:
        """
        Master check - can we trade right now?

        Returns:
            Tuple of (can_trade, reason)
        """
        # 1. Kill switch (highest priority)
        ok, reason = self.check_kill_switch()
        if not ok:
            logger.critical(f"BLOCKED BY KILL SWITCH: {reason}")
            return False, reason

        # 2. Market hours
        ok, reason = self.check_market_hours()
        if not ok:
            logger.warning(f"BLOCKED BY MARKET HOURS: {reason}")
            return False, reason

        # 3. Circuit breakers
        ok, reason = self.check_circuit_breakers()
        if not ok:
            logger.critical(f"BLOCKED BY CIRCUIT BREAKER: {reason}")
            return False, reason

        return True, "OK"

    def update_equity(self, new_equity: float):
        """Update current equity."""
        self.current_equity = new_equity

    def record_trade(self, is_win: bool):
        """Record a completed trade."""
        self.trades_today += 1

        if is_win:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def reset_daily(self):
        """Reset daily counters (call at market open)."""
        self.session_start_equity = self.current_equity
        self.trades_today = 0
        self.consecutive_losses = 0

        # Only reset halt if not due to total loss
        if "Total loss" not in self._halt_reason:
            self._halted = False
            self._halt_reason = ""

    def get_status(self) -> dict:
        """Get current safety status."""
        return {
            "kill_switch": self.check_kill_switch()[0],
            "market_open": self.check_market_hours()[0],
            "circuit_breaker_ok": self.check_circuit_breakers()[0],
            "can_trade": self.can_trade()[0],
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "halted": self._halted,
            "halt_reason": self._halt_reason,
            "equity": self.current_equity,
            "daily_pnl_pct": (
                (self.current_equity - self.session_start_equity) / self.session_start_equity
                if self.session_start_equity > 0 else 0
            )
        }


# Global instance
_safety: Optional[SafetyWrapper] = None


def get_safety(
    initial_equity: float = 100000,
    **kwargs
) -> SafetyWrapper:
    """Get or create global safety wrapper."""
    global _safety
    if _safety is None:
        _safety = SafetyWrapper(initial_equity=initial_equity, **kwargs)
    return _safety


def can_trade() -> Tuple[bool, str]:
    """Quick check if trading is allowed."""
    safety = get_safety()
    return safety.can_trade()


def activate_kill_switch(reason: str = "Manual"):
    """Activate the kill switch."""
    path = SafetyWrapper.KILL_SWITCH_PATH
    content = f"1\n{datetime.now().isoformat()}\n{reason}"
    path.write_text(content)
    logger.critical(f"KILL SWITCH ACTIVATED: {reason}")


def deactivate_kill_switch():
    """Deactivate the kill switch."""
    path = SafetyWrapper.KILL_SWITCH_PATH
    path.write_text(f"0\nDeactivated at {datetime.now().isoformat()}")
    logger.info("Kill switch deactivated")

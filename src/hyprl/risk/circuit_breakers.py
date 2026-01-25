"""
Circuit Breakers for HyprL Trading System.
Multi-level protection against catastrophic losses.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable
import threading
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Trading halted
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker level."""
    name: str
    threshold: float  # Negative for loss (e.g., -0.05 = -5%)
    window_minutes: int  # 0 = from session start, 1440 = daily
    cooldown_minutes: int  # How long to stay triggered
    action: str  # "reduce_size", "close_new", "halt_trading", "close_all"
    description: str = ""


@dataclass
class CircuitBreakerEvent:
    """Record of a circuit breaker trigger."""
    level: str
    trigger_time: datetime
    equity_at_trigger: float
    loss_pct: float
    action_taken: str


class CircuitBreakerManager:
    """
    Manages multi-level circuit breakers.

    Inspired by NYSE circuit breakers (7%, 13%, 20%) but adapted
    for algorithmic trading with more granular levels.

    Levels:
    - Level 1 (-2%): Warning, reduce position size
    - Level 2 (-5%): Caution, stop new positions
    - Level 3 (-10%): Critical, close all and halt for day
    - Level 4 (-15%): Catastrophic, halt for week
    """

    DEFAULT_LEVELS = [
        CircuitBreakerConfig(
            name="LEVEL_1_WARNING",
            threshold=-0.02,
            window_minutes=1440,
            cooldown_minutes=15,
            action="reduce_size",
            description="Daily loss > 2%: Reduce position sizes by 50%"
        ),
        CircuitBreakerConfig(
            name="LEVEL_2_CAUTION",
            threshold=-0.05,
            window_minutes=1440,
            cooldown_minutes=60,
            action="close_new",
            description="Daily loss > 5%: Stop opening new positions"
        ),
        CircuitBreakerConfig(
            name="LEVEL_3_CRITICAL",
            threshold=-0.10,
            window_minutes=1440,
            cooldown_minutes=1440,
            action="halt_trading",
            description="Daily loss > 10%: Close all and halt for rest of day"
        ),
        CircuitBreakerConfig(
            name="LEVEL_4_CATASTROPHIC",
            threshold=-0.15,
            window_minutes=0,  # From initial equity
            cooldown_minutes=10080,  # 1 week
            action="close_all",
            description="Total loss > 15%: Close all and halt for 1 week"
        ),
    ]

    def __init__(
        self,
        initial_equity: float,
        levels: Optional[list[CircuitBreakerConfig]] = None,
        on_trigger: Optional[Callable[[str, str], None]] = None
    ):
        """
        Initialize circuit breaker manager.

        Args:
            initial_equity: Starting account equity
            levels: Custom circuit breaker levels (uses defaults if None)
            on_trigger: Callback function(level_name, action) when triggered
        """
        self.initial_equity = initial_equity
        self.session_start_equity = initial_equity
        self.current_equity = initial_equity
        self.state = CircuitState.CLOSED
        self.current_level: Optional[str] = None
        self.last_trigger_time: Optional[datetime] = None
        self.levels = levels or self.DEFAULT_LEVELS
        self.on_trigger = on_trigger
        self._lock = threading.Lock()

        # Trade tracking
        self.consecutive_losses = 0
        self.max_consecutive_losses = 5
        self.trades_today = 0
        self.max_trades_per_day = 20

        # Event history
        self.events: list[CircuitBreakerEvent] = []
        self.triggers_today = 0

        # Position size multiplier (reduced when Level 1 triggered)
        self.size_multiplier = 1.0

    def update_equity(self, new_equity: float) -> list[str]:
        """
        Update current equity and check circuit breakers.

        Args:
            new_equity: Current account equity

        Returns:
            List of actions to take
        """
        with self._lock:
            self.current_equity = new_equity
            actions = []

            # Calculate returns
            daily_return = (new_equity - self.session_start_equity) / self.session_start_equity
            total_return = (new_equity - self.initial_equity) / self.initial_equity

            for level in self.levels:
                # Use daily return for window > 0, total return for window = 0
                threshold_return = daily_return if level.window_minutes > 0 else total_return

                if threshold_return <= level.threshold:
                    if self._can_trigger(level):
                        action = self._trigger_breaker(level, threshold_return)
                        actions.append(action)

            return actions

    def _can_trigger(self, level: CircuitBreakerConfig) -> bool:
        """Check if this level can be triggered."""
        # If already triggered at higher severity, don't re-trigger lower levels
        if self.current_level:
            current_idx = next(
                (i for i, l in enumerate(self.levels) if l.name == self.current_level),
                -1
            )
            level_idx = next(
                (i for i, l in enumerate(self.levels) if l.name == level.name),
                -1
            )
            if level_idx <= current_idx:
                return False

        # Check cooldown
        if self.last_trigger_time:
            elapsed = datetime.now() - self.last_trigger_time
            if elapsed < timedelta(minutes=level.cooldown_minutes):
                return False

        return True

    def _trigger_breaker(self, level: CircuitBreakerConfig, loss_pct: float) -> str:
        """Trigger a circuit breaker."""
        self.state = CircuitState.OPEN
        self.current_level = level.name
        self.last_trigger_time = datetime.now()
        self.triggers_today += 1

        # Record event
        event = CircuitBreakerEvent(
            level=level.name,
            trigger_time=datetime.now(),
            equity_at_trigger=self.current_equity,
            loss_pct=loss_pct,
            action_taken=level.action
        )
        self.events.append(event)

        # Apply action
        if level.action == "reduce_size":
            self.size_multiplier = 0.5
        elif level.action in ("halt_trading", "close_all", "close_new"):
            self.size_multiplier = 0.0

        action_msg = f"CIRCUIT BREAKER {level.name} TRIGGERED at {loss_pct:.1%}: {level.action}"
        logger.critical(action_msg)

        # Call callback
        if self.on_trigger:
            try:
                self.on_trigger(level.name, level.action)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return action_msg

    def record_trade_result(self, is_win: bool, pnl: float = 0) -> Optional[str]:
        """
        Record result of a completed trade.

        Returns action message if a breaker is triggered.
        """
        with self._lock:
            self.trades_today += 1

            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1

            # Check consecutive losses breaker
            if self.consecutive_losses >= self.max_consecutive_losses:
                self.state = CircuitState.OPEN
                msg = f"CIRCUIT BREAKER: {self.consecutive_losses} consecutive losses - halting for 1 hour"
                logger.critical(msg)
                return msg

            # Check max trades breaker
            if self.trades_today >= self.max_trades_per_day:
                self.state = CircuitState.OPEN
                msg = f"CIRCUIT BREAKER: Max daily trades ({self.max_trades_per_day}) reached"
                logger.warning(msg)
                return msg

            return None

    def can_trade(self) -> tuple[bool, str]:
        """
        Check if trading is currently allowed.

        Returns:
            Tuple of (is_allowed, reason)
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                return False, f"Circuit breaker {self.current_level} is OPEN"

            if self.trades_today >= self.max_trades_per_day:
                return False, "Max daily trades reached"

            if self.consecutive_losses >= self.max_consecutive_losses:
                return False, "Max consecutive losses reached"

            return True, "OK"

    def can_open_new_position(self) -> tuple[bool, str]:
        """Check if new positions can be opened."""
        can, reason = self.can_trade()
        if not can:
            return can, reason

        # Check if close_new action is active
        if self.current_level == "LEVEL_2_CAUTION":
            return False, "Level 2 active: No new positions"

        return True, "OK"

    def get_position_size_multiplier(self) -> float:
        """Get current position size multiplier."""
        return self.size_multiplier

    def reset_daily(self):
        """
        Reset daily counters (call at market open).

        Note: Does NOT reset Level 4 (catastrophic) triggers.
        """
        with self._lock:
            self.session_start_equity = self.current_equity
            self.triggers_today = 0
            self.trades_today = 0
            self.consecutive_losses = 0

            # Only reset if not Level 4
            if self.current_level != "LEVEL_4_CATASTROPHIC":
                self.state = CircuitState.CLOSED
                self.current_level = None
                self.size_multiplier = 1.0

            logger.info("Daily circuit breaker reset")

    def try_recovery(self) -> bool:
        """
        Attempt to recover from OPEN state.

        Returns True if recovery successful.
        """
        with self._lock:
            if self.state != CircuitState.OPEN:
                return True

            if not self.last_trigger_time:
                return False

            # Find current level config
            level_config = next(
                (l for l in self.levels if l.name == self.current_level),
                None
            )

            if not level_config:
                return False

            # Check if cooldown has passed
            elapsed = datetime.now() - self.last_trigger_time
            if elapsed >= timedelta(minutes=level_config.cooldown_minutes):
                self.state = CircuitState.HALF_OPEN
                logger.info(f"Circuit breaker {self.current_level} entering HALF_OPEN state")

                # After successful trade in HALF_OPEN, will transition to CLOSED
                return True

            return False

    def get_status(self) -> dict:
        """Get current circuit breaker status."""
        daily_return = 0
        total_return = 0

        if self.session_start_equity > 0:
            daily_return = (self.current_equity - self.session_start_equity) / self.session_start_equity
        if self.initial_equity > 0:
            total_return = (self.current_equity - self.initial_equity) / self.initial_equity

        return {
            "state": self.state.value,
            "current_level": self.current_level,
            "daily_return": f"{daily_return:.2%}",
            "total_return": f"{total_return:.2%}",
            "size_multiplier": self.size_multiplier,
            "trades_today": self.trades_today,
            "consecutive_losses": self.consecutive_losses,
            "triggers_today": self.triggers_today,
            "can_trade": self.can_trade()[0]
        }


class KillSwitch:
    """
    Emergency kill switch - immediate total stop.

    Stored in external file to prevent modification by bot itself.
    """

    DEFAULT_PATH = Path("/home/kyo/HyprL/.kill_switch")

    def __init__(self, path: Optional[Path] = None):
        self.path = path or self.DEFAULT_PATH

    def is_killed(self) -> bool:
        """Check if kill switch is active."""
        try:
            if not self.path.exists():
                return False

            content = self.path.read_text().strip().lower()
            first_line = content.split('\n')[0]
            return first_line in ('1', 'true', 'kill', 'stop', 'yes')
        except Exception as e:
            logger.error(f"Error reading kill switch: {e}")
            return False  # Fail open to allow manual intervention

    def activate(self, reason: str = "Manual activation"):
        """Activate the kill switch."""
        content = f"1\n{datetime.now().isoformat()}\n{reason}"
        self.path.write_text(content)
        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

    def deactivate(self):
        """Deactivate the kill switch."""
        self.path.write_text("0\nDeactivated at " + datetime.now().isoformat())
        logger.info("Kill switch deactivated")

    def get_status(self) -> dict:
        """Get kill switch status."""
        if not self.path.exists():
            return {"active": False, "reason": None, "timestamp": None}

        try:
            lines = self.path.read_text().strip().split('\n')
            active = lines[0] in ('1', 'true', 'kill', 'stop', 'yes')
            timestamp = lines[1] if len(lines) > 1 else None
            reason = lines[2] if len(lines) > 2 else None

            return {
                "active": active,
                "timestamp": timestamp,
                "reason": reason
            }
        except Exception as e:
            return {"active": False, "error": str(e)}

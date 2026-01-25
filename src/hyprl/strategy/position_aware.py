"""
Position-Aware Trading Module for HyprL
Tracks positions, monitors sentiment changes, and generates dynamic exit signals.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
import json
from pathlib import Path


@dataclass
class Position:
    """Tracked position with full context."""
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    entry_time: datetime
    quantity: float
    stop_loss: float
    take_profit: float

    # Sentiment at entry
    entry_sentiment: float = 0.0
    entry_sentiment_confidence: float = 0.0

    # Current state (updated dynamically)
    current_price: float = 0.0
    current_sentiment: float = 0.0
    current_sentiment_confidence: float = 0.0

    @property
    def duration_minutes(self) -> float:
        """How long we've been in this position."""
        now = datetime.now(timezone.utc)
        if self.entry_time.tzinfo is None:
            entry = self.entry_time.replace(tzinfo=timezone.utc)
        else:
            entry = self.entry_time
        return (now - entry).total_seconds() / 60

    @property
    def duration_hours(self) -> float:
        return self.duration_minutes / 60

    @property
    def unrealized_pnl(self) -> float:
        """Current unrealized P/L in dollars."""
        if self.current_price == 0:
            return 0
        if self.side == "long":
            return (self.current_price - self.entry_price) * self.quantity
        else:  # short
            return (self.entry_price - self.current_price) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        """Current unrealized P/L as percentage."""
        if self.entry_price == 0:
            return 0
        if self.side == "long":
            return (self.current_price - self.entry_price) / self.entry_price * 100
        else:
            return (self.entry_price - self.current_price) / self.entry_price * 100

    @property
    def sentiment_delta(self) -> float:
        """How much sentiment has changed since entry."""
        return self.current_sentiment - self.entry_sentiment

    @property
    def sentiment_direction(self) -> str:
        """Is sentiment moving for or against our position?"""
        delta = self.sentiment_delta

        if self.side == "long":
            if delta > 0.2:
                return "improving"
            elif delta < -0.2:
                return "deteriorating"
        else:  # short
            if delta < -0.2:
                return "improving"  # bearish sentiment = good for short
            elif delta > 0.2:
                return "deteriorating"  # bullish sentiment = bad for short

        return "stable"

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "entry_time": self.entry_time.isoformat(),
            "quantity": self.quantity,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_sentiment": self.entry_sentiment,
            "current_price": self.current_price,
            "current_sentiment": self.current_sentiment,
            "duration_hours": self.duration_hours,
            "unrealized_pnl": self.unrealized_pnl,
            "unrealized_pnl_pct": self.unrealized_pnl_pct,
            "sentiment_delta": self.sentiment_delta,
            "sentiment_direction": self.sentiment_direction
        }


@dataclass
class ExitSignal:
    """Signal to exit a position."""
    symbol: str
    action: str  # "close", "reduce", "hold", "add"
    reason: str
    urgency: str  # "immediate", "soon", "monitor"
    confidence: float
    details: dict = field(default_factory=dict)


class PositionTracker:
    """Tracks all open positions with context."""

    def __init__(self, state_file: Optional[Path] = None):
        self.positions: dict[str, Position] = {}
        self.sentiment_history: dict[str, list[tuple[datetime, float]]] = {}
        self.state_file = state_file or Path("data/position_state.json")
        self._load_state()

    def _load_state(self):
        """Load saved state from disk."""
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    data = json.load(f)
                    # Restore positions
                    for sym, pos_data in data.get("positions", {}).items():
                        pos_data["entry_time"] = datetime.fromisoformat(pos_data["entry_time"])
                        self.positions[sym] = Position(**{
                            k: v for k, v in pos_data.items()
                            if k in Position.__dataclass_fields__
                        })
            except Exception as e:
                print(f"Error loading position state: {e}")

    def _save_state(self):
        """Save state to disk."""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "positions": {sym: pos.to_dict() for sym, pos in self.positions.items()},
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Error saving position state: {e}")

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        sentiment_score: float = 0.0,
        sentiment_confidence: float = 0.0
    ) -> Position:
        """Record a new position."""
        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            entry_time=datetime.now(timezone.utc),
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_sentiment=sentiment_score,
            entry_sentiment_confidence=sentiment_confidence,
            current_price=entry_price,
            current_sentiment=sentiment_score,
            current_sentiment_confidence=sentiment_confidence
        )
        self.positions[symbol] = pos
        self._save_state()
        return pos

    def close_position(self, symbol: str) -> Optional[Position]:
        """Remove a position (closed)."""
        pos = self.positions.pop(symbol, None)
        self._save_state()
        return pos

    def update_position(
        self,
        symbol: str,
        current_price: Optional[float] = None,
        current_sentiment: Optional[float] = None,
        current_sentiment_confidence: Optional[float] = None
    ):
        """Update position with current market data."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        if current_price is not None:
            pos.current_price = current_price

        if current_sentiment is not None:
            pos.current_sentiment = current_sentiment
            # Track sentiment history
            if symbol not in self.sentiment_history:
                self.sentiment_history[symbol] = []
            self.sentiment_history[symbol].append(
                (datetime.now(timezone.utc), current_sentiment)
            )
            # Keep only last 24h
            cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
            self.sentiment_history[symbol] = [
                (t, s) for t, s in self.sentiment_history[symbol] if t > cutoff
            ]

        if current_sentiment_confidence is not None:
            pos.current_sentiment_confidence = current_sentiment_confidence

        self._save_state()

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def get_all_positions(self) -> list[Position]:
        return list(self.positions.values())

    def has_position(self, symbol: str) -> bool:
        return symbol in self.positions


class DynamicExitRules:
    """Generate exit signals based on position state and sentiment."""

    # Configurable thresholds
    PROFIT_TAKE_PCT = 2.0  # Take profit at 2%
    SENTIMENT_REVERSAL_THRESHOLD = 0.3  # Significant sentiment change
    MAX_HOLD_HOURS = 24  # Max time to hold
    LOSS_CUT_PCT = -1.5  # Cut loss at -1.5%

    @classmethod
    def evaluate(cls, position: Position) -> ExitSignal:
        """Evaluate a position and generate exit signal."""

        signals = []

        # Rule 1: Profit + Sentiment Reversal → Take Profit
        if position.unrealized_pnl_pct > 1.0:  # In profit
            if position.sentiment_direction == "deteriorating":
                signals.append(ExitSignal(
                    symbol=position.symbol,
                    action="close",
                    reason="profit_sentiment_reversal",
                    urgency="soon",
                    confidence=0.8,
                    details={
                        "pnl_pct": position.unrealized_pnl_pct,
                        "sentiment_delta": position.sentiment_delta,
                        "message": f"En profit +{position.unrealized_pnl_pct:.1f}% mais sentiment retourne ({position.sentiment_delta:+.2f})"
                    }
                ))

        # Rule 2: Strong Profit → Secure Gains
        if position.unrealized_pnl_pct > cls.PROFIT_TAKE_PCT:
            signals.append(ExitSignal(
                symbol=position.symbol,
                action="close",
                reason="profit_target",
                urgency="soon",
                confidence=0.7,
                details={
                    "pnl_pct": position.unrealized_pnl_pct,
                    "message": f"Objectif de profit atteint: +{position.unrealized_pnl_pct:.1f}%"
                }
            ))

        # Rule 3: Loss + Sentiment Worsening → Cut Loss
        if position.unrealized_pnl_pct < -0.5:  # In loss
            if position.sentiment_direction == "deteriorating":
                signals.append(ExitSignal(
                    symbol=position.symbol,
                    action="close",
                    reason="loss_sentiment_worsening",
                    urgency="immediate",
                    confidence=0.9,
                    details={
                        "pnl_pct": position.unrealized_pnl_pct,
                        "sentiment_delta": position.sentiment_delta,
                        "message": f"En perte {position.unrealized_pnl_pct:.1f}% ET sentiment empire ({position.sentiment_delta:+.2f})"
                    }
                ))

        # Rule 4: Max Loss → Hard Stop
        if position.unrealized_pnl_pct < cls.LOSS_CUT_PCT:
            signals.append(ExitSignal(
                symbol=position.symbol,
                action="close",
                reason="max_loss",
                urgency="immediate",
                confidence=1.0,
                details={
                    "pnl_pct": position.unrealized_pnl_pct,
                    "message": f"Stop loss atteint: {position.unrealized_pnl_pct:.1f}%"
                }
            ))

        # Rule 5: Too Long in Position
        if position.duration_hours > cls.MAX_HOLD_HOURS:
            if position.unrealized_pnl_pct > 0:
                signals.append(ExitSignal(
                    symbol=position.symbol,
                    action="close",
                    reason="time_limit_profit",
                    urgency="soon",
                    confidence=0.6,
                    details={
                        "hours": position.duration_hours,
                        "pnl_pct": position.unrealized_pnl_pct,
                        "message": f"Position ouverte depuis {position.duration_hours:.1f}h, prendre profit"
                    }
                ))

        # Rule 6: Loss + Improving Sentiment → Hold
        if position.unrealized_pnl_pct < 0 and position.sentiment_direction == "improving":
            signals.append(ExitSignal(
                symbol=position.symbol,
                action="hold",
                reason="loss_but_sentiment_improving",
                urgency="monitor",
                confidence=0.5,
                details={
                    "pnl_pct": position.unrealized_pnl_pct,
                    "sentiment_delta": position.sentiment_delta,
                    "message": f"En perte mais sentiment s'améliore ({position.sentiment_delta:+.2f})"
                }
            ))

        # Return highest priority signal
        if not signals:
            return ExitSignal(
                symbol=position.symbol,
                action="hold",
                reason="no_exit_trigger",
                urgency="monitor",
                confidence=0.5,
                details={"message": "Aucun signal de sortie"}
            )

        # Sort by urgency and confidence
        urgency_order = {"immediate": 0, "soon": 1, "monitor": 2}
        signals.sort(key=lambda s: (urgency_order[s.urgency], -s.confidence))

        return signals[0]


class PositionAwareTrading:
    """Main class integrating position tracking with sentiment-based exits."""

    def __init__(self, state_file: Optional[Path] = None):
        self.tracker = PositionTracker(state_file)
        self.exit_rules = DynamicExitRules()

    def on_trade_entry(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        sentiment_score: float = 0.0,
        sentiment_confidence: float = 0.0
    ):
        """Called when entering a trade."""
        return self.tracker.open_position(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence
        )

    def on_trade_exit(self, symbol: str):
        """Called when exiting a trade."""
        return self.tracker.close_position(symbol)

    def update_market_data(
        self,
        symbol: str,
        current_price: float,
        sentiment_score: Optional[float] = None,
        sentiment_confidence: Optional[float] = None
    ):
        """Update with latest market data."""
        self.tracker.update_position(
            symbol=symbol,
            current_price=current_price,
            current_sentiment=sentiment_score,
            current_sentiment_confidence=sentiment_confidence
        )

    def evaluate_position(self, symbol: str) -> Optional[ExitSignal]:
        """Evaluate if we should exit a position."""
        pos = self.tracker.get_position(symbol)
        if not pos:
            return None
        return self.exit_rules.evaluate(pos)

    def evaluate_all_positions(self) -> list[ExitSignal]:
        """Evaluate all open positions."""
        signals = []
        for pos in self.tracker.get_all_positions():
            signal = self.exit_rules.evaluate(pos)
            signals.append(signal)
        return signals

    def get_status_report(self) -> str:
        """Get human-readable status of all positions."""
        positions = self.tracker.get_all_positions()

        if not positions:
            return "Aucune position ouverte."

        lines = ["=== POSITIONS OUVERTES ===\n"]

        for pos in positions:
            signal = self.exit_rules.evaluate(pos)

            # P/L emoji
            if pos.unrealized_pnl_pct > 1:
                pnl_emoji = "🟢"
            elif pos.unrealized_pnl_pct < -1:
                pnl_emoji = "🔴"
            else:
                pnl_emoji = "🟡"

            # Sentiment direction emoji
            if pos.sentiment_direction == "improving":
                sent_emoji = "📈"
            elif pos.sentiment_direction == "deteriorating":
                sent_emoji = "📉"
            else:
                sent_emoji = "➡️"

            lines.append(f"--- {pos.symbol} ({pos.side.upper()}) ---")
            lines.append(f"  Entrée: ${pos.entry_price:.2f} il y a {pos.duration_hours:.1f}h")
            lines.append(f"  Actuel: ${pos.current_price:.2f}")
            lines.append(f"  {pnl_emoji} P/L: ${pos.unrealized_pnl:.2f} ({pos.unrealized_pnl_pct:+.2f}%)")
            lines.append(f"  {sent_emoji} Sentiment: {pos.entry_sentiment:+.2f} → {pos.current_sentiment:+.2f} ({pos.sentiment_delta:+.2f})")
            lines.append(f"  📊 Signal: {signal.action.upper()} - {signal.reason}")
            if signal.details.get("message"):
                lines.append(f"     → {signal.details['message']}")
            lines.append("")

        return "\n".join(lines)


# CLI for testing
if __name__ == "__main__":
    # Demo
    trading = PositionAwareTrading()

    # Simulate opening a position
    trading.on_trade_entry(
        symbol="NVDA",
        side="short",
        price=182.81,
        quantity=165,
        stop_loss=186.0,
        take_profit=175.0,
        sentiment_score=-0.3,
        sentiment_confidence=0.6
    )

    # Simulate market update
    trading.update_market_data(
        symbol="NVDA",
        current_price=184.50,
        sentiment_score=0.7,  # Sentiment turned bullish!
        sentiment_confidence=0.5
    )

    # Get report
    print(trading.get_status_report())

    # Evaluate
    signal = trading.evaluate_position("NVDA")
    print(f"\nRecommandation: {signal.action.upper()}")
    print(f"Raison: {signal.reason}")
    print(f"Urgence: {signal.urgency}")

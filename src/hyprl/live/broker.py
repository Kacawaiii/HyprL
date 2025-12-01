from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from hyprl.live.types import Position, TradeSignal, ExitReason, Bar


TRADE_CSV_COLUMNS = [
    "entry_timestamp",
    "exit_timestamp",
    "direction",
    "probability_up",
    "threshold",
    "entry_price",
    "stop_price",
    "take_profit_price",
    "trailing_stop_activation_price",
    "trailing_stop_distance_price",
    "exit_price",
    "exit_reason",
    "position_size",
    "pnl",
    "return_pct",
    "equity_after",
    "risk_amount",
    "expected_pnl",
    "risk_profile",
    "effective_long_threshold",
    "effective_short_threshold",
    "regime_name",
]


@dataclass(slots=True)
class TradeRecordLive:
    symbol: str
    direction: str
    entry_timestamp: datetime
    probability_up: float
    threshold: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    trailing_stop_activation_price: float | None
    trailing_stop_distance_price: float | None
    position_size: float
    risk_amount: float
    expected_pnl: float
    risk_profile: str | None
    effective_long_threshold: float
    effective_short_threshold: float
    regime_name: str | None
    exit_timestamp: datetime | None = None
    exit_price: float | None = None
    exit_reason: ExitReason | None = None
    pnl: float | None = None
    return_pct: float | None = None
    equity_after: float | None = None

    def is_open(self) -> bool:
        return self.exit_timestamp is None

    def finalize(
        self,
        *,
        exit_price: float,
        exit_time: datetime,
        exit_reason: ExitReason,
        pnl: float,
        return_pct: float,
        equity_after: float,
    ) -> None:
        self.exit_price = exit_price
        self.exit_timestamp = exit_time
        self.exit_reason = exit_reason
        self.pnl = pnl
        self.return_pct = return_pct
        self.equity_after = equity_after

    def to_csv_row(self) -> dict[str, object]:
        if (
            self.exit_timestamp is None
            or self.exit_price is None
            or self.pnl is None
            or self.return_pct is None
            or self.equity_after is None
        ):
            raise ValueError("Trade not finalized; cannot serialize")
        return {
            "entry_timestamp": self.entry_timestamp.isoformat(),
            "exit_timestamp": self.exit_timestamp.isoformat(),
            "direction": self.direction,
            "probability_up": float(self.probability_up),
            "threshold": float(self.threshold),
            "entry_price": float(self.entry_price),
            "stop_price": float(self.stop_price),
            "take_profit_price": float(self.take_profit_price),
            "trailing_stop_activation_price": self.trailing_stop_activation_price,
            "trailing_stop_distance_price": self.trailing_stop_distance_price,
            "exit_price": float(self.exit_price),
            "exit_reason": str(self.exit_reason),
            "position_size": float(self.position_size),
            "pnl": float(self.pnl),
            "return_pct": float(self.return_pct),
            "equity_after": float(self.equity_after),
            "risk_amount": float(self.risk_amount),
            "expected_pnl": float(self.expected_pnl),
            "risk_profile": self.risk_profile,
            "effective_long_threshold": float(self.effective_long_threshold),
            "effective_short_threshold": float(self.effective_short_threshold),
            "regime_name": self.regime_name,
        }


@dataclass(slots=True)
class OpenTradeState:
    record: TradeRecordLive
    stop_price: float
    take_profit_price: float
    trailing_activation: float | None
    trailing_distance: float | None
    direction: str
    current_stop_price: float = field(init=False)
    trailing_engaged: bool = field(init=False, default=False)
    highest_price: float = field(init=False)
    lowest_price: float = field(init=False)

    def __post_init__(self) -> None:
        self.current_stop_price = float(self.stop_price)
        entry_price = float(self.record.entry_price)
        self.highest_price = entry_price
        self.lowest_price = entry_price

    def evaluate_exit(self, bar: Bar) -> tuple[float, ExitReason] | None:
        high = float(bar.high)
        low = float(bar.low)
        self._maybe_update_trailing(high, low)
        tolerance = max(1e-9, abs(self.stop_price) * 1e-9)
        if self.direction == "long":
            if low <= self.current_stop_price:
                reason: ExitReason = "trailing_stop"
                if not (self.trailing_engaged and self.current_stop_price > self.stop_price + tolerance):
                    reason = "stop_loss"
                return float(self.current_stop_price), reason
            if high >= self.take_profit_price:
                return float(self.take_profit_price), "take_profit"
        else:
            if high >= self.current_stop_price:
                reason = "trailing_stop"
                if not (self.trailing_engaged and self.current_stop_price < self.stop_price - tolerance):
                    reason = "stop_loss"
                return float(self.current_stop_price), reason
            if low <= self.take_profit_price:
                return float(self.take_profit_price), "take_profit"
        return None

    def _maybe_update_trailing(self, high: float, low: float) -> None:
        if self.trailing_activation is None or self.trailing_distance is None:
            return
        if self.direction == "long":
            self.highest_price = max(self.highest_price, high)
            if self.highest_price >= self.trailing_activation:
                new_stop = self.highest_price - self.trailing_distance
                if new_stop > self.current_stop_price:
                    self.current_stop_price = new_stop
                    self.trailing_engaged = True
        else:
            self.lowest_price = min(self.lowest_price, low)
            if self.lowest_price <= self.trailing_activation:
                new_stop = self.lowest_price + self.trailing_distance
                if new_stop < self.current_stop_price:
                    self.current_stop_price = new_stop
                    self.trailing_engaged = True


class TradeLogWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict[str, object]) -> None:
        file_exists = self.path.exists()
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TRADE_CSV_COLUMNS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)


class Broker(Protocol):
    """Execution adapter for live/paper sessions."""

    def get_positions(self) -> list[Position]:
        ...

    def get_balance(self) -> float:
        ...

    def submit_signal(self, signal: TradeSignal, bar: Bar) -> None:
        ...

    def close_position(self, symbol: str, reason: ExitReason, bar: Bar, *, exit_price: float | None = None) -> None:
        ...

    def mark_to_market(self, bar: Bar) -> None:
        ...


@dataclass
class PaperBrokerImpl:
    """Simple in-memory paper broker for live dry-runs."""

    cash: float
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0005
    trade_log_path: Path | str | None = None
    debug_exits: bool = False
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[TradeRecordLive] = field(default_factory=list)
    _log_writer: TradeLogWriter | None = field(init=False, default=None, repr=False)
    _open_states: dict[str, OpenTradeState] = field(init=False, default_factory=dict, repr=False)
    _last_exit_timestamp: dict[str, datetime] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.trade_log_path:
            path = Path(self.trade_log_path)
            self._log_writer = TradeLogWriter(path)
            self.trade_log_path = path
        else:
            self.trade_log_path = None

    def get_positions(self) -> list[Position]:
        return list(self.positions.values())

    def get_balance(self) -> float:
        equity = self.cash
        for pos in self.positions.values():
            if pos.side == "long":
                equity += pos.avg_price * pos.size + pos.unrealized_pnl
            else:
                equity += pos.unrealized_pnl
        return equity

    def submit_signal(self, signal: TradeSignal, bar: Bar) -> None:
        symbol = signal.symbol.upper()
        price = float(bar.close)
        notional = signal.size * price
        if signal.side == "long":
            self.cash -= notional
        else:
            self.cash += notional

        if symbol in self.positions:
            pos = self.positions[symbol]
            if pos.side != signal.side:
                raise NotImplementedError("Position adjustments for opposite side not implemented in PaperBrokerImpl")
            new_size = pos.size + signal.size
            if new_size <= 0:
                return
            avg_price = (pos.avg_price * pos.size + price * signal.size) / new_size
            pos.size = new_size
            pos.avg_price = avg_price
            state = self._open_states.get(symbol)
            record = state.record if state is not None else self._fallback_open_trade(symbol)
            if record is not None:
                record.position_size = new_size
                record.risk_amount = float(record.risk_amount) + float(signal.risk_amount or 0.0)
                record.expected_pnl = float(signal.expected_pnl)
            return

        self.positions[symbol] = Position(
            symbol=symbol,
            side=signal.side,
            size=signal.size,
            avg_price=price,
        )
        record = TradeRecordLive(
            symbol=symbol,
            direction=signal.side,
            entry_timestamp=signal.timestamp,
            probability_up=signal.probability_up,
            threshold=signal.threshold,
            entry_price=price,
            stop_price=float(signal.stop_price),
            take_profit_price=float(signal.take_profit_price),
            trailing_stop_activation_price=signal.trailing_stop_activation_price,
            trailing_stop_distance_price=signal.trailing_stop_distance_price,
            position_size=signal.size,
            risk_amount=signal.risk_amount,
            expected_pnl=signal.expected_pnl,
            risk_profile=signal.risk_profile,
            effective_long_threshold=signal.long_threshold,
            effective_short_threshold=signal.short_threshold,
            regime_name=signal.regime_name,
        )
        self.trades.append(record)

        self._open_states[symbol] = OpenTradeState(
            record=record,
            stop_price=float(signal.stop_price),
            take_profit_price=float(signal.take_profit_price),
            trailing_activation=signal.trailing_stop_activation_price,
            trailing_distance=signal.trailing_stop_distance_price,
            direction=signal.side,
        )

    def close_position(
        self,
        symbol: str,
        reason: ExitReason,
        bar: Bar,
        *,
        exit_price: float | None = None,
    ) -> None:
        symbol_norm = symbol.upper()
        pos = self.positions.get(symbol_norm)
        if pos is None:
            return
        price = float(exit_price if exit_price is not None else bar.close)
        if pos.side == "long":
            pos.unrealized_pnl = (price - pos.avg_price) * pos.size
        else:
            pos.unrealized_pnl = (pos.avg_price - price) * pos.size
        equity_before = self.get_balance()
        pos = self.positions.pop(symbol_norm)
        entry_notional = pos.size * pos.avg_price
        exit_notional = pos.size * price
        if pos.side == "long":
            gross_pnl = exit_notional - entry_notional
        else:
            gross_pnl = entry_notional - exit_notional
        total_rate = self.commission_pct + self.slippage_pct
        cost = 2.0 * total_rate * entry_notional if entry_notional > 0 else 0.0
        net_pnl = gross_pnl - cost
        if pos.side == "long":
            self.cash += exit_notional
        else:
            self.cash -= exit_notional
        if cost > 0:
            self.cash -= cost
        equity_after = self.get_balance()
        return_pct = net_pnl / equity_before if equity_before > 0 else 0.0
        state = self._open_states.pop(symbol_norm, None)
        trade = state.record if state is not None else self._fallback_open_trade(symbol_norm)
        if trade is not None:
            trade.finalize(
                exit_price=price,
                exit_time=bar.timestamp,
                exit_reason=reason,
                pnl=float(net_pnl),
                return_pct=float(return_pct),
                equity_after=float(equity_after),
            )
            self._persist_trade(trade)
        self._last_exit_timestamp[symbol_norm] = bar.timestamp

    def mark_to_market(self, bar: Bar) -> None:
        pos = self.positions.get(bar.symbol.upper())
        if not pos:
            return
        if pos.side == "long":
            pos.unrealized_pnl = (bar.close - pos.avg_price) * pos.size
        else:
            pos.unrealized_pnl = (pos.avg_price - bar.close) * pos.size

        self._maybe_close_on_bar(bar)

    def _maybe_close_on_bar(self, bar: Bar) -> None:
        symbol_norm = bar.symbol.upper()
        state = self._open_states.get(symbol_norm)
        if state is None:
            return
        decision = state.evaluate_exit(bar)
        if decision is None:
            return
        exit_price, exit_reason = decision
        if self.debug_exits:
            print(
                f"[EXIT_DEBUG] symbol={symbol_norm} reason={exit_reason} "
                f"exit_price={exit_price:.4f} bar_ts={bar.timestamp.isoformat()}"
            )
        self.close_position(symbol_norm, exit_reason, bar, exit_price=exit_price)

    def _persist_trade(self, trade: TradeRecordLive) -> None:
        if self._log_writer is None:
            return
        self._log_writer.append(trade.to_csv_row())

    def _fallback_open_trade(self, symbol: str) -> TradeRecordLive | None:
        for trade in reversed(self.trades):
            if trade.symbol.upper() == symbol.upper() and trade.is_open():
                return trade
        return None

    def exited_on_bar(self, symbol: str, timestamp: datetime) -> bool:
        last_ts = self._last_exit_timestamp.get(symbol.upper())
        return last_ts == timestamp if last_ts is not None else False

    def get_open_risk_amounts(self) -> dict[str, float]:
        """Return current open risk allocations keyed by symbol."""
        risk: dict[str, float] = {}
        for symbol, state in self._open_states.items():
            amount = float(state.record.risk_amount) if state and state.record else 0.0
            sym = symbol.upper()
            risk[sym] = risk.get(sym, 0.0) + amount
        return risk

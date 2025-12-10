"""Dry-run broker that simulates orders locally with JSON-backed state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import BrokerClient, BrokerOrderResult, Position


@dataclass(slots=True)
class _State:
    cash: float
    positions: dict[str, dict[str, float]]
    last_order_id: int
    last_signals: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": self.positions,
            "last_order_id": self.last_order_id,
            "last_signals": self.last_signals,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_State":
        return cls(
            cash=float(data.get("cash", 0.0) or 0.0),
            positions={
                k.upper(): {
                    "qty": float(v.get("qty", 0.0) or 0.0),
                    "avg_price": float(v.get("avg_price", 0.0) or 0.0),
                }
                for k, v in (data.get("positions") or {}).items()
            },
            last_order_id=int(data.get("last_order_id", 0) or 0),
            last_signals={k.upper(): str(v) for k, v in (data.get("last_signals") or {}).items()},
        )


class DryRunBroker(BrokerClient):
    """Simulate broker orders locally, persisting state to JSON."""

    def __init__(self, state_path: Path, *, persist: bool = True) -> None:
        self.state_path = state_path
        self.persist = persist
        self._state = self._load_state()

    def _default_state(self) -> _State:
        return _State(cash=0.0, positions={}, last_order_id=0, last_signals={})

    def _load_state(self) -> _State:
        try:
            raw = self.state_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                return self._default_state()
            return _State.from_dict(data)
        except FileNotFoundError:
            return self._default_state()
        except (OSError, json.JSONDecodeError):
            return self._default_state()

    def _save_state(self) -> None:
        if not self.persist:
            return
        payload = json.dumps(self._state.to_dict(), indent=2, sort_keys=True)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(f"{self.state_path}.tmp", "w", encoding="utf-8") as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_name = tmp.name
        os.replace(temp_name, self.state_path)

    def get_positions(self) -> list[Position]:
        positions: list[Position] = []
        for symbol, data in self._state.positions.items():
            positions.append(Position(symbol=symbol, qty=float(data.get("qty", 0.0)), avg_price=float(data.get("avg_price", 0.0))))
        return positions

    def get_cash(self) -> float:
        return self._state.cash

    def _next_order_id(self) -> str:
        self._state.last_order_id += 1
        return f"dry-{self._state.last_order_id}"

    def _update_position(self, symbol: str, side: str, qty: float, price: float | None) -> None:
        symbol = symbol.upper()
        record = self._state.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0})
        current_qty = float(record.get("qty", 0.0))
        current_avg = float(record.get("avg_price", 0.0))
        price = price if price is not None else 0.0

        if side == "BUY":
            new_qty = current_qty + qty
            if price > 0 and new_qty > 0:
                new_avg = ((current_qty * current_avg) + (qty * price)) / new_qty
            else:
                new_avg = current_avg
            record = {"qty": new_qty, "avg_price": new_avg}
            if price > 0:
                self._state.cash -= qty * price
        elif side == "SELL":
            new_qty = current_qty - qty
            if new_qty <= 0:
                record = {"qty": 0.0, "avg_price": 0.0}
            else:
                record = {"qty": new_qty, "avg_price": current_avg}
            if price > 0:
                self._state.cash += qty * price
        else:
            return

        if record["qty"] <= 0:
            self._state.positions.pop(symbol, None)
        else:
            self._state.positions[symbol] = record

    def submit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_type: str = "MKT",
        time_in_force: str = "DAY",
        price: float | None = None,
        meta: dict[str, Any] | None = None,
    ) -> BrokerOrderResult:
        side = side.upper()
        order_id = self._next_order_id()
        ts = datetime.now(timezone.utc).isoformat()
        self._update_position(symbol, side, qty, price)
        result = BrokerOrderResult(
            order_id=order_id,
            status="filled",
            filled_qty=qty,
            avg_price=price or 0.0,
            timestamp=ts,
            raw={
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "order_type": order_type,
                "time_in_force": time_in_force,
                "price": price,
                "meta": meta or {},
            },
        )
        self._save_state()
        return result

    @property
    def last_signals(self) -> dict[str, str]:
        return self._state.last_signals

    def set_last_signal(self, ticker: str, signal_id: str) -> None:
        self._state.last_signals[ticker.upper()] = signal_id
        self._save_state()

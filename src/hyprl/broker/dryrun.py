"""Dry-run broker that simulates orders locally with JSON-backed state."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base import (
    Account,
    BrokerBase,
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    TimeInForce,
    Clock,
)


@dataclass(slots=True)
class _State:
    cash: float
    positions: dict[str, dict[str, float]]
    orders: dict[str, dict[str, Any]]
    last_order_id: int
    last_signals: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "cash": self.cash,
            "positions": self.positions,
            "orders": self.orders,
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
                    "side": str(v.get("side", "long")),
                }
                for k, v in (data.get("positions") or {}).items()
            },
            orders={str(k): dict(v) for k, v in (data.get("orders") or {}).items()},
            last_order_id=int(data.get("last_order_id", 0) or 0),
            last_signals={k.upper(): str(v) for k, v in (data.get("last_signals") or {}).items()},
        )


class DryRunBroker(BrokerBase):
    """Simulate broker orders locally, persisting state to JSON."""

    def __init__(self, state_path: Path, *, persist: bool = True) -> None:
        self.state_path = state_path
        self.persist = persist
        self._state = self._load_state()

    def _default_state(self) -> _State:
        return _State(cash=0.0, positions={}, orders={}, last_order_id=0, last_signals={})

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

    def _next_order_id(self) -> str:
        self._state.last_order_id += 1
        return f"dry-{self._state.last_order_id}"

    def _parse_dt(self, value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

    def _serialize_dt(self, value: Optional[datetime]) -> Optional[str]:
        if not value:
            return None
        return value.isoformat()

    def _record_order(self, order: Order) -> None:
        self._state.orders[order.id] = {
            "id": order.id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol,
            "qty": order.qty,
            "filled_qty": order.filled_qty,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "status": order.status.value,
            "time_in_force": order.time_in_force.value,
            "limit_price": order.limit_price,
            "stop_price": order.stop_price,
            "filled_avg_price": order.filled_avg_price,
            "created_at": self._serialize_dt(order.created_at),
            "updated_at": self._serialize_dt(order.updated_at),
            "filled_at": self._serialize_dt(order.filled_at),
            "canceled_at": self._serialize_dt(order.canceled_at),
            "failed_at": self._serialize_dt(order.failed_at),
            "asset_class": order.asset_class,
            "raw": order.raw,
        }

    def _order_from_record(self, record: dict[str, Any]) -> Order:
        return Order(
            id=str(record.get("id")),
            client_order_id=str(record.get("client_order_id", "")),
            symbol=str(record.get("symbol", "")),
            qty=float(record.get("qty", 0.0) or 0.0),
            filled_qty=float(record.get("filled_qty", 0.0) or 0.0),
            side=OrderSide(str(record.get("side", OrderSide.BUY.value))),
            order_type=OrderType(str(record.get("order_type", OrderType.MARKET.value))),
            status=OrderStatus(str(record.get("status", OrderStatus.NEW.value))),
            time_in_force=TimeInForce(str(record.get("time_in_force", TimeInForce.DAY.value))),
            limit_price=record.get("limit_price"),
            stop_price=record.get("stop_price"),
            filled_avg_price=record.get("filled_avg_price"),
            created_at=self._parse_dt(record.get("created_at")),
            updated_at=self._parse_dt(record.get("updated_at")),
            filled_at=self._parse_dt(record.get("filled_at")),
            canceled_at=self._parse_dt(record.get("canceled_at")),
            failed_at=self._parse_dt(record.get("failed_at")),
            asset_class=str(record.get("asset_class", "us_equity")),
            raw=record.get("raw") or {},
        )

    def _update_position(self, symbol: str, side: OrderSide, qty: float, price: float) -> None:
        symbol = symbol.upper()
        record = self._state.positions.get(symbol, {"qty": 0.0, "avg_price": 0.0, "side": "long"})
        current_qty = float(record.get("qty", 0.0))
        current_avg = float(record.get("avg_price", 0.0))
        current_side = str(record.get("side", "long"))

        if side == OrderSide.BUY:
            if current_side == "short":
                new_qty = current_qty - qty
                if new_qty < 0:
                    record = {"qty": abs(new_qty), "avg_price": price, "side": "long"}
                elif new_qty == 0:
                    record = {"qty": 0.0, "avg_price": 0.0, "side": "long"}
                else:
                    record = {"qty": new_qty, "avg_price": current_avg, "side": "short"}
            else:
                new_qty = current_qty + qty
                new_avg = ((current_qty * current_avg) + (qty * price)) / new_qty if new_qty else 0.0
                record = {"qty": new_qty, "avg_price": new_avg, "side": "long"}
            self._state.cash -= qty * price
        elif side == OrderSide.SELL:
            if current_side == "long":
                new_qty = current_qty - qty
                if new_qty < 0:
                    record = {"qty": abs(new_qty), "avg_price": price, "side": "short"}
                elif new_qty == 0:
                    record = {"qty": 0.0, "avg_price": 0.0, "side": "long"}
                else:
                    record = {"qty": new_qty, "avg_price": current_avg, "side": "long"}
            else:
                new_qty = current_qty + qty
                new_avg = ((current_qty * current_avg) + (qty * price)) / new_qty if new_qty else 0.0
                record = {"qty": new_qty, "avg_price": new_avg, "side": "short"}
            self._state.cash += qty * price

        if record["qty"] <= 0:
            self._state.positions.pop(symbol, None)
        else:
            self._state.positions[symbol] = record

    def _position_to_dataclass(self, symbol: str, data: dict[str, float]) -> Position:
        qty = float(data.get("qty", 0.0))
        side = str(data.get("side", "long"))
        entry_price = float(data.get("avg_price", 0.0))
        current_price = entry_price
        market_value = qty * current_price
        cost_basis = qty * entry_price
        unrealized_pnl = market_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100.0) if cost_basis else 0.0
        return Position(
            symbol=symbol,
            qty=qty,
            side=side,
            entry_price=entry_price,
            current_price=current_price,
            market_value=market_value,
            cost_basis=cost_basis,
            unrealized_pnl=unrealized_pnl,
            unrealized_pnl_pct=unrealized_pnl_pct,
        )

    # =====================================================================
    # BrokerBase implementations
    # =====================================================================

    def get_account(self) -> Account:
        positions = self.list_positions()
        equity = self._state.cash + sum(p.market_value for p in positions)
        return Account(
            equity=equity,
            cash=self._state.cash,
            buying_power=equity,
            raw={"dryrun": True},
        )

    def get_position(self, symbol: str) -> Optional[Position]:
        data = self._state.positions.get(symbol.upper())
        if not data:
            return None
        return self._position_to_dataclass(symbol.upper(), data)

    def list_positions(self) -> List[Position]:
        return [self._position_to_dataclass(sym, data) for sym, data in self._state.positions.items()]

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        order_type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Order:
        side_enum = side if isinstance(side, OrderSide) else OrderSide(str(side).lower())
        order_type_enum = (
            order_type if isinstance(order_type, OrderType) else OrderType(str(order_type).lower())
        )
        tif_enum = (
            time_in_force if isinstance(time_in_force, TimeInForce) else TimeInForce(str(time_in_force).lower())
        )
        order_id = self._next_order_id()
        now = datetime.now(timezone.utc)
        client_order_id = client_order_id or self.generate_client_order_id()
        fill_price = limit_price or stop_price or 0.0

        self._update_position(symbol, side_enum, qty, fill_price)

        order = Order(
            id=order_id,
            client_order_id=client_order_id,
            symbol=symbol.upper(),
            qty=qty,
            filled_qty=qty,
            side=side_enum,
            order_type=order_type_enum,
            status=OrderStatus.FILLED,
            time_in_force=tif_enum,
            limit_price=limit_price,
            stop_price=stop_price,
            filled_avg_price=fill_price,
            created_at=now,
            updated_at=now,
            filled_at=now,
            asset_class="us_equity",
            raw={"dryrun": True},
        )
        self._record_order(order)
        self._save_state()
        return order

    def get_order(self, order_id: str) -> Optional[Order]:
        record = self._state.orders.get(order_id)
        if not record:
            return None
        return self._order_from_record(record)

    def get_order_by_client_id(self, client_order_id: str) -> Optional[Order]:
        for record in self._state.orders.values():
            if str(record.get("client_order_id", "")) == client_order_id:
                return self._order_from_record(record)
        return None

    def list_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[datetime] = None,
    ) -> List[Order]:
        orders = [self._order_from_record(rec) for rec in self._state.orders.values()]
        if status == "open":
            orders = [o for o in orders if o.is_open]
        elif status == "closed":
            orders = [o for o in orders if o.is_terminal]
        if after is not None:
            orders = [o for o in orders if o.created_at and o.created_at >= after]
        return orders[:limit]

    def cancel_order(self, order_id: str) -> bool:
        record = self._state.orders.get(order_id)
        if not record:
            return False
        if record.get("status") in (OrderStatus.CANCELED.value, OrderStatus.FILLED.value):
            return False
        record["status"] = OrderStatus.CANCELED.value
        record["canceled_at"] = self._serialize_dt(datetime.now(timezone.utc))
        self._state.orders[order_id] = record
        self._save_state()
        return True

    def cancel_all_orders(self) -> int:
        count = 0
        for order_id in list(self._state.orders.keys()):
            if self.cancel_order(order_id):
                count += 1
        return count

    def cancel_open_orders(self, symbol: str) -> int:
        """Cancel all open orders for a specific symbol."""
        symbol = symbol.upper()
        count = 0
        for order_id, record in list(self._state.orders.items()):
            if record.get("symbol", "").upper() == symbol:
                if record.get("status") not in (OrderStatus.CANCELED.value, OrderStatus.FILLED.value):
                    if self.cancel_order(order_id):
                        count += 1
        return count

    def close_position(self, symbol: str) -> Optional[Order]:
        position = self.get_position(symbol)
        if not position:
            return None
        side = OrderSide.SELL if position.side == "long" else OrderSide.BUY
        return self.submit_order(
            symbol=symbol,
            qty=position.qty,
            side=side,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.DAY,
            limit_price=position.current_price,
        )

    def close_all_positions(self) -> List[Order]:
        orders: List[Order] = []
        for position in self.list_positions():
            order = self.close_position(position.symbol)
            if order:
                orders.append(order)
        return orders

    def get_clock(self) -> Clock:
        now = datetime.now(timezone.utc)
        return Clock(timestamp=now, is_open=True, next_open=now, next_close=now)

    def is_market_open(self) -> bool:
        return True

    # =====================================================================
    # DryRun-specific helpers
    # =====================================================================

    @property
    def last_signals(self) -> dict[str, str]:
        return self._state.last_signals

    def set_last_signal(self, ticker: str, signal_id: str) -> None:
        self._state.last_signals[ticker.upper()] = signal_id
        self._save_state()

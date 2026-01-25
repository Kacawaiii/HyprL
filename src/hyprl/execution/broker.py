from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Protocol

OrderSide = Literal["buy", "sell"]
OrderType = Literal["market", "limit"]


class BrokerLike(Protocol):
    def get_balance(self) -> float: ...

    def get_positions(self) -> Dict[str, Dict[str, float]]: ...

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = "market",
    ) -> Dict[str, float]: ...

    def cancel_order(self, order_id: str) -> None: ...

    def get_open_orders(self) -> List[Dict[str, str]]: ...


@dataclass
class PaperBroker:
    """Simple in-memory broker used for paper trading sessions."""

    initial_balance: float

    def __post_init__(self) -> None:
        self.cash: float = float(self.initial_balance)
        self.positions: Dict[str, Dict[str, float]] = {}
        self.trade_log: List[Dict[str, float]] = []
        self._order_counter = 0

    def get_balance(self) -> float:
        return self.cash

    def get_positions(self) -> Dict[str, Dict[str, float]]:
        return {ticker: pos.copy() for ticker, pos in self.positions.items()}

    def get_trade_log(self) -> List[Dict[str, float]]:
        return list(self.trade_log)

    def get_open_orders(self) -> List[Dict[str, str]]:
        return []

    def cancel_order(self, order_id: str) -> None:  # pragma: no cover - not used yet
        return None

    def place_order(
        self,
        ticker: str,
        side: OrderSide,
        quantity: float,
        price: Optional[float] = None,
        order_type: OrderType = "market",
    ) -> Dict[str, float]:
        if order_type != "market":  # pragma: no cover - future extension
            raise ValueError("PaperBroker currently supports only market orders.")
        if price is None:
            raise ValueError("price is required for paper fills")
        if quantity <= 0:
            raise ValueError("quantity must be positive")

        self._order_counter += 1
        order_id = f"paper-{self._order_counter}"

        delta = quantity if side == "buy" else -quantity
        if side == "buy":
            self.cash -= quantity * price
        else:
            self.cash += quantity * price

        realized_pnl = self._update_position(ticker, delta, price)
        record = {
            "order_id": order_id,
            "ticker": ticker,
            "side": 1.0 if side == "buy" else -1.0,
            "quantity": quantity,
            "price": price,
            "realized_pnl": realized_pnl,
            "cash_after": self.cash,
        }
        self.trade_log.append(record)
        return record

    def _update_position(self, ticker: str, delta: float, price: float) -> float:
        pos = self.positions.get(ticker, {"quantity": 0.0, "avg_price": price})
        qty_old = pos["quantity"]
        avg = pos["avg_price"]
        realized = 0.0

        if qty_old == 0 or qty_old * delta > 0:
            qty_new = qty_old + delta
            avg_new = price if qty_new == 0 else (
                (abs(qty_old) * avg + abs(delta) * price) / abs(qty_new)
            )
        else:
            if abs(delta) <= abs(qty_old):
                qty_new = qty_old + delta
                if qty_old > 0:  # closing part of a long
                    realized += abs(delta) * (price - avg)
                else:  # closing part of a short
                    realized += abs(delta) * (avg - price)
                avg_new = avg if qty_new != 0 else 0.0
            else:
                closing_qty = abs(qty_old)
                if qty_old > 0:
                    realized += closing_qty * (price - avg)
                else:
                    realized += closing_qty * (avg - price)
                qty_new = delta + qty_old
                avg_new = price if qty_new != 0 else 0.0

        if qty_new == 0:
            self.positions.pop(ticker, None)
        else:
            self.positions[ticker] = {"quantity": qty_new, "avg_price": avg_new}
        return realized

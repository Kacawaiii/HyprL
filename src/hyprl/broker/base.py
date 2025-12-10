"""Broker abstractions used by HyprL dry-run bridges."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class BrokerOrderResult:
    order_id: str
    status: str
    filled_qty: float
    avg_price: float
    timestamp: str
    raw: dict[str, Any] | None = None


@dataclass(slots=True)
class Position:
    symbol: str
    qty: float
    avg_price: float


class BrokerClient(Protocol):
    """Minimal broker client interface."""

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
        ...

    def get_positions(self) -> list[Position]:
        ...

    def get_cash(self) -> float:
        ...

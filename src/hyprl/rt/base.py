"""Realtime interfaces for HyprL MVP."""

from __future__ import annotations

from typing import AsyncIterator, Literal, Protocol, TypedDict


class MarketEvent(TypedDict, total=False):
    ts: float
    symbol: str
    price: float
    bid: float | None
    ask: float | None
    volume: float | None


class MarketSource(Protocol):
    """Async producer of market events."""

    def subscribe(self, symbols: list[str]) -> None:
        ...

    async def aiter(self) -> AsyncIterator[MarketEvent]:
        ...


class PaperBroker(Protocol):
    """Interface for paper-trading brokers."""

    async def submit_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: int,
        type: Literal["market", "limit"] = "market",
        limit_price: float | None = None,
        stop_price: float | None = None,
        take_profit: float | None = None,
        client_id: str | None = None,
    ) -> dict:
        ...

    async def get_positions(self) -> list[dict]:
        ...

    async def cancel_all(self) -> None:
        ...

    async def submit_bracket(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        take_profit: float,
        stop_loss: float,
        client_id: str | None = None,
    ) -> dict:  # pragma: no cover - optional capability
        ...

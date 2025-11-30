"""Alpaca paper trading connectors (MVP)."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import AsyncIterator, Literal

import httpx

try:  # pragma: no cover - optional in tests
    import websockets
except ModuleNotFoundError:  # pragma: no cover
    websockets = None  # type: ignore

from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker


class AlpacaAuthError(RuntimeError):
    ...


class AlpacaSource(MarketSource):
    """Minimal websocket source for Alpaca market data."""

    def __init__(self, ws_url: str | None = None, key: str | None = None, secret: str | None = None):
        self.ws_url = ws_url or os.environ.get("ALPACA_DATA_WS", "wss://stream.data.alpaca.markets/v2/iex")
        self.key = key or os.environ.get("ALPACA_API_KEY_ID")
        self.secret = secret or os.environ.get("ALPACA_API_SECRET_KEY")
        if not self.key or not self.secret:
            raise AlpacaAuthError("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        if websockets is None:
            raise RuntimeError("websockets package required for AlpacaSource")
        self._symbols: list[str] = []

    def subscribe(self, symbols: list[str]) -> None:
        self._symbols = [sym.upper() for sym in symbols]

    async def _connect(self):  # pragma: no cover - exercised in integration
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "hyprl-rt-mvp",
        }
        ws = await websockets.connect(self.ws_url, extra_headers=headers)
        await ws.send(json.dumps({"action": "auth", "key": self.key, "secret": self.secret}))
        auth_resp = json.loads(await ws.recv())
        if isinstance(auth_resp, list) and auth_resp and auth_resp[0].get("status") != "authorized":
            raise AlpacaAuthError(auth_resp)
        if self._symbols:
            await ws.send(json.dumps({"action": "subscribe", "trades": self._symbols}))
        return ws

    async def aiter(self) -> AsyncIterator[MarketEvent]:  # pragma: no cover - network loop
        backoff = 1.0
        while True:
            try:
                ws = await self._connect()
                async for raw in ws:
                    payload = json.loads(raw)
                    for event in payload:
                        if event.get("T") not in {"t", "q", "b"}:  # trade/quote/bar
                            continue
                        symbol = event.get("S")
                        price = float(event.get("p") or event.get("bp") or event.get("ap") or event.get("c"))
                        market_event: MarketEvent = {
                            "ts": float(event.get("t", 0)) / 1e9 if event.get("t") else asyncio.get_event_loop().time(),
                            "symbol": symbol,
                            "price": price,
                            "bid": float(event.get("bp")) if event.get("bp") is not None else None,
                            "ask": float(event.get("ap")) if event.get("ap") is not None else None,
                            "volume": float(event.get("s") or event.get("v") or 0.0),
                        }
                        yield market_event
                await ws.close()
            except Exception:  # pragma: no cover - network
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
                continue
            backoff = 1.0


class AlpacaPaperBroker(PaperBroker):
    """REST client for Alpaca paper trading."""

    def __init__(self, base_url: str | None = None, key: str | None = None, secret: str | None = None):
        self.base_url = base_url or os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        self.key = key or os.environ.get("ALPACA_API_KEY_ID")
        self.secret = secret or os.environ.get("ALPACA_API_SECRET_KEY")
        if not self.key or not self.secret:
            raise AlpacaAuthError("Missing Alpaca API credentials")
        self._client = httpx.AsyncClient(base_url=self.base_url, headers=self._headers)

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "Content-Type": "application/json",
        }

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
        payload = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "type": type,
            "time_in_force": "gtc",
            "order_class": "simple",
        }
        if limit_price is not None:
            payload["limit_price"] = limit_price
        if stop_price is not None:
            payload["stop_price"] = stop_price
        if take_profit is not None:
            payload.setdefault("take_profit", {})["limit_price"] = take_profit
        if client_id:
            payload["client_order_id"] = client_id
        response = await self._client.post("/v2/orders", json=payload)
        response.raise_for_status()
        return response.json()

    async def get_positions(self) -> list[dict]:
        response = await self._client.get("/v2/positions")
        response.raise_for_status()
        return response.json()

    async def cancel_all(self) -> None:
        response = await self._client.delete("/v2/orders")
        response.raise_for_status()

    async def close(self) -> None:
        await self._client.aclose()

    async def submit_bracket(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        take_profit: float,
        stop_loss: float,
        client_id: str | None = None,
    ) -> dict:
        timestamp = time.time()
        oco_id = uuid.uuid4().hex
        base_payload = {
            "simulated": True,
            "oco_group_id": oco_id,
            "submitted": timestamp,
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "tp": take_profit,
            "sl": stop_loss,
        }
        payload = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "type": "market",
            "time_in_force": "gtc",
            "order_class": "bracket",
            "take_profit": {"limit_price": take_profit},
            "stop_loss": {"stop_price": stop_loss},
        }
        if client_id:
            payload["client_order_id"] = client_id
        try:  # pragma: no cover - network path
            response = await self._client.post("/v2/orders", json=payload)
            response.raise_for_status()
            data = response.json()
            remote_id = data.get("client_order_id") or data.get("id") or oco_id
            return {
                **base_payload,
                "simulated": False,
                "oco_group_id": remote_id,
                "broker_response": data,
            }
        except Exception:
            return base_payload

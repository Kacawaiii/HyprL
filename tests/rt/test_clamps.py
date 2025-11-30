from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger


class QueueSource(MarketSource):
    def __init__(self) -> None:
        self.queue: asyncio.Queue[MarketEvent | None] = asyncio.Queue()
        self.symbols: list[str] = []

    def subscribe(self, symbols: list[str]) -> None:
        self.symbols = symbols

    async def push(self, event: MarketEvent) -> None:
        await self.queue.put(event)

    async def stop(self) -> None:
        await self.queue.put(None)

    async def aiter(self):
        while True:
            item = await self.queue.get()
            if item is None:
                break
            yield item


class DummyBroker(PaperBroker):
    def __init__(self) -> None:
        self.orders = []

    async def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"status": "accepted"}

    async def get_positions(self):  # pragma: no cover
        return []

    async def cancel_all(self):  # pragma: no cover
        return None


@pytest.mark.asyncio
async def test_rate_cap_and_qty_clamp(tmp_path: Path) -> None:
    source = QueueSource()
    broker = DummyBroker()
    logger = LiveLogger(root=tmp_path, session_id="clamp-session", config={})
    cfg = LiveConfig(
        symbols=["AAA"],
        max_orders_per_min=1,
        per_symbol_cap=1,
        min_qty=5,
        risk_pct=0.01,
        warmup_bars=10,
    )

    async def producer():
        for i in range(40):
            price = 100 + i * 0.1
            event: MarketEvent = {
                "ts": float(i * 30),
                "symbol": "AAA",
                "price": price,
                "volume": 10,
            }
            await source.push(event)
        await source.stop()

    prod_task = asyncio.create_task(producer())
    await run_realtime_paper(source, broker, cfg, logger)
    await prod_task
    logger.close()

    pred_path = tmp_path / "clamp-session" / "predictions.jsonl"
    lines = [line for line in pred_path.read_text().splitlines() if line.strip()]
    assert lines
    reasons = {__import__('json').loads(line).get("reason") for line in lines}
    assert "rate_cap" in reasons or "qty_clamp" in reasons

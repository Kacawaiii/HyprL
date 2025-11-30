from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger


class QueueSource(MarketSource):
    def __init__(self) -> None:
        self.queue: asyncio.Queue[MarketEvent | None] = asyncio.Queue()
        self._symbols: list[str] = []

    def subscribe(self, symbols: list[str]) -> None:
        self._symbols = symbols

    async def push(self, event: MarketEvent) -> None:
        await self.queue.put(event)

    async def stop(self) -> None:
        await self.queue.put(None)

    async def aiter(self):
        while True:
            event = await self.queue.get()
            if event is None:
                break
            yield event


class DummyBroker(PaperBroker):
    def __init__(self) -> None:
        self.orders: list[dict] = []

    async def submit_order(self, **kwargs):
        self.orders.append(kwargs)
        return {"status": "accepted"}

    async def get_positions(self):  # pragma: no cover - not used
        return []

    async def cancel_all(self):  # pragma: no cover - not used
        return None


@pytest.mark.asyncio
async def test_queue_source_loop(tmp_path: Path) -> None:
    source = QueueSource()
    broker = DummyBroker()
    logger = LiveLogger(root=tmp_path, session_id="test-session", config={})
    cfg = LiveConfig(symbols=["AAA"], warmup_bars=60, interval="1m", threshold=0.5, risk_pct=0.1)

    async def producer():
        base_ts = 0.0
        for i in range(120):
            event = MarketEvent(ts=base_ts + i * 30, symbol="AAA", price=100 + i * 0.1, bid=None, ask=None, volume=1.0)
            await source.push(event)
        await source.stop()

    producer_task = asyncio.create_task(producer())
    await run_realtime_paper(source, broker, cfg, logger)
    await producer_task
    logger.close()

    predictions = tmp_path / "test-session" / "predictions.jsonl"
    assert predictions.exists()
    orders = tmp_path / "test-session" / "orders.jsonl"
    assert orders.exists()
    with predictions.open() as fh:
        lines = [json.loads(line) for line in fh if line.strip()]
    assert lines, "predictions should not be empty"
    reasons = {record.get("reason") for record in lines}
    assert "warmup" in reasons or "signal" in reasons
    order_records = [json.loads(line) for line in orders.read_text().splitlines() if line.strip()]
    assert order_records, "orders should be logged even in dry-run"

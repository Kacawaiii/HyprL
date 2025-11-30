from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import pytest

from hyprl.rt import engine as rt_engine
from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger


class QueueSource(MarketSource):
    def __init__(self) -> None:
        self.queue: asyncio.Queue[MarketEvent | None] = asyncio.Queue()

    def subscribe(self, symbols: list[str]) -> None:
        self.symbols = symbols

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
        self.brackets: list[dict] = []

    async def submit_order(self, **kwargs):
        return {"status": "accepted"}

    async def submit_bracket(self, **kwargs):
        payload = {
            "simulated": True,
            "oco_group_id": "SIM-OCO",
            "submitted": time.time(),
        }
        payload.update(kwargs)
        self.brackets.append(payload)
        return payload

    async def get_positions(self):  # pragma: no cover - unused
        return []

    async def cancel_all(self):  # pragma: no cover - unused
        return None


class StubProbabilityModel:
    @staticmethod
    def create() -> "StubProbabilityModel":
        return StubProbabilityModel()

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - no-op
        return None

    def predict_proba(self, *_args, **_kwargs):
        return [0.9]


@pytest.mark.asyncio
async def test_simulated_oco_bracket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = QueueSource()
    broker = DummyBroker()
    logger = LiveLogger(root=tmp_path, session_id="oco-session", config={})
    cfg = LiveConfig(
        symbols=["AAA"],
        warmup_bars=1,
        interval="1m",
        threshold=0.5,
        risk_pct=0.1,
        enable_paper=True,
        min_qty=1,
    )

    def fake_compute_features(bar_df):
        close = float(bar_df["close"].iloc[-1])
        return {
            "atr": 1.0,
            "sma_short": close,
            "sma_long": close,
            "rsi_raw": 50.0,
            "bb_width": 0.1,
            "close": close,
            "open": close,
        }

    def fake_size_position(**kwargs):
        price = kwargs["price"]
        return 5, price - 1.0, price + 1.0

    monkeypatch.setattr(rt_engine, "compute_features", fake_compute_features)
    monkeypatch.setattr(rt_engine, "ProbabilityModel", StubProbabilityModel)
    monkeypatch.setattr(rt_engine, "size_position", fake_size_position)

    async def produce():
        prices = [100.0, 102.0, 103.0]
        for idx, price in enumerate(prices):
            event = MarketEvent(ts=float(idx * 60), symbol="AAA", price=price, volume=1.0)
            await source.push(event)
        await source.stop()

    producer_task = asyncio.create_task(produce())
    await run_realtime_paper(source, broker, cfg, logger)
    await producer_task
    logger.close()

    session_dir = tmp_path / "oco-session"
    orders = [json.loads(line) for line in (session_dir / "orders.jsonl").read_text().splitlines() if line.strip()]
    fills = [json.loads(line) for line in (session_dir / "fills.jsonl").read_text().splitlines() if line.strip()]
    preds = [json.loads(line) for line in (session_dir / "predictions.jsonl").read_text().splitlines() if line.strip()]
    assert any(order.get("oco_group_id") == "SIM-OCO" for order in orders)
    tp_closes = [fill for fill in fills if fill.get("reason") == "tp"]
    assert len(tp_closes) == 1, "OCO should close exactly once"
    assert any(record.get("event") == "oco_close" and record.get("reason") == "tp" for record in preds)

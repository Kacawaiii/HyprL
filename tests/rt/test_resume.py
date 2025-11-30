from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

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
    async def submit_order(self, **kwargs):  # pragma: no cover - unused
        return {"status": "accepted"}

    async def get_positions(self):  # pragma: no cover - unused
        return []

    async def cancel_all(self):  # pragma: no cover - unused
        return None


def _write_session(root: Path, session_id: str, last_ts: int) -> None:
    session_dir = root / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    bars = [{"symbol": "AAA", "bucket": ts, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 1} for ts in range(last_ts - 120, last_ts + 60, 60)]
    (session_dir / "bars.jsonl").write_text("".join(json.dumps(row) + "\n" for row in bars), encoding="utf-8")
    equities = [{"ts": ts, "equity": 10000} for ts in range(last_ts - 120, last_ts + 60, 60)]
    (session_dir / "equity.jsonl").write_text("".join(json.dumps(row) + "\n" for row in equities), encoding="utf-8")
    manifest = {"session_id": session_id, "created_at": last_ts}
    (session_dir / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.mark.asyncio
async def test_resume_session(tmp_path: Path) -> None:
    resume_id = "test_session_001"
    _write_session(tmp_path, resume_id, last_ts=180)
    source = QueueSource()
    broker = DummyBroker()
    logger = LiveLogger(root=tmp_path, session_id=resume_id, config={})
    cfg = LiveConfig(symbols=["AAA"], warmup_bars=1, interval="1m", session_id=resume_id, resume_session=resume_id)

    async def producer():
        for ts in [240, 300]:
            await source.push(MarketEvent(ts=float(ts), symbol="AAA", price=100.0, volume=1.0))
        await source.stop()

    producer_task = asyncio.create_task(producer())
    await run_realtime_paper(source, broker, cfg, logger, None, None)
    await producer_task
    logger.close()

    bars = (tmp_path / resume_id / "bars.jsonl").read_text().splitlines()
    buckets = [json.loads(line)["bucket"] for line in bars]
    assert max(buckets) > 180
    preds = tmp_path / resume_id / "predictions.jsonl"
    events = [json.loads(line).get("event") for line in preds.read_text().splitlines() if line.strip()]
    assert "resume" in events

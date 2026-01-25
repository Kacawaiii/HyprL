from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyprl.rt.base import MarketSource, PaperBroker
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger


class FileTailSource(MarketSource):
    def __init__(self, path: Path) -> None:
        self.path = path
        self._symbols: list[str] = []

    def subscribe(self, symbols: list[str]) -> None:
        self._symbols = symbols

    async def aiter(self):
        self.path.write_text("", encoding="utf-8")
        with self.path.open("r", encoding="utf-8") as fh:
            while True:
                line = fh.readline()
                if not line:
                    await asyncio.sleep(0.01)
                    continue
                event = json.loads(line)
                if event.get("stop"):
                    break
                yield event


class NullBroker(PaperBroker):
    async def submit_order(self, **kwargs):
        return {"status": "dry"}

    async def get_positions(self):  # pragma: no cover
        return []

    async def cancel_all(self):  # pragma: no cover
        return None


@pytest.mark.asyncio
async def test_file_tail_source(tmp_path: Path) -> None:
    path = tmp_path / "stream.ndjson"
    source = FileTailSource(path)
    broker = NullBroker()
    logger = LiveLogger(root=tmp_path, session_id="tail-session", config={})
    cfg = LiveConfig(symbols=["BBB"], warmup_bars=60, interval="1m")

    async def writer():
        for idx in range(80):
            event = {"ts": float(idx * 30), "symbol": "BBB", "price": 101 + idx, "volume": 1.0}
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(event) + "\n")
            await asyncio.sleep(0.01)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"stop": True}) + "\n")

    producer_task = asyncio.create_task(writer())
    await run_realtime_paper(source, broker, cfg, logger)
    await producer_task
    logger.close()

    assert (tmp_path / "tail-session" / "events.jsonl").exists()

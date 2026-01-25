from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyprl.rt import engine as rt_engine
from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger, load_manifest


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
        self.cancel_calls = 0

    async def submit_order(self, **kwargs):
        return {"status": "accepted"}

    async def get_positions(self):  # pragma: no cover - unused
        return []

    async def cancel_all(self):
        self.cancel_calls += 1


class StubProbabilityModel:
    @staticmethod
    def create() -> "StubProbabilityModel":
        return StubProbabilityModel()

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover - no-op
        return None

    def predict_proba(self, *_args, **_kwargs):
        return [0.9]


@pytest.mark.asyncio
async def test_manifest_persists_kill_switch_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = QueueSource()
    broker = DummyBroker()
    session_id = "manifest-session"
    logger = LiveLogger(root=tmp_path, session_id=session_id, config={})
    cfg = LiveConfig(
        symbols=["AAA"],
        warmup_bars=1,
        interval="1m",
        threshold=0.5,
        risk_pct=0.25,
        kill_switch_dd=0.05,
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
        return 500, price - 1.0, price + 1.0

    monkeypatch.setattr(rt_engine, "compute_features", fake_compute_features)
    monkeypatch.setattr(rt_engine, "ProbabilityModel", StubProbabilityModel)
    monkeypatch.setattr(rt_engine, "size_position", fake_size_position)

    async def crash_prices():
        prices = [100.0, 80.0, 60.0]
        for idx, price in enumerate(prices):
            event = MarketEvent(ts=float(idx * 60), symbol="AAA", price=price, volume=1.0)
            await source.push(event)
        await source.stop()

    producer_task = asyncio.create_task(crash_prices())
    await run_realtime_paper(source, broker, cfg, logger)
    await producer_task
    logger.close()

    manifest_path = tmp_path / session_id / "session_manifest.json"
    manifest = load_manifest(manifest_path)
    assert manifest["killswitch"]["triggered"] is True
    assert manifest["killswitch"]["dd_at_trigger"] is not None
    predictions = [json.loads(line) for line in (tmp_path / session_id / "predictions.jsonl").read_text().splitlines() if line.strip()]
    kill_events = [record for record in predictions if record.get("event") == "kill_switch"]
    assert len(kill_events) == 1
    assert broker.cancel_calls == 1
    last_bar_before = manifest.get("last_bar_ts")

    # resume with new logger reusing same session
    resume_source = QueueSource()
    resume_broker = DummyBroker()
    resume_logger = LiveLogger(root=tmp_path, session_id=session_id, config={})
    resume_cfg = LiveConfig(
        symbols=["AAA"],
        warmup_bars=1,
        interval="1m",
        threshold=cfg.threshold,
        risk_pct=cfg.risk_pct,
        kill_switch_dd=0.05,
        resume_session=session_id,
    )

    async def resume_prices():
        prices = [200.0, 150.0, 120.0]
        start_ts = (last_bar_before or 0.0) + 60.0
        for idx, price in enumerate(prices):
            event = MarketEvent(ts=float(start_ts + idx * 60), symbol="AAA", price=price, volume=1.0)
            await resume_source.push(event)
        await resume_source.stop()

    producer_task = asyncio.create_task(resume_prices())
    await run_realtime_paper(resume_source, resume_broker, resume_cfg, resume_logger)
    await producer_task
    resume_logger.close()

    resume_manifest = load_manifest(manifest_path)
    predictions = [json.loads(line) for line in (tmp_path / session_id / "predictions.jsonl").read_text().splitlines() if line.strip()]
    assert len([record for record in predictions if record.get("event") == "kill_switch"]) == 1, "No double kill event"
    assert resume_broker.cancel_calls == 0, "Resume run must not cancel twice"
    assert resume_manifest["last_bar_ts"] > last_bar_before

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hyprl.rt import engine as rt_engine
from hyprl.rt.base import MarketEvent, MarketSource, PaperBroker
from hyprl.rt.engine import LiveConfig, run_realtime_paper
from hyprl.rt.logging import LiveLogger
from hyprl.rt.tuner import Tuner


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
    async def submit_order(self, **kwargs):  # pragma: no cover - dry run
        return {"status": "accepted"}

    async def get_positions(self):  # pragma: no cover - unused
        return []

    async def cancel_all(self):  # pragma: no cover - unused
        return None


class StubProbabilityModel:
    @staticmethod
    def create() -> "StubProbabilityModel":
        return StubProbabilityModel()

    def fit(self, *args, **kwargs) -> None:  # pragma: no cover
        return None

    def predict_proba(self, *_args, **_kwargs):
        return [0.9]


@pytest.mark.asyncio
async def test_tuner_updates_threshold_and_risk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = QueueSource()
    broker = DummyBroker()
    logger = LiveLogger(root=tmp_path, session_id="tuner-session", config={})
    base_threshold = 0.5
    base_risk = 0.2
    cfg = LiveConfig(
        symbols=["AAA"],
        warmup_bars=1,
        interval="1m",
        threshold=base_threshold,
        risk_pct=base_risk,
        kill_switch_dd=0.9,
        per_symbol_cap=1,
    )
    cfg.tuner = Tuner(
        thr_min=0.45,
        thr_max=0.65,
        thr_step=0.05,
        risk_min=0.05,
        risk_max=0.3,
        risk_step=0.05,
        cooldown_bars=5,
        thr=cfg.threshold,
        risk=cfg.risk_pct,
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
        return 150, price - 1.0, price + 1.0

    monkeypatch.setattr(rt_engine, "compute_features", fake_compute_features)
    monkeypatch.setattr(rt_engine, "ProbabilityModel", StubProbabilityModel)
    monkeypatch.setattr(rt_engine, "size_position", fake_size_position)

    async def produce():
        prices = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0]
        for idx, price in enumerate(prices):
            event = MarketEvent(ts=float(idx * 60), symbol="AAA", price=price, volume=1.0)
            await source.push(event)
        await source.stop()

    producer_task = asyncio.create_task(produce())
    await run_realtime_paper(source, broker, cfg, logger)
    await producer_task
    logger.close()

    predictions_path = tmp_path / "tuner-session" / "predictions.jsonl"
    tuner_events = []
    for line in predictions_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") == "tuner":
            tuner_events.append(record)
    assert len(tuner_events) == 1, "Tuner should honor cooldown and log once."
    assert cfg.threshold > base_threshold
    assert cfg.risk_pct < base_risk

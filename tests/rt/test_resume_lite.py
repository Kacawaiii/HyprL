from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyprl.rt.engine import _load_resume_state, _log_resume_event


class DummyLogger:
    def __init__(self, session_dir: Path):
        self.session_dir = session_dir
        self.root = session_dir.parent
        session_dir.mkdir(parents=True, exist_ok=True)
        for name in ["bars.jsonl", "equity.jsonl", "predictions.jsonl"]:
            path = session_dir / name
            path.touch(exist_ok=True)

    def _append(self, name: str, obj: dict) -> None:
        with (self.session_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(obj) + "\n")

    def log_prediction(
        self,
        symbol: str | None = None,
        prob_up: float | None = None,
        threshold: float | None = None,
        direction: str | None = None,
        extra: dict | None = None,
    ) -> None:
        payload = {
            "symbol": symbol,
            "prob_up": prob_up,
            "threshold": threshold,
            "direction": direction,
        }
        if extra:
            payload.update(extra)
        self._append("predictions.jsonl", payload)

    def log_bar(self, bar: dict) -> None:
        self._append("bars.jsonl", bar)


@pytest.mark.asyncio
async def test_resume_lite_skips_old_bars_and_appends_new(tmp_path: Path) -> None:
    session_dir = tmp_path / "data" / "live" / "sessions" / "test_session_resume"
    session_dir.mkdir(parents=True, exist_ok=True)
    bars_path = session_dir / "bars.jsonl"
    with bars_path.open("w", encoding="utf-8") as handle:
        for ts in [100.0, 160.0, 220.0]:
            handle.write(json.dumps({"ts": ts, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0}) + "\n")
    equity_path = session_dir / "equity.jsonl"
    with equity_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": 220.0, "equity": 10000.0}) + "\n")

    logger = DummyLogger(session_dir)

    old_bars = [
        {"ts": 180.0, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
        {"ts": 200.0, "open": 1, "high": 1, "low": 1, "close": 1, "volume": 0},
    ]
    new_bars = [
        {"ts": 280.0, "open": 2, "high": 2, "low": 2, "close": 2, "volume": 0},
        {"ts": 340.0, "open": 3, "high": 3, "low": 3, "close": 3, "volume": 0},
    ]

    last_bar_ts, equity_peak = _load_resume_state(session_dir)
    assert last_bar_ts == 220.0
    assert equity_peak == 10000.0

    _log_resume_event(logger, "test_session_resume", last_bar_ts)

    for bar in old_bars + new_bars:
        if last_bar_ts is not None and bar["ts"] <= last_bar_ts:
            continue
        logger.log_bar(bar)

    with (session_dir / "predictions.jsonl").open("r", encoding="utf-8") as handle:
        resumes = [json.loads(line) for line in handle if line.strip()]
    resume_events = [entry for entry in resumes if entry.get("event") == "resume"]
    assert len(resume_events) == 1

    with (session_dir / "bars.jsonl").open("r", encoding="utf-8") as handle:
        bars = [json.loads(line) for line in handle if line.strip()]
    ts_values = [bar["ts"] for bar in bars]
    assert ts_values.count(180.0) == 0
    assert ts_values.count(200.0) == 0
    assert any(abs(ts - 280.0) < 1e-9 for ts in ts_values)
    assert any(abs(ts - 340.0) < 1e-9 for ts in ts_values)

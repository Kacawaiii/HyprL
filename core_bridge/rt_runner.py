"""Runtime session runners bridging stub and real engines."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _merge(target: dict, patch: dict) -> dict:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge(target[key], value)
        else:
            target[key] = value
    return target


def _update_manifest(path: Path | None, patch: dict) -> None:
    if path is None:
        return
    manifest = _load_manifest(path)
    updated = _merge(manifest or {}, patch)
    path.write_text(json.dumps(updated, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


async def run_stub_session(
    config: dict,
    session_dir: Path,
    stop_event: asyncio.Event,
    manifest_path: Path | None,
) -> str:
    _update_manifest(manifest_path, {"impl": "stub"})
    symbols = [sym.upper() for sym in config.get("symbols", ["AAPL"])] or ["AAPL"]
    base_prices = {symbol: 100.0 + idx * 3.0 for idx, symbol in enumerate(symbols)}
    equity = float(config.get("initial_balance", 10_000.0))
    _append_jsonl(session_dir / "equity.jsonl", {"ts": time.time(), "equity": equity})
    counters = {"bars": 0, "predictions": 0, "fills": 0}
    rng = random.Random(sum(ord(ch) for ch in config.get("session_id", "stub")))
    last_event_ts = time.time()
    kill_switch_dd = config.get("kill_switch_dd")
    kill_triggered = False
    equity_peak = equity

    iteration = 0
    while not stop_event.is_set():
        now = time.time()
        for idx, symbol in enumerate(symbols):
            base = base_prices[symbol]
            wave = math.sin(iteration / 5.0 + idx) * 0.5
            noise = rng.uniform(-0.05, 0.05)
            price = round(base + wave + noise + iteration * 0.01, 4)
            bar_payload = {
                "ts": now,
                "symbol": symbol,
                "open": price,
                "high": price + 0.02,
                "low": price - 0.02,
                "close": price,
                "volume": float(100 + iteration),
            }
            _append_jsonl(session_dir / "bars.jsonl", bar_payload)
            counters["bars"] += 1
            last_event_ts = now

            reason = "signal" if (iteration + idx) % 3 else "rate_cap"
            prediction_payload = {
                "ts": now,
                "symbol": symbol,
                "prob_up": round(0.51 + 0.1 * math.sin(iteration / 7.0), 3),
                "threshold": config.get("threshold", 0.5),
                "direction": "UP" if iteration % 2 == 0 else "DOWN",
                "reason": reason,
                "close": price,
            }
            _append_jsonl(session_dir / "predictions.jsonl", prediction_payload)
            counters["predictions"] += 1

            if (iteration + idx) % 5 == 0:
                pnl = round((rng.random() - 0.4) * 2.0, 4)
                equity = round(equity + pnl, 4)
                equity_payload = {"ts": now, "equity": equity, "symbol": symbol, "pnl": pnl}
                _append_jsonl(
                    session_dir / "fills.jsonl",
                    {
                        "ts": now,
                        "symbol": symbol,
                        "qty": 1,
                        "price": price,
                        "pnl": pnl,
                    },
                )
                _append_jsonl(session_dir / "equity.jsonl", equity_payload)
                counters["fills"] += 1
                equity_peak = max(equity_peak, equity)
                if kill_switch_dd is not None and equity_peak > 0:
                    dd = 1.0 - (equity / equity_peak)
                    if dd >= kill_switch_dd:
                        kill_triggered = True
                        _update_manifest(
                            manifest_path,
                            {
                                "killswitch": {
                                    "triggered": True,
                                    "triggered_at_ts": now,
                                    "dd_at_trigger": dd,
                                }
                            },
                        )
                        break

            _update_manifest(
                manifest_path,
                {
                    "last_event_ts": last_event_ts,
                    "equity_peak": equity_peak,
                    "counters": counters.copy(),
                },
            )

            if stop_event.is_set() or kill_triggered:
                break

        if stop_event.is_set() or kill_triggered:
            break
        iteration += 1
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=0.05)
        except asyncio.TimeoutError:
            continue

    return "stub"


class _SyntheticSource:
    def __init__(self, stop_event: asyncio.Event, interval: float = 0.25):
        self._stop_event = stop_event
        self._interval = interval
        self._symbols: list[str] = ["AAPL"]

    def subscribe(self, symbols: List[str]) -> None:
        cleaned = [sym.upper() for sym in symbols if sym]
        if cleaned:
            self._symbols = cleaned

    async def aiter(self):
        idx = 0
        while not self._stop_event.is_set():
            ts = time.time()
            for offset, symbol in enumerate(self._symbols):
                price = 120.0 + math.sin(idx / 4.0 + offset) * 0.8 + idx * 0.01
                yield {
                    "ts": ts,
                    "symbol": symbol,
                    "price": price,
                    "volume": 10.0 + offset,
                }
            idx += 1
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._interval)
            except asyncio.TimeoutError:
                continue


class _PaperBroker:
    def __init__(self):
        self._positions: Dict[str, float] = defaultdict(float)
        self._open_orders: list[dict] = []

    async def submit_order(self, symbol: str, side: str, qty: int, **kwargs) -> dict:
        direction = 1 if side.lower() == "buy" else -1
        self._positions[symbol] += direction * qty
        order = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "ts": time.time(),
            "extra": kwargs,
        }
        self._open_orders.append(order)
        return order

    async def get_positions(self) -> list[dict]:
        return [
            {"symbol": symbol, "qty": qty}
            for symbol, qty in self._positions.items()
            if abs(qty) > 0
        ]

    async def cancel_all(self) -> None:
        self._open_orders.clear()

    async def submit_bracket(self, symbol: str, side: str, qty: float, take_profit: float, stop_loss: float, **kwargs) -> dict:
        bracket = {
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
            "ts": time.time(),
        }
        self._open_orders.append(bracket)
        return bracket


async def run_real_session(
    config: dict,
    session_dir: Path,
    stop_event: asyncio.Event,
    manifest_path: Path | None,
) -> str:
    try:
        from hyprl.rt.engine import LiveConfig, run_realtime_paper
        from hyprl.rt.logging import LiveLogger
    except Exception as exc:  # pragma: no cover - resilience path
        logger.warning("[RT] fallback stub due to missing realtime deps: %s", exc)
        _update_manifest(manifest_path, {"impl": "stub"})
        return await run_stub_session(config, session_dir, stop_event, manifest_path)

    live_cfg = LiveConfig(
        symbols=[sym.upper() for sym in config.get("symbols", ["AAPL"])],
        interval=config.get("interval", "1m"),
        threshold=float(config.get("threshold", 0.55)),
        risk_pct=float(config.get("risk_pct", 0.1)),
        enable_paper=bool(config.get("enable_paper", False)),
        session_id=config.get("session_id"),
        kill_switch_dd=config.get("kill_switch_dd"),
        resume_session=config.get("resume_session"),
    )

    source = _SyntheticSource(stop_event)
    source.subscribe(live_cfg.symbols)
    broker = _PaperBroker()

    logger_rt = LiveLogger(root=session_dir.parent, session_id=session_dir.name, config=config)
    try:
        runner_task = asyncio.create_task(
            run_realtime_paper(source, broker, live_cfg, logger_rt),
            name=f"hyprl-rt-{session_dir.name}",
        )
        stop_task = asyncio.create_task(stop_event.wait())
        done, _ = await asyncio.wait({runner_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
        if stop_task in done and not runner_task.done():
            runner_task.cancel()
        if stop_task not in done:
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass
        else:
            await stop_task
        try:
            await runner_task
        except asyncio.CancelledError:
            pass
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("[RT] real session crashed (%s); switching to stub", exc)
        _update_manifest(manifest_path, {"impl": "stub"})
        return await run_stub_session(config, session_dir, stop_event, manifest_path)
    finally:
        logger_rt.close()

    _update_manifest(manifest_path, {"impl": "real"})
    return "real"

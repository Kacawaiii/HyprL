"""Async session manager orchestrating realtime workers."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set

from sqlalchemy.orm import Session

from api import repo
from core_bridge import rt_runner

logger = logging.getLogger(__name__)

SESSION_COST = 50
USAGE_EVENT_KEY = "v2/sessions.start"


class SessionNotFoundError(Exception):
    """Raised when a session id is unknown to the manager."""


class SessionAccessError(Exception):
    """Raised when an account attempts to access a foreign session."""


@dataclass
class SessionHandle:
    session_id: str
    session_dir: Path
    account_id: str
    token_id: str
    impl: str
    started_at: float
    stop_event: asyncio.Event
    task: asyncio.Task | None = None
    status: str = "running"
    finished_at: float | None = None


def _safe_symbols(symbols: Iterable[str]) -> List[str]:
    return [sym.upper() for sym in symbols if sym]


class SessionManager:
    def __init__(self, root: Path | str | None = None):
        self._root = Path(root or "data/live/sessions").resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, SessionHandle] = {}
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return self._root

    async def start_session(
        self,
        payload,
        *,
        account_id: str,
        token_id: str,
        db: Session,
        billable: bool = True,
    ) -> dict:
        impl_env = os.getenv("HYPRL_RT_IMPL", "stub").strip().lower()
        impl = "real" if impl_env == "real" else "stub"
        runner = rt_runner.run_real_session if impl == "real" else rt_runner.run_stub_session

        async with self._lock:
            try:
                if billable:
                    repo.debit_credits(db, account_id, SESSION_COST)
                    repo.record_usage_event(
                        db,
                        account_id=account_id,
                        token_id=token_id,
                        endpoint=USAGE_EVENT_KEY,
                        cost=SESSION_COST,
                    )

                session_id = self._generate_session_id()
                session_dir = self._session_dir(session_id)
                session_dir.mkdir(parents=True, exist_ok=False)
                manifest = self._initial_manifest(
                    session_id=session_id,
                    payload=payload,
                    account_id=account_id,
                    token_id=token_id,
                    impl=impl,
                )
                self._write_manifest(session_dir, manifest)
                stop_event = asyncio.Event()
                handle = SessionHandle(
                    session_id=session_id,
                    session_dir=session_dir,
                    account_id=account_id,
                    token_id=token_id,
                    impl=impl,
                    started_at=manifest["started_at"],
                    stop_event=stop_event,
                )
                config = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload)
                config.update({
                    "session_id": session_id,
                    "account_id": account_id,
                    "token_id": token_id,
                })
                manifest_path = session_dir / "session_manifest.json"
                task = asyncio.create_task(
                    self._run_session(handle, runner, config, manifest_path),
                    name=f"hyprl-session-{session_id}",
                )
                handle.task = task
                self._handles[session_id] = handle

                def _cleanup(_):
                    self._handles.pop(session_id, None)

                task.add_done_callback(_cleanup)
                db.commit()
            except Exception:
                db.rollback()
                raise

        return {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "impl": impl,
        }

    async def get_status(
        self,
        session_id: str,
        *,
        account_id: str,
        scopes: Set[str],
    ) -> dict:
        session_dir = self._existing_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if not manifest:
            raise SessionNotFoundError(session_id)
        self._enforce_access(manifest, account_id, scopes)
        handle = self._handles.get(session_id)
        status = manifest.get("status")
        if handle:
            if handle.task and handle.task.done():
                status = handle.status
            else:
                status = "running"
        status = status or "finished"
        counters = manifest.get("counters") or self._count_logs(session_dir)
        kill_switch = manifest.get("killswitch", {})
        metrics = self._basic_metrics(session_dir)
        last_event_ts = manifest.get("last_event_ts") or self._last_event_ts(session_dir)
        return {
            "session_id": session_id,
            "status": status,
            "last_event_ts": last_event_ts,
            "counters": counters,
            "kill_switch_triggered": bool(kill_switch.get("triggered")),
            "metrics": metrics,
        }

    async def stop_session(
        self,
        session_id: str,
        *,
        account_id: str,
        scopes: Set[str],
    ) -> bool:
        session_dir = self._existing_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if not manifest:
            raise SessionNotFoundError(session_id)
        self._enforce_access(manifest, account_id, scopes)
        handle = self._handles.get(session_id)
        if handle is None:
            self._update_manifest(session_dir, {"status": "stopped", "completed_at": time.time()})
            return True
        handle.stop_event.set()
        if handle.task:
            try:
                await asyncio.wait_for(handle.task, timeout=5.0)
            except asyncio.TimeoutError:
                handle.task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await handle.task
        handle.status = "stopped"
        self._update_manifest(handle.session_dir, {"status": "stopped", "completed_at": time.time()})
        return True

    async def build_report(
        self,
        session_id: str,
        *,
        account_id: str,
        scopes: Set[str],
    ) -> dict:
        session_dir = self._existing_dir(session_id)
        manifest = self._read_manifest(session_dir)
        if not manifest:
            raise SessionNotFoundError(session_id)
        self._enforce_access(manifest, account_id, scopes)
        metrics = self._basic_metrics(session_dir) or {"pf": 0.0, "sharpe": 0.0, "dd": 0.0}
        exposure_data = self._prediction_stats(session_dir)
        fills_data = self._fill_stats(session_dir)
        top_rejections = exposure_data["top_rejections"]
        duration_s = self._session_duration(manifest, session_dir)
        return {
            "session_id": session_id,
            "metrics": {
                "pf": float(metrics.get("pf") or 0.0),
                "sharpe": float(metrics.get("sharpe") or 0.0),
                "dd": float(metrics.get("dd") or 0.0),
                "winrate": fills_data["winrate"],
                "exposure": exposure_data["exposure"],
                "avg_hold_bars": exposure_data["avg_hold_bars"],
            },
            "top_rejections": top_rejections,
            "duration_s": duration_s,
        }

    async def shutdown(self) -> None:
        handles = list(self._handles.values())
        for handle in handles:
            with contextlib.suppress(SessionNotFoundError):
                await self.stop_session(
                    handle.session_id,
                    account_id=handle.account_id,
                    scopes={"admin:*"},
                )

    async def _run_session(
        self,
        handle: SessionHandle,
        runner: Callable,
        config: dict,
        manifest_path: Path,
    ) -> None:
        status = "finished"
        try:
            impl_used = await runner(config, handle.session_dir, handle.stop_event, manifest_path)
            handle.impl = impl_used
            if handle.stop_event.is_set():
                status = "stopped"
        except asyncio.CancelledError:
            status = "stopped"
            raise
        except Exception as exc:  # pragma: no cover - safety net
            status = "failed"
            logger.exception("Session %s crashed: %s", handle.session_id, exc)
        finally:
            handle.status = status
            handle.finished_at = time.time()
            self._update_manifest(
                handle.session_dir,
                {"status": status, "completed_at": handle.finished_at, "impl": handle.impl},
            )

    def _generate_session_id(self) -> str:
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        suffix = uuid.uuid4().hex[:6]
        return f"sess_{ts}_{suffix}"

    def _session_dir(self, session_id: str) -> Path:
        path = (self._root / session_id).resolve()
        path.relative_to(self._root)
        return path

    def _existing_dir(self, session_id: str) -> Path:
        path = self._session_dir(session_id)
        if not path.exists():
            raise SessionNotFoundError(session_id)
        return path

    def _initial_manifest(self, session_id: str, payload, account_id: str, token_id: str, impl: str) -> dict:
        started_at = time.time()
        return {
            "session_id": session_id,
            "started_at": started_at,
            "account_id": account_id,
            "token_id": token_id,
            "impl": impl,
            "status": "running",
            "symbols": _safe_symbols(payload.symbols),
            "interval": payload.interval,
            "threshold": payload.threshold,
            "risk_pct": payload.risk_pct,
            "kill_switch_dd": payload.kill_switch_dd,
            "resume_session": payload.resume_session,
            "enable_paper": payload.enable_paper,
            "counters": {"bars": 0, "predictions": 0, "fills": 0},
            "killswitch": {
                "triggered": False,
                "triggered_at_ts": None,
                "dd_at_trigger": None,
            },
            "last_event_ts": None,
            "equity_peak": None,
        }

    def _write_manifest(self, session_dir: Path, manifest: dict) -> None:
        path = session_dir / "session_manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _read_manifest(self, session_dir: Path) -> dict:
        path = session_dir / "session_manifest.json"
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _update_manifest(self, session_dir: Path, patch: dict) -> None:
        manifest = self._read_manifest(session_dir)
        if not manifest:
            manifest = {}
        self._merge(manifest, patch)
        self._write_manifest(session_dir, manifest)

    def _merge(self, target: dict, patch: dict) -> dict:
        for key, value in patch.items():
            if isinstance(value, dict) and isinstance(target.get(key), dict):
                self._merge(target[key], value)
            else:
                target[key] = value
        return target

    def _count_logs(self, session_dir: Path) -> dict:
        return {
            "bars": self._count_lines(session_dir / "bars.jsonl"),
            "predictions": self._count_lines(session_dir / "predictions.jsonl"),
            "fills": self._count_lines(session_dir / "fills.jsonl"),
        }

    def _count_lines(self, path: Path) -> int:
        if not path.exists():
            return 0
        count = 0
        with path.open("r", encoding="utf-8") as handle:
            for _ in handle:
                count += 1
        return count

    def _last_event_ts(self, session_dir: Path) -> float | None:
        max_ts: float | None = None
        for name in ("bars", "predictions", "fills", "equity"):
            path = session_dir / f"{name}.jsonl"
            ts = self._last_ts_from_file(path)
            if ts is None:
                continue
            max_ts = ts if max_ts is None else max(max_ts, ts)
        return max_ts

    def _last_ts_from_file(self, path: Path) -> float | None:
        if not path.exists():
            return None
        last_ts: float | None = None
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts_val = payload.get("ts")
                if isinstance(ts_val, (int, float)):
                    last_ts = float(ts_val)
        return last_ts

    def _basic_metrics(self, session_dir: Path) -> dict | None:
        path = session_dir / "equity.jsonl"
        if not path.exists():
            return None
        equity: List[float] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                eq = payload.get("equity")
                if isinstance(eq, (int, float)):
                    equity.append(float(eq))
        if len(equity) < 2:
            return None
        returns: List[float] = []
        for prev, current in zip(equity, equity[1:]):
            if prev == 0:
                continue
            returns.append((current - prev) / prev)
        positives = sum(val for val in returns if val > 0)
        negatives = sum(val for val in returns if val < 0)
        pf = positives / abs(negatives) if negatives < 0 else None
        mean_ret = sum(returns) / len(returns) if returns else 0.0
        variance = sum((val - mean_ret) ** 2 for val in returns) / max(len(returns) - 1, 1)
        sharpe = mean_ret / math.sqrt(variance) * math.sqrt(len(returns)) if variance > 0 else None
        peak = equity[0]
        max_dd = 0.0
        for value in equity:
            peak = max(peak, value)
            if peak <= 0:
                continue
            dd = 1.0 - (value / peak)
            max_dd = max(max_dd, dd)
        return {"pf": pf, "sharpe": sharpe, "dd": max_dd}

    def _prediction_stats(self, session_dir: Path) -> dict:
        path = session_dir / "predictions.jsonl"
        total = 0
        signals = 0
        rejections: Dict[str, int] = {}
        with path.open("r", encoding="utf-8") if path.exists() else contextlib.nullcontext(None) as handle:
            if handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    total += 1
                    reason = payload.get("reason")
                    if reason == "signal":
                        signals += 1
                    elif reason:
                        rejections[reason] = rejections.get(reason, 0) + 1
        exposure = (signals / total) if total else 0.0
        avg_hold = (total / max(signals, 1)) if total else 0.0
        top = sorted(rejections.items(), key=lambda item: item[1], reverse=True)[:5]
        return {
            "exposure": float(exposure),
            "avg_hold_bars": float(avg_hold),
            "top_rejections": [(reason, count) for reason, count in top],
        }

    def _fill_stats(self, session_dir: Path) -> dict:
        path = session_dir / "fills.jsonl"
        total = 0
        wins = 0
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pnl = payload.get("pnl")
                    if isinstance(pnl, (int, float)):
                        total += 1
                        if pnl > 0:
                            wins += 1
        winrate = (wins / total) if total else 0.0
        return {"winrate": float(winrate)}

    def _session_duration(self, manifest: dict, session_dir: Path) -> float:
        started = manifest.get("started_at")
        if started is None:
            return 0.0
        end_ts = manifest.get("last_event_ts") or manifest.get("completed_at") or self._last_event_ts(session_dir)
        if end_ts is None:
            return 0.0
        return max(0.0, float(end_ts) - float(started))

    def _enforce_access(self, manifest: dict, account_id: str, scopes: Set[str]) -> None:
        owner = manifest.get("account_id")
        if owner and owner != account_id and "admin:*" not in scopes:
            raise SessionAccessError(account_id)


session_manager = SessionManager()

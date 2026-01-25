"""Realtime session logging helpers."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Dict


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _json_safe(value):
    if isinstance(value, (str, int, float)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return str(value)


class LiveLogger:
    def __init__(self, root: Path | None = None, session_id: str | None = None, config: dict | None = None):
        self.root = Path(root or "data/live/sessions")
        self.session_id = session_id or f"session_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        self.session_dir = self.root / self.session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self._handles: dict[str, Any] = {}
        self._manifest = load_manifest(self.session_dir / "session_manifest.json")
        if not self._manifest:
            self._manifest = {
                "session_id": self.session_id,
                "created_at": time.time(),
                "resumed_from": None,
                "last_bar_ts": None,
                "equity_peak": 0.0,
                "killswitch": {
                    "enabled": False,
                    "dd_limit": None,
                    "triggered": False,
                    "triggered_at_ts": None,
                    "dd_at_trigger": None,
                },
                "config": _json_safe(config or {}),
            }
        self._manifest_path = self.session_dir / "session_manifest.json"
        self._write_manifest()

    @property
    def manifest(self) -> dict:
        return self._manifest

    def update_manifest(self, patch: Dict[str, Any]) -> None:
        def _merge(target: dict, updates: dict) -> dict:
            for key, value in updates.items():
                if isinstance(value, dict) and isinstance(target.get(key), dict):
                    _merge(target[key], value)
                else:
                    target[key] = value
            return target

        _merge(self._manifest, patch)
        self._write_manifest()

    def _write_manifest(self) -> None:
        save_manifest(self._manifest_path, self._manifest)

    def _writer(self, name: str):
        if name not in self._handles:
            path = self.session_dir / f"{name}.jsonl"
            self._handles[name] = path.open("a", encoding="utf-8")
        return self._handles[name]

    def _log(self, name: str, payload: dict) -> None:
        payload.setdefault("ts", time.time())
        handle = self._writer(name)
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()

    def log_event(self, event: dict) -> None:
        self._log("events", event)

    def log_bar(self, bar: dict) -> None:
        self._log("bars", bar)

    def log_features(self, features: dict) -> None:
        self._log("features", features)

    def log_prediction(
        self,
        prediction: dict | None = None,
        *,
        symbol: str | None = None,
        prob_up: float | None = None,
        threshold: float | None = None,
        direction: str | None = None,
        extra: dict | None = None,
    ) -> None:
        payload: dict = {}
        if prediction:
            payload.update(prediction)
        if any(field is not None for field in (symbol, prob_up, threshold, direction)):
            payload.setdefault("symbol", symbol)
            payload.setdefault("prob_up", prob_up)
            payload.setdefault("threshold", threshold)
            payload.setdefault("direction", direction)
        if extra:
            payload.update(extra)
        self._log("predictions", payload)

    def log_order(self, order: dict) -> None:
        self._log("orders", order)

    def log_fill(self, fill: dict) -> None:
        self._log("fills", fill)

    def log_equity(self, equity: dict) -> None:
        self._log("equity", equity)

    def close(self) -> None:
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()
        self._write_manifest()

    def __del__(self):  # pragma: no cover - best effort
        self.close()

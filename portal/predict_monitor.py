"""Helpers shared by the Predict Monitor page and offline tests."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Iterable


def parse_symbols(raw: str) -> list[str]:
    return [symbol.strip().upper() for symbol in raw.split(",") if symbol.strip()]


def parse_optional_float(raw: str) -> float | None:
    value = raw.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:  # pragma: no cover - validation handled upstream
        raise ValueError(f"Impossible de convertir '{raw}' en float") from exc


def _parse_timestamp(raw: float | int | str | None) -> float | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    if isinstance(raw, str):
        try:
            if raw.endswith("Z"):
                raw = raw.replace("Z", "+00:00")
            return datetime.fromisoformat(raw).timestamp()
        except ValueError:
            return None
    return None


def append_predict_history(history: list[dict], response: dict, *, timestamp: float | None = None) -> list[dict]:
    ts_base = timestamp or time.time()
    results: Iterable[dict] = response.get("results", []) or []
    new_entries: list[dict] = []
    prev_ts = _parse_timestamp(history[-1].get("ts")) if history else None
    for idx, result in enumerate(results):
        entry_ts = ts_base + idx * 1e-3
        created_at = result.get("created_at")
        parsed_ts = _parse_timestamp(created_at)
        if parsed_ts is not None:
            entry_ts = parsed_ts + idx * 1e-6
        if prev_ts is not None and entry_ts <= prev_ts:
            entry_ts = prev_ts + 1e-6
        prev_ts = entry_ts
        new_entries.append(
            {
                "prediction_id": result.get("prediction_id"),
                "ts": entry_ts,
                "created_at": created_at,
                "closed_at": result.get("closed_at"),
                "symbol": str(result.get("symbol", "")).upper(),
                "prob_up": float(result.get("prob_up", 0.0)),
                "direction": str(result.get("direction", "UNKNOWN")).upper(),
                "threshold": result.get("threshold"),
                "tp": result.get("tp"),
                "sl": result.get("sl"),
                "risk_pct": result.get("risk_pct"),
                "closed": result.get("closed"),
                "outcome": result.get("outcome"),
                "pnl": result.get("pnl"),
            }
        )
    return [*history, *new_entries]


__all__ = ["append_predict_history", "parse_optional_float", "parse_symbols"]

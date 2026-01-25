from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class SignalTraceWriter:
    """Utility to persist per-bar signal diagnostics to CSV."""

    FIELDNAMES = [
        "source",
        "symbol",
        "timestamp",
        "decision",
        "reason",
        "direction",
        "probability_up",
        "long_threshold",
        "short_threshold",
        "expected_pnl",
        "min_ev",
        "trend_ok",
        "sentiment_ok",
        "atr_value",
        "position_size",
        "risk_amount",
        "equity",
        "meta",
        "feature_idx",
        "design_rows",
        "feature_signature",
    ]

    def __init__(self, path: str | Path, *, source: str, symbol: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.source = source
        self.symbol = symbol.upper()
        self._handle = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._handle, fieldnames=self.FIELDNAMES)
        self._writer.writeheader()

    def log(self, **row: Any) -> None:
        payload: dict[str, Any] = {field: None for field in self.FIELDNAMES}
        payload["source"] = self.source
        payload["symbol"] = self.symbol
        extras: list[str] = []
        for key, value in row.items():
            if key in payload:
                payload[key] = value
            else:
                extras.append(f"{key}={value}")
        if extras:
            payload["meta"] = " | ".join(extras)
        timestamp = payload.get("timestamp")
        if timestamp is not None and hasattr(timestamp, "isoformat"):
            payload["timestamp"] = timestamp.isoformat()
        self._writer.writerow(payload)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> "SignalTraceWriter":  # pragma: no cover - convenience
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # pragma: no cover - convenience
        self.close()

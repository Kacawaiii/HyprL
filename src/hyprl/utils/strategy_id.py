"""Utilities for stable strategy identity hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Tuple


@dataclass(frozen=True)
class StrategyIdentity:
    tickers: Tuple[str, ...]
    interval: str
    model_id: str
    label_mode: str | None = None
    label_horizon: int | None = None
    feature_set_id: str | None = None
    risk_profile: str | None = None
    risk_pct: float | None = None
    tp_multiple: float | None = None
    sl_multiple: float | None = None
    trailing: bool | None = None
    execution_mode: str = "backtest"
    extra: dict[str, Any] = field(default_factory=dict)


def compute_strategy_id(identity: StrategyIdentity, *, length: int = 16) -> str:
    """Return a deterministic truncated SHA256 hash for the provided identity."""
    payload = json.dumps(asdict(identity), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:length]


__all__ = ["StrategyIdentity", "compute_strategy_id"]

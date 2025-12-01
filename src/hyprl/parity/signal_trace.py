from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hyprl.logging.signals import SignalTraceWriter

PARITY_TRACE_ENV = "HYPRL_PARITY_TRACE"


def _to_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "None"):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _to_bool(value: str | None) -> bool | None:
    if value in (None, ""):
        return None
    lowered = value.lower()
    if lowered in ("true", "1", "yes"):
        return True
    if lowered in ("false", "0", "no"):
        return False
    return None


@dataclass(slots=True)
class SignalTrace:
    timestamp: str
    symbol: str
    source: str
    decision: str | None
    reason: str | None
    direction: str | None
    probability_up: float | None
    long_threshold: float | None
    short_threshold: float | None
    expected_pnl: float | None
    min_ev: float | None
    trend_ok: bool | None
    sentiment_ok: bool | None
    atr_value: float | None
    position_size: float | None
    risk_amount: float | None
    equity: float | None
    meta: str | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "SignalTrace":
        return cls(
            timestamp=row.get("timestamp", ""),
            symbol=row.get("symbol", "").upper(),
            source=row.get("source", ""),
            decision=row.get("decision"),
            reason=row.get("reason"),
            direction=row.get("direction"),
            probability_up=_to_float(row.get("probability_up")),
            long_threshold=_to_float(row.get("long_threshold")),
            short_threshold=_to_float(row.get("short_threshold")),
            expected_pnl=_to_float(row.get("expected_pnl")),
            min_ev=_to_float(row.get("min_ev")),
            trend_ok=_to_bool(row.get("trend_ok")),
            sentiment_ok=_to_bool(row.get("sentiment_ok")),
            atr_value=_to_float(row.get("atr_value")),
            position_size=_to_float(row.get("position_size")),
            risk_amount=_to_float(row.get("risk_amount")),
            equity=_to_float(row.get("equity")),
            meta=row.get("meta"),
        )


@dataclass(slots=True)
class SignalTracePair:
    timestamp: str
    symbol: str
    backtest: SignalTrace | None
    replay: SignalTrace | None

    def probability_diff(self) -> float:
        if not self.backtest or not self.replay:
            return float("inf")
        pb = self.backtest.probability_up or 0.0
        pr = self.replay.probability_up or 0.0
        return abs(pb - pr)

    def decision_match(self) -> bool:
        if not self.backtest or not self.replay:
            return False
        return (self.backtest.decision or "").lower() == (self.replay.decision or "").lower()

    def position_size_diff(self) -> float:
        if not self.backtest or not self.replay:
            return float("inf")
        pb = self.backtest.position_size or 0.0
        pr = self.replay.position_size or 0.0
        return abs(pb - pr)


def load_signal_log(path: str | Path) -> list[SignalTrace]:
    entries: list[SignalTrace] = []
    with Path(path).open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            entries.append(SignalTrace.from_row(row))
    return entries


def pair_traces(
    backtest_traces: Sequence[SignalTrace],
    replay_traces: Sequence[SignalTrace],
) -> list[SignalTracePair]:
    bt_lookup = {(trace.timestamp, trace.symbol): trace for trace in backtest_traces}
    replay_lookup = {(trace.timestamp, trace.symbol): trace for trace in replay_traces}
    all_keys = sorted(set(bt_lookup.keys()) | set(replay_lookup.keys()))
    pairs: list[SignalTracePair] = []
    for key in all_keys:
        bt = bt_lookup.get(key)
        rp = replay_lookup.get(key)
        pairs.append(SignalTracePair(timestamp=key[0], symbol=key[1], backtest=bt, replay=rp))
    return pairs


class ParityTraceHandle:
    """Runtime helper that mirrors signal logs when HYPRL_PARITY_TRACE is set."""

    def __init__(self, symbol: str, source: str) -> None:
        self.symbol = symbol.upper()
        self.source = source
        path_hint = os.getenv(PARITY_TRACE_ENV)
        self._writer: SignalTraceWriter | None
        if not path_hint:
            self._writer = None
            return
        target = Path(path_hint)
        if target.suffix and target.suffix.lower() == ".csv" and not target.is_dir():
            output_path = target
        else:
            if target.is_file():
                output_path = target
            else:
                target.mkdir(parents=True, exist_ok=True)
                output_path = target / f"{self.symbol.lower()}_{self.source}_parity.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = SignalTraceWriter(output_path, source=self.source, symbol=self.symbol)

    @property
    def active(self) -> bool:
        return self._writer is not None

    def log(self, **payload: object) -> None:
        if self._writer is None:
            return
        self._writer.log(**payload)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        self.close()


def attach_parity_trace(
    symbol: str,
    source: str,
    base_callback: Callable[..., None] | None,
) -> tuple[Callable[..., None] | None, ParityTraceHandle | None]:
    handle = ParityTraceHandle(symbol, source)
    if not handle.active:
        return base_callback, None

    def _callback(**payload: object) -> None:
        if base_callback is not None:
            base_callback(**payload)
        handle.log(**payload)

    return _callback, handle

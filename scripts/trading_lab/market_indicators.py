"""Phase 2B: a compact causal indicator library over Phase 2A series.

Every function here is pure and takes a `MarketSeries` that Phase 2A already
proved. Nothing in this module opens a database, materializes a snapshot,
reads a clock, or mutates its input -- the only way data enters is through a
series whose provenance is already fixed.

Two conventions carry the whole safety argument:

* **Causality by segment.** A missing opening is a break in knowledge, not a
  shortcut. The series is split into maximal runs of strictly adjacent
  openings, and every recursive indicator (EMA, ATR, RSI) restarts its
  warm-up at the beginning of each run. A `previous_close` never crosses a
  gap; a window that would span one yields None. Nothing is forward-filled.
* **Deterministic arithmetic.** Divisions run inside a local Decimal context
  fixed by this module, so results do not change because a caller altered
  `decimal.getcontext()` elsewhere in the process.

Outputs are always the same length as the source series and aligned index by
index, with None wherever the required causal history is not yet available.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal, localcontext
import hashlib
import json

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.market_series import MarketSeries

INDICATOR_SCHEMA_VERSION = "trading-lab.market-indicator.v1"
# High enough that ordinary price arithmetic is exact well past any realistic
# significance, and FIXED here so a caller's global context cannot move it.
INDICATOR_PRECISION = 34
MAX_INDICATOR_PERIOD = 1_000

_HUNDRED = Decimal(100)
_FIFTY = Decimal(50)
_ZERO = Decimal(0)


class MarketIndicatorError(RuntimeError):
    """Raised when an indicator cannot be computed safely."""


@dataclass(frozen=True)
class IndicatorSpec:
    """The DEFINITION of a computation -- never the values it produced.

    Two runs sharing a spec share a `spec_hash`; changing the period, the
    version or the name changes it. That makes a stored feature traceable to
    the exact formula that produced it.
    """

    name: str
    version: str
    parameters: tuple[tuple[str, object], ...]

    def canonical(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "version": self.version,
                "parameters": dict(self.parameters),
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    @property
    def spec_hash(self) -> str:
        return hashlib.sha256(self.canonical().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class IndicatorResult:
    """One indicator over one series: what was computed, and by what rule."""

    spec: IndicatorSpec
    spec_hash: str
    values: tuple[Decimal | None, ...]


def _require_period(period: object) -> int:
    # bool is an int subclass; True must not read as period 1 here.
    if type(period) is not int or not 1 <= period <= MAX_INDICATOR_PERIOD:
        raise MarketIndicatorError(
            f"period must be an int in 1..{MAX_INDICATOR_PERIOD}, got {period!r}"
        )
    return period


def _spec(name: str, **parameters: object) -> IndicatorSpec:
    return IndicatorSpec(
        name=name,
        version=INDICATOR_SCHEMA_VERSION,
        parameters=tuple(sorted(parameters.items())),
    )


def _result(spec: IndicatorSpec, values: list[Decimal | None]) -> IndicatorResult:
    return IndicatorResult(spec=spec, spec_hash=spec.spec_hash, values=tuple(values))


def contiguous_segments(series: MarketSeries) -> tuple[tuple[int, int], ...]:
    """Maximal runs of strictly adjacent openings, as [start, end) indices.

    This is where the gap policy lives: every indicator below iterates over
    these runs, so no state and no `previous_close` can survive a gap.
    """
    if series.timeframe not in TIMEFRAME_DURATIONS:
        raise MarketIndicatorError(f"unsupported timeframe {series.timeframe!r}")
    duration: timedelta = TIMEFRAME_DURATIONS[series.timeframe]
    points = series.points
    if not points:
        return ()
    segments: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(points)):
        previous = datetime.fromisoformat(points[index - 1].bar_open_at)
        current = datetime.fromisoformat(points[index].bar_open_at)
        if current - previous != duration:
            segments.append((start, index))
            start = index
    segments.append((start, len(points)))
    return tuple(segments)


def simple_moving_average(series: MarketSeries, *, period: int) -> IndicatorResult:
    """Mean close over `period` strictly adjacent observations ending at i."""
    period = _require_period(period)
    values: list[Decimal | None] = [None] * len(series.points)
    with localcontext() as context:
        context.prec = INDICATOR_PRECISION
        for start, end in contiguous_segments(series):
            closes = [point.close for point in series.points[start:end]]
            for offset in range(period - 1, len(closes)):
                window = closes[offset - period + 1 : offset + 1]
                values[start + offset] = sum(window) / Decimal(period)
    return _result(_spec("sma", period=period), values)


def exponential_moving_average(series: MarketSeries, *, period: int) -> IndicatorResult:
    """EMA with alpha = 2 / (period + 1), seeded by the exact SMA of the
    first `period` observations of each contiguous segment.

    The seed matters: initialising from the first close instead would make
    early values depend on where the segment happens to start."""
    period = _require_period(period)
    values: list[Decimal | None] = [None] * len(series.points)
    with localcontext() as context:
        context.prec = INDICATOR_PRECISION
        alpha = Decimal(2) / Decimal(period + 1)
        for start, end in contiguous_segments(series):
            closes = [point.close for point in series.points[start:end]]
            if len(closes) < period:
                continue
            previous = sum(closes[:period]) / Decimal(period)
            values[start + period - 1] = previous
            for offset in range(period, len(closes)):
                previous = alpha * closes[offset] + (Decimal(1) - alpha) * previous
                values[start + offset] = previous
    return _result(_spec("ema", period=period), values)


def true_range(series: MarketSeries) -> IndicatorResult:
    """TR, with `previous_close` never reaching across a gap.

    The first bar of each contiguous segment has no usable predecessor, so
    it falls back to high - low rather than borrowing the close from the
    other side of the hole."""
    values: list[Decimal | None] = [None] * len(series.points)
    with localcontext() as context:
        context.prec = INDICATOR_PRECISION
        for start, end in contiguous_segments(series):
            for index in range(start, end):
                point = series.points[index]
                if index == start:
                    values[index] = point.high - point.low
                    continue
                previous_close = series.points[index - 1].close
                values[index] = max(
                    point.high - point.low,
                    abs(point.high - previous_close),
                    abs(point.low - previous_close),
                )
    return _result(_spec("true_range"), values)


def average_true_range(series: MarketSeries, *, period: int) -> IndicatorResult:
    """Wilder ATR: seeded by the mean of the first `period` true ranges of a
    segment, then ATR_t = (ATR_{t-1} * (period - 1) + TR_t) / period."""
    period = _require_period(period)
    ranges = true_range(series).values
    values: list[Decimal | None] = [None] * len(series.points)
    with localcontext() as context:
        context.prec = INDICATOR_PRECISION
        for start, end in contiguous_segments(series):
            segment = [ranges[index] for index in range(start, end)]
            if len(segment) < period:
                continue
            previous = sum(segment[:period]) / Decimal(period)
            values[start + period - 1] = previous
            for offset in range(period, len(segment)):
                previous = (previous * Decimal(period - 1) + segment[offset]) / Decimal(period)
                values[start + offset] = previous
    return _result(_spec("atr", period=period), values)


def relative_strength_index(series: MarketSeries, *, period: int) -> IndicatorResult:
    """Wilder RSI over deltas between strictly adjacent closes.

    Boundary conventions, fixed here so they never drift:
    average_loss == 0 and average_gain > 0 -> 100;
    average_gain == 0 and average_loss > 0 -> 0;
    both zero (a perfectly flat window) -> 50, the neutral reading, rather
    than an undefined division."""
    period = _require_period(period)
    values: list[Decimal | None] = [None] * len(series.points)
    with localcontext() as context:
        context.prec = INDICATOR_PRECISION
        for start, end in contiguous_segments(series):
            closes = [point.close for point in series.points[start:end]]
            if len(closes) <= period:
                continue
            deltas = [closes[step] - closes[step - 1] for step in range(1, len(closes))]
            gains = [delta if delta > 0 else _ZERO for delta in deltas]
            losses = [-delta if delta < 0 else _ZERO for delta in deltas]
            average_gain = sum(gains[:period]) / Decimal(period)
            average_loss = sum(losses[:period]) / Decimal(period)
            values[start + period] = _wilder_rsi(average_gain, average_loss)
            for offset in range(period, len(deltas)):
                average_gain = (average_gain * Decimal(period - 1) + gains[offset]) / Decimal(period)
                average_loss = (average_loss * Decimal(period - 1) + losses[offset]) / Decimal(period)
                values[start + offset + 1] = _wilder_rsi(average_gain, average_loss)
    return _result(_spec("rsi", period=period), values)


def _wilder_rsi(average_gain: Decimal, average_loss: Decimal) -> Decimal:
    if average_loss == _ZERO:
        return _FIFTY if average_gain == _ZERO else _HUNDRED
    if average_gain == _ZERO:
        return _ZERO
    return _HUNDRED - (_HUNDRED / (Decimal(1) + average_gain / average_loss))

"""Phase 2A: turn an immutable causal snapshot into an analysis-ready series.

This is the first consumer of the Phase 1 foundation and the boundary every
later Phase 2 capability must go through. It reads ONLY through
`replay_snapshot`, so every bar it returns has already been proven against
its receipt, and it carries the snapshot's provenance (`snapshot_id`,
`as_of`, `entries_content_hash`) on the result.

Two rules make it safe to build features on top:

* **Causality is structural.** Every derived value at index i is computed
  from points 0..i only. There is no place in this module where a later
  point can reach an earlier one, so "no future leakage" is a property of
  the shape of the code, not of a convention future callers must remember.
* **Gaps are reported, never filled.** A missing opening stays missing --
  no forward-fill, no synthetic bar. Callers decide what an incomplete
  window means; this module refuses to invent data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.market_snapshots import load_snapshot, replay_snapshot

MARKET_SERIES_SCHEMA_VERSION = "trading-lab.market-series.v1"
MAX_TRAILING_WINDOW = 1_000


class MarketSeriesError(RuntimeError):
    """Raised when a series or a derived value cannot be produced safely."""


@dataclass(frozen=True)
class SeriesPoint:
    """One proven bar, in grid order."""

    bar_open_at: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    content_sha256: str


@dataclass(frozen=True)
class MarketSeries:
    """An ordered series plus the snapshot provenance it came from."""

    schema_version: str
    snapshot_id: str
    snapshot_request_id: str
    entries_content_hash: str
    provider: str
    product_id: str
    timeframe: str
    range_start: str
    range_end: str
    as_of: str
    points: tuple[SeriesPoint, ...]
    missing_openings: tuple[str, ...]


@dataclass(frozen=True)
class FeaturePoint:
    """Derived values for one opening, computed from points 0..i ONLY.

    `complete` is False when the trailing window was not fully available --
    at the start of the series, or across a gap. The values are then None
    rather than borrowed from a shorter window or from later data.
    """

    bar_open_at: str
    close: Decimal
    simple_return: Decimal | None
    rolling_mean: Decimal | None
    rolling_stdev: Decimal | None
    window: int
    complete: bool


def _expected_openings(range_start: str, range_end: str, timeframe: str) -> list[str]:
    duration: timedelta = TIMEFRAME_DURATIONS[timeframe]
    start = datetime.fromisoformat(range_start)
    end = datetime.fromisoformat(range_end)
    openings: list[str] = []
    current = start
    while current < end:
        openings.append(current.isoformat())
        current += duration
    return openings


def load_market_series(connection, *, snapshot_id: str) -> MarketSeries:
    """Build the ordered series a snapshot attests to.

    Goes through `replay_snapshot`, so nothing reaches the caller that has
    not been rebuilt and proven against its stored receipt. Openings the
    snapshot does not cover are reported in `missing_openings`; they are
    never filled in.
    """
    manifest = load_snapshot(connection, snapshot_id=snapshot_id).manifest
    if manifest.timeframe not in TIMEFRAME_DURATIONS:
        raise MarketSeriesError(
            f"snapshot {snapshot_id!r} declares timeframe {manifest.timeframe!r}, "
            "which this build cannot place on a grid"
        )
    bars = replay_snapshot(connection, snapshot_id=snapshot_id)
    points = tuple(
        SeriesPoint(
            bar_open_at=bar["bar_open_at"],
            open=Decimal(bar["open"]),
            high=Decimal(bar["high"]),
            low=Decimal(bar["low"]),
            close=Decimal(bar["close"]),
            volume=Decimal(bar["volume"]),
            content_sha256=bar["content_sha256"],
        )
        for bar in bars
    )
    # Grid order is inherited, not re-established: `replay_snapshot` returns
    # entries ordered by bar_open_at and Phase 1 proves it. Re-sorting here
    # would paper over an upstream regression, and re-asserting it would be
    # dead code -- a reordering upstream is refused by the snapshot layer
    # before a single bar reaches this function.
    present = {point.bar_open_at for point in points}
    missing = tuple(
        opening
        for opening in _expected_openings(
            manifest.range_start, manifest.range_end, manifest.timeframe
        )
        if opening not in present
    )
    return MarketSeries(
        schema_version=MARKET_SERIES_SCHEMA_VERSION,
        snapshot_id=manifest.snapshot_id,
        snapshot_request_id=manifest.snapshot_request_id,
        entries_content_hash=manifest.entries_content_hash,
        provider=manifest.provider,
        product_id=manifest.product_id,
        timeframe=manifest.timeframe,
        range_start=manifest.range_start,
        range_end=manifest.range_end,
        as_of=manifest.as_of,
        points=points,
        missing_openings=missing,
    )


def causal_features(series: MarketSeries, *, window: int) -> tuple[FeaturePoint, ...]:
    """Trailing-window features, causal by construction.

    Index i sees `series.points[i - window + 1 : i + 1]` and nothing else --
    never i + 1, never the tail of the series, never an average of the whole
    range. A window that would span a gap, or that reaches before the start,
    yields None values with `complete=False`.
    """
    if type(window) is not int or not 1 <= window <= MAX_TRAILING_WINDOW:
        raise MarketSeriesError(
            f"window must be an int in 1..{MAX_TRAILING_WINDOW}, got {window!r}"
        )
    duration: timedelta = TIMEFRAME_DURATIONS[series.timeframe]
    features: list[FeaturePoint] = []
    for index, point in enumerate(series.points):
        start = index - window + 1
        trailing = series.points[start : index + 1] if start >= 0 else ()
        contiguous = len(trailing) == window and all(
            datetime.fromisoformat(trailing[step + 1].bar_open_at)
            - datetime.fromisoformat(trailing[step].bar_open_at)
            == duration
            for step in range(len(trailing) - 1)
        )
        simple_return = None
        if index > 0:
            previous = series.points[index - 1]
            gap_free = (
                datetime.fromisoformat(point.bar_open_at)
                - datetime.fromisoformat(previous.bar_open_at)
                == duration
            )
            if gap_free and previous.close != 0:
                simple_return = (point.close - previous.close) / previous.close
        mean = stdev = None
        if contiguous:
            closes = [item.close for item in trailing]
            mean = sum(closes) / Decimal(window)
            if window > 1:
                variance = sum((value - mean) ** 2 for value in closes) / Decimal(window - 1)
                stdev = variance.sqrt()
            else:
                stdev = Decimal(0)
        features.append(
            FeaturePoint(
                bar_open_at=point.bar_open_at,
                close=point.close,
                simple_return=simple_return,
                rolling_mean=mean,
                rolling_stdev=stdev,
                window=window,
                complete=contiguous,
            )
        )
    return tuple(features)

"""Protected research intervals per product, and the decisions that stay clear of them.

Everything here is computed from the registered contracts, never restated:

* crypto: `protected_holdout.PROTECTED_WINDOW_V1` (range from `research_holdout`);
* equity: `equity_research.EQUITY_CONFIRMATORY_HOLDOUT_V1`.

A decision is admissible only when no data it reads touches a protected interval:
the bars behind its price features, the bars of its label window, and the event
window of the event features. Price warm-ups are measured by running the real
indicators on a synthetic series, so a changed feature set changes the answer.
Offline: no price rows, no network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from scripts.trading_lab.coinbase_candles import TIMEFRAME_DURATIONS
from scripts.trading_lab.equity_calendar import USEquityRegularCalendar
from scripts.trading_lab.equity_research import (
    EQUITY_CONFIRMATORY_HOLDOUT_V1, EQUITY_RESEARCH_SPEC_V1)
from scripts.trading_lab.market_dataset import DatasetConfig, build_dataset
from scripts.trading_lab.market_indicators import (
    atr_percent, ema_spread, relative_strength_index, return_over_period)
from scripts.trading_lab.market_series import MarketSeries, SeriesPoint
from scripts.trading_lab.paper_model import FEATURE_SET_V2
from scripts.trading_lab.protected_holdout import PROTECTED_WINDOW_V1
from scripts.trading_lab.signal_engine import SIGNAL_SPEC_V1

EVENT_WINDOW_DAYS = 30  # the longest event window of the event features (7 and 30 days)
CRYPTO_BAR = timedelta(hours=1)
UTC = timezone.utc


def iso_z(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _day(text: str) -> datetime:
    return datetime.fromisoformat(text[:10]).replace(tzinfo=UTC)


def _first_complete_index(columns: list[tuple[Decimal | None, ...]]) -> int:
    for index in range(len(columns[0])):
        if all(column[index] is not None for column in columns):
            return index
    raise ValueError("no row ever has every feature")


def _series(timeframe: str, count: int) -> MarketSeries:
    """A synthetic contiguous series with wiggling closes; only its shape is used."""
    points = []
    for index in range(count):
        close = Decimal(100) + Decimal(index % 7) - Decimal(index % 3)
        points.append(SeriesPoint(
            bar_open_at=(datetime(2000, 1, 1, tzinfo=UTC) + index * TIMEFRAME_DURATIONS[timeframe]).isoformat(),
            open=close, high=close + 2, low=close - 2, close=close,
            volume=Decimal(1), content_sha256=""))
    return MarketSeries(
        schema_version="synthetic", snapshot_id="s", snapshot_request_id="s", entries_content_hash="",
        provider="synthetic", product_id="X", timeframe=timeframe, range_start="", range_end="",
        as_of="", points=tuple(points), missing_openings=())


def crypto_price_warmup() -> int:
    """Bars before the first row whose features all exist (the frozen crypto feature set)."""
    series = _series("1h", 120)
    config = DatasetConfig(features=FEATURE_SET_V2)
    rows = build_dataset(series, config=config).rows
    return next(i for i, row in enumerate(rows) if all(v is not None for _, v in row.features))


def equity_price_warmup() -> int:
    """Same measurement for the six frozen equity features (see equity_dataset.build_dataset)."""
    series = _series("1d", 120)
    return _first_complete_index([
        return_over_period(series, period=1).values, return_over_period(series, period=5).values,
        return_over_period(series, period=20).values,
        ema_spread(series, fast_period=10, slow_period=20).values,
        relative_strength_index(series, period=14).values, atr_percent(series, period=14).values])


@dataclass(frozen=True)
class Interval:
    """[start, end_exclusive) in market time, with the contract it comes from."""

    holdout_id: str
    products: tuple[str, ...]
    start: datetime
    end_exclusive: datetime
    bounds: str
    contract: str
    identity_hash: str
    detail: dict

    def touches(self, first: datetime, last: datetime) -> bool:
        """Does any instant of the closed range [first, last] fall inside the interval?"""
        return first < self.end_exclusive and last >= self.start

    def payload(self) -> dict:
        return {"holdout_id": self.holdout_id, "products": list(self.products), "start": iso_z(self.start),
                "end_exclusive": iso_z(self.end_exclusive), "bounds": self.bounds,
                "contract": self.contract, "identity_hash": self.identity_hash, **self.detail}


def crypto_interval() -> Interval:
    window = PROTECTED_WINDOW_V1
    return Interval(
        window.holdout_id, tuple(window.products), window.start_at, window.closes_at,
        "start inclusive; last protected bar OPENING " + window.end + " inclusive; the interval ends "
        "(exclusive) one bar later, when no protected bar can still be forming",
        "scripts/trading_lab/protected_holdout.py PROTECTED_WINDOW_V1 (range: research_holdout.CONFIRMATORY_HOLDOUT_V2)",
        window.holdout_hash,
        {"timeframe": window.timeframe, "last_protected_bar_open": window.end, "single_use": window.single_use,
         "observed": window.observed, "unit": "bar opening instants"})


def equity_interval() -> Interval:
    holdout = EQUITY_CONFIRMATORY_HOLDOUT_V1
    return Interval(
        holdout.holdout_id, tuple(i.split(":")[1] for i in holdout.instruments), _day(holdout.start),
        _day(holdout.end) + timedelta(days=1),
        "dates inclusive: " + holdout.start[:10] + " through " + holdout.end[:10] +
        " (the contract compares session dates: covers() reads the first ten characters)",
        "scripts/trading_lab/equity_research.py EQUITY_CONFIRMATORY_HOLDOUT_V1",
        holdout.spec_hash,
        {"timeframe": holdout.timeframe, "calendar_id": holdout.calendar_id, "declared_end": holdout.end,
         "research_spec_hash": EQUITY_RESEARCH_SPEC_V1.spec_hash, "single_use": True,
         "observed": holdout.observed, "unit": "session dates"})


def protection_table() -> dict[str, list[dict]]:
    """Product -> its protected intervals. A product absent from every contract has none."""
    table: dict[str, list[dict]] = {}
    for interval in (crypto_interval(), equity_interval()):
        for product in interval.products:
            table.setdefault(product, []).append(interval.payload())
    return dict(sorted(table.items()))


def _runs(opens: list[datetime], interval: Interval, *, by_date: bool) -> list[list[datetime]]:
    """Maximal runs of bars outside the interval; a feature series never spans a protected bar."""
    runs, current = [], []
    for moment in opens:
        stamp = _day(iso_z(moment)) if by_date else moment
        if interval.start <= stamp < interval.end_exclusive:
            if current:
                runs.append(current)
            current = []
        else:
            current.append(moment)
    return runs + ([current] if current else [])


def _decision_range(run, *, warmup: int, horizon: int, close_of) -> dict | None:
    """First/last admissible decision of one run: features need `warmup` earlier bars, the label
    `horizon` later bars, all inside the run."""
    first, last = warmup, len(run) - 1 - horizon
    if last < first:
        return None
    return {"first_bar_open": iso_z(run[first]), "first_decision_at": iso_z(close_of(run[first])),
            "last_bar_open": iso_z(run[last]), "last_decision_at": iso_z(close_of(run[last])),
            "decisions": last - first + 1, "run_bars": len(run)}


def crypto_ranges(corpus_start: datetime, horizon_end_exclusive: datetime) -> dict:
    """Admissible crypto decisions for a series that may run from `corpus_start` to the horizon end."""
    interval, warmup = crypto_interval(), crypto_price_warmup()
    label = SIGNAL_SPEC_V1.prediction_horizon
    count = int((horizon_end_exclusive - corpus_start) / CRYPTO_BAR)
    opens = [corpus_start + i * CRYPTO_BAR for i in range(count)]
    runs = _runs(opens, interval, by_date=False)
    return {"decision_clock": "bar close = bar opening + 1h", "price_warmup_bars": warmup,
            "label_horizon_bars": label,
            "runs": [r for r in (_decision_range(run, warmup=warmup, horizon=label,
                                                 close_of=lambda o: o + CRYPTO_BAR) for run in runs) if r],
            "event_window_clean_from": iso_z(interval.end_exclusive + timedelta(days=EVENT_WINDOW_DAYS))}


def equity_ranges(corpus_start: str, horizon_end: str) -> dict:
    """Same for equity: sessions from the pinned calendar, decision at the session close."""
    interval = equity_interval()
    warmup = equity_price_warmup()
    label = EQUITY_RESEARCH_SPEC_V1.target.horizon_sessions
    calendar = USEquityRegularCalendar()
    sessions = calendar.sessions_between(_day(corpus_start), _day(horizon_end))
    by_open = {s.open_at: s for s in sessions}
    runs = _runs(sorted(by_open), interval, by_date=True)
    return {"decision_clock": "session close (calendar close_at, early closes honoured)",
            "price_warmup_sessions": warmup, "label_horizon_sessions": label,
            "runs": [r for r in (_decision_range(run, warmup=warmup, horizon=label,
                                                 close_of=lambda o: by_open[o].close_at) for run in runs) if r],
            "event_window_clean_from": iso_z(interval.end_exclusive + timedelta(days=EVENT_WINDOW_DAYS))}


def protection_flags(product: str, T: datetime) -> dict:
    """Which protected intervals of this product contain T, or are touched by its 30-day event window."""
    inside, touched = [], []
    for interval in (crypto_interval(), equity_interval()):
        if product not in interval.products:
            continue
        if interval.start <= T < interval.end_exclusive:
            inside.append(interval.holdout_id)
        if T >= interval.start and T - timedelta(days=EVENT_WINDOW_DAYS) < interval.end_exclusive:
            touched.append(interval.holdout_id)
    return {"inside": inside, "event_window_touches": touched}


__all__ = ["EVENT_WINDOW_DAYS", "Interval", "crypto_interval", "crypto_price_warmup", "crypto_ranges",
           "equity_interval", "equity_price_warmup", "equity_ranges", "iso_z",
           "protection_flags", "protection_table"]

"""Turning the frozen local corpus into a causal feature matrix.

The whole file exists to make one guarantee checkable by reading it: every
number on row `t` was knowable at `session_close(t)`, and the only thing that
looks forward is the target, which is supposed to.

Two mechanisms carry that weight.

**Session-ordinal re-indexing.** The repository's indicators define adjacency
as "exactly one timeframe later", which is correct for a market that never
closes. US equities close every weekend, so on real daily bars that rule
fragments 501 sessions into 113 runs of at most five and `return20` never
becomes computable. Here each bar is placed at its ordinal in the frozen
calendar's session list and given a synthetic uniform timestamp derived from
that ordinal. Consecutive sessions become adjacent; a genuinely missing
session still leaves a real hole, because the ordinal gap is real. The
indicator maths is then reused verbatim rather than reimplemented, which is
the point -- a second RSI would eventually disagree with the first.

**Eligibility rather than filling.** A row with an unavailable dependency is
dropped, never zero-filled and never forward-filled. A zero is a number a
model will happily learn from, and it is not the number that was missing.
"""

from __future__ import annotations

import json
import pathlib
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from scripts.trading_lab.equity_research import (
    FEATURE_NAMES, EquityResearchError, sha256_canonical)
from scripts.trading_lab.market_indicators import (
    atr_percent, ema_spread, relative_strength_index, return_over_period)
from scripts.trading_lab.market_series import MarketSeries, SeriesPoint

EQUITY_DATASET_SCHEMA_VERSION = "trading-lab.equity-dataset.v1"

# The synthetic clock the session ordinals are projected onto. Arbitrary and
# fixed: only the SPACING matters, and it never leaves this module.
_SESSION_EPOCH = datetime(2000, 1, 3, tzinfo=timezone.utc)
_SESSION_STEP = timedelta(days=1)


class EquityDatasetError(EquityResearchError):
    """Raised when a dataset cannot be built causally."""


def _session_ordinals(calendar_id: str, start: str, end: str) -> dict[str, int]:
    """Every session the frozen calendar produces, numbered in order.

    The calendar is the authority on which days exist, so a corpus that is
    missing one produces a gap in this numbering rather than a silently
    shorter but contiguous-looking series.
    """
    from scripts.trading_lab.trading_calendar import get_calendar

    sessions = get_calendar(calendar_id).sessions_between(
        f"{start}T00:00:00Z", f"{end}T23:59:59Z")
    return {session.session_date: index for index, session in enumerate(sessions)}


def load_analytical_view(instrument_id: str, *, registry, spec) -> tuple:
    """Verified RAW rows -> SPLIT_ADJUSTED analytical rows.

    Goes through the real `apply_splits`, with the splits the capture actually
    recorded, so the identity of the output is SPLIT_ADJUSTED by construction
    rather than by assertion. Today there are no splits in range and the
    numbers are unchanged; the day there is one, this is already correct.
    """
    from datetime import datetime as _dt

    from scripts.trading_lab.equity_market import (
        ADJUSTMENT_RAW, ADJUSTMENT_SPLIT_ADJUSTED, EquityMarketBar, apply_splits)
    from scripts.trading_lab.instruments import InstrumentId

    rows = registry.read_bars(instrument_id)
    identifier = InstrumentId.coerce(instrument_id)

    def _moment(text: str):
        return _dt.fromisoformat(str(text).replace("Z", "+00:00"))

    bars = tuple(
        EquityMarketBar(
            instrument_id=identifier, timeframe="1d",
            provider_id=spec.source_corpus_id,
            adjustment_policy=ADJUSTMENT_RAW,
            bar_open_at=_moment(row["bar_open_at"]),
            bar_close_at=_moment(row["bar_close_at"]),
            open=Decimal(row["open"]), high=Decimal(row["high"]),
            low=Decimal(row["low"]), close=Decimal(row["close"]),
            volume=Decimal(row["volume"]), session_date=row["session_date"])
        for row in rows)

    splits = _recorded_splits(instrument_id, registry)
    adjusted = apply_splits(bars, splits)
    if adjusted and adjusted[0].adjustment_policy != ADJUSTMENT_SPLIT_ADJUSTED:
        raise EquityDatasetError(
            "the analytical view did not come back SPLIT_ADJUSTED")
    return adjusted


def _recorded_splits(instrument_id: str, registry) -> tuple:
    """The split events the capture stored beside the bars. Usually none.

    Read from the corpus's own corporate-actions artefact rather than
    re-derived, so the adjustment reflects exactly what the capture recorded
    and hashed. Zero events in this range today; the path is the same when
    there is one.
    """
    from scripts.trading_lab.equity_market import StockSplit
    from scripts.trading_lab.instruments import InstrumentId

    path = (registry.corpus_root / "corporate_actions"
            / f"{instrument_id.replace(':', '_')}.json")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    identifier = InstrumentId.coerce(instrument_id)
    return tuple(
        StockSplit(instrument_id=identifier,
                   effective_date=item["effective_date"],
                   ratio_numerator=int(item["ratio_numerator"]),
                   ratio_denominator=int(item["ratio_denominator"]),
                   source="local-corpus")
        for item in payload.get("splits", []))


def analytical_view_hash(instrument_id: str, bars) -> str:
    """Content identity of the analytical view. Adjustment is inside it."""
    return sha256_canonical({
        "instrument_id": instrument_id,
        "adjustment_policy": bars[0].adjustment_policy if bars else None,
        "rows": [
            {"session_date": bar.session_date,
             "open": str(bar.open), "high": str(bar.high),
             "low": str(bar.low), "close": str(bar.close),
             "volume": str(bar.volume)}
            for bar in bars],
    })


def _session_indexed_series(instrument_id: str, bars, ordinals) -> MarketSeries:
    """Project bars onto the session ordinal, then hand them to the indicators.

    The synthetic timestamp is a function of the ordinal alone, so two
    consecutive sessions are exactly one step apart no matter how many
    weekends or holidays sit between them in real time -- and two sessions
    with a missing one between them are two steps apart, which is what makes
    a real gap still break an indicator's state.
    """
    points = []
    for bar in bars:
        ordinal = ordinals.get(bar.session_date)
        if ordinal is None:
            raise EquityDatasetError(
                f"{bar.session_date} is not a session in the frozen calendar")
        moment = _SESSION_EPOCH + ordinal * _SESSION_STEP
        points.append(SeriesPoint(
            bar_open_at=moment.isoformat(), open=bar.open, high=bar.high,
            low=bar.low, close=bar.close, volume=bar.volume,
            content_sha256=""))
    return MarketSeries(
        schema_version=EQUITY_DATASET_SCHEMA_VERSION, snapshot_id="local",
        snapshot_request_id="local", entries_content_hash="", provider="local",
        product_id=instrument_id, timeframe="1d", range_start="", range_end="",
        as_of="", points=tuple(points), missing_openings=())


def build_dataset(instrument_id: str, *, registry, spec) -> dict:
    """Eligible rows only: six causal features and a 5-session forward target.

    A row survives when every feature exists AND the session five ahead exists
    in the exploratory corpus. The final five sessions therefore have no
    target and are dropped rather than given a shortened horizon, which would
    be a different experiment measured under the same name.
    """
    bars = load_analytical_view(instrument_id, registry=registry, spec=spec)
    if not bars:
        raise EquityDatasetError(f"no analytical rows for {instrument_id}")
    ordinals = _session_ordinals(spec.calendar_id, spec.exploratory_start,
                                 spec.exploratory_end)
    series = _session_indexed_series(instrument_id, bars, ordinals)

    columns = {
        "return1": return_over_period(series, period=1).values,
        "return5": return_over_period(series, period=5).values,
        "return20": return_over_period(series, period=20).values,
        "ema_spread10_20": ema_spread(series, fast_period=10,
                                      slow_period=20).values,
        "rsi14": relative_strength_index(series, period=14).values,
        "atr_pct14": atr_percent(series, period=14).values,
    }
    if tuple(columns) != FEATURE_NAMES:
        raise EquityDatasetError("feature columns drifted from the frozen spec")

    horizon = spec.target.horizon_sessions
    session_dates = [bar.session_date for bar in bars]
    closes = [bar.close for bar in bars]
    ordinal_of = [ordinals[date] for date in session_dates]

    rows = []
    for index in range(len(bars)):
        values = [columns[name][index] for name in FEATURE_NAMES]
        if any(value is None or not value.is_finite() for value in values):
            continue
        ahead = index + horizon
        if ahead >= len(bars):
            continue
        # The horizon must be five SESSIONS, which on a contiguous corpus is
        # five ordinals. Checked rather than assumed: a corpus with a hole
        # would otherwise silently measure a longer horizon.
        if ordinal_of[ahead] - ordinal_of[index] != horizon:
            continue
        if closes[index] == 0:
            continue
        target = closes[ahead] / closes[index] - Decimal(1)
        if not target.is_finite():
            continue
        rows.append({
            "session_date": session_dates[index],
            "session_ordinal": ordinal_of[index],
            "target_session_date": session_dates[ahead],
            "features": [str(value) for value in values],
            "target": str(target),
        })

    return {
        "schema_version": EQUITY_DATASET_SCHEMA_VERSION,
        "instrument_id": instrument_id,
        "adjustment_policy": bars[0].adjustment_policy,
        "analytical_view_hash": analytical_view_hash(instrument_id, bars),
        "source_sessions": len(bars),
        "eligible_rows": len(rows),
        "feature_names": list(FEATURE_NAMES),
        "rows": rows,
        "feature_matrix_hash": sha256_canonical(
            [row["features"] for row in rows]),
        "target_vector_hash": sha256_canonical([row["target"] for row in rows]),
    }


__all__ = ["EQUITY_DATASET_SCHEMA_VERSION", "EquityDatasetError",
           "analytical_view_hash", "build_dataset", "load_analytical_view"]

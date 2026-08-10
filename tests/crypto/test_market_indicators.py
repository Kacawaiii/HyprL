"""Phase 2B: the causal indicator library, and the proofs it cannot look ahead."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal, getcontext, localcontext
import importlib
import json

import pytest


GRID = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 3, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2026, 8, 3, 13, 0, tzinfo=timezone.utc)
T3 = datetime(2026, 8, 3, 14, 0, tzinfo=timezone.utc)
T4 = datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc)


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.market_data_store")


@pytest.fixture
def snapshots_module():
    return importlib.import_module("scripts.trading_lab.market_snapshots")


@pytest.fixture
def series_module():
    return importlib.import_module("scripts.trading_lab.market_series")


@pytest.fixture
def indicators():
    return importlib.import_module("scripts.trading_lab.market_indicators")


def _iso(moment): return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _publish(store, opens, ohlc, *, at):
    rows = [
        [int(o.timestamp()), low, high, open_, close, "1.0"]
        for o, (low, high, open_, close) in zip(opens, ohlc)
    ]
    return store.ingest_coinbase_response(
        json.dumps(rows, separators=(",", ":")).encode("utf-8"),
        product_id="BTC-USD", timeframe="1h",
        available_at=_iso(at), ingested_at=_iso(at + timedelta(seconds=1)),
    )


def _flat(closes):
    """low, high, open, close -- a wide band so any close is valid."""
    return [("1.0", "100000.0", c, c) for c in closes]


def _series_for(store, snapshots_module, series_module, as_of, opens):
    connection = store._connect()
    try:
        result = snapshots_module._materialize_snapshot(
            connection,
            provider="coinbase_exchange_rest", product_id="BTC-USD", timeframe="1h",
            range_start=_iso(opens[0]), range_end=_iso(opens[-1] + timedelta(hours=1)),
            as_of=_iso(as_of),
        )
        return series_module.load_market_series(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def _build(store_module, snapshots_module, series_module, tmp_path, *, name,
           closes, skip=(), ohlc=None, at=T1, as_of=T2):
    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + timedelta(hours=i) for i in range(len(closes))]
    kept = [(o, v) for index, (o, v) in enumerate(zip(opens, ohlc or _flat(closes)))
            if index not in skip]
    _publish(store, [o for o, _ in kept], [v for _, v in kept], at=at)
    return store, opens, _series_for(store, snapshots_module, series_module, as_of, opens)


# --- SMA -----------------------------------------------------------------


def test_sma_warms_up_then_matches_a_hand_computed_mean(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="sma", closes=["10", "20", "30", "40", "50"])
    values = indicators.simple_moving_average(series, period=3).values
    assert values[0] is None and values[1] is None
    assert values[2] == Decimal(20)   # (10+20+30)/3
    assert values[3] == Decimal(30)
    assert values[4] == Decimal(40)
    assert len(values) == len(series.points)


# --- EMA -----------------------------------------------------------------


def test_ema_is_seeded_by_the_exact_sma_then_recurses_forward(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ema", closes=["10", "20", "30", "40"])
    values = indicators.exponential_moving_average(series, period=3).values
    assert values[0] is None and values[1] is None
    assert values[2] == Decimal(20)  # seed = SMA(10,20,30)
    alpha = Decimal(2) / Decimal(4)
    assert values[3] == alpha * Decimal(40) + (Decimal(1) - alpha) * Decimal(20)


# --- True Range / ATR ----------------------------------------------------


def test_true_range_uses_the_previous_close_only_inside_a_segment(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    ohlc = [("10", "20", "15", "15"), ("30", "40", "35", "35"), ("5", "12", "8", "8")]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="tr", closes=["15", "35", "8"], ohlc=ohlc)
    values = indicators.true_range(series).values
    assert values[0] == Decimal(10)                       # first bar: high - low
    assert values[1] == max(Decimal(10), abs(Decimal(40) - Decimal(15)),
                            abs(Decimal(30) - Decimal(15)))
    assert values[2] == max(Decimal(7), abs(Decimal(12) - Decimal(35)),
                            abs(Decimal(5) - Decimal(35)))


def test_atr_uses_wilder_smoothing_after_its_warm_up(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    ohlc = [("10", "20", "15", "15")] * 4
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="atr", closes=["15"] * 4, ohlc=ohlc)
    ranges = indicators.true_range(series).values
    values = indicators.average_true_range(series, period=3).values
    assert values[0] is None and values[1] is None
    assert values[2] == sum(ranges[:3]) / Decimal(3)
    assert values[3] == (values[2] * Decimal(2) + ranges[3]) / Decimal(3)


# --- RSI -----------------------------------------------------------------


def test_rsi_reaches_its_boundary_conventions(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    rising = _build(store_module, snapshots_module, series_module, tmp_path,
                    name="rsi_up", closes=["10", "20", "30", "40"])[2]
    falling = _build(store_module, snapshots_module, series_module, tmp_path,
                     name="rsi_down", closes=["40", "30", "20", "10"])[2]
    flat = _build(store_module, snapshots_module, series_module, tmp_path,
                  name="rsi_flat", closes=["25", "25", "25", "25"])[2]
    assert indicators.relative_strength_index(rising, period=3).values[3] == Decimal(100)
    assert indicators.relative_strength_index(falling, period=3).values[3] == Decimal(0)
    assert indicators.relative_strength_index(flat, period=3).values[3] == Decimal(50)


def test_rsi_warms_up_on_deltas_not_on_closes(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """period deltas require period + 1 closes: the first reading sits at
    index `period`, never earlier."""
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="rsi_warm", closes=["10", "11", "12", "13", "14"])
    values = indicators.relative_strength_index(series, period=3).values
    assert values[:3] == (None, None, None)
    assert values[3] == Decimal(100)


# --- gaps: the recursive state must not survive one --------------------


def test_every_recursive_indicator_restarts_after_a_gap(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = ["10", "20", "30", "40", "50", "60", "70"]
    _, opens, series = _build(store_module, snapshots_module, series_module, tmp_path,
                              name="gap", closes=closes, skip={3})
    assert series.missing_openings == (opens[3].isoformat(),)
    assert indicators.contiguous_segments(series) == ((0, 3), (3, 6))

    sma = indicators.simple_moving_average(series, period=3).values
    ema = indicators.exponential_moving_average(series, period=3).values
    atr = indicators.average_true_range(series, period=3).values
    rsi = indicators.relative_strength_index(series, period=2).values
    tr = indicators.true_range(series).values

    # index 3 is the first bar AFTER the gap: every window restarts there.
    assert sma[2] is not None and sma[3] is None and sma[4] is None and sma[5] is not None
    assert ema[3] is None and ema[4] is None and ema[5] is not None
    assert atr[3] is None and atr[5] is not None
    assert rsi[3] is None                      # no delta across the gap
    # TR at the segment boundary falls back to high - low, never a close
    # borrowed from the other side of the hole.
    assert tr[3] == series.points[3].high - series.points[3].low


# --- anti-future-leakage -------------------------------------------------


def test_indicators_do_not_change_when_a_later_revision_lands(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    store, opens, series = _build(store_module, snapshots_module, series_module, tmp_path,
                                  name="leak", closes=["10", "20", "30", "40", "50"])
    def compute(source):
        return {
            "sma": indicators.simple_moving_average(source, period=3).values,
            "ema": indicators.exponential_moving_average(source, period=3).values,
            "atr": indicators.average_true_range(source, period=3).values,
            "rsi": indicators.relative_strength_index(source, period=3).values,
            "tr": indicators.true_range(source).values,
        }
    before = compute(series)

    _publish(store, opens, _flat(["999"] * 5), at=T3)  # the future arrives

    recomputed = _series_for(store, snapshots_module, series_module, T2, opens)
    assert compute(recomputed) == before

    later = _series_for(store, snapshots_module, series_module, T4, opens)
    assert compute(later) != before


def test_truncating_the_series_leaves_earlier_indicator_values_untouched(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """The structural proof: if any indicator reached forward, cutting the
    series short would change values it had already produced."""
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="trunc", closes=["10", "13", "12", "18", "17", "25", "24"])
    calls = (
        ("sma", lambda s: indicators.simple_moving_average(s, period=3).values),
        ("ema", lambda s: indicators.exponential_moving_average(s, period=3).values),
        ("tr", lambda s: indicators.true_range(s).values),
        ("atr", lambda s: indicators.average_true_range(s, period=3).values),
        ("rsi", lambda s: indicators.relative_strength_index(s, period=3).values),
    )
    for name, call in calls:
        full = call(series)
        for cut in range(1, len(series.points) + 1):
            truncated = call(replace(series, points=series.points[:cut]))
            assert truncated == full[:cut], (name, cut)


# --- identity, determinism, validation -----------------------------------


def test_the_spec_hash_describes_the_definition_not_the_values(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="spec_a", closes=["10", "20", "30", "40"])
    other = _build(store_module, snapshots_module, series_module, tmp_path,
                   name="spec_b", closes=["99", "98", "97", "96"])[2]
    first = indicators.simple_moving_average(series, period=3)
    same_spec_other_data = indicators.simple_moving_average(other, period=3)
    other_period = indicators.simple_moving_average(series, period=2)
    other_name = indicators.exponential_moving_average(series, period=3)

    assert first.spec_hash == same_spec_other_data.spec_hash != ""
    assert first.values != same_spec_other_data.values
    assert first.spec_hash != other_period.spec_hash
    assert first.spec_hash != other_name.spec_hash
    assert len(first.spec_hash) == 64
    # A different version yields a different hash, definition-only.
    bumped = replace(first.spec, version="trading-lab.market-indicator.v2")
    assert bumped.spec_hash != first.spec_hash
    # Parameter order cannot change the identity.
    shuffled = replace(first.spec, parameters=tuple(reversed(first.spec.parameters)))
    assert shuffled.spec_hash == first.spec_hash


def test_results_do_not_depend_on_the_callers_decimal_context(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ctx", closes=["10", "13", "12", "18", "17", "25"])
    def compute():
        return (
            indicators.exponential_moving_average(series, period=3).values,
            indicators.average_true_range(series, period=3).values,
            indicators.relative_strength_index(series, period=3).values,
        )
    reference = compute()
    original = getcontext().prec
    try:
        getcontext().prec = 6      # a caller sabotaging the global context
        assert compute() == reference
        getcontext().prec = 60
        assert compute() == reference
    finally:
        getcontext().prec = original


@pytest.mark.parametrize("period", [0, -1, 1.0, True, "3", None, 1001])
def test_an_invalid_period_is_refused(
    tmp_path, store_module, snapshots_module, series_module, indicators, period
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="badp", closes=["10", "20", "30"])
    for call in (indicators.simple_moving_average, indicators.exponential_moving_average,
                 indicators.average_true_range, indicators.relative_strength_index):
        with pytest.raises(indicators.MarketIndicatorError):
            call(series, period=period)


def test_indicators_never_mutate_the_series_and_stay_aligned(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="pure", closes=["10", "20", "30", "40", "50"])
    snapshot_of_input = replace(series)
    for values in (
        indicators.simple_moving_average(series, period=2).values,
        indicators.exponential_moving_average(series, period=2).values,
        indicators.true_range(series).values,
        indicators.average_true_range(series, period=2).values,
        indicators.relative_strength_index(series, period=2).values,
    ):
        assert len(values) == len(series.points)
        assert all(value is None or isinstance(value, Decimal) for value in values)
    assert series == snapshot_of_input


def test_true_range_does_not_reach_across_a_gap(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """Narrow bands and a jump across the hole, so the two policies give
    different numbers: 2 if the segment restarts, 41 if `previous_close`
    is allowed to cross the gap."""
    ohlc = [("10", "12", "11", "11")] * 4 + [("50", "52", "51", "51")]
    _, opens, series = _build(store_module, snapshots_module, series_module, tmp_path,
                              name="tr_gap", closes=["11", "11", "11", "11", "51"],
                              ohlc=ohlc, skip={3})
    assert series.missing_openings == (opens[3].isoformat(),)
    assert indicators.contiguous_segments(series) == ((0, 3), (3, 4))
    values = indicators.true_range(series).values
    assert values[3] == Decimal(2)           # high - low, segment restart
    assert values[3] != Decimal(41)          # what crossing the gap would give


# --- one-bar causal return (Phase 4B) -------------------------------------


def test_the_one_bar_return_matches_hand_computed_values(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret", closes=["100", "110", "99"])
    result = indicators.simple_return(series)
    assert result.values[0] is None                    # nothing precedes the first bar
    with localcontext() as context:
        context.prec = indicators.INDICATOR_PRECISION
        assert result.values[1] == (Decimal("110") - Decimal("100")) / Decimal("100")
        assert result.values[2] == (Decimal("99") - Decimal("110")) / Decimal("110")
    assert result.values[1] == Decimal("0.1")
    assert result.values[2] == Decimal("-0.1")


def test_the_one_bar_return_never_reaches_across_a_gap(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """The first bar after a hole has no predecessor it may legitimately use."""
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-gap", closes=["100", "110", "0", "99"], skip={2})
    assert len(series.points) == 3
    result = indicators.simple_return(series)
    assert result.values[0] is None
    assert result.values[1] == Decimal("0.1")
    assert result.values[2] is None                    # 99 follows the gap
    # a naive implementation would have produced 99/110 - 1 here
    assert Decimal("-0.1") not in [v for v in result.values if v is not None]


def test_the_indicator_agrees_exactly_with_the_established_feature_point(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """One definition of "return", not two that drift apart.

    `causal_features` runs at the CALLER's Decimal precision while indicators
    pin themselves to INDICATOR_PRECISION, so the reference has to be taken at
    the module's precision or the comparison would fail for arithmetic reasons
    that say nothing about the semantics.
    """
    closes = [str(100 + (index * 7) % 23) for index in range(20)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-equiv", closes=closes, skip={6, 13})
    assert len(series.missing_openings) == 2
    result = indicators.simple_return(series)
    with localcontext() as context:
        context.prec = indicators.INDICATOR_PRECISION
        reference = series_module.causal_features(series, window=3)
    assert len(reference) == len(result.values)
    assert [point.simple_return for point in reference] == list(result.values)
    # and the agreement is not vacuous: real values on both sides, Nones at the seams
    assert sum(1 for value in result.values if value is not None) >= 14
    assert sum(1 for value in result.values if value is None) == 3


def test_the_one_bar_return_is_invariant_under_truncation(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """Structural anti-lookahead: later bars cannot change earlier returns."""
    closes = [str(100 + (index * 11) % 29) for index in range(18)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-trunc", closes=closes)
    full = indicators.simple_return(series).values
    for cut in (3, 7, 12, 17):
        truncated = replace(series, points=series.points[: cut + 1])
        assert indicators.simple_return(truncated).values == full[: cut + 1]


def test_the_one_bar_return_ignores_the_callers_decimal_context(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = ["100", "103", "107", "102"]     # produces repeating decimals
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-prec", closes=closes)
    original = getcontext().prec
    seen = set()
    try:
        for precision in (6, 28, 34, 60):
            getcontext().prec = precision
            seen.add(tuple(str(value) for value in
                           indicators.simple_return(series).values))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_the_one_bar_return_does_not_mutate_its_source(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-immutable", closes=["100", "110", "99", "105"])
    snapshot = tuple(series.points)
    indicators.simple_return(series)
    assert series.points == snapshot


def test_the_return_spec_hash_describes_the_definition(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-spec", closes=["100", "110", "99"])
    first = indicators.simple_return(series)
    assert first.spec.name == "simple_return"
    assert first.spec.parameters == ()
    assert first.spec_hash == indicators.simple_return(series).spec_hash
    assert len(first.spec_hash) == 64
    # a different definition is a different hash; market values never enter it
    other = replace(first.spec, version="trading-lab.market-indicator.v2")
    assert other.spec_hash != first.spec_hash
    assert first.spec_hash != indicators.true_range(series).spec_hash
    moved = _build(store_module, snapshots_module, series_module, tmp_path,
                   name="ret-spec2", closes=["500", "550", "495"])[2]
    assert indicators.simple_return(moved).spec_hash == first.spec_hash


# --- relative / dimensionless primitives (Phase 4D) ---

# These fixtures run far longer series than the shared T1/T2 anchors cover,
# so they declare a later availability instant. Contract A treats it as a
# declared historical value, exactly as elsewhere.
V2_AT = datetime(2026, 8, 14, 12, 0, tzinfo=timezone.utc)
V2_ASOF = datetime(2026, 8, 14, 13, 0, tzinfo=timezone.utc)


def test_the_multi_bar_return_matches_hand_computed_values(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-n", closes=["100", "110", "99", "121", "88"], at=V2_AT, as_of=V2_ASOF)
    result = indicators.return_over_period(series, period=2)
    assert result.values[0] is None and result.values[1] is None    # warm-up
    with localcontext() as context:
        context.prec = indicators.INDICATOR_PRECISION
        assert result.values[2] == (Decimal("99") - Decimal("100")) / Decimal("100")
        assert result.values[3] == (Decimal("121") - Decimal("110")) / Decimal("110")
        assert result.values[4] == (Decimal("88") - Decimal("99")) / Decimal("99")


def test_the_multi_bar_return_agrees_with_the_frozen_one_bar_primitive(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    """One definition of a 1-bar return, reachable through two names.

    `simple_return` stays parameterless because Benchmark V1 committed to its
    spec hash. The new primitive must nevertheless compute the same thing at
    period=1, or the repository would hold two disagreeing definitions.
    """
    closes = [str(100 + (index * 13) % 31) for index in range(20)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="ret-equiv-n", closes=closes, skip={5, 12}, at=V2_AT, as_of=V2_ASOF)
    assert len(series.missing_openings) == 2
    assert indicators.return_over_period(series, period=1).values == \
        indicators.simple_return(series).values
    # ... while remaining a DIFFERENT identity, so V1 keeps its own hash
    assert indicators.return_over_period(series, period=1).spec_hash != \
        indicators.simple_return(series).spec_hash


@pytest.mark.parametrize("period", [1, 4, 12])
def test_the_multi_bar_return_never_reaches_across_a_gap(
    tmp_path, store_module, snapshots_module, series_module, indicators, period
) -> None:
    closes = [str(100 + index) for index in range(24)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name=f"ret-gap-{period}", closes=closes, skip={10}, at=V2_AT, as_of=V2_ASOF)
    segments = indicators.contiguous_segments(series)
    assert len(segments) == 2
    values = indicators.return_over_period(series, period=period).values
    for start, end in segments:
        # the first `period` observations of every segment have no usable history
        assert all(values[index] is None for index in range(start, min(start + period, end)))
        assert all(values[index] is not None for index in range(start + period, end))


@pytest.mark.parametrize("period", [1, 4, 12])
def test_the_multi_bar_return_is_invariant_under_truncation(
    tmp_path, store_module, snapshots_module, series_module, indicators, period
) -> None:
    closes = [str(100 + (index * 7) % 19) for index in range(30)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name=f"ret-trunc-{period}", closes=closes, at=V2_AT, as_of=V2_ASOF)
    full = indicators.return_over_period(series, period=period).values
    for cut in (14, 21, 29):
        truncated = replace(series, points=series.points[: cut + 1])
        assert indicators.return_over_period(truncated, period=period).values == full[: cut + 1]


def test_the_ema_spread_is_built_from_the_official_emas(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = [str(100 + (index * 11) % 37) for index in range(60)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="spread", closes=closes, at=V2_AT, as_of=V2_ASOF)
    fast = indicators.exponential_moving_average(series, period=12).values
    slow = indicators.exponential_moving_average(series, period=26).values
    spread = indicators.ema_spread(series, fast_period=12, slow_period=26).values
    with localcontext() as context:
        context.prec = indicators.INDICATOR_PRECISION
        expected = [None if (a is None or b is None or b == 0) else (a - b) / b
                    for a, b in zip(fast, slow)]
    assert list(spread) == expected
    # the warm-up is the SLOWER ema's, not the faster one's
    assert spread[24] is None and slow[24] is None
    assert spread[25] is not None and slow[25] is not None


def test_the_ema_spread_restarts_after_a_gap_because_the_emas_do(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = [str(100 + (index * 5) % 23) for index in range(80)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="spread-gap", closes=closes, skip={40}, at=V2_AT, as_of=V2_ASOF)
    segments = indicators.contiguous_segments(series)
    assert len(segments) == 2
    spread = indicators.ema_spread(series, fast_period=12, slow_period=26).values
    second_start = segments[1][0]
    # no value is borrowed across the hole: the second segment warms up again
    assert all(spread[index] is None
               for index in range(second_start, second_start + 25))
    assert spread[second_start + 25] is not None


def test_the_ema_spread_refuses_an_inverted_or_malformed_configuration(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="spread-bad", closes=[str(100 + i) for i in range(40)], at=V2_AT, as_of=V2_ASOF)
    for kwargs in ({"fast_period": 26, "slow_period": 12},
                   {"fast_period": 12, "slow_period": 12}):
        with pytest.raises(indicators.MarketIndicatorError, match="shorter than"):
            indicators.ema_spread(series, **kwargs)
    for kwargs in ({"fast_period": 0, "slow_period": 26},
                   {"fast_period": True, "slow_period": 26},
                   {"fast_period": 12, "slow_period": 2.0}):
        with pytest.raises(indicators.MarketIndicatorError, match="period"):
            indicators.ema_spread(series, **kwargs)


def test_the_percentage_atr_is_the_official_atr_over_the_same_close(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = [str(100 + (index * 9) % 29) for index in range(50)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="atrpct", closes=closes, at=V2_AT, as_of=V2_ASOF)
    atr = indicators.average_true_range(series, period=14).values
    percent = indicators.atr_percent(series, period=14).values
    with localcontext() as context:
        context.prec = indicators.INDICATOR_PRECISION
        expected = [None if value is None else value / point.close
                    for value, point in zip(atr, series.points)]
    assert list(percent) == expected
    # identical warm-up and gap behaviour, by construction
    assert [value is None for value in percent] == [value is None for value in atr]


def test_the_percentage_atr_keeps_the_atr_gap_semantics(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = [str(100 + (index * 3) % 17) for index in range(70)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="atrpct-gap", closes=closes, skip={35}, at=V2_AT, as_of=V2_ASOF)
    atr = indicators.average_true_range(series, period=14).values
    percent = indicators.atr_percent(series, period=14).values
    assert [value is None for value in percent] == [value is None for value in atr]
    assert any(value is not None for value in percent)


@pytest.mark.parametrize("builder", [
    lambda m, s: m.return_over_period(s, period=4),
    lambda m, s: m.return_over_period(s, period=12),
    lambda m, s: m.ema_spread(s, fast_period=12, slow_period=26),
    lambda m, s: m.atr_percent(s, period=14),
])
def test_the_new_primitives_ignore_the_callers_decimal_context(
    tmp_path, store_module, snapshots_module, series_module, indicators, builder
) -> None:
    closes = [str(100 + (index * 7) % 23) for index in range(60)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="v2-prec", closes=closes, at=V2_AT, as_of=V2_ASOF)
    original = getcontext().prec
    seen = set()
    try:
        for precision in (6, 28, 34, 60):
            getcontext().prec = precision
            seen.add(tuple(str(value) for value in builder(indicators, series).values))
    finally:
        getcontext().prec = original
    assert len(seen) == 1


def test_the_new_primitives_do_not_mutate_their_source(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = [str(100 + (index * 7) % 23) for index in range(60)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="v2-immutable", closes=closes, at=V2_AT, as_of=V2_ASOF)
    snapshot = tuple(series.points)
    indicators.return_over_period(series, period=4)
    indicators.ema_spread(series, fast_period=12, slow_period=26)
    indicators.atr_percent(series, period=14)
    assert series.points == snapshot


def test_the_new_primitive_spec_hashes_describe_their_parameters(
    tmp_path, store_module, snapshots_module, series_module, indicators
) -> None:
    closes = [str(100 + (index * 7) % 23) for index in range(60)]
    _, _, series = _build(store_module, snapshots_module, series_module, tmp_path,
                          name="v2-spec", closes=closes, at=V2_AT, as_of=V2_ASOF)
    four = indicators.return_over_period(series, period=4)
    twelve = indicators.return_over_period(series, period=12)
    assert four.spec.parameters == (("period", 4),)
    assert four.spec_hash != twelve.spec_hash
    assert four.spec_hash == indicators.return_over_period(series, period=4).spec_hash

    spread = indicators.ema_spread(series, fast_period=12, slow_period=26)
    assert spread.spec.parameters == (("fast_period", 12), ("slow_period", 26))
    assert spread.spec_hash != indicators.ema_spread(
        series, fast_period=8, slow_period=26).spec_hash
    assert spread.spec_hash != indicators.ema_spread(
        series, fast_period=12, slow_period=30).spec_hash

    percent = indicators.atr_percent(series, period=14)
    assert percent.spec.parameters == (("period", 14),)
    assert percent.spec_hash != indicators.average_true_range(series, period=14).spec_hash
    # market values never enter an indicator identity
    moved = _build(store_module, snapshots_module, series_module, tmp_path,
                   name="v2-spec2", closes=[str(9000 + (i * 7) % 23) for i in range(60)], at=V2_AT, as_of=V2_ASOF)[2]
    assert indicators.atr_percent(moved, period=14).spec_hash == percent.spec_hash

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

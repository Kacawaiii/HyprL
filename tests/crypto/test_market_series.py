"""Phase 2A: the causal series layer, and the proof it cannot leak the future."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import importlib
import json
from pathlib import Path

import pytest


GRID = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)
T1 = datetime(2026, 8, 2, 20, 0, tzinfo=timezone.utc)   # version A published
T2 = datetime(2026, 8, 2, 21, 0, tzinfo=timezone.utc)   # a read between them
T3 = datetime(2026, 8, 2, 22, 0, tzinfo=timezone.utc)   # version B published
T4 = datetime(2026, 8, 2, 23, 0, tzinfo=timezone.utc)   # a read after both


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.market_data_store")


@pytest.fixture
def snapshots_module():
    return importlib.import_module("scripts.trading_lab.market_snapshots")


@pytest.fixture
def series_module():
    return importlib.import_module("scripts.trading_lab.market_series")


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _publish(store, opens, closes, *, at: datetime):
    rows = [
        [int(open_at.timestamp()), "1.0", "100000.0", "105.0", close, "1.0"]
        for open_at, close in zip(opens, closes)
    ]
    return store.ingest_coinbase_response(
        json.dumps(rows, separators=(",", ":")).encode("utf-8"),
        product_id="BTC-USD", timeframe="1h",
        available_at=_iso(at), ingested_at=_iso(at + timedelta(seconds=1)),
    )


def _request(as_of: datetime, *, opens):
    return {
        "provider": "coinbase_exchange_rest", "product_id": "BTC-USD", "timeframe": "1h",
        "range_start": _iso(opens[0]), "range_end": _iso(opens[-1] + timedelta(hours=1)),
        "as_of": _iso(as_of),
    }


def _series(store, snapshots_module, series_module, as_of, opens):
    connection = store._connect()
    try:
        result = snapshots_module._materialize_snapshot(connection, **_request(as_of, opens=opens))
        return series_module.load_market_series(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def _five_bar_store(store_module, tmp_path, *, name):
    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    opens = [GRID + timedelta(hours=index) for index in range(5)]
    _publish(store, opens, ["100.0", "110.0", "120.0", "130.0", "140.0"], at=T1)
    return store, opens


def test_a_snapshot_becomes_an_ordered_provenance_carrying_series(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    store, opens = _five_bar_store(store_module, tmp_path, name="basic")
    series = _series(store, snapshots_module, series_module, T2, opens)

    assert series.schema_version == "trading-lab.market-series.v1"
    assert [point.bar_open_at for point in series.points] == [o.isoformat() for o in opens]
    assert [point.close for point in series.points] == [
        Decimal(value) for value in ("100.0", "110.0", "120.0", "130.0", "140.0")
    ]
    assert series.missing_openings == ()
    # Provenance travels with the data, so a downstream artefact can always
    # name the exact causal state it was computed from.
    assert series.as_of == _iso(T2).replace("Z", "+00:00")
    assert len(series.entries_content_hash) == 64
    assert series.snapshot_id.startswith("hyprl-market-snapshot-")
    assert all(len(point.content_sha256) == 64 for point in series.points)


def test_a_missing_opening_is_reported_and_never_filled(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    store = store_module.MarketDataStore(tmp_path / "gap.sqlite3")
    opens = [GRID + timedelta(hours=index) for index in range(5)]
    kept = [opens[0], opens[1], opens[3], opens[4]]  # 11:00 never published
    _publish(store, kept, ["100.0", "110.0", "130.0", "140.0"], at=T1)
    series = _series(store, snapshots_module, series_module, T2, opens)

    assert len(series.points) == 4
    assert series.missing_openings == (opens[2].isoformat(),)
    assert opens[2].isoformat() not in [point.bar_open_at for point in series.points]


# --- the Phase 2 invariant: no future leakage ----------------------------


def test_features_at_a_past_as_of_do_not_change_when_a_later_revision_lands(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    """The negative proof. Compute at as_of=T2, publish a correction known
    only at T3, recompute at as_of=T2: byte-identical. A read after T3 may
    differ -- and does."""
    store, opens = _five_bar_store(store_module, tmp_path, name="leak")
    before = _series(store, snapshots_module, series_module, T2, opens)
    features_before = series_module.causal_features(before, window=3)

    _publish(store, opens, ["999.0"] * 5, at=T3)  # the future arrives

    recomputed = _series(store, snapshots_module, series_module, T2, opens)
    assert recomputed == before
    assert series_module.causal_features(recomputed, window=3) == features_before

    later = _series(store, snapshots_module, series_module, T4, opens)
    assert [point.close for point in later.points] == [Decimal("999.0")] * 5
    assert series_module.causal_features(later, window=3) != features_before
    assert later.snapshot_id != before.snapshot_id


def test_a_trailing_feature_never_reads_a_later_point(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    """Truncating the series after index i must leave index i untouched. If
    any feature reached forward, the truncated run would disagree."""
    store, opens = _five_bar_store(store_module, tmp_path, name="trailing")
    series = _series(store, snapshots_module, series_module, T2, opens)
    full = series_module.causal_features(series, window=3)

    from dataclasses import replace
    for cut in range(1, len(series.points) + 1):
        truncated = replace(series, points=series.points[:cut])
        assert series_module.causal_features(truncated, window=3) == full[:cut], cut


def test_an_incomplete_window_yields_none_rather_than_a_shorter_one(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    store, opens = _five_bar_store(store_module, tmp_path, name="incomplete")
    features = series_module.causal_features(
        _series(store, snapshots_module, series_module, T2, opens), window=3
    )
    assert [f.complete for f in features] == [False, False, True, True, True]
    assert features[0].rolling_mean is None and features[1].rolling_mean is None
    assert features[0].simple_return is None
    # 100,110,120 -> mean 110 exactly; nothing borrowed from 130/140.
    assert features[2].rolling_mean == Decimal("110")
    assert features[1].simple_return == Decimal("110.0") / Decimal("100.0") - 1


def test_a_window_spanning_a_gap_is_marked_incomplete(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    """A gap is a hole in knowledge, not a shortcut: no window may silently
    bridge it, and no return may be computed across it."""
    store = store_module.MarketDataStore(tmp_path / "gapwin.sqlite3")
    opens = [GRID + timedelta(hours=index) for index in range(5)]
    kept = [opens[0], opens[1], opens[3], opens[4]]
    _publish(store, kept, ["100.0", "110.0", "130.0", "140.0"], at=T1)
    features = series_module.causal_features(
        _series(store, snapshots_module, series_module, T2, opens), window=2
    )
    assert [f.bar_open_at for f in features] == [o.isoformat() for o in kept]
    # index 2 is 12:00, whose predecessor in the series is 10:00 -> not adjacent.
    assert features[2].complete is False
    assert features[2].simple_return is None
    assert features[3].complete is True


@pytest.mark.parametrize("window", [0, -1, 1.0, True, "3", 1001, None])
def test_an_invalid_window_is_refused(
    tmp_path, store_module, snapshots_module, series_module, window
) -> None:
    store, opens = _five_bar_store(store_module, tmp_path, name="badwin")
    series = _series(store, snapshots_module, series_module, T2, opens)
    with pytest.raises(series_module.MarketSeriesError):
        series_module.causal_features(series, window=window)


def test_the_series_and_its_features_are_deterministic(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    store, opens = _five_bar_store(store_module, tmp_path, name="det")
    runs = [
        (lambda s: (s, series_module.causal_features(s, window=3)))(
            _series(store, snapshots_module, series_module, T2, opens)
        )
        for _ in range(3)
    ]
    assert all(run == runs[0] for run in runs)
    # A database built independently produces the same series and features.
    twin, twin_opens = _five_bar_store(store_module, tmp_path, name="twin")
    twin_series = _series(twin, snapshots_module, series_module, T2, twin_opens)
    assert twin_series == runs[0][0]


def test_building_a_series_leaves_the_source_snapshot_untouched(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    store, opens = _five_bar_store(store_module, tmp_path, name="readonly")
    connection = store._connect()
    try:
        result = snapshots_module._materialize_snapshot(connection, **_request(T2, opens=opens))
        loaded_before = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        series_module.load_market_series(connection, snapshot_id=result.snapshot_id)
        series_module.causal_features(
            series_module.load_market_series(connection, snapshot_id=result.snapshot_id), window=2
        )
        assert snapshots_module.load_snapshot(
            connection, snapshot_id=result.snapshot_id
        ) == loaded_before
        assert connection.in_transaction is False
    finally:
        connection.close()


def test_a_corrupted_payload_is_refused_before_any_series_is_built(
    tmp_path, store_module, snapshots_module, series_module
) -> None:
    """The series layer inherits fail-closed: it never sees an unproven bar."""
    import sqlite3

    store, opens = _five_bar_store(store_module, tmp_path, name="corrupt")
    connection = store._connect()
    try:
        result = snapshots_module._materialize_snapshot(connection, **_request(T2, opens=opens))
        sha = connection.execute(
            "SELECT content_sha256 FROM market_snapshot_entries WHERE snapshot_id = ? "
            "ORDER BY bar_open_at LIMIT 1", (result.snapshot_id,)
        ).fetchone()[0]
    finally:
        connection.close()
    raw = sqlite3.connect(store.database_path)
    payload = json.loads(raw.execute(
        "SELECT payload_json FROM market_bar_receipts WHERE content_sha256 = ?", (sha,)
    ).fetchone()[0])
    raw.close()
    payload["close"] = "0.01"
    tamper = sqlite3.connect(store.database_path)
    try:
        for name in ("market_bar_receipts",):
            tamper.execute(f"DROP TRIGGER IF EXISTS {name}_no_update")
        tamper.execute("PRAGMA foreign_keys = OFF")
        tamper.execute(
            "UPDATE market_bar_receipts SET payload_json = ? WHERE content_sha256 = ?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), sha),
        )
        tamper.commit()
    finally:
        tamper.close()
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            series_module.load_market_series(connection, snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
    finally:
        connection.close()

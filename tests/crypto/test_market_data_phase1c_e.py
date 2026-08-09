"""Phase 1C-E: the Phase 1 market-data foundation proven as ONE system.

Every other test file proves a component. This one proves the seam: real
ingestion through the public store API, causal as-of selection, immutable
snapshot materialization, load/list/replay, restart, WAL concurrency and
fail-closed behaviour, exercised end to end on real file databases.

The invariant this file exists for above all others is ANTI-FUTURE-LEAKAGE:
a read at as_of=T may never use knowledge that did not exist at T.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]

# One deliberately boring hourly grid, so every assertion below is about
# causality rather than about calendar arithmetic.
GRID = datetime(2026, 8, 2, 9, 0, tzinfo=timezone.utc)
OPENS = [GRID + timedelta(hours=index) for index in range(3)]
T1 = datetime(2026, 8, 2, 12, 0, tzinfo=timezone.utc)   # version A published
T2 = datetime(2026, 8, 2, 13, 0, tzinfo=timezone.utc)   # a read between them
T3 = datetime(2026, 8, 2, 14, 0, tzinfo=timezone.utc)   # version B published
T4 = datetime(2026, 8, 2, 15, 0, tzinfo=timezone.utc)   # a read after both

VERSION_A_CLOSE = "106.0"
VERSION_B_CLOSE = "999.0"


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.market_data_store")


@pytest.fixture
def snapshots_module():
    return importlib.import_module("scripts.trading_lab.market_snapshots")


def _iso(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _candle(open_at: datetime, close: str) -> list:
    return [int(open_at.timestamp()), "100.0", "1000.0", "105.0", close, "1.0"]


def _payload(rows: list) -> bytes:
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


def _publish(store, opens, close, *, at: datetime):
    """One real provider response, through the public ingestion API."""
    return store.ingest_coinbase_response(
        _payload([_candle(open_at, close) for open_at in opens]),
        product_id="BTC-USD",
        timeframe="1h",
        available_at=_iso(at),
        ingested_at=_iso(at + timedelta(seconds=1)),
    )


def _two_version_store(store_module, tmp_path, *, name="p1ce"):
    """Version A of three bars known at T1, version B of the same three
    bars known only at T3. The whole point of the file."""
    store = store_module.MarketDataStore(tmp_path / f"{name}.sqlite3")
    _publish(store, OPENS, VERSION_A_CLOSE, at=T1)
    _publish(store, OPENS, VERSION_B_CLOSE, at=T3)
    return store


def _request(as_of: datetime, *, opens=None) -> dict:
    opens = opens or OPENS
    return {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": _iso(opens[0]),
        "range_end": _iso(opens[-1] + timedelta(hours=1)),
        "as_of": _iso(as_of),
    }


def _materialize(snapshots_module, store, as_of, **overrides):
    values = _request(as_of)
    values.update(overrides)
    connection = store._connect()
    try:
        return snapshots_module._materialize_snapshot(connection, **values)
    finally:
        connection.close()


def _with_connection(store, call):
    connection = store._connect()
    try:
        return call(connection)
    finally:
        connection.close()


def _closes(bars) -> list[Decimal]:
    """Prices come back in the canonical 10-decimal form, so compare values
    rather than the strings that carry them."""
    return [Decimal(bar["close"]) for bar in bars]


def _close(value: str) -> Decimal:
    return Decimal(value)


def _digest(loaded, bars) -> str:
    """A single value standing for everything a snapshot attests to."""
    return hashlib.sha256(
        json.dumps(
            {
                "manifest": loaded.manifest.__dict__,
                "entries": [entry.__dict__ for entry in loaded.entries],
                "bars": list(bars),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sabotage(database: Path, statements: list[tuple[str, tuple]]) -> None:
    connection = sqlite3.connect(database)
    try:
        for name in ("market_snapshot_entries", "market_snapshot_manifests",
                     "market_bar_receipts", "market_ingestions"):
            connection.execute(f"DROP TRIGGER IF EXISTS {name}_no_update")
            connection.execute(f"DROP TRIGGER IF EXISTS {name}_no_delete")
        connection.execute("DROP TRIGGER IF EXISTS market_snapshot_entries_no_late_insert")
        connection.execute("PRAGMA foreign_keys = OFF")
        for sql, params in statements:
            connection.execute(sql, params)
        connection.commit()
    finally:
        connection.close()


# --- 4. the whole pipeline, once, readably --------------------------------


def test_the_whole_phase1_pipeline_runs_end_to_end(
    tmp_path, store_module, snapshots_module
) -> None:
    """ingest -> persist -> causal select -> materialize -> load -> list ->
    replay, on one file database, through the public surfaces."""
    store = _two_version_store(store_module, tmp_path, name="e2e")

    result = _materialize(snapshots_module, store, T2)
    assert result.created is True
    assert result.entry_count == 3
    assert result.snapshot_id.startswith("hyprl-market-snapshot-")
    assert len(result.snapshot_id) == 86
    assert len(result.snapshot_request_id) == 94
    assert len(result.entries_content_hash) == 64

    loaded = _with_connection(
        store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=result.snapshot_id)
    )
    assert loaded.manifest.entry_count == 3
    assert loaded.manifest.as_of == _iso(T2).replace("Z", "+00:00")
    assert loaded.manifest.entries_content_hash == result.entries_content_hash
    # Openings are ordered, complete and exactly the requested range.
    assert [entry.bar_open_at for entry in loaded.entries] == [
        open_at.isoformat() for open_at in OPENS
    ]

    page = _with_connection(
        store,
        lambda c: snapshots_module.list_snapshot_manifests(
            c, snapshot_request_id=result.snapshot_request_id, limit=10
        ),
    )
    assert [item.snapshot_id for item in page.items] == [result.snapshot_id]
    assert page.next_after_snapshot_id is None

    bars = _with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=result.snapshot_id)
    )
    assert len(bars) == 3
    assert [bar["bar_open_at"] for bar in bars] == [o.isoformat() for o in OPENS]
    # The replay reproduces the SNAPSHOT's content, not the latest content.
    assert _closes(bars) == [_close(VERSION_A_CLOSE)] * 3
    assert _close(VERSION_B_CLOSE) not in _closes(bars)
    # Replay and load agree on every content hash, in the same order.
    assert [bar["content_sha256"] for bar in bars] == [
        entry.content_sha256 for entry in loaded.entries
    ]
    # Repeating the identical request is idempotent, not a second snapshot.
    again = _materialize(snapshots_module, store, T2)
    assert (again.snapshot_id, again.created) == (result.snapshot_id, False)
    assert again.entries_content_hash == result.entries_content_hash


# --- 5. anti-future-leakage: the invariant this phase exists for ----------


def test_a_causal_read_never_sees_a_revision_published_later(
    tmp_path, store_module, snapshots_module
) -> None:
    """as_of=T2 sits strictly between the two publications. Every bar must
    resolve to version A -- not the physically-last row, not MAX(rowid), not
    the newest ingestion."""
    store = _two_version_store(store_module, tmp_path, name="leak")

    before = _materialize(snapshots_module, store, T2)
    after = _materialize(snapshots_module, store, T4)

    bars_before = _with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=before.snapshot_id)
    )
    bars_after = _with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=after.snapshot_id)
    )

    # Several bars, so a bug that only trivialises one opening cannot pass.
    assert _closes(bars_before) == [_close(VERSION_A_CLOSE)] * 3, _closes(bars_before)
    assert _closes(bars_after) == [_close(VERSION_B_CLOSE)] * 3, _closes(bars_after)
    assert before.snapshot_id != after.snapshot_id
    assert before.entries_content_hash != after.entries_content_hash

    # And the selection layer itself, not just the materialized artefact.
    selected = _with_connection(
        store,
        lambda c: snapshots_module._select_snapshot_receipts(c, **_request(T2)),
    )
    assert len(selected) == 3
    assert all(item.ingested_at <= _iso(T2).replace("Z", "+00:00") for item in selected), selected
    assert {item.content_sha256 for item in selected} == {
        entry.content_sha256 for entry in _with_connection(
            store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=before.snapshot_id)
        ).entries
    }


def test_a_read_exactly_at_the_publication_instant_includes_it(
    tmp_path, store_module, snapshots_module
) -> None:
    """The boundary is inclusive: as_of == ingested_at is knowledge that DID
    exist. Pinning it stops the guard drifting to a strict comparison."""
    store = _two_version_store(store_module, tmp_path, name="edge")
    at_publication = T3 + timedelta(seconds=1)  # exactly version B's ingested_at
    bars = _with_connection(
        store,
        lambda c: snapshots_module.replay_snapshot(
            c,
            snapshot_id=snapshots_module._materialize_snapshot(
                c, **_request(at_publication)
            ).snapshot_id,
        ),
    )
    assert _closes(bars) == [_close(VERSION_B_CLOSE)] * 3
    one_microsecond_earlier = T3 + timedelta(seconds=1) - timedelta(microseconds=1)
    earlier = _with_connection(
        store,
        lambda c: snapshots_module.replay_snapshot(
            c,
            snapshot_id=snapshots_module._materialize_snapshot(
                c, **_request(one_microsecond_earlier)
            ).snapshot_id,
        ),
    )
    assert _closes(earlier) == [_close(VERSION_A_CLOSE)] * 3


# --- 6. immutability and idempotence -------------------------------------


def test_an_existing_snapshot_never_absorbs_data_published_after_it(
    tmp_path, store_module, snapshots_module
) -> None:
    store = store_module.MarketDataStore(tmp_path / "immutable.sqlite3")
    _publish(store, OPENS, VERSION_A_CLOSE, at=T1)

    original = _materialize(snapshots_module, store, T2)
    loaded_before = _with_connection(
        store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=original.snapshot_id)
    )
    bars_before = _with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=original.snapshot_id)
    )

    # Newer knowledge arrives AFTER the snapshot exists.
    _publish(store, OPENS, VERSION_B_CLOSE, at=T3)

    loaded_after = _with_connection(
        store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=original.snapshot_id)
    )
    bars_after = _with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=original.snapshot_id)
    )
    assert loaded_after == loaded_before
    assert bars_after == bars_before
    assert _closes(bars_after) == [_close(VERSION_A_CLOSE)] * 3

    # Re-running the same causal request still resolves to the same snapshot.
    repeated = _materialize(snapshots_module, store, T2)
    assert (repeated.snapshot_id, repeated.created) == (original.snapshot_id, False)

    # A LATER causal request may see the new data -- without touching S.
    later = _materialize(snapshots_module, store, T4)
    assert later.snapshot_id != original.snapshot_id
    assert _closes(_with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=later.snapshot_id)
    )) == [_close(VERSION_B_CLOSE)] * 3
    assert _with_connection(
        store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=original.snapshot_id)
    ) == loaded_before


# --- 7. restart: the truth lives in the file ------------------------------


def test_a_fresh_process_reproduces_the_snapshot_exactly(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _two_version_store(store_module, tmp_path, name="restart")
    result = _materialize(snapshots_module, store, T2)
    loaded = _with_connection(
        store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=result.snapshot_id)
    )
    bars = _with_connection(
        store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=result.snapshot_id)
    )
    expected = _digest(loaded, bars)

    script = (
        "import importlib, json, sys\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "sm = importlib.import_module('scripts.trading_lab.market_data_store')\n"
        "ms = importlib.import_module('scripts.trading_lab.market_snapshots')\n"
        f"store = sm.MarketDataStore({str(store.database_path)!r})\n"
        "c = store._connect()\n"
        f"loaded = ms.load_snapshot(c, snapshot_id={result.snapshot_id!r})\n"
        f"bars = ms.replay_snapshot(c, snapshot_id={result.snapshot_id!r})\n"
        f"again = ms._materialize_snapshot(c, **{_request(T2)!r})\n"
        "c.close()\n"
        "print(json.dumps({'manifest': loaded.manifest.__dict__,\n"
        "                  'entries': [e.__dict__ for e in loaded.entries],\n"
        "                  'bars': list(bars),\n"
        "                  'snapshot_id': again.snapshot_id,\n"
        "                  'created': again.created}, sort_keys=True, separators=(',', ':')))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    reopened = json.loads(completed.stdout)
    assert reopened["snapshot_id"] == result.snapshot_id
    assert reopened["created"] is False  # nothing re-created on reopen
    assert hashlib.sha256(
        json.dumps(
            {k: reopened[k] for k in ("manifest", "entries", "bars")},
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest() == expected


# --- 8. WAL concurrency, through the production surfaces ------------------


def test_a_concurrent_publication_cannot_tear_a_causal_read(
    tmp_path, store_module, snapshots_module
) -> None:
    """A reader holding one coherent view keeps it whole while a real
    ingestion commits on another connection."""
    store = _two_version_store(store_module, tmp_path, name="wal")
    reader = store._connect()
    try:
        reader.execute("BEGIN")
        before = snapshots_module._select_snapshot_receipts(reader, **_request(T4))
        assert len(before) == 3

        # A real publication, via the public API, on its own connection.
        _publish(store, OPENS, "777.0", at=T4 + timedelta(hours=1))

        during = snapshots_module._select_snapshot_receipts(reader, **_request(T4))
        assert during == before, "the open read view must not shift under it"
        reader.commit()

        # A NEW causal request, late enough, sees the new publication.
        after = snapshots_module._select_snapshot_receipts(
            reader, **_request(T4 + timedelta(hours=2))
        )
        assert {item.content_sha256 for item in after} != {
            item.content_sha256 for item in before
        }
    finally:
        reader.close()


# --- 9. fail-closed on three representative corruptions -------------------


def test_a_manifest_hash_that_no_longer_describes_its_entries_is_refused(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _two_version_store(store_module, tmp_path, name="c1")
    result = _materialize(snapshots_module, store, T2)
    _sabotage(store.database_path, [
        ("UPDATE market_snapshot_manifests SET entries_content_hash = ?", ("0" * 64,)),
    ])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
        assert connection.execute("SELECT 1").fetchone() == (1,)
    finally:
        connection.close()


def test_an_impossible_entry_cardinality_is_refused(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _two_version_store(store_module, tmp_path, name="c2")
    result = _materialize(snapshots_module, store, T2)
    _sabotage(store.database_path, [
        ("DELETE FROM market_snapshot_entries WHERE snapshot_id = ? AND bar_open_at = ?",
         (result.snapshot_id, OPENS[1].isoformat())),
    ])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption) as excinfo:
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        assert "declares 3 entries but stores 2" in str(excinfo.value)
        assert connection.in_transaction is False
    finally:
        connection.close()


def test_a_tampered_payload_is_refused_at_replay(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _two_version_store(store_module, tmp_path, name="c3")
    result = _materialize(snapshots_module, store, T2)
    connection = store._connect()
    try:
        content_sha256 = connection.execute(
            "SELECT content_sha256 FROM market_snapshot_entries WHERE snapshot_id = ? "
            "ORDER BY bar_open_at LIMIT 1",
            (result.snapshot_id,),
        ).fetchone()[0]
    finally:
        connection.close()
    original = json.loads(sqlite3.connect(store.database_path).execute(
        "SELECT payload_json FROM market_bar_receipts WHERE content_sha256 = ?",
        (content_sha256,),
    ).fetchone()[0])
    original["close"] = "0.01"
    _sabotage(store.database_path, [
        ("UPDATE market_bar_receipts SET payload_json = ? WHERE content_sha256 = ?",
         (json.dumps(original, sort_keys=True, separators=(",", ":")), content_sha256)),
    ])
    connection = store._connect()
    try:
        # Loading only reads references; replay is what proves the bytes.
        snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
    finally:
        connection.close()


# --- 10. bounds and volume, once ------------------------------------------


def test_a_large_range_stays_ordered_bounded_and_linear(
    tmp_path, store_module, snapshots_module
) -> None:
    """1 200 openings: more than one SNAPSHOT_LOAD_FETCH_SIZE chunk, so the
    bounded drains are crossed for real."""
    store = store_module.MarketDataStore(tmp_path / "volume.sqlite3")
    total = 1_200
    opens = [GRID + timedelta(hours=index) for index in range(total)]
    published_at = GRID + timedelta(days=400)
    for start in range(0, total, 300):  # MAX_CANDLES_PER_RESPONSE
        _publish(store, opens[start:start + 300], VERSION_A_CLOSE, at=published_at)

    values = _request(published_at + timedelta(days=1), opens=opens)
    connection = store._connect()
    queries: list[str] = []
    connection.set_trace_callback(queries.append)
    try:
        result = snapshots_module._materialize_snapshot(connection, **values)
        loaded = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.set_trace_callback(None)
        connection.close()
    assert result.entry_count == total
    assert len(loaded.entries) == total
    assert [entry.bar_open_at for entry in loaded.entries] == [o.isoformat() for o in opens]
    # No N+1: the number of SELECTs must not scale with the number of bars.
    selects = [q for q in queries if q.lstrip().upper().startswith("SELECT")]
    assert len(selects) < 50, len(selects)


def test_a_range_wider_than_the_maximum_is_refused_before_any_read(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _two_version_store(store_module, tmp_path, name="toowide")
    limit = snapshots_module.MAX_SNAPSHOT_RANGE_OPENS
    values = _request(T4)
    values["range_end"] = _iso(OPENS[0] + timedelta(hours=limit + 1))
    connection = store._connect()
    queries: list[str] = []
    connection.set_trace_callback(queries.append)
    try:
        with pytest.raises(snapshots_module.MarketSnapshotError) as excinfo:
            snapshots_module._materialize_snapshot(connection, **values)
        assert "exceeds the maximum" in str(excinfo.value)
        # Refused on the request itself: no scan, no materialization, no write.
        assert not any("market_bar_receipts" in q for q in queries), queries
        assert connection.in_transaction is False
    finally:
        connection.set_trace_callback(None)
        connection.close()


# --- 11. determinism ------------------------------------------------------


def test_the_same_causal_request_is_deterministic_across_connections(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _two_version_store(store_module, tmp_path, name="det")
    digests = set()
    identifiers = set()
    for _ in range(3):
        result = _materialize(snapshots_module, store, T2)
        loaded = _with_connection(
            store, lambda c: snapshots_module.load_snapshot(c, snapshot_id=result.snapshot_id)
        )
        bars = _with_connection(
            store, lambda c: snapshots_module.replay_snapshot(c, snapshot_id=result.snapshot_id)
        )
        identifiers.add((result.snapshot_id, result.snapshot_request_id,
                         result.entries_content_hash))
        digests.add(_digest(loaded, bars))
    assert len(identifiers) == 1, identifiers
    assert len(digests) == 1, digests
    # And two independently built databases agree, so nothing depends on
    # rowids, insertion order or filesystem order.
    twin = _two_version_store(store_module, tmp_path, name="twin")
    twin_result = _materialize(snapshots_module, twin, T2)
    assert twin_result.snapshot_id == next(iter(identifiers))[0]

"""Phase 1C-A: causal eligibility and deterministic revision selection.

Historical causal snapshots based on declared ingestion time (Contract A):
`as_of` is a cutoff over the DECLARED historical `ingested_at` of existing
receipts, never a live wall-clock, never a promise of lookahead-free
real-time knowledge. This file tests only the pure selection mechanism --
no snapshot manifest is created or persisted anywhere in this file.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import sqlite3

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def snapshots_module():
    return importlib.import_module("scripts.trading_lab.market_snapshots")


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.market_data_store")


def _store(module, tmp_path):
    return module.MarketDataStore(tmp_path / "market-data.sqlite3")


def _row(iso_open: str, *, low: str, high: str, open_: str, close: str, volume: str) -> list:
    epoch = int(datetime.fromisoformat(iso_open).timestamp())
    return [epoch, low, high, open_, close, volume]


def _bar(iso_open: str, volume: str = "1.0") -> list:
    return _row(iso_open, low="100.0", high="110.0", open_="105.0", close="106.0", volume=volume)


def _candles_payload(rows: list) -> bytes:
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


def _ingest(store, payload: bytes, **overrides):
    values = {
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "available_at": "2026-08-05T10:05:00Z",
        "ingested_at": "2026-08-05T10:06:00Z",
    }
    values.update(overrides)
    return store.ingest_coinbase_response(payload, **values)


def _seed_foreign_bar_receipt(
    database: Path,
    *,
    provider: str,
    product_id: str,
    timeframe: str,
    bar_open_at: str,
    bar_close_at: str,
) -> None:
    """Directly seed one receipt row to simulate a bar known under a
    provider the public ingestion API cannot itself produce (PROVIDER is a
    fixed module constant), so that isolation of the selection query can
    still be proven against a foreign provider."""
    seed_id = f"{provider}|{product_id}|{timeframe}|{bar_open_at}"
    payload_sha256 = hashlib.sha256(f"seed-payload|{seed_id}".encode()).hexdigest()
    ingestion_id = f"seed-ingestion|{seed_id}"
    content_sha256 = hashlib.sha256(f"seed-content|{seed_id}".encode()).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO raw_market_payloads (
                payload_sha256, provider, payload_bytes, byte_count, first_stored_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (payload_sha256, provider, b"seed", 4, bar_close_at),
        )
        connection.execute(
            """
            INSERT INTO market_ingestions (
                ingestion_id, schema_version, provider, product_id, timeframe,
                available_at, ingested_at, raw_payload_sha256, metadata_json, bar_count
            ) VALUES (?, 'seed', ?, ?, ?, ?, ?, ?, '{}', 1)
            """,
            (ingestion_id, provider, product_id, timeframe, bar_close_at, bar_close_at, payload_sha256),
        )
        connection.execute(
            """
            INSERT INTO market_bar_receipts (
                content_sha256, ingestion_id, bar_id, bar_version_id,
                bar_open_at, bar_close_at, available_at, ingested_at,
                raw_payload_sha256, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            (
                content_sha256,
                ingestion_id,
                f"seed-bar|{seed_id}",
                f"seed-version|{seed_id}",
                bar_open_at,
                bar_close_at,
                bar_close_at,
                bar_close_at,
                payload_sha256,
            ),
        )


def _seed_bar_receipt(
    database: Path,
    *,
    provider: str,
    product_id: str,
    timeframe: str,
    bar_open_at: str,
    available_at: str,
    ingested_at: str,
) -> None:
    """Directly seed one receipt row bypassing the public ingestion API and
    its causal validation entirely, so DB states build_market_bar would
    never allow (e.g. available_at > as_of while ingested_at <= as_of) can
    be constructed to prove this module's OWN defensive checks, independent
    of whatever Phase 1B already guarantees."""
    seed_id = f"{provider}|{product_id}|{timeframe}|{bar_open_at}|{available_at}|{ingested_at}"
    payload_sha256 = hashlib.sha256(f"seed-payload|{seed_id}".encode()).hexdigest()
    ingestion_id = f"seed-ingestion|{seed_id}"
    content_sha256 = hashlib.sha256(f"seed-content|{seed_id}".encode()).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO raw_market_payloads (payload_sha256, provider, payload_bytes, byte_count, first_stored_at) VALUES (?, ?, ?, ?, ?)",
            (payload_sha256, provider, b"seed", 4, ingested_at),
        )
        connection.execute(
            """
            INSERT INTO market_ingestions (
                ingestion_id, schema_version, provider, product_id, timeframe,
                available_at, ingested_at, raw_payload_sha256, metadata_json, bar_count
            ) VALUES (?, 'seed', ?, ?, ?, ?, ?, ?, '{}', 1)
            """,
            (ingestion_id, provider, product_id, timeframe, available_at, ingested_at, payload_sha256),
        )
        connection.execute(
            """
            INSERT INTO market_bar_receipts (
                content_sha256, ingestion_id, bar_id, bar_version_id,
                bar_open_at, bar_close_at, available_at, ingested_at,
                raw_payload_sha256, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')
            """,
            (
                content_sha256,
                ingestion_id,
                f"seed-bar|{seed_id}",
                f"seed-version|{seed_id}",
                bar_open_at,
                available_at,  # bar_close_at: not causally meaningful for this seed
                available_at,
                ingested_at,
                payload_sha256,
            ),
        )


def _select(snapshots_module, database, **overrides):
    values = {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": "2026-08-05T00:00:00Z",
        "range_end": "2026-08-06T00:00:00Z",
        "as_of": "2026-08-05T23:59:59Z",
    }
    values.update(overrides)
    with sqlite3.connect(database) as connection:
        return snapshots_module._select_snapshot_receipts(connection, **values)


# --- 1-3: basic eligibility -------------------------------------------


def test_earlier_version_selected_when_revision_is_after_as_of(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T10:06:00Z")

    assert len(selected) == 1
    assert selected[0].bar_open_at == "2026-08-05T09:00:00+00:00"
    assert selected[0].ingested_at == "2026-08-05T10:06:00+00:00"


def test_only_future_version_produces_no_entry(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T10:00:00Z")

    assert selected == ()


def test_available_before_as_of_but_ingested_after_excludes_receipt(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T12:00:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T11:00:00Z")

    assert selected == ()


# --- F2/F7: corrupted-invariant defensive guard, typed errors only -------


def test_corrupted_row_with_available_at_after_as_of_raises_typed_error(
    tmp_path, store_module, snapshots_module
) -> None:
    """Phase 1B's own API can never produce available_at > ingested_at, but
    this module must not silently trust that invariant on data it did not
    itself validate. Seed a row that violates it directly, bypassing the
    public ingestion API entirely."""
    store = _store(store_module, tmp_path)
    _seed_bar_receipt(
        store.database_path,
        provider="coinbase_exchange_rest",
        product_id="BTC-USD",
        timeframe="1h",
        bar_open_at="2026-08-05T09:00:00+00:00",
        available_at="2026-08-05T20:00:00+00:00",
        ingested_at="2026-08-05T10:00:00+00:00",
    )

    with pytest.raises(snapshots_module.MarketSnapshotError, match="invariant"):
        _select(snapshots_module, store.database_path, as_of="2026-08-05T12:00:00Z")


def test_corrupted_row_with_unparseable_available_at_raises_typed_error_not_value_error(
    tmp_path, store_module, snapshots_module
) -> None:
    """A raw ValueError from datetime parsing must never escape this module
    untyped: even a garbage available_at value must surface as
    MarketSnapshotError so a caller catching only that type is safe."""
    store = _store(store_module, tmp_path)
    _seed_bar_receipt(
        store.database_path,
        provider="coinbase_exchange_rest",
        product_id="BTC-USD",
        timeframe="1h",
        bar_open_at="2026-08-05T09:00:00+00:00",
        available_at="NOT-A-TIMESTAMP",
        ingested_at="2026-08-05T10:00:00+00:00",
    )

    with pytest.raises(snapshots_module.MarketSnapshotError):
        _select(snapshots_module, store.database_path, as_of="2026-08-05T12:00:00Z")


def test_valid_receipts_are_not_affected_by_the_defensive_available_at_check(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    assert len(selected) == 1
    assert selected[0].bar_open_at == "2026-08-05T09:00:00+00:00"


# --- 4-5: revision selection and tie-break ------------------------------


def test_largest_eligible_ingested_at_wins_among_different_versions(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T12:00:00Z")

    assert len(selected) == 1
    assert selected[0].ingested_at == "2026-08-05T11:06:00+00:00"


def test_same_ingested_at_same_version_picks_max_content_sha256(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload([_bar("2026-08-05T09:00:00+00:00")])
    # Same raw bytes, same ingested_at, DIFFERENT available_at -> different
    # content_sha256 (available_at is part of the hashed record) but the
    # SAME bar_version_id (OHLCV unchanged).
    _ingest(store, payload, available_at="2026-08-05T10:05:00Z", ingested_at="2026-08-05T11:00:00Z")
    _ingest(store, payload, available_at="2026-08-05T10:07:00Z", ingested_at="2026-08-05T11:00:00Z")

    with sqlite3.connect(store.database_path) as connection:
        rows = connection.execute(
            "SELECT content_sha256, bar_version_id FROM market_bar_receipts ORDER BY content_sha256"
        ).fetchall()
    assert len(rows) == 2
    assert rows[0][1] == rows[1][1]  # same bar_version_id
    expected_winner = max(rows, key=lambda row: row[0])[0]

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    assert len(selected) == 1
    assert selected[0].content_sha256 == expected_winner


# --- 6-8: revision conflicts --------------------------------------------


def test_two_distinct_versions_at_same_max_ingested_at_raise_conflict(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T11:00:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T10:07:00Z",
        ingested_at="2026-08-05T11:00:00Z",
    )

    with pytest.raises(snapshots_module.SnapshotSelectionConflict) as excinfo:
        _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    error = excinfo.value
    assert error.provider == "coinbase_exchange_rest"
    assert error.product_id == "BTC-USD"
    assert error.timeframe == "1h"
    assert error.bar_open_at == "2026-08-05T09:00:00+00:00"
    assert error.ingested_at == "2026-08-05T11:00:00+00:00"
    assert len(error.bar_version_ids) == 2


def test_conflict_in_the_middle_of_a_range_returns_no_partial_result(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T10:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T10:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T11:06:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    with pytest.raises(snapshots_module.SnapshotSelectionConflict):
        _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")


def test_older_conflict_is_ignored_when_a_later_unambiguous_version_exists(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T10:06:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="3.0")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    assert len(selected) == 1
    assert selected[0].ingested_at == "2026-08-05T11:06:00+00:00"


# --- 9: isolation --------------------------------------------------------


def test_isolated_from_other_provider(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)
    _seed_foreign_bar_receipt(
        store.database_path,
        provider="other_provider_rest",
        product_id="BTC-USD",
        timeframe="1h",
        bar_open_at="2026-08-05T09:00:00+00:00",
        bar_close_at="2026-08-05T10:00:00+00:00",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    assert selected == ()


def test_isolated_from_other_product_id(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        product_id="ETH-USD",
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    assert selected == ()


def test_isolated_from_other_timeframe(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T00:00:00+00:00")]),
        timeframe="1d",
        available_at="2026-08-06T00:05:00Z",
        ingested_at="2026-08-06T00:06:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-06T23:59:59Z")

    assert selected == ()


# --- 10-11: determinism under insertion order / replay ------------------


def test_reversed_insertion_order_gives_identical_result(
    tmp_path, store_module, snapshots_module
) -> None:
    forward = _store(store_module, tmp_path / "forward")
    _ingest(
        forward,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        forward,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )

    reversed_store = _store(store_module, tmp_path / "reversed")
    _ingest(
        reversed_store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="2.0")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )
    _ingest(
        reversed_store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00", volume="1.0")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )

    forward_result = _select(snapshots_module, forward.database_path, as_of="2026-08-05T23:59:59Z")
    reversed_result = _select(snapshots_module, reversed_store.database_path, as_of="2026-08-05T23:59:59Z")

    assert forward_result == reversed_result
    assert len(forward_result) == 1
    assert forward_result[0].ingested_at == "2026-08-05T11:06:00+00:00"


def test_repeated_calls_are_stable(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )

    first = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")
    second = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")
    third = _select(snapshots_module, store.database_path, as_of="2026-08-05T23:59:59Z")

    assert first == second == third


# --- F6: canonical ASC order of the result, discriminating -------------


def test_result_sequence_is_strictly_ascending_by_bar_open_at(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    # Ingested out of chronological order: 11:00, then 09:00, then 10:00.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T10:00:00+00:00")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )

    selected = _select(
        snapshots_module,
        store.database_path,
        range_start="2026-08-05T09:00:00Z",
        range_end="2026-08-05T12:00:00Z",
        as_of="2026-08-05T23:59:59Z",
    )

    assert [entry.bar_open_at for entry in selected] == [
        "2026-08-05T09:00:00+00:00",
        "2026-08-05T10:00:00+00:00",
        "2026-08-05T11:00:00+00:00",
    ]


# --- 12-13: empty ranges, inclusive/exclusive boundaries -----------------


def test_empty_range_produces_empty_tuple(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)

    selected = _select(snapshots_module, store.database_path)

    assert selected == ()


def test_range_start_inclusive_range_end_exclusive(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T10:00:00+00:00")]),
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )

    selected = _select(
        snapshots_module,
        store.database_path,
        range_start="2026-08-05T09:00:00Z",
        range_end="2026-08-05T10:00:00Z",
        as_of="2026-08-05T23:59:59Z",
    )

    assert len(selected) == 1
    assert selected[0].bar_open_at == "2026-08-05T09:00:00+00:00"


# --- 14-15: timestamp normalization --------------------------------------


def test_equivalent_instants_with_different_utc_offsets_behave_identically(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T12:00:00Z",
    )

    via_z = _select(snapshots_module, store.database_path, as_of="2026-08-05T12:00:00Z")
    via_offset = _select(snapshots_module, store.database_path, as_of="2026-08-05T14:00:00+02:00")

    assert via_z == via_offset
    assert len(via_z) == 1


def test_selection_excludes_receipt_whose_utc_instant_is_after_as_of_despite_lexical_order(
    tmp_path, store_module, snapshots_module
) -> None:
    """Discriminating case (F9): ingested_at is stored as UTC '13:00:00+00:00';
    as_of is expressed as '14:00:00+02:00' (= 12:00 UTC, i.e. BEFORE
    ingested_at). A naive lexical comparison of the raw digit sequences
    ('13' < '14') would wrongly conclude the receipt is eligible without
    ever normalizing the +02:00 offset; the real UTC instant comparison
    must exclude it. Unlike test_equivalent_instants_with_different_utc_offsets_*
    above, which uses offsets landing on the SAME side of the cutoff either
    way, this case only passes if normalization is genuinely applied.
    """
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T13:00:00Z",
    )

    selected = _select(snapshots_module, store.database_path, as_of="2026-08-05T14:00:00+02:00")

    assert selected == ()


def test_naive_timestamp_is_rejected_before_any_query(tmp_path, store_module, snapshots_module) -> None:
    class _ForbiddenConnection:
        def execute(self, *_args, **_kwargs):
            pytest.fail("a query was executed despite an invalid/naive timestamp input")

    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module._select_snapshot_receipts(
            _ForbiddenConnection(),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start="2026-08-05T00:00:00Z",
            range_end="2026-08-06T00:00:00Z",
            as_of=datetime(2026, 8, 5, 12, 0, 0),  # naive, no tzinfo
        )


def test_invalid_range_end_before_range_start_is_rejected_before_any_query(
    tmp_path, store_module, snapshots_module
) -> None:
    class _ForbiddenConnection:
        def execute(self, *_args, **_kwargs):
            pytest.fail("a query was executed despite an invalid range")

    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module._select_snapshot_receipts(
            _ForbiddenConnection(),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start="2026-08-06T00:00:00Z",
            range_end="2026-08-05T00:00:00Z",
            as_of="2026-08-05T12:00:00Z",
        )


# --- identity helpers -----------------------------------------------------


def _request_kwargs(**overrides):
    values = {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": "2026-08-05T00:00:00Z",
        "range_end": "2026-08-06T00:00:00Z",
        "as_of": "2026-08-05T23:59:59Z",
    }
    values.update(overrides)
    return values


def test_snapshot_request_id_is_deterministic(snapshots_module) -> None:
    first = snapshots_module.build_snapshot_request_id(**_request_kwargs())
    second = snapshots_module.build_snapshot_request_id(**_request_kwargs())
    assert first == second
    assert first.startswith("hyprl-market-snapshot-request-")


def test_snapshot_request_id_changes_with_any_parameter(snapshots_module) -> None:
    baseline = snapshots_module.build_snapshot_request_id(**_request_kwargs())
    variants = [
        _request_kwargs(provider="other_provider_rest"),
        _request_kwargs(product_id="ETH-USD"),
        _request_kwargs(timeframe="1d"),
        _request_kwargs(range_start="2026-08-04T00:00:00Z"),
        _request_kwargs(range_end="2026-08-07T00:00:00Z"),
        _request_kwargs(as_of="2026-08-05T00:00:00Z"),
        _request_kwargs(selection_policy_version="some-other-policy-v2"),
    ]
    ids = {snapshots_module.build_snapshot_request_id(**kwargs) for kwargs in variants}
    assert baseline not in ids
    assert len(ids) == len(variants)


def test_snapshot_request_id_equivalent_offsets_produce_same_id(snapshots_module) -> None:
    a = snapshots_module.build_snapshot_request_id(**_request_kwargs(as_of="2026-08-05T23:59:59Z"))
    b = snapshots_module.build_snapshot_request_id(
        **_request_kwargs(as_of="2026-08-06T01:59:59+02:00")
    )
    assert a == b


def test_snapshot_request_id_rejects_range_end_equal_to_range_start(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(range_start="2026-08-05T00:00:00Z", range_end="2026-08-05T00:00:00Z")
        )


def test_snapshot_request_id_rejects_range_end_before_range_start(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(range_start="2026-08-06T00:00:00Z", range_end="2026-08-05T00:00:00Z")
        )


def test_entries_content_hash_is_deterministic_and_order_independent(snapshots_module) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    e1 = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )
    e2 = Entry(
        bar_open_at="2026-08-05T10:00:00+00:00", content_sha256="b" * 64,
        bar_id="bar-2", bar_version_id="ver-2",
        ingested_at="2026-08-05T11:06:00+00:00", available_at="2026-08-05T11:05:00+00:00",
    )

    forward = snapshots_module.build_entries_content_hash((e1, e2))
    reversed_order = snapshots_module.build_entries_content_hash((e2, e1))
    repeated = snapshots_module.build_entries_content_hash((e1, e2))

    assert forward == reversed_order == repeated
    assert isinstance(forward, str)
    assert len(forward) == 64


def test_entries_content_hash_changes_when_a_receipt_changes(snapshots_module) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    baseline = (
        Entry(
            bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
            bar_id="bar-1", bar_version_id="ver-1",
            ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
        ),
    )
    changed = (
        Entry(
            bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="c" * 64,
            bar_id="bar-1", bar_version_id="ver-1",
            ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
        ),
    )
    assert snapshots_module.build_entries_content_hash(baseline) != snapshots_module.build_entries_content_hash(changed)


def test_entries_content_hash_of_empty_snapshot_is_stable(snapshots_module) -> None:
    a = snapshots_module.build_entries_content_hash(())
    b = snapshots_module.build_entries_content_hash(())
    assert a == b
    assert len(a) == 64


# --- F1: structural uniqueness of entries by bar_open_at -----------------


def test_entries_content_hash_rejects_duplicate_bar_open_at_identical_content(
    snapshots_module,
) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    d1 = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )
    d2 = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )

    with pytest.raises(snapshots_module.MarketSnapshotError, match="2026-08-05T09:00:00"):
        snapshots_module.build_entries_content_hash((d1, d2))


def test_entries_content_hash_rejects_conflicting_content_sha256_same_bar_open_at(
    snapshots_module,
) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    d1 = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )
    d2 = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="f" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )

    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module.build_entries_content_hash((d1, d2))
    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module.build_entries_content_hash((d2, d1))


# --- F3: explicit kind/schema_version envelope for entries_content_hash --


def test_entries_content_hash_uses_kind_and_schema_version_envelope(snapshots_module) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    entry = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )

    actual = snapshots_module.build_entries_content_hash((entry,))
    expected_envelope = {
        "kind": "market_snapshot_entries",
        "schema_version": snapshots_module.SNAPSHOT_SCHEMA_VERSION,
        "entries": [["2026-08-05T09:00:00+00:00", "a" * 64]],
    }
    expected = hashlib.sha256(
        json.dumps(
            expected_envelope, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    ).hexdigest()
    assert actual == expected


def test_entries_content_hash_changes_when_schema_version_changes(
    snapshots_module, monkeypatch
) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    entry = Entry(
        bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )

    baseline = snapshots_module.build_entries_content_hash((entry,))
    monkeypatch.setattr(snapshots_module, "SNAPSHOT_SCHEMA_VERSION", "some-other-schema-v2")
    changed = snapshots_module.build_entries_content_hash((entry,))

    assert baseline != changed


def test_entries_content_hash_kind_field_provides_real_domain_separation(snapshots_module) -> None:
    empty = snapshots_module.build_entries_content_hash(())
    naive_without_kind = hashlib.sha256(
        json.dumps(
            {"schema_version": snapshots_module.SNAPSHOT_SCHEMA_VERSION, "entries": []},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    assert empty != naive_without_kind


# --- F4: bar_open_at canonicalized before hashing -------------------------


def test_entries_content_hash_canonicalizes_equivalent_bar_open_at_offsets(
    snapshots_module,
) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    via_utc = Entry(
        bar_open_at="2026-08-03T12:00:00+00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )
    via_offset = Entry(
        bar_open_at="2026-08-03T14:00:00+02:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )

    assert snapshots_module.build_entries_content_hash(
        (via_utc,)
    ) == snapshots_module.build_entries_content_hash((via_offset,))


def test_entries_content_hash_rejects_naive_bar_open_at(snapshots_module) -> None:
    Entry = snapshots_module.SelectedSnapshotReceipt
    naive = Entry(
        bar_open_at="2026-08-05T09:00:00", content_sha256="a" * 64,
        bar_id="bar-1", bar_version_id="ver-1",
        ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
    )

    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module.build_entries_content_hash((naive,))


def test_snapshot_id_combines_request_and_result_identity(snapshots_module) -> None:
    request_id = snapshots_module.build_snapshot_request_id(**_request_kwargs())
    entries_hash = snapshots_module.build_entries_content_hash(())

    snapshot_id = snapshots_module.build_snapshot_id(
        snapshot_request_id=request_id, entries_content_hash=entries_hash
    )
    same_again = snapshots_module.build_snapshot_id(
        snapshot_request_id=request_id, entries_content_hash=entries_hash
    )
    different_result = snapshots_module.build_snapshot_id(
        snapshot_request_id=request_id,
        entries_content_hash=snapshots_module.build_entries_content_hash(
            (
                snapshots_module.SelectedSnapshotReceipt(
                    bar_open_at="2026-08-05T09:00:00+00:00", content_sha256="a" * 64,
                    bar_id="bar-1", bar_version_id="ver-1",
                    ingested_at="2026-08-05T10:06:00+00:00", available_at="2026-08-05T10:05:00+00:00",
                ),
            )
        ),
    )

    assert snapshot_id == same_again
    assert snapshot_id != different_result
    assert snapshot_id.startswith("hyprl-market-snapshot-")
    assert not snapshot_id.startswith("hyprl-market-snapshot-request-")


def test_snapshot_id_kind_field_provides_real_domain_separation(snapshots_module) -> None:
    request_id = "hyprl-market-snapshot-request-" + "a" * 64
    entries_hash = "b" * 64

    snapshot_id = snapshots_module.build_snapshot_id(
        snapshot_request_id=request_id, entries_content_hash=entries_hash
    )
    naive_hash_without_kind = hashlib.sha256(
        json.dumps(
            {"snapshot_request_id": request_id, "entries_content_hash": entries_hash},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert snapshot_id != f"hyprl-market-snapshot-{naive_hash_without_kind}"

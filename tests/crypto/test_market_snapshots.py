"""Phase 1C-A: causal eligibility and deterministic revision selection.

Historical causal snapshots based on declared ingestion time (Contract A):
`as_of` is a cutoff over the DECLARED historical `ingested_at` of existing
receipts, never a live wall-clock, never a promise of lookahead-free
real-time knowledge. This file tests only the pure selection mechanism --
no snapshot manifest is created or persisted anywhere in this file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import random
import re
import sqlite3
import tracemalloc

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


# =========================================================================
# Phase 1C-B: range alignment, cardinality bounds, bounded streaming read,
# and the SQLite query/index contract.
# =========================================================================


SNAPSHOT_DOMAIN_INDEX = "market_bar_receipts_snapshot_domain_lookup"
GRID_BASE = datetime(2026, 8, 5, tzinfo=timezone.utc)
INGEST_BASE = GRID_BASE + timedelta(days=100)


def _iso(moment: datetime) -> str:
    return moment.isoformat()


class _ForbiddenConnection:
    """Fails the test if any query is issued, proving a preflight rejection
    happened strictly BEFORE SQLite was ever touched."""

    def execute(self, *_args, **_kwargs):
        pytest.fail("a query was executed despite a range that must be rejected first")


class _EmptyCursor:
    def fetchmany(self, _size):
        return []

    def fetchall(self):
        pytest.fail("fetchall() must never be used: the read must stay bounded")

    def close(self):
        return None


class _CapturingConnection:
    """Records the exact SQL and parameters the selector issues, and serves
    an empty result so the call still completes normally."""

    def __init__(self) -> None:
        self.sql: str | None = None
        self.parameters: tuple | None = None

    def execute(self, sql, parameters=()):
        self.sql = sql
        self.parameters = parameters
        return _EmptyCursor()


class _ReorderingCursor:
    def __init__(self, rows: list, sizes: list[int]) -> None:
        self._rows = rows
        self._offset = 0
        self._sizes = sizes

    def fetchmany(self, size):
        self._sizes.append(size)
        chunk = self._rows[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def fetchall(self):
        pytest.fail("fetchall() must never be used: the read must stay bounded")

    def close(self):
        return None


class _ReorderingConnection:
    """Wraps a real connection, replays the selector's OWN query, then serves
    the resulting rows back in a caller-chosen order through fetchmany() only.

    Deliberately does not reimplement the query: the rows are exactly those
    SQLite returns, only their order (and therefore their distribution across
    chunks) is adversarial.
    """

    def __init__(self, connection: sqlite3.Connection, reorder) -> None:
        self._connection = connection
        self._reorder = reorder
        self.fetchmany_sizes: list[int] = []
        self.row_count = 0

    def execute(self, sql, parameters=()):
        rows = list(self._connection.execute(sql, parameters))
        self.row_count = len(rows)
        return _ReorderingCursor(list(self._reorder(rows)), self.fetchmany_sizes)


def _reference_selection(rows: list) -> tuple:
    """Independent oracle: Phase 1C-A's original non-streaming algorithm,
    re-expressed here (group everything, then MAX(ingested_at), then
    MAX(content_sha256)). Any streaming implementation must agree with it.

    Returns (selection, conflict) where exactly one is meaningful.
    """
    by_open: dict[str, list] = {}
    for row in rows:
        by_open.setdefault(row[0], []).append(row)
    selection = []
    for bar_open_at in sorted(by_open):
        candidates = by_open[bar_open_at]
        max_ingested_at = max(row[1] for row in candidates)
        winners = [row for row in candidates if row[1] == max_ingested_at]
        versions = sorted({row[4] for row in winners})
        if len(versions) > 1:
            return (), (bar_open_at, max_ingested_at, tuple(versions))
        selection.append(max(winners, key=lambda row: row[2]))
    return tuple(selection), None


def _seed_grid(
    database: Path,
    *,
    provider: str = "coinbase_exchange_rest",
    product_id: str = "BTC-USD",
    timeframe: str = "1h",
    openings: int,
    revisions: int,
    first_open: datetime = GRID_BASE,
    step: timedelta = timedelta(hours=1),
    ingest_base: datetime = INGEST_BASE,
    tag: str = "grid",
) -> None:
    """Bulk-seed `revisions` full re-ingestions of `openings` consecutive
    bars, bypassing the public ingestion API so realistic revision
    cardinalities can be built without driving the whole Phase 1B pipeline
    (which is exercised on its own elsewhere). Revision k declares
    ingested_at = ingest_base + k minutes."""
    payload_sha256 = hashlib.sha256(f"{tag}|payload".encode()).hexdigest()
    ingestion_rows = []
    receipt_rows = []
    for revision in range(revisions):
        ingestion_id = f"{tag}-ing-{revision:06d}"
        ingested_at = _iso(ingest_base + timedelta(minutes=revision))
        ingestion_rows.append(
            (
                ingestion_id, "seed", provider, product_id, timeframe,
                ingested_at, ingested_at, payload_sha256, "{}", openings,
            )
        )
        for index in range(openings):
            receipt_rows.append(
                (
                    hashlib.sha256(f"{tag}|{revision}|{index}".encode()).hexdigest(),
                    ingestion_id,
                    f"{tag}-bar-{index:06d}",
                    f"{tag}-ver-{revision:06d}-{index:06d}",
                    _iso(first_open + step * index),
                    _iso(first_open + step * (index + 1)),
                    ingested_at,
                    ingested_at,
                    payload_sha256,
                    "{}",
                )
            )
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO raw_market_payloads (payload_sha256, provider, "
            "payload_bytes, byte_count, first_stored_at) VALUES (?, ?, ?, ?, ?)",
            (payload_sha256, provider, b"seed", 4, _iso(ingest_base)),
        )
        connection.executemany(
            "INSERT INTO market_ingestions (ingestion_id, schema_version, provider, "
            "product_id, timeframe, available_at, ingested_at, raw_payload_sha256, "
            "metadata_json, bar_count) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ingestion_rows,
        )
        connection.executemany(
            "INSERT INTO market_bar_receipts (content_sha256, ingestion_id, bar_id, "
            "bar_version_id, bar_open_at, bar_close_at, available_at, ingested_at, "
            "raw_payload_sha256, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            receipt_rows,
        )


def _seed_receipt(
    database: Path,
    *,
    provider: str = "coinbase_exchange_rest",
    product_id: str = "BTC-USD",
    timeframe: str = "1h",
    bar_open_at: str,
    ingested_at: str,
    bar_version_id: str,
    tag: str,
) -> None:
    """Seed exactly one extra receipt in its own ingestion (revisions of the
    same bar always come from distinct ingestions in Phase 1B, which enforces
    UNIQUE(ingestion_id, bar_id))."""
    payload_sha256 = hashlib.sha256(f"{tag}|payload".encode()).hexdigest()
    with sqlite3.connect(database) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO raw_market_payloads (payload_sha256, provider, "
            "payload_bytes, byte_count, first_stored_at) VALUES (?, ?, ?, ?, ?)",
            (payload_sha256, provider, b"seed", 4, ingested_at),
        )
        connection.execute(
            "INSERT INTO market_ingestions (ingestion_id, schema_version, provider, "
            "product_id, timeframe, available_at, ingested_at, raw_payload_sha256, "
            "metadata_json, bar_count) VALUES (?, 'seed', ?, ?, ?, ?, ?, ?, '{}', 1)",
            (f"{tag}-ing", provider, product_id, timeframe, ingested_at, ingested_at, payload_sha256),
        )
        connection.execute(
            "INSERT INTO market_bar_receipts (content_sha256, ingestion_id, bar_id, "
            "bar_version_id, bar_open_at, bar_close_at, available_at, ingested_at, "
            "raw_payload_sha256, payload_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, '{}')",
            (
                hashlib.sha256(f"{tag}|content".encode()).hexdigest(),
                f"{tag}-ing",
                f"{tag}-bar",
                bar_version_id,
                bar_open_at,
                bar_open_at,
                ingested_at,
                ingested_at,
                payload_sha256,
            ),
        )


def _build_limit_store(tmp_path_factory, *, name: str, with_conflict: bool) -> dict:
    """One store holding EXACTLY MAX_SNAPSHOT_ELIGIBLE_RECEIPTS eligible
    receipts under `as_of_at_limit`, and exactly one more under
    `as_of_over_limit`. Built once per module: the real limit is exercised at
    its real scale, following this project's convention of never faking a
    cardinality bound."""
    store_module = importlib.import_module("scripts.trading_lab.market_data_store")
    snapshots_module = importlib.import_module("scripts.trading_lab.market_snapshots")
    limit = snapshots_module.MAX_SNAPSHOT_ELIGIBLE_RECEIPTS
    database = tmp_path_factory.mktemp(name) / "market-data.sqlite3"
    store = store_module.MarketDataStore(database)

    openings = 1000
    if with_conflict:
        # 999 openings fully revised, plus one final opening whose two most
        # recent receipts share the SAME ingested_at with different
        # bar_version_id -- a genuine contradiction sitting at the very end
        # of the eligible set.
        grid_openings = openings - 1
        revisions = limit // openings
        _seed_grid(
            database, openings=grid_openings, revisions=revisions, tag="conf",
        )
        last_open = _iso(GRID_BASE + timedelta(hours=grid_openings))
        for extra in range(revisions):
            # The last two share one ingested_at: MAX(ingested_at) is ambiguous.
            minute = min(extra, revisions - 2)
            _seed_receipt(
                database,
                bar_open_at=last_open,
                ingested_at=_iso(INGEST_BASE + timedelta(minutes=minute)),
                bar_version_id=f"conflict-ver-{extra:06d}",
                tag=f"conf-tail-{extra:06d}",
            )
    else:
        revisions = limit // openings
        _seed_grid(database, openings=openings, revisions=revisions, tag="clean")

    # Must cover every grid revision (minute 0..revisions-1) while excluding
    # the single over-limit receipt seeded one day later.
    as_of_at_limit = _iso(INGEST_BASE + timedelta(hours=12))
    _seed_receipt(
        database,
        bar_open_at=_iso(GRID_BASE),
        ingested_at=_iso(INGEST_BASE + timedelta(days=1)),
        bar_version_id="over-limit-ver",
        tag=f"{name}-over",
    )
    return {
        "database": database,
        "store": store,
        "limit": limit,
        "openings": openings,
        "range_start": _iso(GRID_BASE),
        "range_end": _iso(GRID_BASE + timedelta(hours=openings)),
        "as_of_at_limit": as_of_at_limit,
        "as_of_over_limit": _iso(INGEST_BASE + timedelta(days=2)),
    }


@pytest.fixture(scope="module")
def limit_store(tmp_path_factory):
    return _build_limit_store(tmp_path_factory, name="atlimit", with_conflict=False)


@pytest.fixture(scope="module")
def conflicting_limit_store(tmp_path_factory):
    return _build_limit_store(tmp_path_factory, name="conflimit", with_conflict=True)


def _select_on(snapshots_module, database, **overrides):
    values = {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
    }
    values.update(overrides)
    with sqlite3.connect(database) as connection:
        return snapshots_module._select_snapshot_receipts(connection, **values)


def _index_names(database: Path, table: str) -> set[str]:
    with sqlite3.connect(database) as connection:
        return {
            name
            for (name,) in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND tbl_name = ? "
                "AND sql IS NOT NULL",
                (table,),
            )
        }


def _query_plan(database: Path, sql: str, parameters) -> str:
    with sqlite3.connect(database) as connection:
        rows = connection.execute(f"EXPLAIN QUERY PLAN {sql}", parameters).fetchall()
    return " ".join(row[3] for row in rows)


def _capture_selection_sql(snapshots_module, **overrides):
    values = {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": "2026-08-05T00:00:00Z",
        "range_end": "2026-08-06T00:00:00Z",
        "as_of": "2026-08-05T23:59:59Z",
    }
    values.update(overrides)
    connection = _CapturingConnection()
    assert snapshots_module._select_snapshot_receipts(connection, **values) == ()
    return connection


# --- 1C-B alignment -------------------------------------------------------


def test_aligned_hourly_range_is_accepted(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-05T09:00:00+00:00")]))

    selected = _select_on(
        snapshots_module,
        store.database_path,
        range_start="2026-08-05T09:00:00Z",
        range_end="2026-08-05T12:00:00Z",
        as_of="2026-08-05T23:00:00Z",
    )

    assert len(selected) == 1


def test_unaligned_hourly_range_start_is_rejected_before_any_query(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module._select_snapshot_receipts(
            _ForbiddenConnection(),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start="2026-08-05T10:30:00Z",
            range_end="2026-08-05T12:00:00Z",
            as_of="2026-08-05T23:00:00Z",
        )


def test_unaligned_hourly_range_end_is_rejected_before_any_query(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module._select_snapshot_receipts(
            _ForbiddenConnection(),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start="2026-08-05T10:00:00Z",
            range_end="2026-08-05T12:30:00Z",
            as_of="2026-08-05T23:00:00Z",
        )


def test_daily_range_on_utc_midnight_is_accepted(snapshots_module) -> None:
    request_id = snapshots_module.build_snapshot_request_id(
        **_request_kwargs(
            timeframe="1d",
            range_start="2026-08-05T00:00:00Z",
            range_end="2026-08-07T00:00:00Z",
        )
    )
    assert request_id.startswith("hyprl-market-snapshot-request-")


def test_daily_range_aligned_only_to_the_hour_is_rejected(snapshots_module) -> None:
    hourly_bounds = {
        "range_start": "2026-08-05T01:00:00Z",
        "range_end": "2026-08-06T01:00:00Z",
    }
    # The very same bounds are a perfectly legal 1h range: only the daily
    # grid rejects them, which is exactly the point.
    assert snapshots_module.build_snapshot_request_id(
        **_request_kwargs(timeframe="1h", **hourly_bounds)
    )
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(timeframe="1d", **hourly_bounds)
        )


def test_residual_seconds_in_a_range_bound_are_rejected(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(
                range_start="2026-08-05T10:00:01Z", range_end="2026-08-05T12:00:00Z"
            )
        )


def test_residual_microseconds_in_a_range_bound_are_rejected(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(
                range_start="2026-08-05T10:00:00Z",
                range_end="2026-08-05T12:00:00.000001Z",
            )
        )


def test_divisible_but_off_grid_range_is_rejected(snapshots_module) -> None:
    """[10:30, 12:30) spans exactly two hours, so a naive divisibility check
    accepts it -- yet no stored bar_open_at can ever fall on it."""
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(
                range_start="2026-08-05T10:30:00Z", range_end="2026-08-05T12:30:00Z"
            )
        )


def test_daily_divisible_but_off_grid_range_is_rejected(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError, match="aligned"):
        snapshots_module.build_snapshot_request_id(
            **_request_kwargs(
                timeframe="1d",
                range_start="2026-08-05T01:00:00Z",
                range_end="2026-08-06T01:00:00Z",
            )
        )


def test_equivalent_utc_offsets_align_identically(snapshots_module) -> None:
    utc = snapshots_module.build_snapshot_request_id(
        **_request_kwargs(
            timeframe="1d",
            range_start="2026-08-05T00:00:00Z",
            range_end="2026-08-06T00:00:00Z",
        )
    )
    shifted = snapshots_module.build_snapshot_request_id(
        **_request_kwargs(
            timeframe="1d",
            range_start="2026-08-04T19:00:00-05:00",
            range_end="2026-08-05T19:00:00-05:00",
        )
    )
    assert utc == shifted


def test_leap_day_and_year_boundaries_are_aligned(snapshots_module) -> None:
    for timeframe, start, end in (
        ("1d", "2024-02-29T00:00:00Z", "2024-03-01T00:00:00Z"),
        ("1d", "2026-12-31T00:00:00Z", "2027-01-01T00:00:00Z"),
        ("1h", "2026-12-31T23:00:00Z", "2027-01-01T00:00:00Z"),
    ):
        assert snapshots_module.build_snapshot_request_id(
            **_request_kwargs(timeframe=timeframe, range_start=start, range_end=end)
        )


def test_unknown_timeframe_is_rejected_before_any_query(snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError):
        snapshots_module._select_snapshot_receipts(
            _ForbiddenConnection(),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="4h",
            range_start="2026-08-05T00:00:00Z",
            range_end="2026-08-06T00:00:00Z",
            as_of="2026-08-05T23:00:00Z",
        )


def _market_bar_accepts_open(market_bar_module, *, timeframe: str, bar_open_at: str) -> bool:
    duration = market_bar_module.TIMEFRAME_DURATIONS[timeframe]
    opened = datetime.fromisoformat(bar_open_at.replace("Z", "+00:00"))
    try:
        market_bar_module.build_market_bar(
            asset="BTC/USD",
            venue="coinbase_exchange",
            provider="coinbase_exchange",
            timeframe=timeframe,
            bar_open_at=opened,
            bar_close_at=opened + duration,
            available_at=opened + duration,
            ingested_at=opened + duration,
            open_price="100.0",
            high_price="110.0",
            low_price="90.0",
            close_price="105.0",
            volume="1.0",
            raw_payload_sha256="a" * 64,
        )
    except ValueError:
        return False
    return True


def _preflight_accepts_bound(snapshots_module, market_bar_module, *, timeframe, bar_open_at) -> bool:
    duration = market_bar_module.TIMEFRAME_DURATIONS[timeframe]
    opened = datetime.fromisoformat(bar_open_at.replace("Z", "+00:00"))
    try:
        snapshots_module.build_snapshot_request_id(
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe=timeframe,
            range_start=opened,
            range_end=opened + duration,
            as_of=opened + duration,
        )
    except snapshots_module.MarketSnapshotError:
        return False
    return True


def test_alignment_rules_match_market_bar_v1_exactly(snapshots_module) -> None:
    """Bidirectional cross-check: the snapshot range preflight must accept a
    bound if and only if build_market_bar would accept it as a bar_open_at.
    Any divergence would let a snapshot request name a grid position that can
    never hold a stored bar."""
    market_bar_module = importlib.import_module("scripts.trading_lab.market_bar")
    bounds = [
        "2026-08-05T00:00:00+00:00",
        "2026-08-05T10:00:00+00:00",
        "2026-08-05T10:30:00+00:00",
        "2026-08-05T10:00:30+00:00",
        "2026-08-05T10:00:00.000001+00:00",
        "2026-08-04T19:00:00-05:00",
        "2024-02-29T00:00:00+00:00",
        "2026-12-31T23:00:00+00:00",
    ]
    for timeframe in ("1h", "1d"):
        for bound in bounds:
            assert _preflight_accepts_bound(
                snapshots_module, market_bar_module, timeframe=timeframe, bar_open_at=bound
            ) is _market_bar_accepts_open(
                market_bar_module, timeframe=timeframe, bar_open_at=bound
            ), f"divergence on timeframe={timeframe} bound={bound}"


# --- 1C-B open-count limit ------------------------------------------------


def test_one_open_below_the_range_limit_is_accepted(snapshots_module) -> None:
    limit = snapshots_module.MAX_SNAPSHOT_RANGE_OPENS
    assert snapshots_module.build_snapshot_request_id(
        **_request_kwargs(
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=limit - 1)),
        )
    )


def test_exactly_the_range_open_limit_is_accepted(snapshots_module) -> None:
    limit = snapshots_module.MAX_SNAPSHOT_RANGE_OPENS
    assert snapshots_module.build_snapshot_request_id(
        **_request_kwargs(
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=limit)),
        )
    )


def test_one_open_above_the_range_limit_is_rejected_before_any_query(snapshots_module) -> None:
    limit = snapshots_module.MAX_SNAPSHOT_RANGE_OPENS
    with pytest.raises(snapshots_module.MarketSnapshotError, match="10000|10,000|maximum"):
        snapshots_module._select_snapshot_receipts(
            _ForbiddenConnection(),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=limit + 1)),
            as_of=_iso(INGEST_BASE),
        )


def test_enormous_range_is_rejected_without_materializing_a_grid(snapshots_module) -> None:
    """A ~87-million-open range must be rejected by O(1) integer arithmetic:
    materializing the grid first would cost gigabytes."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        with pytest.raises(snapshots_module.MarketSnapshotError):
            snapshots_module._select_snapshot_receipts(
                _ForbiddenConnection(),
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                range_start="0001-01-01T00:00:00+00:00",
                range_end="9999-01-01T00:00:00+00:00",
                as_of="2026-08-05T00:00:00+00:00",
            )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 100_000, f"preflight allocated {peak} bytes, grid likely materialized"


def test_non_empty_range_without_any_data_returns_an_empty_tuple(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)

    selected = _select_on(
        snapshots_module,
        store.database_path,
        range_start="2026-08-05T00:00:00Z",
        range_end="2026-08-06T00:00:00Z",
        as_of="2026-08-05T23:00:00Z",
    )

    assert selected == ()


def test_request_id_and_selection_share_the_same_range_preflight(snapshots_module) -> None:
    """Both entry points must reject exactly the same ranges: an id must
    never be minted for a range the selector would refuse to compute."""
    limit = snapshots_module.MAX_SNAPSHOT_RANGE_OPENS
    rejected = [
        ("1h", "2026-08-05T10:30:00Z", "2026-08-05T12:30:00Z"),
        ("1d", "2026-08-05T01:00:00Z", "2026-08-06T01:00:00Z"),
        ("1h", _iso(GRID_BASE), _iso(GRID_BASE + timedelta(hours=limit + 1))),
    ]
    for timeframe, range_start, range_end in rejected:
        with pytest.raises(snapshots_module.MarketSnapshotError):
            snapshots_module.build_snapshot_request_id(
                **_request_kwargs(
                    timeframe=timeframe, range_start=range_start, range_end=range_end
                )
            )
        with pytest.raises(snapshots_module.MarketSnapshotError):
            snapshots_module._select_snapshot_receipts(
                _ForbiddenConnection(),
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe=timeframe,
                range_start=range_start,
                range_end=range_end,
                as_of=_iso(INGEST_BASE),
            )


# --- 1C-B eligible-receipt limit -----------------------------------------


def _count_eligible(database: Path, *, range_start: str, range_end: str, as_of: str) -> int:
    """Independent count of the rows the selector is entitled to read, so a
    cardinality test can never silently pass on a smaller set than it claims
    to exercise."""
    with sqlite3.connect(database) as connection:
        (count,) = connection.execute(
            "SELECT COUNT(*) FROM market_bar_receipts r WHERE r.ingestion_id IN ("
            "SELECT i.ingestion_id FROM market_ingestions i WHERE i.provider = ? "
            "AND i.product_id = ? AND i.timeframe = ?) AND r.bar_open_at >= ? "
            "AND r.bar_open_at < ? AND r.ingested_at <= ?",
            ("coinbase_exchange_rest", "BTC-USD", "1h", range_start, range_end, as_of),
        ).fetchone()
    return count


def test_the_limit_fixtures_really_sit_on_the_boundary(
    limit_store, conflicting_limit_store, snapshots_module
) -> None:
    """Guards every other cardinality test in this file: both stores must
    hold exactly the limit under one as_of and exactly one more under the
    other."""
    limit = snapshots_module.MAX_SNAPSHOT_ELIGIBLE_RECEIPTS
    for store in (limit_store, conflicting_limit_store):
        assert _count_eligible(
            store["database"],
            range_start=store["range_start"],
            range_end=store["range_end"],
            as_of=store["as_of_at_limit"],
        ) == limit
        assert _count_eligible(
            store["database"],
            range_start=store["range_start"],
            range_end=store["range_end"],
            as_of=store["as_of_over_limit"],
        ) == limit + 1


def test_exactly_the_eligible_receipt_limit_is_accepted(limit_store, snapshots_module) -> None:
    selected = _select_on(
        snapshots_module,
        limit_store["database"],
        range_start=limit_store["range_start"],
        range_end=limit_store["range_end"],
        as_of=limit_store["as_of_at_limit"],
    )

    assert len(selected) == limit_store["openings"]


def test_one_receipt_above_the_eligible_limit_is_rejected(limit_store, snapshots_module) -> None:
    with pytest.raises(snapshots_module.SnapshotEligibilityLimitExceeded):
        _select_on(
            snapshots_module,
            limit_store["database"],
            range_start=limit_store["range_start"],
            range_end=limit_store["range_end"],
            as_of=limit_store["as_of_over_limit"],
        )


def test_eligibility_limit_error_is_a_market_snapshot_error(limit_store, snapshots_module) -> None:
    with pytest.raises(snapshots_module.MarketSnapshotError) as excinfo:
        _select_on(
            snapshots_module,
            limit_store["database"],
            range_start=limit_store["range_start"],
            range_end=limit_store["range_end"],
            as_of=limit_store["as_of_over_limit"],
        )

    error = excinfo.value
    assert isinstance(error, snapshots_module.SnapshotEligibilityLimitExceeded)
    assert error.provider == "coinbase_exchange_rest"
    assert error.product_id == "BTC-USD"
    assert error.timeframe == "1h"
    assert error.limit == snapshots_module.MAX_SNAPSHOT_ELIGIBLE_RECEIPTS
    assert error.range_start == limit_store["range_start"]
    assert error.range_end == limit_store["range_end"]
    assert error.as_of == limit_store["as_of_over_limit"]
    assert "payload" not in str(error)


def test_eligibility_overflow_returns_no_partial_result(limit_store, snapshots_module) -> None:
    """Nothing derived from the first MAX rows may ever reach the caller."""
    outcome = "not-set"
    try:
        outcome = _select_on(
            snapshots_module,
            limit_store["database"],
            range_start=limit_store["range_start"],
            range_end=limit_store["range_end"],
            as_of=limit_store["as_of_over_limit"],
        )
    except snapshots_module.SnapshotEligibilityLimitExceeded:
        pass
    assert outcome == "not-set"


def test_conflict_within_the_eligible_limit_is_still_detected(
    conflicting_limit_store, snapshots_module
) -> None:
    """At exactly the limit, every eligible row must still be processed: a
    contradiction must surface as a conflict, never be skipped."""
    with pytest.raises(snapshots_module.SnapshotSelectionConflict):
        _select_on(
            snapshots_module,
            conflicting_limit_store["database"],
            range_start=conflicting_limit_store["range_start"],
            range_end=conflicting_limit_store["range_end"],
            as_of=conflicting_limit_store["as_of_at_limit"],
        )


def test_overflow_takes_precedence_over_a_conflict_beyond_the_limit(
    conflicting_limit_store, snapshots_module
) -> None:
    """Documented V1 precedence: once the eligibility bound is exceeded, the
    result is rejected for cardinality, not for the conflict it also has.
    Both branches are fail-closed and neither returns a partial result."""
    with pytest.raises(snapshots_module.SnapshotEligibilityLimitExceeded):
        _select_on(
            snapshots_module,
            conflicting_limit_store["database"],
            range_start=conflicting_limit_store["range_start"],
            range_end=conflicting_limit_store["range_end"],
            as_of=conflicting_limit_store["as_of_over_limit"],
        )


def test_limit_is_never_used_as_functional_truncation(limit_store, snapshots_module) -> None:
    """At the limit the full result is produced (no truncation); one row over,
    nothing at all is produced (no prefix-derived snapshot)."""
    complete = _select_on(
        snapshots_module,
        limit_store["database"],
        range_start=limit_store["range_start"],
        range_end=limit_store["range_end"],
        as_of=limit_store["as_of_at_limit"],
    )
    assert len(complete) == limit_store["openings"]
    assert len({entry.bar_open_at for entry in complete}) == limit_store["openings"]

    with pytest.raises(snapshots_module.SnapshotEligibilityLimitExceeded):
        _select_on(
            snapshots_module,
            limit_store["database"],
            range_start=limit_store["range_start"],
            range_end=limit_store["range_end"],
            as_of=limit_store["as_of_over_limit"],
        )


def test_memory_is_bounded_by_openings_not_by_eligible_receipts(
    limit_store, snapshots_module
) -> None:
    """100 000 eligible receipts collapse to 1 000 openings: peak allocation
    must follow the openings, not the rows (a fetchall of the same result set
    costs tens of megabytes)."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        selected = _select_on(
            snapshots_module,
            limit_store["database"],
            range_start=limit_store["range_start"],
            range_end=limit_store["range_end"],
            as_of=limit_store["as_of_at_limit"],
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(selected) == limit_store["openings"]
    assert peak < 20_000_000, f"peak allocation {peak} bytes suggests an unbounded read"


# --- 1C-B bounded streaming read -----------------------------------------


def test_selection_never_calls_fetchall(tmp_path, store_module, snapshots_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-05T09:00:00+00:00")]))

    with sqlite3.connect(store.database_path) as connection:
        recorder = _ReorderingConnection(connection, lambda rows: rows)
        selected = snapshots_module._select_snapshot_receipts(
            recorder,
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start="2026-08-05T09:00:00Z",
            range_end="2026-08-05T12:00:00Z",
            as_of="2026-08-05T23:00:00Z",
        )

    assert len(selected) == 1
    assert recorder.fetchmany_sizes
    assert set(recorder.fetchmany_sizes) == {snapshots_module.SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE}


def test_streaming_result_is_independent_of_row_order_and_chunking(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    """Neither the order SQLite returns rows in, nor the chunk boundaries,
    may change the selection: the same opening and the same MAX(ingested_at)
    are deliberately forced to straddle several chunks."""
    store = _store(store_module, tmp_path)
    for revision in range(5):
        _ingest(
            store,
            _candles_payload(
                [
                    _bar("2026-08-05T09:00:00+00:00", volume=f"{revision + 1}.0"),
                    _bar("2026-08-05T10:00:00+00:00", volume=f"{revision + 1}.5"),
                ]
            ),
            available_at=_iso(INGEST_BASE + timedelta(minutes=revision)),
            ingested_at=_iso(INGEST_BASE + timedelta(minutes=revision)),
        )

    with sqlite3.connect(store.database_path) as connection:
        baseline_rows = list(
            connection.execute(
                "SELECT bar_open_at, ingested_at, content_sha256, bar_id, "
                "bar_version_id, available_at FROM market_bar_receipts"
            )
        )
    expected, conflict = _reference_selection(baseline_rows)
    assert conflict is None
    expected_keys = [(row[0], row[2]) for row in expected]

    orderings = {
        "as-is": lambda rows: rows,
        "reversed": lambda rows: list(reversed(rows)),
        "by-content": lambda rows: sorted(rows, key=lambda row: row[2]),
        "by-ingested": lambda rows: sorted(rows, key=lambda row: row[1]),
    }
    for seed in range(6):
        orderings[f"shuffled-{seed}"] = (
            lambda rows, seed=seed: random.Random(seed).sample(rows, len(rows))
        )

    for chunk_size in (1, 2, 3, 7, 1000):
        monkeypatch.setattr(
            snapshots_module, "SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE", chunk_size
        )
        for label, reorder in orderings.items():
            with sqlite3.connect(store.database_path) as connection:
                selected = snapshots_module._select_snapshot_receipts(
                    _ReorderingConnection(connection, reorder),
                    provider="coinbase_exchange_rest",
                    product_id="BTC-USD",
                    timeframe="1h",
                    range_start="2026-08-05T09:00:00Z",
                    range_end="2026-08-05T12:00:00Z",
                    as_of=_iso(INGEST_BASE + timedelta(hours=1)),
                )
            assert [
                (entry.bar_open_at, entry.content_sha256) for entry in selected
            ] == expected_keys, f"chunk={chunk_size} order={label}"


def test_older_conflict_is_ignored_when_a_newer_unambiguous_version_exists(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    """A contradiction that a later revision supersedes must never be raised,
    even when the rows arrive in the worst possible chunk arrangement."""
    store = _store(store_module, tmp_path)
    conflicting_at = _iso(INGEST_BASE)
    for suffix in ("a", "b"):
        _seed_receipt(
            store.database_path,
            bar_open_at="2026-08-05T09:00:00+00:00",
            ingested_at=conflicting_at,
            bar_version_id=f"old-conflict-{suffix}",
            tag=f"old-{suffix}",
        )
    _seed_receipt(
        store.database_path,
        bar_open_at="2026-08-05T09:00:00+00:00",
        ingested_at=_iso(INGEST_BASE + timedelta(hours=1)),
        bar_version_id="newer-unambiguous",
        tag="newer",
    )

    for chunk_size in (1, 2, 1000):
        monkeypatch.setattr(
            snapshots_module, "SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE", chunk_size
        )
        for reorder in (lambda rows: rows, lambda rows: list(reversed(rows))):
            with sqlite3.connect(store.database_path) as connection:
                selected = snapshots_module._select_snapshot_receipts(
                    _ReorderingConnection(connection, reorder),
                    provider="coinbase_exchange_rest",
                    product_id="BTC-USD",
                    timeframe="1h",
                    range_start="2026-08-05T09:00:00Z",
                    range_end="2026-08-05T10:00:00Z",
                    as_of=_iso(INGEST_BASE + timedelta(hours=2)),
                )
            assert len(selected) == 1
            assert selected[0].bar_version_id == "newer-unambiguous"


def test_conflict_at_the_maximum_is_detected_across_chunk_boundaries(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    _seed_receipt(
        store.database_path,
        bar_open_at="2026-08-05T09:00:00+00:00",
        ingested_at=_iso(INGEST_BASE - timedelta(hours=1)),
        bar_version_id="superseded",
        tag="old",
    )
    for suffix in ("a", "b"):
        _seed_receipt(
            store.database_path,
            bar_open_at="2026-08-05T09:00:00+00:00",
            ingested_at=_iso(INGEST_BASE),
            bar_version_id=f"max-conflict-{suffix}",
            tag=f"max-{suffix}",
        )

    for chunk_size in (1, 2, 1000):
        monkeypatch.setattr(
            snapshots_module, "SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE", chunk_size
        )
        for reorder in (lambda rows: rows, lambda rows: list(reversed(rows))):
            with sqlite3.connect(store.database_path) as connection:
                with pytest.raises(snapshots_module.SnapshotSelectionConflict) as excinfo:
                    snapshots_module._select_snapshot_receipts(
                        _ReorderingConnection(connection, reorder),
                        provider="coinbase_exchange_rest",
                        product_id="BTC-USD",
                        timeframe="1h",
                        range_start="2026-08-05T09:00:00Z",
                        range_end="2026-08-05T10:00:00Z",
                        as_of=_iso(INGEST_BASE + timedelta(hours=2)),
                    )
            assert excinfo.value.bar_version_ids == ("max-conflict-a", "max-conflict-b")


def test_streaming_selection_matches_the_reference_algorithm(
    tmp_path, store_module, snapshots_module
) -> None:
    """Equivalence oracle: the streaming implementation must agree with the
    plain group-then-max algorithm on a matrix of revision shapes."""
    for case, (openings, revisions) in enumerate(
        ((1, 1), (1, 5), (3, 1), (3, 4), (10, 3), (5, 20))
    ):
        database = tmp_path / f"case-{case}.sqlite3"
        store_module.MarketDataStore(database)
        _seed_grid(database, openings=openings, revisions=revisions, tag=f"case{case}")

        with sqlite3.connect(database) as connection:
            rows = list(
                connection.execute(
                    "SELECT bar_open_at, ingested_at, content_sha256, bar_id, "
                    "bar_version_id, available_at FROM market_bar_receipts"
                )
            )
        expected, conflict = _reference_selection(rows)
        assert conflict is None

        selected = _select_on(
            snapshots_module,
            database,
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=openings)),
            as_of=_iso(INGEST_BASE + timedelta(days=1)),
        )
        assert [
            (entry.bar_open_at, entry.ingested_at, entry.content_sha256, entry.bar_version_id)
            for entry in selected
        ] == [(row[0], row[1], row[2], row[4]) for row in expected], f"case {case}"


def test_selection_result_is_strictly_sorted_by_bar_open_at(
    tmp_path, store_module, snapshots_module
) -> None:
    database = tmp_path / "sorted.sqlite3"
    store_module.MarketDataStore(database)
    _seed_grid(database, openings=25, revisions=3, tag="sorted")

    selected = _select_on(
        snapshots_module,
        database,
        range_start=_iso(GRID_BASE),
        range_end=_iso(GRID_BASE + timedelta(hours=25)),
        as_of=_iso(INGEST_BASE + timedelta(days=1)),
    )

    opens = [entry.bar_open_at for entry in selected]
    assert opens == sorted(opens)
    assert len(opens) == len(set(opens)) == 25


# --- 1C-B SQL and index contract -----------------------------------------


def test_selection_sql_has_no_order_by_and_a_detection_only_limit(snapshots_module) -> None:
    connection = _capture_selection_sql(snapshots_module)
    sql = connection.sql

    assert "ORDER BY" not in sql.upper()
    assert "payload_json" not in sql
    assert "rowid" not in sql.lower()
    # The detection bound is a BOUND PARAMETER, never a literal interpolated
    # at import time: a literal could drift away from the constant the
    # counter reads (F3).
    assert "LIMIT ?" in sql
    assert str(snapshots_module.MAX_SNAPSHOT_ELIGIBLE_RECEIPTS) not in sql
    assert connection.parameters == (
        "coinbase_exchange_rest",
        "BTC-USD",
        "1h",
        "2026-08-05T00:00:00+00:00",
        "2026-08-06T00:00:00+00:00",
        "2026-08-05T23:59:59+00:00",
        snapshots_module.MAX_SNAPSHOT_ELIGIBLE_RECEIPTS + 1,
    )


def test_query_limit_constant_is_exactly_one_above_the_eligibility_limit(
    snapshots_module,
) -> None:
    assert (
        snapshots_module.SNAPSHOT_ELIGIBLE_RECEIPT_QUERY_LIMIT
        == snapshots_module.MAX_SNAPSHOT_ELIGIBLE_RECEIPTS + 1
    )


def test_selection_sql_placeholder_count_matches_the_parameter_count(
    snapshots_module,
) -> None:
    connection = _capture_selection_sql(snapshots_module)

    assert connection.sql.count("?") == len(connection.parameters) == 7


def test_sql_limit_and_row_counter_share_one_source_of_truth(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    """Discriminating probe for F3: moving the single derived bound must move
    the SQL LIMIT and the Python counter together.

    If the SQL carried its own interpolated literal, the query would keep
    returning up to the original bound while the counter refused at the new
    one -- a silent divergence between the detector and the threshold it is
    supposed to detect.
    """
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=1, revisions=6, tag="f3")
    select_kwargs = {
        "range_start": _iso(GRID_BASE),
        "range_end": _iso(GRID_BASE + timedelta(hours=1)),
        "as_of": _iso(INGEST_BASE + timedelta(days=1)),
    }
    # All 6 revisions are eligible with the production bound.
    assert len(_select_on(snapshots_module, store.database_path, **select_kwargs)) == 1

    monkeypatch.setattr(snapshots_module, "SNAPSHOT_ELIGIBLE_RECEIPT_QUERY_LIMIT", 5)

    captured = _CapturingConnection()
    snapshots_module._select_snapshot_receipts(
        captured,
        provider="coinbase_exchange_rest",
        product_id="BTC-USD",
        timeframe="1h",
        **select_kwargs,
    )
    # The SQL now carries the moved bound...
    assert captured.parameters[6] == 5

    # ...and the counter refuses at exactly that same bound, reporting the
    # eligibility limit derived from it rather than a stale constant.
    with pytest.raises(snapshots_module.SnapshotEligibilityLimitExceeded) as excinfo:
        _select_on(snapshots_module, store.database_path, **select_kwargs)
    assert excinfo.value.limit == 4


def test_selection_plan_uses_the_covering_snapshot_index_without_temp_btree(
    tmp_path, store_module, snapshots_module
) -> None:
    database = tmp_path / "plan.sqlite3"
    store_module.MarketDataStore(database)
    _seed_grid(database, openings=50, revisions=3, tag="plan")
    captured = _capture_selection_sql(snapshots_module)

    plan = _query_plan(database, captured.sql, captured.parameters)

    assert re.search(
        rf"\bSEARCH\b\s+r\b\s+USING\s+COVERING\s+INDEX\s+{SNAPSHOT_DOMAIN_INDEX}\b", plan
    ), plan
    assert not re.search(r"\bSCAN\b(?:\s+TABLE)?\s+(?:\S+\s+AS\s+)?r\b", plan), plan
    assert "TEMP B-TREE" not in plan, plan


def test_selection_plan_does_not_depend_on_out_of_range_history(
    tmp_path, store_module, snapshots_module
) -> None:
    captured = _capture_selection_sql(snapshots_module)
    plans = []
    for case, extra_history in enumerate((0, 5000)):
        database = tmp_path / f"history-{case}.sqlite3"
        store_module.MarketDataStore(database)
        _seed_grid(database, openings=24, revisions=2, tag=f"hist{case}")
        if extra_history:
            _seed_grid(
                database,
                openings=extra_history,
                revisions=1,
                first_open=GRID_BASE - timedelta(hours=extra_history + 24),
                tag=f"old{case}",
            )
        plans.append(_query_plan(database, captured.sql, captured.parameters))

    assert plans[0] == plans[1], plans
    assert "TEMP B-TREE" not in plans[1]


def test_store_creates_exactly_one_new_receipt_index_and_none_on_ingestions(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)

    assert _index_names(store.database_path, "market_bar_receipts") == {
        "market_bar_receipts_lookup",
        "market_bar_receipts_open_lookup",
        SNAPSHOT_DOMAIN_INDEX,
    }
    assert _index_names(store.database_path, "market_ingestions") == set()


def test_reopening_an_existing_database_recreates_the_snapshot_index(
    tmp_path, store_module
) -> None:
    """A Phase 1B database created before this sub-gate must gain the index on
    reopen: CREATE INDEX IF NOT EXISTS runs on every MarketDataStore init."""
    store = _store(store_module, tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(f"DROP INDEX {SNAPSHOT_DOMAIN_INDEX}")
    assert SNAPSHOT_DOMAIN_INDEX not in _index_names(store.database_path, "market_bar_receipts")

    store_module.MarketDataStore(store.database_path)

    assert SNAPSHOT_DOMAIN_INDEX in _index_names(store.database_path, "market_bar_receipts")


def test_phase_1b_open_lookup_plan_is_not_hijacked_by_the_new_index(
    tmp_path, store_module
) -> None:
    """The new index must not steal the queries Phase 1B asserts are served by
    market_bar_receipts_open_lookup."""
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-05T09:00:00+00:00")]))

    with sqlite3.connect(store.database_path) as connection:
        plan = " ".join(
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT bar_open_at, ingested_at FROM "
                "market_bar_receipts r WHERE r.bar_open_at = ?",
                ("2026-08-05T09:00:00+00:00",),
            )
        )

    assert "market_bar_receipts_open_lookup" in plan, plan
    assert SNAPSHOT_DOMAIN_INDEX not in plan, plan

"""Phase 1C-A: causal eligibility and deterministic revision selection.

Historical causal snapshots based on declared ingestion time (Contract A):
`as_of` is a cutoff over the DECLARED historical `ingested_at` of existing
receipts, never a live wall-clock, never a promise of lookahead-free
real-time knowledge. This file tests only the pure selection mechanism --
no snapshot manifest is created or persisted anywhere in this file.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
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


# =========================================================================
# Phase 1C-C: atomic, idempotent and immutable materialization of a causal
# snapshot. Storage only -- no public create/load/list API exists yet.
# =========================================================================


def _write_connection(store):
    """A connection carrying the PRAGMAs the write path requires."""
    return store._connect()


def _materialize(snapshots_module, store, **overrides):
    values = {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": _iso(GRID_BASE),
        "range_end": _iso(GRID_BASE + timedelta(hours=3)),
        "as_of": _iso(INGEST_BASE + timedelta(days=1)),
    }
    values.update(overrides)
    connection = _write_connection(store)
    try:
        return snapshots_module._materialize_snapshot(connection, **values)
    finally:
        connection.close()


def _manifests(database: Path) -> list[tuple]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT snapshot_id, snapshot_request_id, entries_content_hash, "
            "snapshot_schema_version, selection_policy_version, provider, product_id, "
            "timeframe, range_start, range_end, as_of, entry_count "
            "FROM market_snapshot_manifests ORDER BY snapshot_id"
        ).fetchall()


def _entries(database: Path, snapshot_id: str | None = None) -> list[tuple]:
    with sqlite3.connect(database) as connection:
        if snapshot_id is None:
            return connection.execute(
                "SELECT snapshot_id, bar_open_at, content_sha256 FROM market_snapshot_entries "
                "ORDER BY snapshot_id, bar_open_at"
            ).fetchall()
        return connection.execute(
            "SELECT bar_open_at, content_sha256 FROM market_snapshot_entries "
            "WHERE snapshot_id = ? ORDER BY bar_open_at",
            (snapshot_id,),
        ).fetchall()


def _seed_manifest(database: Path, *, snapshot_id: str, entries: list[tuple], **overrides) -> None:
    """Persist a snapshot directly (entries first, manifest last) so states
    the primitive would never itself produce -- a missing entry, an extra
    one, a wrong hash -- can be built and detected."""
    values = {
        "snapshot_request_id": "req",
        "entries_content_hash": "e" * 64,
        "snapshot_schema_version": "trading-lab.market-snapshot.v1",
        "selection_policy_version": "trading-lab.market-snapshot-selection.v1",
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": _iso(GRID_BASE),
        "range_end": _iso(GRID_BASE + timedelta(hours=3)),
        "as_of": _iso(INGEST_BASE + timedelta(days=1)),
        "entry_count": len(entries),
    }
    values.update(overrides)
    connection = sqlite3.connect(database)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA recursive_triggers = ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.executemany(
            "INSERT INTO market_snapshot_entries (snapshot_id, bar_open_at, content_sha256) "
            "VALUES (?, ?, ?)",
            [(snapshot_id, bar_open_at, content_sha256) for bar_open_at, content_sha256 in entries],
        )
        connection.execute(
            "INSERT INTO market_snapshot_manifests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id, values["snapshot_request_id"], values["entries_content_hash"],
                values["snapshot_schema_version"], values["selection_policy_version"],
                values["provider"], values["product_id"], values["timeframe"],
                values["range_start"], values["range_end"], values["as_of"],
                values["entry_count"],
            ),
        )
        connection.commit()
    finally:
        connection.close()


def _expected_identities(snapshots_module, store, **overrides) -> tuple[str, str, str, tuple]:
    """Recompute what the primitive must produce, independently of it."""
    values = {
        "provider": "coinbase_exchange_rest",
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "range_start": _iso(GRID_BASE),
        "range_end": _iso(GRID_BASE + timedelta(hours=3)),
        "as_of": _iso(INGEST_BASE + timedelta(days=1)),
    }
    values.update(overrides)
    with sqlite3.connect(store.database_path) as connection:
        selected = snapshots_module._select_snapshot_receipts(connection, **values)
    request_id = snapshots_module.build_snapshot_request_id(**values)
    content_hash = snapshots_module.build_entries_content_hash(selected)
    snapshot_id = snapshots_module.build_snapshot_id(
        snapshot_request_id=request_id, entries_content_hash=content_hash
    )
    return snapshot_id, request_id, content_hash, selected


# --- 1C-C: write-context guard -------------------------------------------


def test_materialization_refuses_a_connection_without_the_write_pragmas(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=1, tag="ctx")

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        with pytest.raises(snapshots_module.SnapshotWriteContextError):
            snapshots_module._materialize_snapshot(
                connection,
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                range_start=_iso(GRID_BASE),
                range_end=_iso(GRID_BASE + timedelta(hours=3)),
                as_of=_iso(INGEST_BASE + timedelta(days=1)),
            )

    assert _manifests(store.database_path) == []
    assert _entries(store.database_path) == []


def test_materialization_refuses_a_connection_already_in_a_transaction(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    connection = _write_connection(store)
    try:
        connection.execute("BEGIN IMMEDIATE")
        assert connection.in_transaction
        with pytest.raises(snapshots_module.SnapshotWriteContextError):
            snapshots_module._materialize_snapshot(
                connection,
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                range_start=_iso(GRID_BASE),
                range_end=_iso(GRID_BASE + timedelta(hours=3)),
                as_of=_iso(INGEST_BASE + timedelta(days=1)),
            )
        connection.rollback()
    finally:
        connection.close()


def test_write_context_is_checked_before_any_query(
    tmp_path, store_module, snapshots_module
) -> None:
    """The guard must fire before the causal selection runs: a bad write
    context must cost nothing and touch nothing."""
    class _Probe:
        def __init__(self) -> None:
            self.statements: list[str] = []
            self.in_transaction = False

        def execute(self, sql, parameters=()):
            self.statements.append(sql)
            if "PRAGMA" in sql.upper():
                class _R:
                    def fetchone(self_inner):
                        return (0,)
                return _R()
            pytest.fail(f"a non-PRAGMA statement ran despite a bad write context: {sql!r}")

    probe = _Probe()
    with pytest.raises(snapshots_module.SnapshotWriteContextError):
        snapshots_module._materialize_snapshot(
            probe,
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=3)),
            as_of=_iso(INGEST_BASE + timedelta(days=1)),
        )
    assert all("PRAGMA" in sql.upper() for sql in probe.statements), probe.statements


# --- 1C-C: creation -------------------------------------------------------


def test_materializing_a_snapshot_persists_manifest_and_entries(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=2, tag="create")
    snapshot_id, request_id, content_hash, selected = _expected_identities(
        snapshots_module, store
    )

    result = _materialize(snapshots_module, store)

    assert result.created is True
    assert result.snapshot_id == snapshot_id
    assert result.snapshot_request_id == request_id
    assert result.entries_content_hash == content_hash
    assert result.entry_count == 3
    assert _manifests(store.database_path) == [
        (
            snapshot_id, request_id, content_hash,
            snapshots_module.SNAPSHOT_SCHEMA_VERSION,
            snapshots_module.SELECTION_POLICY_VERSION,
            "coinbase_exchange_rest", "BTC-USD", "1h",
            _iso(GRID_BASE), _iso(GRID_BASE + timedelta(hours=3)),
            _iso(INGEST_BASE + timedelta(days=1)), 3,
        )
    ]
    assert _entries(store.database_path, snapshot_id) == [
        (entry.bar_open_at, entry.content_sha256) for entry in selected
    ]


def test_materialized_entries_copy_no_market_data(
    tmp_path, store_module, snapshots_module
) -> None:
    """An entry is a reference, never a copy: OHLCV, payload_json, bar_id and
    bar_version_id must live only in the Phase 1B receipt."""
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=2, revisions=1, tag="nocopy")

    _materialize(snapshots_module, store)

    with sqlite3.connect(store.database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(market_snapshot_entries)")
        }
    assert columns == {"snapshot_id", "bar_open_at", "content_sha256"}


def test_materializing_an_empty_range_persists_a_valid_empty_snapshot(
    tmp_path, store_module, snapshots_module
) -> None:
    """An empty snapshot means 'no eligible receipt at this as_of', never a
    confirmed provider gap."""
    store = _store(store_module, tmp_path)
    snapshot_id, _, content_hash, selected = _expected_identities(snapshots_module, store)
    assert selected == ()

    result = _materialize(snapshots_module, store)

    assert result.created is True
    assert result.entry_count == 0
    assert result.entries_content_hash == content_hash
    assert [row[0] for row in _manifests(store.database_path)] == [snapshot_id]
    assert _entries(store.database_path) == []


def test_materializing_ten_thousand_entries_stays_within_the_range_limit(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    openings = snapshots_module.MAX_SNAPSHOT_RANGE_OPENS
    _seed_grid(store.database_path, openings=openings, revisions=1, tag="big")

    result = _materialize(
        snapshots_module, store, range_end=_iso(GRID_BASE + timedelta(hours=openings))
    )

    assert result.created is True
    assert result.entry_count == openings
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM market_snapshot_entries"
        ).fetchone()[0] == openings


# --- 1C-C: idempotence ----------------------------------------------------


def test_second_materialization_of_the_same_request_is_a_no_op(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=2, tag="idem")

    first = _materialize(snapshots_module, store)
    manifests_after_first = _manifests(store.database_path)
    entries_after_first = _entries(store.database_path)
    second = _materialize(snapshots_module, store)

    assert first.created is True
    assert second.created is False
    assert second.snapshot_id == first.snapshot_id
    assert second.entry_count == first.entry_count
    assert _manifests(store.database_path) == manifests_after_first
    assert _entries(store.database_path) == entries_after_first


def test_divergent_persisted_manifest_is_detected_as_corruption(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=2, revisions=1, tag="corrupt")
    snapshot_id, request_id, content_hash, selected = _expected_identities(
        snapshots_module, store
    )
    _seed_manifest(
        store.database_path,
        snapshot_id=snapshot_id,
        entries=[(entry.bar_open_at, entry.content_sha256) for entry in selected],
        snapshot_request_id=request_id,
        entries_content_hash=content_hash,
        timeframe="1d",  # divergent from the request that produced this id
    )

    with pytest.raises(snapshots_module.SnapshotStateCorruption):
        _materialize(snapshots_module, store)


def test_missing_extra_and_wrong_entries_are_all_detected_as_corruption(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=1, tag="entries")
    snapshot_id, request_id, content_hash, selected = _expected_identities(
        snapshots_module, store
    )
    exact = [(entry.bar_open_at, entry.content_sha256) for entry in selected]
    other_sha = [row for row in exact if row[1] != exact[0][1]][0][1]

    corruptions = {
        "missing entry": exact[:-1],
        "extra entry": exact + [(_iso(GRID_BASE + timedelta(hours=9)), exact[0][1])],
        "wrong content_sha256": [(exact[0][0], other_sha)] + exact[1:],
        "wrong bar_open_at": [(_iso(GRID_BASE + timedelta(hours=7)), exact[0][1])] + exact[1:],
    }
    for index, (label, entries) in enumerate(corruptions.items()):
        database = tmp_path / f"corrupt-{index}.sqlite3"
        broken = store_module.MarketDataStore(database)
        _seed_grid(database, openings=3, revisions=1, tag="entries")
        _seed_manifest(
            database,
            snapshot_id=snapshot_id,
            entries=entries,
            snapshot_request_id=request_id,
            entries_content_hash=content_hash,
            entry_count=len(exact),
        )
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            _materialize(snapshots_module, broken)


def test_wrong_persisted_entries_content_hash_is_detected(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=2, revisions=1, tag="hash")
    snapshot_id, request_id, _, selected = _expected_identities(snapshots_module, store)
    _seed_manifest(
        store.database_path,
        snapshot_id=snapshot_id,
        entries=[(entry.bar_open_at, entry.content_sha256) for entry in selected],
        snapshot_request_id=request_id,
        entries_content_hash="f" * 64,
    )

    with pytest.raises(snapshots_module.SnapshotStateCorruption):
        _materialize(snapshots_module, store)


# --- 1C-C: backfill -------------------------------------------------------


def test_backfill_produces_a_new_snapshot_and_leaves_the_old_one_intact(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=1, tag="before")
    first = _materialize(snapshots_module, store)
    first_entries = _entries(store.database_path, first.snapshot_id)

    # A retrodated revision changes the selected content for the same request.
    _seed_grid(
        store.database_path, openings=3, revisions=1, tag="after",
        ingest_base=INGEST_BASE + timedelta(hours=2),
    )
    second = _materialize(snapshots_module, store)

    assert second.created is True
    assert second.snapshot_id != first.snapshot_id
    assert second.entries_content_hash != first.entries_content_hash
    assert second.snapshot_request_id == first.snapshot_request_id
    assert len(_manifests(store.database_path)) == 2
    assert _entries(store.database_path, first.snapshot_id) == first_entries


# --- 1C-C: atomicity ------------------------------------------------------


def test_selection_conflict_persists_nothing(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    for suffix in ("a", "b"):
        _seed_receipt(
            store.database_path,
            bar_open_at=_iso(GRID_BASE),
            ingested_at=_iso(INGEST_BASE),
            bar_version_id=f"conflict-{suffix}",
            tag=f"conf-{suffix}",
        )

    with pytest.raises(snapshots_module.SnapshotSelectionConflict):
        _materialize(snapshots_module, store)

    assert _manifests(store.database_path) == []
    assert _entries(store.database_path) == []


def test_eligibility_limit_persists_nothing(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=2, revisions=4, tag="limit")
    monkeypatch.setattr(snapshots_module, "SNAPSHOT_ELIGIBLE_RECEIPT_QUERY_LIMIT", 3)

    with pytest.raises(snapshots_module.SnapshotEligibilityLimitExceeded):
        _materialize(snapshots_module, store)

    assert _manifests(store.database_path) == []
    assert _entries(store.database_path) == []


def test_failure_while_sealing_rolls_back_every_entry(
    tmp_path, store_module, snapshots_module
) -> None:
    """Entries are inserted before the manifest; if the seal never lands, the
    entries must not survive."""
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=1, tag="rollback")

    class _FailOnManifest:
        def __init__(self, connection):
            self._connection = connection

        @property
        def in_transaction(self):
            return self._connection.in_transaction

        def execute(self, sql, parameters=()):
            if "INSERT INTO market_snapshot_manifests" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._connection.execute(sql, parameters)

        def executemany(self, sql, seq):
            return self._connection.executemany(sql, seq)

        def commit(self):
            return self._connection.commit()

        def rollback(self):
            return self._connection.rollback()

    connection = _write_connection(store)
    try:
        with pytest.raises(snapshots_module.SnapshotPersistenceError):
            snapshots_module._materialize_snapshot(
                _FailOnManifest(connection),
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                range_start=_iso(GRID_BASE),
                range_end=_iso(GRID_BASE + timedelta(hours=3)),
                as_of=_iso(INGEST_BASE + timedelta(days=1)),
            )
        assert connection.in_transaction is False
    finally:
        connection.close()

    assert _manifests(store.database_path) == []
    assert _entries(store.database_path) == []


def test_retry_after_a_rollback_succeeds_cleanly(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    for suffix in ("a", "b"):
        _seed_receipt(
            store.database_path,
            bar_open_at=_iso(GRID_BASE),
            ingested_at=_iso(INGEST_BASE),
            bar_version_id=f"retry-{suffix}",
            tag=f"retry-{suffix}",
        )
    with pytest.raises(snapshots_module.SnapshotSelectionConflict):
        _materialize(snapshots_module, store)

    # A later unambiguous revision resolves the contradiction.
    _seed_receipt(
        store.database_path,
        bar_open_at=_iso(GRID_BASE),
        ingested_at=_iso(INGEST_BASE + timedelta(hours=1)),
        bar_version_id="retry-resolved",
        tag="retry-resolved",
    )
    result = _materialize(snapshots_module, store)

    assert result.created is True
    assert result.entry_count == 1
    assert len(_manifests(store.database_path)) == 1


# --- 1C-C: concurrency ----------------------------------------------------


def test_two_writers_creating_the_same_snapshot_produce_one_manifest(
    tmp_path, store_module, snapshots_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=1, tag="race")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result()
            for future in [
                pool.submit(_materialize, snapshots_module, store),
                pool.submit(_materialize, snapshots_module, store),
            ]
        ]

    assert {result.snapshot_id for result in results} == {results[0].snapshot_id}
    assert sorted(result.created for result in results) == [False, True]
    assert len(_manifests(store.database_path)) == 1
    assert len(_entries(store.database_path)) == 3


# --- 1C-C: guarantees the first mutation round found untested -------------


class _ProxyConnection:
    """Minimal pass-through the write primitive can drive, so a probe can be
    injected at a chosen point of its sequence."""

    def __init__(self, connection) -> None:
        self._connection = connection

    @property
    def in_transaction(self):
        return self._connection.in_transaction

    def execute(self, sql, parameters=()):
        return self._connection.execute(sql, parameters)

    def executemany(self, sql, seq):
        return self._connection.executemany(sql, seq)

    def commit(self):
        return self._connection.commit()

    def rollback(self):
        return self._connection.rollback()


def test_write_lock_is_held_before_the_causal_selection_runs(
    tmp_path, store_module, snapshots_module
) -> None:
    """BEGIN IMMEDIATE, not BEGIN: the write lock must already be held while
    the selection reads, otherwise a concurrent backfill landing between the
    selection and the insert would produce a manifest attesting a state that
    never existed. Probed from a second connection at that exact moment."""
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=2, revisions=1, tag="lock")
    observed: list[str] = []

    class _ProbeDuringSelection(_ProxyConnection):
        def execute(self, sql, parameters=()):
            if "FROM market_bar_receipts r" in sql and not observed:
                other = sqlite3.connect(store.database_path, timeout=0.2)
                try:
                    other.execute("PRAGMA busy_timeout = 200")
                    try:
                        other.execute("BEGIN IMMEDIATE")
                        observed.append("write lock was NOT held")
                        other.rollback()
                    except sqlite3.OperationalError:
                        observed.append("write lock held")
                finally:
                    other.close()
            return super().execute(sql, parameters)

    connection = _write_connection(store)
    try:
        snapshots_module._materialize_snapshot(
            _ProbeDuringSelection(connection),
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=3)),
            as_of=_iso(INGEST_BASE + timedelta(days=1)),
        )
    finally:
        connection.close()

    assert observed == ["write lock held"], observed


def test_snapshot_inserts_never_suppress_a_conflict(snapshots_module) -> None:
    """Idempotence is proven by verifying the persisted state, never by
    letting SQLite swallow a conflicting insert: OR IGNORE would turn a real
    collision into a silent partial write, OR REPLACE would breach
    immutability."""
    for sql in (snapshots_module._INSERT_ENTRY_SQL, snapshots_module._INSERT_MANIFEST_SQL):
        upper = sql.upper()
        assert "OR IGNORE" not in upper, sql
        assert "OR REPLACE" not in upper, sql
        assert "OR FAIL" not in upper, sql


def _foreign_receipt(database: Path, *, product_id: str) -> tuple[str, str]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT r.content_sha256, r.bar_open_at FROM market_bar_receipts r "
            "JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id "
            "WHERE i.product_id = ? LIMIT 1",
            (product_id,),
        ).fetchone()


def test_post_insert_validation_rejects_a_receipt_from_another_domain(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    """The pre-commit validation exists to catch a selection bug, so it is
    tested against one: the selection is forced to return a receipt that
    belongs to a different product."""
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=1, revisions=1, tag="own")
    _seed_grid(store.database_path, openings=1, revisions=1, tag="foreign", product_id="ETH-USD")
    content_sha256, bar_open_at = _foreign_receipt(store.database_path, product_id="ETH-USD")
    intruder = snapshots_module.SelectedSnapshotReceipt(
        bar_open_at=bar_open_at, content_sha256=content_sha256,
        bar_id="", bar_version_id="", ingested_at="", available_at="",
    )
    monkeypatch.setattr(
        snapshots_module, "_select_snapshot_receipts", lambda *args, **kwargs: (intruder,)
    )

    with pytest.raises(snapshots_module.SnapshotPersistenceError):
        _materialize(snapshots_module, store)

    assert _manifests(store.database_path) == []
    assert _entries(store.database_path) == []


def test_post_insert_validation_rejects_an_entry_whose_opening_does_not_match(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=1, revisions=1, tag="mismatch")
    with sqlite3.connect(store.database_path) as connection:
        content_sha256 = connection.execute(
            "SELECT content_sha256 FROM market_bar_receipts LIMIT 1"
        ).fetchone()[0]
    mismatched = snapshots_module.SelectedSnapshotReceipt(
        bar_open_at=_iso(GRID_BASE + timedelta(hours=2)),  # not this receipt's opening
        content_sha256=content_sha256,
        bar_id="", bar_version_id="", ingested_at="", available_at="",
    )
    monkeypatch.setattr(
        snapshots_module, "_select_snapshot_receipts", lambda *args, **kwargs: (mismatched,)
    )

    with pytest.raises(snapshots_module.SnapshotPersistenceError):
        _materialize(snapshots_module, store)

    assert _manifests(store.database_path) == []
    assert _entries(store.database_path) == []


# --- 1C-C / F1: rollback coverage of the non-sqlite3 handlers -------------


def _write_lock_is_released(database: Path) -> bool:
    """Prove the BEGIN IMMEDIATE write lock is actually gone, on a real file
    database. `connection.in_transaction` alone is a Python-side flag: it
    would not reveal a lock still held against every other writer."""
    other = sqlite3.connect(database, timeout=0.3)
    try:
        other.execute("PRAGMA busy_timeout = 300")
        try:
            other.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError:
            return False
        other.rollback()
        return True
    finally:
        other.close()


@pytest.mark.parametrize("scenario", ["selection_conflict", "state_corruption"])
def test_business_failure_rolls_back_and_releases_the_write_lock(
    tmp_path, store_module, snapshots_module, scenario
) -> None:
    """Both business failures travel through `except MarketSnapshotError`,
    which must roll back before re-raising. Without that rollback the
    exception still surfaces unchanged and nothing is persisted -- so a test
    checking only the exception passes -- while the transaction stays open
    and the write lock is held against every other writer, forever."""
    store = _store(store_module, tmp_path)
    select_kwargs = {
        "range_start": _iso(GRID_BASE),
        "range_end": _iso(GRID_BASE + timedelta(hours=3)),
        "as_of": _iso(INGEST_BASE + timedelta(days=1)),
    }

    if scenario == "selection_conflict":
        for suffix in ("a", "b"):
            _seed_receipt(
                store.database_path,
                bar_open_at=_iso(GRID_BASE),
                ingested_at=_iso(INGEST_BASE),
                bar_version_id=f"f1-conflict-{suffix}",
                tag=f"f1-conf-{suffix}",
            )
        expected_error = snapshots_module.SnapshotSelectionConflict
    else:
        _seed_grid(store.database_path, openings=2, revisions=1, tag="f1corrupt")
        snapshot_id, request_id, content_hash, selected = _expected_identities(
            snapshots_module, store
        )
        _seed_manifest(
            store.database_path,
            snapshot_id=snapshot_id,
            entries=[(entry.bar_open_at, entry.content_sha256) for entry in selected],
            snapshot_request_id=request_id,
            entries_content_hash=content_hash,
            timeframe="1d",  # divergent from the request that produced this id
        )
        expected_error = snapshots_module.SnapshotStateCorruption

    manifests_before = _manifests(store.database_path)
    entries_before = _entries(store.database_path)

    connection = _write_connection(store)
    try:
        with pytest.raises(expected_error) as excinfo:
            snapshots_module._materialize_snapshot(
                connection,
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                **select_kwargs,
            )
        # The business exception must reach the caller unchanged, never
        # wrapped into a persistence error.
        assert type(excinfo.value) is expected_error
        assert connection.in_transaction is False
        assert _write_lock_is_released(store.database_path)

        # Nothing partial: the database holds exactly what it held before.
        assert _manifests(store.database_path) == manifests_before
        assert _entries(store.database_path) == entries_before

        # The same connection is still usable: it reaches the same business
        # failure again instead of a write-context refusal.
        with pytest.raises(expected_error):
            snapshots_module._materialize_snapshot(
                connection,
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                **select_kwargs,
            )
        assert connection.in_transaction is False
    finally:
        connection.close()


def test_unexpected_error_after_writes_rolls_back_and_releases_the_write_lock(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    """An error the primitive does not model at all travels through the
    generic `except Exception` handler. Injected late, once entries AND the
    manifest are already written inside the open transaction, so the
    rollback has real uncommitted state to undo."""
    store = _store(store_module, tmp_path)
    _seed_grid(store.database_path, openings=3, revisions=1, tag="f1generic")
    observed: dict[str, int] = {}

    def _raise_after_writes(connection, **_kwargs):
        # Read inside the still-open transaction: proves the injection point
        # is genuinely after the writes, not before them.
        observed["entries"] = connection.execute(
            "SELECT COUNT(*) FROM market_snapshot_entries"
        ).fetchone()[0]
        observed["manifests"] = connection.execute(
            "SELECT COUNT(*) FROM market_snapshot_manifests"
        ).fetchone()[0]
        raise RuntimeError("injected failure after uncommitted writes")

    monkeypatch.setattr(snapshots_module, "_verify_existing_snapshot", _raise_after_writes)

    connection = _write_connection(store)
    try:
        with pytest.raises(RuntimeError) as excinfo:
            snapshots_module._materialize_snapshot(
                connection,
                provider="coinbase_exchange_rest",
                product_id="BTC-USD",
                timeframe="1h",
                range_start=_iso(GRID_BASE),
                range_end=_iso(GRID_BASE + timedelta(hours=3)),
                as_of=_iso(INGEST_BASE + timedelta(days=1)),
            )

        assert observed == {"entries": 3, "manifests": 1}, observed
        # The generic handler re-raises unchanged: not wrapped, so no cause.
        assert type(excinfo.value) is RuntimeError
        assert not isinstance(excinfo.value, snapshots_module.MarketSnapshotError)
        assert excinfo.value.__cause__ is None

        assert connection.in_transaction is False
        assert _write_lock_is_released(store.database_path)
        assert _manifests(store.database_path) == []
        assert _entries(store.database_path) == []

        # Once the injected fault is gone the very same connection works.
        monkeypatch.undo()
        result = snapshots_module._materialize_snapshot(
            connection,
            provider="coinbase_exchange_rest",
            product_id="BTC-USD",
            timeframe="1h",
            range_start=_iso(GRID_BASE),
            range_end=_iso(GRID_BASE + timedelta(hours=3)),
            as_of=_iso(INGEST_BASE + timedelta(days=1)),
        )
        assert result.created is True
        assert result.entry_count == 3
    finally:
        connection.close()


# =========================================================================
# Phase 1C-D: verified loading, bounded listing and offline replay.
# =========================================================================


def _one_snapshot(store_module, snapshots_module, tmp_path, *, openings=3, name="load"):
    """A store holding one materialized snapshot built from REAL MarketBar
    payloads, so replay can verify content_sha256 against payload_json."""
    database = tmp_path / f"{name}.sqlite3"
    if database.exists():
        database.unlink()
    store = store_module.MarketDataStore(database)
    connection = store._connect()
    ingested = INGEST_BASE + timedelta(days=800)
    connection.execute(
        "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
        ("a" * 64, "coinbase_exchange_rest", b"x", 1, _iso(GRID_BASE)),
    )
    connection.execute(
        "INSERT INTO market_ingestions VALUES ('i0', 's', 'coinbase_exchange_rest', "
        "'BTC-USD', '1h', ?, ?, ?, '{}', ?)",
        (_iso(ingested), _iso(ingested), "a" * 64, openings),
    )
    market_bar = importlib.import_module("scripts.trading_lab.market_bar")
    for index in range(openings):
        bar = market_bar.build_market_bar(
            asset="BTC/USD", venue="coinbase_exchange", provider="coinbase_exchange",
            timeframe="1h",
            bar_open_at=GRID_BASE + timedelta(hours=index),
            bar_close_at=GRID_BASE + timedelta(hours=index + 1),
            available_at=ingested, ingested_at=ingested,
            open_price="64123.12345678901", high_price="64200.00000000004",
            low_price="64000.00000000001", close_price="64150.50000000005",
            volume="12.345678901234567891", raw_payload_sha256="a" * 64,
        )
        connection.execute(
            "INSERT INTO market_bar_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                bar["content_sha256"], "i0", bar["bar_id"], bar["bar_version_id"],
                bar["bar_open_at"], bar["bar_close_at"], bar["available_at"],
                bar["ingested_at"], "a" * 64,
                json.dumps(bar, sort_keys=True, separators=(",", ":")),
            ),
        )
    connection.commit()
    result = snapshots_module._materialize_snapshot(
        connection,
        provider="coinbase_exchange_rest", product_id="BTC-USD", timeframe="1h",
        range_start=_iso(GRID_BASE),
        range_end=_iso(GRID_BASE + timedelta(hours=max(openings, 1))),
        as_of=_iso(ingested + timedelta(days=1)),
    )
    connection.close()
    return store, result


class _ForbiddenBusinessConnection:
    """Allows PRAGMA reads, fails the test on any business statement."""

    def __init__(self) -> None:
        self.in_transaction = False
        self.statements: list[str] = []

    def execute(self, sql, parameters=()):
        self.statements.append(sql)
        upper = sql.strip().upper()
        if upper.startswith("PRAGMA"):
            class _R:
                def fetchone(self_inner):
                    return (1,)
            return _R()
        pytest.fail(f"a business statement ran on invalid input: {sql!r}")


# --- 1C-D: input validation ----------------------------------------------


@pytest.mark.parametrize(
    "bad",
    [None, b"a" * 64, 12345, "", "a" * 64, "hyprl-market-snapshot-" + "A" * 64,
     "hyprl-market-snapshot-" + "a" * 63, "hyprl-market-snapshot-" + "a" * 65,
     "hyprl-market-snapshot-" + "g" * 64, " hyprl-market-snapshot-" + "a" * 64,
     "hyprl-market-snapshot-request-" + "a" * 64],
)
def test_load_snapshot_rejects_invalid_snapshot_id_before_any_query(
    snapshots_module, bad
) -> None:
    connection = _ForbiddenBusinessConnection()
    with pytest.raises(snapshots_module.SnapshotInputError):
        snapshots_module.load_snapshot(connection, snapshot_id=bad)
    assert connection.in_transaction is False


@pytest.mark.parametrize("bad", [None, 12345, "", "hyprl-market-snapshot-" + "a" * 64,
                                 "hyprl-market-snapshot-request-" + "A" * 64])
def test_list_rejects_invalid_request_id_before_any_query(snapshots_module, bad) -> None:
    connection = _ForbiddenBusinessConnection()
    with pytest.raises(snapshots_module.SnapshotInputError):
        snapshots_module.list_snapshot_manifests(connection, snapshot_request_id=bad)


@pytest.mark.parametrize("bad", [True, False, 1.0, "10", 0, -1, 1001])
def test_list_rejects_invalid_limit_before_any_query(snapshots_module, bad) -> None:
    connection = _ForbiddenBusinessConnection()
    with pytest.raises(snapshots_module.SnapshotInputError):
        snapshots_module.list_snapshot_manifests(
            connection,
            snapshot_request_id="hyprl-market-snapshot-request-" + "a" * 64,
            limit=bad,
        )


def test_list_rejects_invalid_cursor_before_any_query(snapshots_module) -> None:
    connection = _ForbiddenBusinessConnection()
    with pytest.raises(snapshots_module.SnapshotInputError):
        snapshots_module.list_snapshot_manifests(
            connection,
            snapshot_request_id="hyprl-market-snapshot-request-" + "a" * 64,
            after_snapshot_id="not-a-snapshot-id",
        )


# --- 1C-D: public dataclasses --------------------------------------------


def test_public_dataclasses_are_frozen_and_minimal(tmp_path, store_module, snapshots_module) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="dc")
    connection = store._connect()
    try:
        loaded = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()

    import dataclasses

    assert dataclasses.is_dataclass(loaded)
    entry = loaded.entries[0]
    # SnapshotEntryRef must expose the reference and nothing internal.
    assert [field.name for field in dataclasses.fields(entry)] == ["bar_open_at", "content_sha256"]
    for internal in ("bar_id", "bar_version_id", "ingested_at", "available_at",
                     "provider", "product_id", "timeframe", "payload_json"):
        assert not hasattr(entry, internal), internal
    with pytest.raises(dataclasses.FrozenInstanceError):
        entry.bar_open_at = "x"
    with pytest.raises(dataclasses.FrozenInstanceError):
        loaded.manifest.provider = "x"
    assert [field.name for field in dataclasses.fields(loaded.manifest)] == [
        "snapshot_id", "snapshot_request_id", "entries_content_hash",
        "snapshot_schema_version", "selection_policy_version", "provider",
        "product_id", "timeframe", "range_start", "range_end", "as_of", "entry_count",
    ]


# --- 1C-D: load nominal ---------------------------------------------------


def test_load_snapshot_returns_manifest_and_ordered_entries(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=5, name="ord")
    connection = store._connect()
    try:
        loaded = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
        again = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()

    assert loaded.manifest.snapshot_id == result.snapshot_id
    assert loaded.manifest.entry_count == 5
    assert loaded.manifest.provider == "coinbase_exchange_rest"
    opens = [entry.bar_open_at for entry in loaded.entries]
    assert opens == sorted(opens) and len(opens) == 5
    assert again == loaded  # connection reusable, deterministic


def test_load_snapshot_of_an_empty_snapshot_returns_no_entries(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=0, name="empty")
    connection = store._connect()
    try:
        loaded = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()
    assert loaded.entries == ()
    assert loaded.manifest.entry_count == 0


def test_load_snapshot_raises_not_found_for_an_unknown_snapshot(
    tmp_path, store_module, snapshots_module
) -> None:
    store, _ = _one_snapshot(store_module, snapshots_module, tmp_path, name="nf")
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotNotFound):
            snapshots_module.load_snapshot(
                connection, snapshot_id="hyprl-market-snapshot-" + "b" * 64
            )
        assert connection.in_transaction is False
    finally:
        connection.close()


def test_load_snapshot_never_reads_a_payload(tmp_path, store_module, snapshots_module) -> None:
    """load_snapshot returns references only: touching payload_json would make
    every load pay the replay cost."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="nopay")
    seen: list[str] = []

    class _Spy:
        def __init__(self, inner): self._inner = inner
        @property
        def in_transaction(self): return self._inner.in_transaction
        def execute(self, sql, parameters=()):
            seen.append(sql)
            return self._inner.execute(sql, parameters)
        def commit(self): return self._inner.commit()
        def rollback(self): return self._inner.rollback()

    connection = store._connect()
    try:
        snapshots_module.load_snapshot(_Spy(connection), snapshot_id=result.snapshot_id)
    finally:
        connection.close()
    assert seen and all("payload_json" not in sql for sql in seen), seen


# --- 1C-D: read context ---------------------------------------------------


def test_read_path_refuses_a_connection_already_in_a_transaction(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="ctx")
    connection = store._connect()
    try:
        connection.execute("BEGIN")
        with pytest.raises(snapshots_module.SnapshotReadContextError):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        connection.rollback()
    finally:
        connection.close()


def test_read_path_refuses_a_connection_without_health_pragmas(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="prag")
    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 0
        with pytest.raises(snapshots_module.SnapshotReadContextError):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)


# --- 1C-D: listing --------------------------------------------------------


def test_list_returns_an_empty_page_for_an_unknown_request(
    tmp_path, store_module, snapshots_module
) -> None:
    store, _ = _one_snapshot(store_module, snapshots_module, tmp_path, name="lst0")
    connection = store._connect()
    try:
        page = snapshots_module.list_snapshot_manifests(
            connection, snapshot_request_id="hyprl-market-snapshot-request-" + "c" * 64
        )
    finally:
        connection.close()
    assert page.items == ()
    assert page.next_after_snapshot_id is None


def test_list_paginates_by_keyset_without_duplicates_or_gaps(
    tmp_path, store_module, snapshots_module
) -> None:
    """Several materializations of the SAME request (backfill) must paginate
    exactly once each, ordered lexicographically by snapshot_id."""
    store, first = _one_snapshot(store_module, snapshots_module, tmp_path, openings=3, name="lstn")
    connection = store._connect()
    ingested = INGEST_BASE + timedelta(days=800)
    market_bar = importlib.import_module("scripts.trading_lab.market_bar")
    expected = {first.snapshot_id}
    try:
        for revision in range(1, 4):
            later = ingested + timedelta(hours=revision)
            connection.execute(
                "INSERT INTO market_ingestions VALUES (?, 's', 'coinbase_exchange_rest', "
                "'BTC-USD', '1h', ?, ?, ?, '{}', 3)",
                (f"i{revision}", _iso(later), _iso(later), "a" * 64),
            )
            for index in range(3):
                bar = market_bar.build_market_bar(
                    asset="BTC/USD", venue="coinbase_exchange", provider="coinbase_exchange",
                    timeframe="1h",
                    bar_open_at=GRID_BASE + timedelta(hours=index),
                    bar_close_at=GRID_BASE + timedelta(hours=index + 1),
                    available_at=later, ingested_at=later,
                    open_price=f"{64123 + revision}.12345678901",
                    high_price="64200.00000000004", low_price="64000.00000000001",
                    close_price="64150.50000000005", volume="12.345678901234567891",
                    raw_payload_sha256="a" * 64,
                )
                connection.execute(
                    "INSERT INTO market_bar_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (bar["content_sha256"], f"i{revision}", bar["bar_id"], bar["bar_version_id"],
                     bar["bar_open_at"], bar["bar_close_at"], bar["available_at"],
                     bar["ingested_at"], "a" * 64,
                     json.dumps(bar, sort_keys=True, separators=(",", ":"))),
                )
            connection.commit()
            expected.add(
                snapshots_module._materialize_snapshot(
                    connection,
                    provider="coinbase_exchange_rest", product_id="BTC-USD", timeframe="1h",
                    range_start=_iso(GRID_BASE), range_end=_iso(GRID_BASE + timedelta(hours=3)),
                    as_of=_iso(ingested + timedelta(days=1)),
                ).snapshot_id
            )

        collected: list[str] = []
        cursor = None
        pages = 0
        while True:
            page = snapshots_module.list_snapshot_manifests(
                connection,
                snapshot_request_id=first.snapshot_request_id,
                after_snapshot_id=cursor,
                limit=2,
            )
            pages += 1
            assert len(page.items) <= 2
            collected.extend(item.snapshot_id for item in page.items)
            cursor = page.next_after_snapshot_id
            if cursor is None:
                break
            assert cursor == page.items[-1].snapshot_id
    finally:
        connection.close()

    assert len(expected) == 4
    assert collected == sorted(expected)
    assert len(collected) == len(set(collected))
    assert pages == 2


def test_list_sql_uses_no_offset_rowid_or_chronology(snapshots_module) -> None:
    for sql in (snapshots_module._LIST_MANIFESTS_SQL,
                snapshots_module._LIST_MANIFESTS_AFTER_SQL):
        upper = sql.upper()
        assert "OFFSET" not in upper
        assert "ROWID" not in upper
        assert "MAX(" not in upper
        assert "LATEST" not in upper
        assert "ORDER BY SNAPSHOT_ID ASC" in " ".join(upper.split())


# --- 1C-D: replay ---------------------------------------------------------


def test_replay_returns_exact_ordered_market_bars(tmp_path, store_module, snapshots_module) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=4, name="rep")
    connection = store._connect()
    try:
        bars = snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
    finally:
        connection.close()

    assert len(bars) == 4
    assert [bar["bar_open_at"] for bar in bars] == sorted(bar["bar_open_at"] for bar in bars)
    market_bar = importlib.import_module("scripts.trading_lab.market_bar")
    for bar in bars:
        assert bar["schema_version"] == market_bar.SCHEMA_VERSION
        unsigned = {key: value for key, value in bar.items() if key != "content_sha256"}
        assert hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest() == bar["content_sha256"]


def test_replay_of_an_empty_snapshot_returns_an_empty_tuple(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=0, name="rep0")
    connection = store._connect()
    try:
        assert snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id) == ()
    finally:
        connection.close()


# --- 1C-D: amendment R1 ---------------------------------------------------


def test_read_transaction_is_closed_before_python_validation(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    """The SQL capture must end before the CPU-bound validation starts: in
    journal_mode=delete a read transaction blocks every writer's COMMIT, so
    holding it through JSON decoding would stall writers for the whole replay."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=4, name="r1")
    observed: dict[str, object] = {}
    original = snapshots_module._validate_captured_snapshot

    def _slow_validation(captured, **kwargs):
        observed["in_transaction"] = kwargs["connection_probe"].in_transaction \
            if "connection_probe" in kwargs else None
        return original(captured, **kwargs)

    connection = store._connect()
    writer = store._connect()
    try:
        real_decode = snapshots_module._decode_market_bar

        def _slow_decode(*args, **kwargs):
            # While decoding, the reader must hold no transaction and a writer
            # must be able to take AND commit the write lock.
            observed["reader_in_transaction"] = connection.in_transaction
            if "writer_committed" not in observed:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
                    ("f" * 64, "coinbase_exchange_rest", b"y", 1, _iso(GRID_BASE)),
                )
                writer.commit()
                observed["writer_committed"] = True
            return real_decode(*args, **kwargs)

        monkeypatch.setattr(snapshots_module, "_decode_market_bar", _slow_decode)
        bars = snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()
        writer.close()

    assert observed["reader_in_transaction"] is False
    assert observed["writer_committed"] is True
    assert len(bars) == 4


# --- 1C-D: budget --------------------------------------------------------


def test_payload_budget_is_preflighted_before_any_payload_query(
    tmp_path, store_module, snapshots_module, monkeypatch
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=4, name="bud")
    monkeypatch.setattr(snapshots_module, "MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES", 10)
    seen: list[str] = []

    class _Spy:
        def __init__(self, inner): self._inner = inner
        @property
        def in_transaction(self): return self._inner.in_transaction
        def execute(self, sql, parameters=()):
            seen.append(sql)
            return self._inner.execute(sql, parameters)
        def commit(self): return self._inner.commit()
        def rollback(self): return self._inner.rollback()

    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotReplayLimitExceeded) as excinfo:
            snapshots_module.replay_snapshot(_Spy(connection), snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
    finally:
        connection.close()

    # The preflight may name payload_json inside length(CAST(... AS BLOB)) --
    # that counts bytes in SQLite without transferring them. What must never
    # run is the query that actually ships the payloads to Python.
    assert snapshots_module._SELECT_SNAPSHOT_PAYLOADS_SQL not in seen, seen
    assert any("length(CAST(r.payload_json AS BLOB))" in sql for sql in seen), seen
    message = str(excinfo.value)
    assert str(snapshots_module.MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES) in message
    assert "{" not in message and "schema_version" not in message


def test_payload_budget_counts_utf8_bytes_not_characters(snapshots_module) -> None:
    text = "é" * 10  # 10 characters, 20 UTF-8 bytes
    assert snapshots_module._payload_byte_length(text) == 20
    assert snapshots_module._payload_byte_length(text) != len(text)


# --- 1C-D: corruption -----------------------------------------------------


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


@pytest.mark.parametrize(
    "label, statements",
    [
        ("receipt manquant", [("DELETE FROM market_bar_receipts WHERE bar_open_at = "
                               "(SELECT MIN(bar_open_at) FROM market_bar_receipts)", ())]),
        ("ingestion manquante", [("DELETE FROM market_ingestions WHERE ingestion_id = 'i0'", ())]),
        ("entry manquante", [("DELETE FROM market_snapshot_entries WHERE bar_open_at = "
                              "(SELECT MAX(bar_open_at) FROM market_snapshot_entries)", ())]),
        ("mauvais provider", [("UPDATE market_ingestions SET provider = 'other_rest'", ())]),
        ("mauvais product_id", [("UPDATE market_ingestions SET product_id = 'ETH-USD'", ())]),
        ("mauvais timeframe", [("UPDATE market_ingestions SET timeframe = '1d'", ())]),
        ("entry_count faux", [("UPDATE market_snapshot_manifests SET entry_count = 99", ())]),
        ("entries_content_hash faux",
         [("UPDATE market_snapshot_manifests SET entries_content_hash = ?", ("f" * 64,))]),
        ("ingested_at > as_of",
         [("UPDATE market_bar_receipts SET ingested_at = '2999-01-01T00:00:00+00:00'", ())]),
        ("available_at > as_of",
         [("UPDATE market_bar_receipts SET available_at = '2999-01-01T00:00:00+00:00'", ())]),
    ],
)
def test_structural_corruption_is_detected_by_load(
    tmp_path, store_module, snapshots_module, label, statements
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name=f"c{abs(hash(label))%9999}")
    _sabotage(store.database_path, statements)
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        assert connection.in_transaction is False
    finally:
        connection.close()


def test_unknown_manifest_version_is_refused_without_conversion(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="ver")
    _sabotage(store.database_path,
              [("UPDATE market_snapshot_manifests SET snapshot_schema_version = ?",
                ("trading-lab.market-snapshot.v99",))])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotUnsupportedVersion):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def test_unknown_selection_policy_is_refused(tmp_path, store_module, snapshots_module) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="pol")
    _sabotage(store.database_path,
              [("UPDATE market_snapshot_manifests SET selection_policy_version = ?", ("",))])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotUnsupportedVersion):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def test_replay_detects_an_invalid_payload_and_returns_nothing(
    tmp_path, store_module, snapshots_module
) -> None:
    """A corrupted LAST bar must prevent every earlier bar from surfacing."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=4, name="badp")
    _sabotage(store.database_path,
              [("UPDATE market_bar_receipts SET payload_json = '{\"broken\":' "
                "WHERE bar_open_at = (SELECT MAX(bar_open_at) FROM market_bar_receipts)", ())])
    connection = store._connect()
    outcome = "not-set"
    try:
        try:
            outcome = snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
        except snapshots_module.SnapshotStateCorruption as exc:
            assert exc.__cause__ is not None
        assert connection.in_transaction is False
    finally:
        connection.close()
    assert outcome == "not-set"


def test_replay_detects_a_semantically_invalid_bar_with_a_consistent_hash(
    tmp_path, store_module, snapshots_module
) -> None:
    """Rehashing a tampered payload defeats hash-only checking: the bar is
    rebuilt through MarketBar V1 itself, so a bad venue is still caught."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="tamper")
    with sqlite3.connect(store.database_path) as connection:
        sha, payload = connection.execute(
            "SELECT content_sha256, payload_json FROM market_bar_receipts LIMIT 1"
        ).fetchone()
    record = json.loads(payload)
    record["venue"] = "INVALID VENUE!"
    unsigned = {key: value for key, value in record.items() if key != "content_sha256"}
    record["content_sha256"] = hashlib.sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    _sabotage(store.database_path,
              [("UPDATE market_bar_receipts SET payload_json = ? WHERE content_sha256 = ?",
                (json.dumps(record, sort_keys=True, separators=(",", ":")), sha))])

    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption):
            snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


# --- 1C-D: plans and ordering --------------------------------------------


def test_read_query_plans_use_the_expected_indexes(tmp_path, store_module, snapshots_module) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="plan1cd")
    with sqlite3.connect(store.database_path) as connection:
        plans = {
            "manifest": " ".join(
                row[3] for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + snapshots_module._SELECT_SNAPSHOT_MANIFEST_SQL,
                    (result.snapshot_id,))),
            "listing": " ".join(
                row[3] for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + snapshots_module._LIST_MANIFESTS_AFTER_SQL,
                    (result.snapshot_request_id, "", 2))),
            "metadata": " ".join(
                row[3] for row in connection.execute(
                    "EXPLAIN QUERY PLAN " + snapshots_module._SELECT_SNAPSHOT_METADATA_SQL,
                    (result.snapshot_id,))),
        }
    assert "sqlite_autoindex_market_snapshot_manifests_1" in plans["manifest"]
    assert "market_snapshot_manifests_request_lookup" in plans["listing"]
    assert "SEARCH e USING PRIMARY KEY" in plans["metadata"]
    for name, plan in plans.items():
        assert "TEMP B-TREE" not in plan, (name, plan)
        assert "SCAN market_snapshot_manifests" not in plan, (name, plan)


def test_results_are_unchanged_under_reverse_unordered_selects(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=6, name="rev")
    connection = store._connect()
    try:
        normal_load = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        normal_replay = snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
        connection.execute("PRAGMA reverse_unordered_selects = ON")
        reversed_load = snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
        reversed_replay = snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
        connection.execute("PRAGMA reverse_unordered_selects = OFF")
    finally:
        connection.close()
    assert reversed_load == normal_load
    assert reversed_replay == normal_replay


def test_read_path_uses_fetchmany_and_never_fetchall(snapshots_module) -> None:
    import inspect
    capture = inspect.getsource(snapshots_module._capture_snapshot)
    drain = inspect.getsource(snapshots_module._fetch_all_bounded)
    listing = inspect.getsource(snapshots_module.list_snapshot_manifests)
    for source in (capture, drain, listing):
        assert "fetchall(" not in source, source
    assert "fetchmany(" in drain
    assert "_fetch_all_bounded(" in capture and "_fetch_all_bounded(" in listing


# --- 1C-D: guarantees the first mutation round found untested -------------


def test_missing_receipt_is_reported_as_a_broken_reference(
    tmp_path, store_module, snapshots_module
) -> None:
    """An INNER JOIN would drop the row and surface a mere count mismatch;
    the LEFT JOIN must name the broken reference itself."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="mref")
    _sabotage(store.database_path, [("DELETE FROM market_bar_receipts WHERE bar_open_at = "
                                     "(SELECT MIN(bar_open_at) FROM market_bar_receipts)", ())])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption, match="missing receipt"):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def test_missing_ingestion_is_reported_as_a_broken_reference(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="mking")
    _sabotage(store.database_path, [("DELETE FROM market_ingestions WHERE ingestion_id = 'i0'", ())])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption, match="ingestion is"):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def test_reading_never_takes_the_write_lock(tmp_path, store_module, snapshots_module) -> None:
    """BEGIN, never BEGIN IMMEDIATE: a reader must succeed while a writer
    already holds the write lock, reading the pre-existing state."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="nolock")
    writer = store._connect()
    reader = store._connect()
    try:
        reader.execute("PRAGMA busy_timeout = 300")
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("e" * 64, "coinbase_exchange_rest", b"z", 1, _iso(GRID_BASE)),
        )
        loaded = snapshots_module.load_snapshot(reader, snapshot_id=result.snapshot_id)
        assert loaded.manifest.snapshot_id == result.snapshot_id
        writer.rollback()
    finally:
        reader.close()
        writer.close()


def test_tampered_manifest_request_id_is_detected(
    tmp_path, store_module, snapshots_module
) -> None:
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="treq")
    _sabotage(store.database_path,
              [("UPDATE market_snapshot_manifests SET snapshot_request_id = ?",
                ("hyprl-market-snapshot-request-" + "9" * 64,))])
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption,
                           match="snapshot_request_id"):
            snapshots_module.load_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


def test_manifest_whose_snapshot_id_does_not_hash_its_own_identities_is_detected(
    tmp_path, store_module, snapshots_module
) -> None:
    """Only the snapshot_id recomputation catches this: the manifest is
    internally consistent apart from its own identifier."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="tid")
    forged = "hyprl-market-snapshot-" + "7" * 64
    connection = sqlite3.connect(store.database_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        row = connection.execute(
            "SELECT * FROM market_snapshot_manifests WHERE snapshot_id = ?", (result.snapshot_id,)
        ).fetchone()
        connection.executemany(
            "INSERT INTO market_snapshot_entries (snapshot_id, bar_open_at, content_sha256) "
            "VALUES (?, ?, ?)",
            [(forged, bar_open_at, content_sha256) for bar_open_at, content_sha256 in
             connection.execute("SELECT bar_open_at, content_sha256 FROM market_snapshot_entries "
                                "WHERE snapshot_id = ?", (result.snapshot_id,)).fetchall()],
        )
        connection.execute(
            "INSERT INTO market_snapshot_manifests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (forged, *row[1:]),
        )
        connection.commit()
    finally:
        connection.close()

    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption, match="identity"):
            snapshots_module.load_snapshot(connection, snapshot_id=forged)
    finally:
        connection.close()


def _replace_payload(database: Path, *, mutate) -> None:
    """Forge a fully self-consistent snapshot around a mutated bar: receipt
    hash, entry reference and manifest identities are ALL recomputed, so only
    a real MarketBar rebuild can still reject it."""
    snapshots = importlib.import_module("scripts.trading_lab.market_snapshots")
    connection = sqlite3.connect(database)
    try:
        for name in ("market_snapshot_entries", "market_snapshot_manifests",
                     "market_bar_receipts"):
            connection.execute(f"DROP TRIGGER IF EXISTS {name}_no_update")
            connection.execute(f"DROP TRIGGER IF EXISTS {name}_no_delete")
        connection.execute("DROP TRIGGER IF EXISTS market_snapshot_entries_no_late_insert")
        connection.execute("PRAGMA foreign_keys = OFF")
        sha, payload = connection.execute(
            "SELECT content_sha256, payload_json FROM market_bar_receipts "
            "ORDER BY bar_open_at LIMIT 1"
        ).fetchone()
        record = mutate(json.loads(payload))
        unsigned = {key: value for key, value in record.items() if key != "content_sha256"}
        record["content_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        connection.execute(
            "UPDATE market_bar_receipts SET content_sha256 = ?, payload_json = ? "
            "WHERE content_sha256 = ?",
            (record["content_sha256"], json.dumps(record, sort_keys=True, separators=(",", ":")), sha),
        )
        connection.execute(
            "UPDATE market_snapshot_entries SET content_sha256 = ? WHERE content_sha256 = ?",
            (record["content_sha256"], sha),
        )
        manifest = connection.execute("SELECT * FROM market_snapshot_manifests").fetchone()
        pairs = connection.execute(
            "SELECT bar_open_at, content_sha256 FROM market_snapshot_entries "
            "WHERE snapshot_id = ? ORDER BY bar_open_at", (manifest[0],)
        ).fetchall()
        entries_hash = snapshots.build_entries_content_hash(
            tuple(snapshots.SelectedSnapshotReceipt(
                bar_open_at=b, content_sha256=c, bar_id="", bar_version_id="",
                ingested_at="", available_at="") for b, c in pairs)
        )
        request_id = manifest[1]
        new_id = snapshots.build_snapshot_id(
            snapshot_request_id=request_id, entries_content_hash=entries_hash
        )
        connection.execute(
            "UPDATE market_snapshot_entries SET snapshot_id = ? WHERE snapshot_id = ?",
            (new_id, manifest[0]),
        )
        connection.execute("DELETE FROM market_snapshot_manifests WHERE snapshot_id = ?",
                           (manifest[0],))
        connection.execute(
            "INSERT INTO market_snapshot_manifests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id, request_id, entries_hash, *manifest[3:]),
        )
        connection.commit()
        return new_id
    finally:
        connection.close()


def test_replay_rejects_a_forged_bar_that_market_bar_rebuilds_differently(
    tmp_path, store_module, snapshots_module
) -> None:
    """Everything is recomputed consistently -- receipt hash, entry, manifest
    identities -- so every hash check passes. Only rebuilding the bar through
    MarketBar V1 exposes that bar_status was forged."""
    store, _ = _one_snapshot(store_module, snapshots_module, tmp_path, name="forge")
    forged_id = _replace_payload(
        store.database_path, mutate=lambda record: {**record, "bar_status": "partial"}
    )
    connection = store._connect()
    try:
        snapshots_module.load_snapshot(connection, snapshot_id=forged_id)  # structurally sound
        with pytest.raises(snapshots_module.SnapshotStateCorruption, match="rebuild"):
            snapshots_module.replay_snapshot(connection, snapshot_id=forged_id)
    finally:
        connection.close()


def test_replay_refuses_an_unknown_market_bar_schema_version(
    tmp_path, store_module, snapshots_module
) -> None:
    store, _ = _one_snapshot(store_module, snapshots_module, tmp_path, name="mbver")
    forged_id = _replace_payload(
        store.database_path,
        mutate=lambda record: {**record, "schema_version": "trading-lab.market-bar.v99"},
    )
    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotUnsupportedVersion):
            snapshots_module.replay_snapshot(connection, snapshot_id=forged_id)
    finally:
        connection.close()


def test_replay_detects_a_payload_whose_self_hash_is_wrong(
    tmp_path, store_module, snapshots_module
) -> None:
    """The payload keeps the content_sha256 it is referenced by, but its
    other fields were changed: only recomputing the payload's own hash
    catches that."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="selfhash")
    connection = sqlite3.connect(store.database_path)
    try:
        for name in ("market_bar_receipts",):
            connection.execute(f"DROP TRIGGER IF EXISTS {name}_no_update")
        sha, payload = connection.execute(
            "SELECT content_sha256, payload_json FROM market_bar_receipts "
            "ORDER BY bar_open_at LIMIT 1"
        ).fetchone()
        record = json.loads(payload)
        record["volume"] = "999.000000000000000000"   # hash NOT recomputed
        connection.execute(
            "UPDATE market_bar_receipts SET payload_json = ? WHERE content_sha256 = ?",
            (json.dumps(record, sort_keys=True, separators=(",", ":")), sha),
        )
        connection.commit()
    finally:
        connection.close()

    connection = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotStateCorruption, match="hash"):
            snapshots_module.replay_snapshot(connection, snapshot_id=result.snapshot_id)
    finally:
        connection.close()


# =========================================================================
# Phase 1C-D / F1-F4: invariants the code upholds but no test protected.
# Each of these dies if its guard is removed -- verified by mutation.
# =========================================================================


class _StaticCursor:
    """Serves a fixed row list through the bounded-read protocol."""

    def __init__(self, rows: list) -> None:
        self._rows = list(rows)
        self._offset = 0

    def fetchmany(self, size):
        chunk = self._rows[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def fetchall(self):
        pytest.fail("fetchall() must never be used on the read path")

    def close(self):
        return None


class _RewritingConnection:
    """Delegates to a real connection but rewrites the rows of one exact
    query, so the two capture passes can be made to disagree."""

    def __init__(self, inner, *, target_sql: str, rewrite) -> None:
        self._inner = inner
        self._target_sql = target_sql
        self._rewrite = rewrite

    @property
    def in_transaction(self):
        return self._inner.in_transaction

    def execute(self, sql, parameters=()):
        cursor = self._inner.execute(sql, parameters)
        if sql == self._target_sql:
            return _StaticCursor(self._rewrite(list(cursor)))
        return cursor

    def commit(self):
        return self._inner.commit()

    def rollback(self):
        return self._inner.rollback()


# --- F1: the payload pass must never be zipped against a different length --


@pytest.mark.parametrize(
    "label, rewrite",
    [
        ("payload row missing", lambda rows: rows[:-1]),
        ("payload row added", lambda rows: rows + [rows[-1]]),
        ("payload order reversed", lambda rows: list(reversed(rows))),
        ("payload bar_open_at differs",
         lambda rows: [("2999-01-01T00:00:00+00:00", *rows[0][1:])] + rows[1:]),
        ("payload content_sha256 differs",
         lambda rows: [(rows[0][0], "f" * 64, *rows[0][2:])] + rows[1:]),
    ],
)
def test_payload_pass_disagreeing_with_metadata_is_refused(
    tmp_path, store_module, snapshots_module, label, rewrite
) -> None:
    """zip() silently stops at the shorter sequence. A length guard must run
    first, and the per-row alignment must be checked too -- otherwise a
    replay could return fewer bars than the snapshot attests to, with no
    error at all."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=3, name="f1")
    outcome = "not-set"
    connection = store._connect()
    try:
        rewriting = _RewritingConnection(
            connection,
            target_sql=snapshots_module._SELECT_SNAPSHOT_PAYLOADS_SQL,
            rewrite=rewrite,
        )
        with pytest.raises(snapshots_module.SnapshotStateCorruption) as excinfo:
            outcome = snapshots_module.replay_snapshot(
                rewriting, snapshot_id=result.snapshot_id
            )
        assert outcome == "not-set", "a partial replay escaped"
        assert connection.in_transaction is False
        message = str(excinfo.value)
        assert "schema_version" not in message and "{" not in message
        # The connection is still usable afterwards.
        assert len(snapshots_module.replay_snapshot(
            connection, snapshot_id=result.snapshot_id)) == 3
    finally:
        connection.close()


# --- F2: limit + 1 on the cursor path ------------------------------------


def _many_snapshots(store_module, snapshots_module, tmp_path, *, count, name):
    """`count` distinct materializations of the SAME request, so listing has
    a real multi-page dataset to paginate."""
    store, first = _one_snapshot(store_module, snapshots_module, tmp_path, openings=2, name=name)
    market_bar = importlib.import_module("scripts.trading_lab.market_bar")
    ingested = INGEST_BASE + timedelta(days=800)
    ids = {first.snapshot_id}
    connection = store._connect()
    try:
        for revision in range(1, count):
            later = ingested + timedelta(hours=revision)
            connection.execute(
                "INSERT INTO market_ingestions VALUES (?, 's', 'coinbase_exchange_rest', "
                "'BTC-USD', '1h', ?, ?, ?, '{}', 2)",
                (f"r{revision}", _iso(later), _iso(later), "a" * 64),
            )
            for index in range(2):
                bar = market_bar.build_market_bar(
                    asset="BTC/USD", venue="coinbase_exchange", provider="coinbase_exchange",
                    timeframe="1h",
                    bar_open_at=GRID_BASE + timedelta(hours=index),
                    bar_close_at=GRID_BASE + timedelta(hours=index + 1),
                    available_at=later, ingested_at=later,
                    open_price=f"{70000 + revision}.12345678901",
                    high_price="80000.00000000004", low_price="60000.00000000001",
                    close_price="70000.50000000005", volume="12.345678901234567891",
                    raw_payload_sha256="a" * 64,
                )
                connection.execute(
                    "INSERT INTO market_bar_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (bar["content_sha256"], f"r{revision}", bar["bar_id"], bar["bar_version_id"],
                     bar["bar_open_at"], bar["bar_close_at"], bar["available_at"],
                     bar["ingested_at"], "a" * 64,
                     json.dumps(bar, sort_keys=True, separators=(",", ":"))),
                )
            connection.commit()
            ids.add(snapshots_module._materialize_snapshot(
                connection,
                provider="coinbase_exchange_rest", product_id="BTC-USD", timeframe="1h",
                range_start=_iso(GRID_BASE), range_end=_iso(GRID_BASE + timedelta(hours=2)),
                as_of=_iso(ingested + timedelta(days=1)),
            ).snapshot_id)
    finally:
        connection.close()
    assert len(ids) == count
    return store, first.snapshot_request_id, sorted(ids)


def test_pagination_spans_four_pages_without_omitting_a_manifest(
    tmp_path, store_module, snapshots_module
) -> None:
    """Seven manifests read two at a time: pages 2 and 3 go through the
    cursor path AND still have a following page, which is exactly where a
    missing `limit + 1` omits a manifest while reporting the end of the
    listing."""
    store, request_id, expected = _many_snapshots(
        store_module, snapshots_module, tmp_path, count=7, name="f2"
    )
    connection = store._connect()
    collected: list[str] = []
    cursors: list[str | None] = []
    try:
        cursor = None
        for page_index in range(4):
            page = snapshots_module.list_snapshot_manifests(
                connection, snapshot_request_id=request_id,
                after_snapshot_id=cursor, limit=2,
            )
            ids = [item.snapshot_id for item in page.items]
            assert ids == sorted(ids)
            if cursor is not None:
                assert all(item > cursor for item in ids), (cursor, ids)
            collected.extend(ids)
            cursors.append(page.next_after_snapshot_id)
            cursor = page.next_after_snapshot_id
            if page_index < 3:
                assert cursor is not None, f"page {page_index} lost its cursor"
                assert cursor == ids[-1]
        assert cursors[3] is None
    finally:
        connection.close()

    assert collected == expected
    assert len(collected) == len(set(collected)) == 7


# --- F3: the capture must be one single SQLite view -----------------------


def test_capture_reads_manifest_and_entries_in_one_view(
    tmp_path, store_module, snapshots_module
) -> None:
    """Without an explicit BEGIN every SELECT is its own view, so a writer
    landing between the manifest and the entries would be half-seen. Proven
    on a real file database with two connections: while the reader holds its
    capture, the writer cannot commit, and the reader sees the old state
    whole."""
    import threading

    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=3, name="f3")
    # The sealing trigger exists precisely to forbid this; drop it for the
    # fixture so a late entry is technically possible to attempt.
    with sqlite3.connect(store.database_path) as setup:
        setup.execute("DROP TRIGGER market_snapshot_entries_no_late_insert")

    reader = store._connect()
    metadata_reached = threading.Event()
    writer_done = threading.Event()
    observed: dict[str, object] = {}

    def _writer() -> None:
        # sqlite3 forbids sharing a connection across threads: this one is
        # created, used and closed entirely here.
        writer = store._connect()
        try:
            writer.execute("PRAGMA busy_timeout = 300")
            metadata_reached.wait(timeout=10)
            try:
                writer.execute("BEGIN IMMEDIATE")
                writer.execute(
                    "INSERT INTO market_snapshot_entries "
                    "(snapshot_id, bar_open_at, content_sha256) "
                    "SELECT ?, '2027-01-01T00:00:00+00:00', content_sha256 "
                    "FROM market_snapshot_entries WHERE snapshot_id = ? LIMIT 1",
                    (result.snapshot_id, result.snapshot_id),
                )
                writer.commit()
                observed["writer_committed"] = True
            except sqlite3.OperationalError as exc:
                observed["writer_committed"] = False
                observed["writer_error"] = str(exc)
                writer.rollback()
        finally:
            writer_done.set()
            writer.close()

    class _PausingConnection(_RewritingConnection):
        def execute(self, sql, parameters=()):
            if sql == snapshots_module._SELECT_SNAPSHOT_METADATA_SQL:
                metadata_reached.set()
                writer_done.wait(timeout=10)
            return self._inner.execute(sql, parameters)

    thread = threading.Thread(target=_writer)
    thread.start()
    try:
        loaded = snapshots_module.load_snapshot(
            _PausingConnection(reader, target_sql="", rewrite=lambda rows: rows),
            snapshot_id=result.snapshot_id,
        )
    finally:
        metadata_reached.set()
        thread.join(timeout=10)
        reader.close()

    # The reader saw the pre-existing state in full, never a torn view.
    assert loaded.manifest.entry_count == 3
    assert len(loaded.entries) == 3
    # And it held a real transaction: the writer could not commit through it.
    assert observed.get("writer_committed") is False, observed


# --- F4: the sqlite3.Error handler must release the read lock -------------


class _FailingOnQueryConnection(_RewritingConnection):
    def execute(self, sql, parameters=()):
        if sql == self._target_sql:
            raise sqlite3.OperationalError("injected disk I/O error")
        return self._inner.execute(sql, parameters)


def test_sqlite_failure_during_capture_releases_the_read_lock(
    tmp_path, store_module, snapshots_module
) -> None:
    """An sqlite3 failure after BEGIN and after a successful manifest read
    must still roll back. Otherwise the read transaction stays open and, in
    journal_mode=delete, blocks every writer's COMMIT until the connection
    dies."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=3, name="f4")
    reader = store._connect()
    outcome = "not-set"
    try:
        failing = _FailingOnQueryConnection(
            reader, target_sql=snapshots_module._SELECT_SNAPSHOT_METADATA_SQL,
            rewrite=lambda rows: rows,
        )
        with pytest.raises(snapshots_module.SnapshotReadError) as excinfo:
            outcome = snapshots_module.load_snapshot(failing, snapshot_id=result.snapshot_id)
        assert outcome == "not-set"
        assert type(excinfo.value.__cause__) is sqlite3.OperationalError
        assert reader.in_transaction is False

        # The lock must be genuinely gone: a second connection writes and commits.
        writer = store._connect()
        try:
            writer.execute("PRAGMA busy_timeout = 300")
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
                ("d" * 64, "coinbase_exchange_rest", b"w", 1, _iso(GRID_BASE)),
            )
            writer.commit()
        finally:
            writer.close()

        # And the original connection still works.
        assert len(snapshots_module.load_snapshot(
            reader, snapshot_id=result.snapshot_id).entries) == 3
    finally:
        reader.close()


# --- Strict input typing --------------------------------------------------


class _StrSubclass(str):
    pass


def test_identifier_validation_rejects_a_str_subclass(
    tmp_path, store_module, snapshots_module
) -> None:
    """`type(x) is str`, not isinstance: a str subclass may carry arbitrary
    behaviour, and an identifier is compared and stored verbatim."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, name="typ")
    connection = store._connect()
    try:
        assert snapshots_module.load_snapshot(
            connection, snapshot_id=result.snapshot_id).manifest.entry_count == 3
        with pytest.raises(snapshots_module.SnapshotInputError):
            snapshots_module.load_snapshot(
                connection, snapshot_id=_StrSubclass(result.snapshot_id)
            )
        with pytest.raises(snapshots_module.SnapshotInputError):
            snapshots_module.list_snapshot_manifests(
                connection, snapshot_request_id=_StrSubclass(result.snapshot_request_id)
            )
    finally:
        connection.close()


@pytest.mark.parametrize(
    "bad", [True, False, bytearray(b"5"), b"5", 5.0, "5"]
)
def test_limit_validation_rejects_anything_that_is_not_exactly_int(
    snapshots_module, bad
) -> None:
    """bool subclasses int, so isinstance would silently accept True as 1."""
    connection = _ForbiddenBusinessConnection()
    with pytest.raises(snapshots_module.SnapshotInputError):
        snapshots_module.list_snapshot_manifests(
            connection,
            snapshot_request_id="hyprl-market-snapshot-request-" + "a" * 64,
            limit=bad,
        )


class _IntSubclass(int):
    pass


def test_limit_validation_rejects_an_int_subclass(snapshots_module) -> None:
    connection = _ForbiddenBusinessConnection()
    with pytest.raises(snapshots_module.SnapshotInputError):
        snapshots_module.list_snapshot_manifests(
            connection,
            snapshot_request_id="hyprl-market-snapshot-request-" + "a" * 64,
            limit=_IntSubclass(10),
        )


class _RaisingOnQueryConnection(_RewritingConnection):
    """Raises an error the primitive does not model at all."""

    def execute(self, sql, parameters=()):
        if sql == self._target_sql:
            raise RuntimeError("injected unexpected failure during capture")
        return self._inner.execute(sql, parameters)


def test_unexpected_error_during_capture_releases_the_read_lock(
    tmp_path, store_module, snapshots_module
) -> None:
    """The generic handler must roll back too. An unmodelled error escaping
    with the read transaction still open would hold the lock against every
    writer, exactly like the sqlite3 case."""
    store, result = _one_snapshot(store_module, snapshots_module, tmp_path, openings=3, name="f4g")
    reader = store._connect()
    outcome = "not-set"
    try:
        raising = _RaisingOnQueryConnection(
            reader, target_sql=snapshots_module._SELECT_SNAPSHOT_METADATA_SQL,
            rewrite=lambda rows: rows,
        )
        with pytest.raises(RuntimeError) as excinfo:
            outcome = snapshots_module.load_snapshot(raising, snapshot_id=result.snapshot_id)
        assert outcome == "not-set"
        # Propagated unchanged, never disguised as corruption.
        assert type(excinfo.value) is RuntimeError
        assert not isinstance(excinfo.value, snapshots_module.MarketSnapshotError)
        assert reader.in_transaction is False

        writer = store._connect()
        try:
            writer.execute("PRAGMA busy_timeout = 300")
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
                ("c" * 64, "coinbase_exchange_rest", b"g", 1, _iso(GRID_BASE)),
            )
            writer.commit()
        finally:
            writer.close()

        assert len(snapshots_module.load_snapshot(
            reader, snapshot_id=result.snapshot_id).entries) == 3
    finally:
        reader.close()


def test_business_error_during_capture_releases_the_read_lock(
    tmp_path, store_module, snapshots_module
) -> None:
    """SnapshotNotFound travels through the MarketSnapshotError handler of the
    capture: it too must leave no transaction and no held lock behind."""
    store, _ = _one_snapshot(store_module, snapshots_module, tmp_path, name="f4b")
    reader = store._connect()
    try:
        with pytest.raises(snapshots_module.SnapshotNotFound):
            snapshots_module.load_snapshot(
                reader, snapshot_id="hyprl-market-snapshot-" + "1" * 64
            )
        assert reader.in_transaction is False

        writer = store._connect()
        try:
            writer.execute("PRAGMA busy_timeout = 300")
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
                ("b" * 64, "coinbase_exchange_rest", b"b", 1, _iso(GRID_BASE)),
            )
            writer.commit()
        finally:
            writer.close()
    finally:
        reader.close()

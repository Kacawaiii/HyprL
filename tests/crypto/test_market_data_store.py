from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import sqlite3

import pytest


ROOT = Path(__file__).resolve().parents[2]
BTC_FIXTURE = ROOT / "tests" / "fixtures" / "crypto" / "coinbase_btc_usd_1h.json"


@pytest.fixture
def store_module():
    return importlib.import_module("scripts.trading_lab.market_data_store")


def _store(module, tmp_path):
    return module.MarketDataStore(tmp_path / "market-data.sqlite3")


def _ingest(store, payload: bytes, **overrides):
    values = {
        "product_id": "BTC-USD",
        "timeframe": "1h",
        "available_at": "2026-08-02T11:00:05Z",
        "ingested_at": "2026-08-02T11:00:06Z",
    }
    values.update(overrides)
    return store.ingest_coinbase_response(payload, **values)


def _counts(database: Path) -> tuple[int, int, int]:
    with sqlite3.connect(database) as connection:
        return tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "raw_market_payloads",
                "market_ingestions",
                "market_bar_receipts",
            )
        )


def _row(iso_open: str, *, low: str, high: str, open_: str, close: str, volume: str) -> list:
    epoch = int(datetime.fromisoformat(iso_open).timestamp())
    return [epoch, low, high, open_, close, volume]


def _candles_payload(rows: list) -> bytes:
    return json.dumps(rows, separators=(",", ":")).encode("utf-8")


def _bar(iso_open: str) -> list:
    return _row(iso_open, low="100.0", high="110.0", open_="105.0", close="106.0", volume="1.0")


def _gap_event_count(database: Path, *, product_id: str | None = None, timeframe: str | None = None) -> int:
    query = "SELECT COUNT(*) FROM market_data_gap_events"
    params: list[str] = []
    conditions = []
    if product_id is not None:
        conditions.append("product_id = ?")
        params.append(product_id)
    if timeframe is not None:
        conditions.append("timeframe = ?")
        params.append(timeframe)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    with sqlite3.connect(database) as connection:
        return connection.execute(query, params).fetchone()[0]


def _gap_events(database: Path) -> list[tuple]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            """
            SELECT expected_bar_open_at, event_type, provider, product_id, timeframe,
                   cause, observed_by_ingestion_id, available_at, ingested_at, event_id
            FROM market_data_gap_events
            ORDER BY expected_bar_open_at, event_type
            """
        ).fetchall()


def test_market_data_store_persists_exact_raw_bytes_and_bar_receipts(
    tmp_path,
    store_module,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()

    result = _ingest(store, payload)

    assert result.raw_payloads_inserted == 1
    assert result.ingestions_inserted == 1
    assert result.bar_receipts_inserted == 2
    assert result.exact_replays == 0
    with sqlite3.connect(store.database_path) as connection:
        stored_payload = connection.execute(
            "SELECT payload_bytes FROM raw_market_payloads"
        ).fetchone()[0]
        rows = connection.execute(
            """
            SELECT bar_id, bar_version_id, content_sha256, payload_json
            FROM market_bar_receipts
            ORDER BY bar_open_at
            """
        ).fetchall()
    assert stored_payload == payload
    assert len(rows) == 2
    for bar_id, version_id, content_hash, payload_json in rows:
        record = json.loads(payload_json)
        assert record["bar_id"] == bar_id
        assert record["bar_version_id"] == version_id
        assert record["content_sha256"] == content_hash


def test_market_data_store_exact_replay_is_a_noop(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    first = _ingest(store, payload)

    replay = _ingest(store, payload)

    assert replay.ingestion_id == first.ingestion_id
    assert replay.raw_payloads_inserted == 0
    assert replay.ingestions_inserted == 0
    assert replay.bar_receipts_inserted == 0
    assert replay.exact_replays == 2
    assert _counts(store.database_path) == (1, 1, 2)


def test_same_raw_payload_at_new_ingestion_time_creates_receipts_not_versions(
    tmp_path,
    store_module,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    _ingest(store, payload)

    later = _ingest(
        store,
        payload,
        ingested_at="2026-08-02T11:05:00Z",
    )

    assert later.raw_payloads_inserted == 0
    assert later.ingestions_inserted == 1
    assert later.bar_receipts_inserted == 2
    with sqlite3.connect(store.database_path) as connection:
        distinct_versions = connection.execute(
            "SELECT COUNT(DISTINCT bar_version_id) FROM market_bar_receipts"
        ).fetchone()[0]
    assert _counts(store.database_path) == (1, 2, 4)
    assert distinct_versions == 2


def test_provider_revision_creates_new_version_only_for_changed_bar(
    tmp_path,
    store_module,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    revised_page = json.loads(payload)
    revised_page[0][4] = "64260.0"
    revised_payload = json.dumps(revised_page, separators=(",", ":")).encode("utf-8")
    _ingest(store, payload)

    _ingest(
        store,
        revised_payload,
        available_at="2026-08-02T11:01:00Z",
        ingested_at="2026-08-02T11:01:01Z",
    )

    with sqlite3.connect(store.database_path) as connection:
        versions_by_open = connection.execute(
            """
            SELECT bar_open_at, COUNT(DISTINCT bar_version_id)
            FROM market_bar_receipts
            GROUP BY bar_open_at
            ORDER BY bar_open_at
            """
        ).fetchall()
    assert versions_by_open == [
        ("2026-08-02T09:00:00+00:00", 1),
        ("2026-08-02T10:00:00+00:00", 2),
    ]


@pytest.mark.parametrize(
    ("payload", "overrides"),
    [
        (b"not json", {}),
        (
            BTC_FIXTURE.read_bytes(),
            {
                "available_at": "2026-08-02T10:30:00Z",
                "ingested_at": "2026-08-02T10:30:01Z",
            },
        ),
    ],
)
def test_invalid_response_leaves_store_empty_atomically(
    tmp_path,
    store_module,
    payload,
    overrides,
) -> None:
    store = _store(store_module, tmp_path)

    with pytest.raises(ValueError):
        _ingest(store, payload, **overrides)

    assert _counts(store.database_path) == (0, 0, 0)


def test_unexpected_post_begin_failure_rolls_back_and_retry_succeeds(
    tmp_path,
    store_module,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER injected_receipt_failure
            BEFORE INSERT ON market_bar_receipts
            BEGIN
                SELECT RAISE(ABORT, 'injected failure');
            END;
            """
        )

    with pytest.raises(store_module.MarketDataStoreError, match="persistence failed"):
        _ingest(store, payload)
    assert _counts(store.database_path) == (0, 0, 0)

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER injected_receipt_failure")
    result = _ingest(store, payload)
    assert result.bar_receipts_inserted == 2
    assert _counts(store.database_path) == (1, 1, 2)


def test_market_data_tables_are_insert_only(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(store, BTC_FIXTURE.read_bytes())

    with sqlite3.connect(store.database_path) as connection:
        for statement in (
            "UPDATE raw_market_payloads SET byte_count = 0",
            "DELETE FROM raw_market_payloads",
            "UPDATE market_ingestions SET bar_count = 0",
            "DELETE FROM market_ingestions",
            "UPDATE market_bar_receipts SET bar_open_at = 'x'",
            "DELETE FROM market_bar_receipts",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
                connection.execute(statement)


def test_concurrent_exact_ingestions_store_one_copy(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _ingest(store, payload), range(2)))

    assert sum(result.raw_payloads_inserted for result in results) == 1
    assert sum(result.ingestions_inserted for result in results) == 1
    assert sum(result.bar_receipts_inserted for result in results) == 2
    assert _counts(store.database_path) == (1, 1, 2)


def test_preexisting_raw_hash_collision_fails_closed(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    import hashlib

    digest = hashlib.sha256(payload).hexdigest()
    with sqlite3.connect(store.database_path) as connection:
        connection.execute(
            """
            INSERT INTO raw_market_payloads (
                payload_sha256, provider, payload_bytes, byte_count, first_stored_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (digest, "coinbase_exchange_rest", b"different", 9, "2026-08-02T00:00:00+00:00"),
        )

    with pytest.raises(store_module.MarketDataConflict, match="immutable"):
        _ingest(store, payload)
    assert _counts(store.database_path) == (1, 0, 0)


def test_store_rejects_adapter_output_not_bound_to_raw_payload(
    tmp_path,
    store_module,
    monkeypatch,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    real_adapter = store_module.adapt_coinbase_candles

    def tampered_adapter(*args, **kwargs):
        records = real_adapter(*args, **kwargs)
        records[0]["raw_payload_sha256"] = "0" * 64
        return records

    monkeypatch.setattr(store_module, "adapt_coinbase_candles", tampered_adapter)
    with pytest.raises(store_module.MarketDataStoreError, match="invalid"):
        _ingest(store, payload)
    assert _counts(store.database_path) == (0, 0, 0)


def test_store_rebuilds_and_rejects_fully_rehashed_tampered_bar(
    tmp_path,
    store_module,
    monkeypatch,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    real_adapter = store_module.adapt_coinbase_candles

    def tampered_adapter(*args, **kwargs):
        records = real_adapter(*args, **kwargs)
        records[0]["open"] = "64150.0000000000"
        unsigned = dict(records[0])
        del unsigned["content_sha256"]
        canonical = json.dumps(
            unsigned,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        import hashlib

        records[0]["content_sha256"] = hashlib.sha256(canonical).hexdigest()
        return records

    monkeypatch.setattr(store_module, "adapt_coinbase_candles", tampered_adapter)
    with pytest.raises(store_module.MarketDataStoreError, match="invalid"):
        _ingest(store, payload)
    assert _counts(store.database_path) == (0, 0, 0)


def test_store_bounds_raw_input_before_calling_adapter(
    tmp_path,
    store_module,
    monkeypatch,
) -> None:
    store = _store(store_module, tmp_path)

    def forbidden_adapter(*_args, **_kwargs):
        pytest.fail("oversized raw payload reached provider adapter")

    monkeypatch.setattr(store_module, "adapt_coinbase_candles", forbidden_adapter)
    with pytest.raises(store_module.MarketDataStoreError, match="too large"):
        _ingest(store, b" " * 1_000_001)
    assert _counts(store.database_path) == (0, 0, 0)


def test_store_bounds_adapter_record_count_before_rebuilding(
    tmp_path,
    store_module,
    monkeypatch,
) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    records = store_module.adapt_coinbase_candles(
        payload,
        product_id="BTC-USD",
        timeframe="1h",
        available_at="2026-08-02T11:00:05Z",
        ingested_at="2026-08-02T11:00:06Z",
    )
    monkeypatch.setattr(
        store_module,
        "adapt_coinbase_candles",
        lambda *_args, **_kwargs: [records[0]] * 301,
    )

    def forbidden_builder(**_kwargs):
        pytest.fail("oversized adapter result reached MarketBar rebuilding")

    monkeypatch.setattr(store_module, "build_market_bar", forbidden_builder)
    with pytest.raises(store_module.MarketDataStoreError, match="invalid"):
        _ingest(store, payload)
    assert _counts(store.database_path) == (0, 0, 0)


def test_contiguous_series_produces_no_gap_events(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()

    _ingest(store, payload)

    assert _gap_event_count(store.database_path) == 0


def test_single_interior_missing_bar_creates_one_detected_gap_event(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload([_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")])

    _ingest(
        store,
        payload,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    events = _gap_events(store.database_path)
    assert len(events) == 1
    (open_at, event_type, provider, product_id, timeframe, cause, ingestion_id, available_at, ingested_at, event_id) = events[0]
    assert open_at == "2026-08-05T10:00:00+00:00"
    assert event_type == "DETECTED"
    assert provider == "coinbase_exchange_rest"
    assert product_id == "BTC-USD"
    assert timeframe == "1h"
    assert cause in ("unknown", "provider_missing")
    assert ingestion_id
    assert available_at == "2026-08-05T12:05:00+00:00"
    assert ingested_at == "2026-08-05T12:06:00+00:00"
    assert event_id


def test_multiple_consecutive_missing_bars_create_one_event_each(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload([_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T13:00:00+00:00")])

    _ingest(
        store,
        payload,
        available_at="2026-08-05T14:05:00Z",
        ingested_at="2026-08-05T14:06:00Z",
    )

    events = _gap_events(store.database_path)
    opens = [event[0] for event in events]
    assert opens == [
        "2026-08-05T10:00:00+00:00",
        "2026-08-05T11:00:00+00:00",
        "2026-08-05T12:00:00+00:00",
    ]
    assert all(event[1] == "DETECTED" for event in events)
    assert len({event[9] for event in events}) == 3


def test_gap_detection_replay_is_idempotent(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload([_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")])
    overrides = {
        "available_at": "2026-08-05T12:05:00Z",
        "ingested_at": "2026-08-05T12:06:00Z",
    }
    _ingest(store, payload, **overrides)

    _ingest(store, payload, **overrides)

    assert _gap_event_count(store.database_path) == 1


def test_late_arrival_resolves_gap_with_single_resolved_event(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    skip_payload = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
    )
    _ingest(
        store,
        skip_payload,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    before = _gap_events(store.database_path)
    assert len(before) == 1
    detected_before = before[0]

    fill_payload = _candles_payload([_bar("2026-08-05T10:00:00+00:00")])
    _ingest(
        store,
        fill_payload,
        available_at="2026-08-05T15:00:00Z",
        ingested_at="2026-08-05T15:01:00Z",
    )

    after = _gap_events(store.database_path)
    assert len(after) == 2
    by_type = {event[1]: event for event in after}
    assert set(by_type) == {"DETECTED", "RESOLVED"}
    assert by_type["DETECTED"] == detected_before
    assert by_type["RESOLVED"][0] == "2026-08-05T10:00:00+00:00"
    assert by_type["RESOLVED"][6] != detected_before[6]


def test_replaying_the_resolution_does_not_duplicate(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    skip_payload = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
    )
    _ingest(
        store,
        skip_payload,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    fill_payload = _candles_payload([_bar("2026-08-05T10:00:00+00:00")])
    fill_overrides = {
        "available_at": "2026-08-05T15:00:00Z",
        "ingested_at": "2026-08-05T15:01:00Z",
    }
    _ingest(store, fill_payload, **fill_overrides)

    _ingest(store, fill_payload, **fill_overrides)

    assert _gap_event_count(store.database_path) == 2


def test_gap_events_are_isolated_by_product_and_timeframe(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    btc_hourly_gap = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
    )
    _ingest(
        store,
        btc_hourly_gap,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    eth_hourly_contiguous = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T10:00:00+00:00")]
    )
    _ingest(
        store,
        eth_hourly_contiguous,
        product_id="ETH-USD",
        timeframe="1h",
        available_at="2026-08-05T11:05:00Z",
        ingested_at="2026-08-05T11:06:00Z",
    )
    btc_daily_single = _candles_payload([_bar("2026-08-05T00:00:00+00:00")])
    _ingest(
        store,
        btc_daily_single,
        timeframe="1d",
        available_at="2026-08-06T00:05:00Z",
        ingested_at="2026-08-06T00:06:00Z",
    )

    assert _gap_event_count(store.database_path, product_id="BTC-USD", timeframe="1h") == 1
    assert _gap_event_count(store.database_path, product_id="ETH-USD", timeframe="1h") == 0
    assert _gap_event_count(store.database_path, product_id="BTC-USD", timeframe="1d") == 0
    assert _gap_event_count(store.database_path) == 1


def test_gap_event_insertion_failure_rolls_back_atomically(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload([_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")])
    overrides = {
        "available_at": "2026-08-05T12:05:00Z",
        "ingested_at": "2026-08-05T12:06:00Z",
    }
    with sqlite3.connect(store.database_path) as connection:
        connection.executescript(
            """
            CREATE TRIGGER injected_gap_event_failure
            BEFORE INSERT ON market_data_gap_events
            BEGIN
                SELECT RAISE(ABORT, 'injected failure');
            END;
            """
        )

    with pytest.raises(store_module.MarketDataStoreError, match="persistence failed"):
        _ingest(store, payload, **overrides)
    assert _counts(store.database_path) == (0, 0, 0)
    assert _gap_event_count(store.database_path) == 0

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER injected_gap_event_failure")
    result = _ingest(store, payload, **overrides)
    assert result.bar_receipts_inserted == 2
    assert _gap_event_count(store.database_path) == 1


def test_market_data_gap_events_table_is_insert_only(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload([_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")])
    _ingest(
        store,
        payload,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    with sqlite3.connect(store.database_path) as connection:
        for statement in (
            "UPDATE market_data_gap_events SET event_type = 'RESOLVED'",
            "DELETE FROM market_data_gap_events",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="insert-only"):
                connection.execute(statement)


def test_isolated_bar_creates_no_boundary_gap_events(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    lone_bar = _candles_payload([_bar("2026-08-05T11:00:00+00:00")])

    _ingest(
        store,
        lone_bar,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    assert _gap_event_count(store.database_path) == 0


def test_bar_revision_does_not_create_or_resolve_gap_events(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    payload = BTC_FIXTURE.read_bytes()
    revised_page = json.loads(payload)
    revised_page[0][4] = "64260.0"
    revised_payload = json.dumps(revised_page, separators=(",", ":")).encode("utf-8")
    _ingest(store, payload)
    assert _gap_event_count(store.database_path) == 0

    _ingest(
        store,
        revised_payload,
        available_at="2026-08-02T11:01:00Z",
        ingested_at="2026-08-02T11:01:01Z",
    )

    assert _gap_event_count(store.database_path) == 0


GAP_BASE = datetime(2020, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


# --- F1: bounded gap cardinality ---------------------------------------


def test_count_missing_opens_is_arithmetic_not_enumerative(store_module) -> None:
    huge_gap = store_module._count_missing_opens(
        GAP_BASE, GAP_BASE + timedelta(days=36525), timedelta(hours=1)
    )
    assert huge_gap == 36525 * 24 - 1


def test_pathological_century_gap_is_rejected_atomically(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    upper = GAP_BASE + timedelta(days=36525)
    payload = _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))])

    with pytest.raises(store_module.MarketDataStoreError, match="exceeds the maximum"):
        _ingest(
            store,
            payload,
            available_at=_iso(upper + timedelta(hours=1, minutes=5)),
            ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
        )

    assert _counts(store.database_path) == (0, 0, 0)
    assert _gap_event_count(store.database_path) == 0


def test_gap_limit_is_checked_before_materializing_events(
    tmp_path, store_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    upper = GAP_BASE + timedelta(days=36525)
    payload = _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))])

    def forbidden_record(*_args, **_kwargs):
        pytest.fail("gap events were materialized before the cardinality preflight rejected the ingestion")

    monkeypatch.setattr(store_module, "_record_gap_event", forbidden_record)

    with pytest.raises(store_module.MarketDataStoreError, match="exceeds the maximum"):
        _ingest(
            store,
            payload,
            available_at=_iso(upper + timedelta(hours=1, minutes=5)),
            ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
        )

    assert _counts(store.database_path) == (0, 0, 0)


def test_exactly_the_gap_limit_is_accepted(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    limit = store_module.MAX_GAP_CANDIDATES_PER_INGESTION
    upper = GAP_BASE + timedelta(hours=limit + 1)
    payload = _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))])

    result = _ingest(
        store,
        payload,
        available_at=_iso(upper + timedelta(hours=1, minutes=5)),
        ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
    )

    assert result.gap_events_detected == limit
    assert _gap_event_count(store.database_path) == limit


def test_one_more_than_the_gap_limit_is_rejected_atomically(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    limit = store_module.MAX_GAP_CANDIDATES_PER_INGESTION
    upper = GAP_BASE + timedelta(hours=limit + 2)
    payload = _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))])

    with pytest.raises(store_module.MarketDataStoreError, match="exceeds the maximum"):
        _ingest(
            store,
            payload,
            available_at=_iso(upper + timedelta(hours=1, minutes=5)),
            ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
        )

    assert _counts(store.database_path) == (0, 0, 0)
    assert _gap_event_count(store.database_path) == 0


def test_multi_segment_total_over_limit_is_rejected_before_partial_insert(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    a = GAP_BASE
    b = a + timedelta(hours=4001)
    c = b + timedelta(hours=7001)
    payload = _candles_payload([_bar(_iso(a)), _bar(_iso(b)), _bar(_iso(c))])

    with pytest.raises(store_module.MarketDataStoreError, match="exceeds the maximum"):
        _ingest(
            store,
            payload,
            available_at=_iso(c + timedelta(hours=1, minutes=5)),
            ingested_at=_iso(c + timedelta(hours=1, minutes=6)),
        )

    assert _counts(store.database_path) == (0, 0, 0)
    assert _gap_event_count(store.database_path) == 0


def test_valid_ingestion_after_pathological_rejection_succeeds(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    upper = GAP_BASE + timedelta(days=36525)
    pathological_payload = _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))])
    with pytest.raises(store_module.MarketDataStoreError, match="exceeds the maximum"):
        _ingest(
            store,
            pathological_payload,
            available_at=_iso(upper + timedelta(hours=1, minutes=5)),
            ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
        )

    result = _ingest(store, BTC_FIXTURE.read_bytes())

    assert result.bar_receipts_inserted == 2
    assert _counts(store.database_path) == (1, 1, 2)
    assert _gap_event_count(store.database_path) == 0


# --- F2: causal semantics of DETECTED / RESOLVED ------------------------


def test_first_seen_returns_earliest_declared_ingested_at_not_insert_order(
    tmp_path, store_module
) -> None:
    """Contract A (ingested_at is a caller-DECLARED historical timestamp, not
    this store's real receipt time or a local insert order): three receipts
    for the SAME bar_open_at are inserted, in this physical order:
      R1 (inserted 1st): ingested_at=12:06
      R2 (inserted 2nd): ingested_at=10:06  <- the true minimum, expected
      R3 (inserted 3rd, last): ingested_at=14:06

    R2 is deliberately neither the first nor the last row physically
    written, and 14:06 is deliberately both "last inserted" and the maximum
    declared value. This eliminates, in a single test, every plausible wrong
    strategy at once: first-physically-inserted (12:06), last-physically-
    inserted (14:06), MAX(ingested_at) (14:06), rowid ASC (12:06) and rowid
    DESC (14:06) all disagree with the correct answer. Only MIN(declared
    ingested_at) -- R2's 10:06 -- is correct.
    """
    store = _store(store_module, tmp_path)
    # R1: physically inserted FIRST, declared ingested_at=12:06.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    # R2: physically inserted SECOND -- neither first nor last -- declared
    # ingested_at=10:06, the true historical minimum and expected answer.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    # R3: physically inserted LAST, declared ingested_at=14:06 -- also the
    # maximum declared value. A "last inserted", "rowid DESC" or "MAX"
    # implementation would each wrongly return this value.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T14:06:00Z",
    )

    with sqlite3.connect(store.database_path) as connection:
        available_at, ingested_at = store_module._first_seen(
            connection,
            product_id="BTC-USD",
            timeframe="1h",
            bar_open_at="2026-08-05T09:00:00+00:00",
        )

    first_physically_inserted = "2026-08-05T12:06:00+00:00"  # R1: wrong
    last_physically_inserted = "2026-08-05T14:06:00+00:00"  # R3: wrong
    maximum_declared_ingested_at = "2026-08-05T14:06:00+00:00"  # R3: wrong
    assert ingested_at != first_physically_inserted
    assert ingested_at != last_physically_inserted
    assert ingested_at != maximum_declared_ingested_at
    # Only the minimum DECLARED historical ingested_at is correct (R2).
    assert available_at == "2026-08-05T10:05:00+00:00"
    assert ingested_at == "2026-08-05T10:06:00+00:00"


def test_non_monotone_ingestion_order_detected_causal_at_uses_latest_bound(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    # Call 1: the UPPER bound, ingested_at=12:06 (the larger value).
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    # Call 2: the LOWER bound, ingested_at=10:06 -- SMALLER than call 1's
    # 12:06. The sequence of ingested_at across successive calls is
    # genuinely non-monotone (12:06 -> 10:06), not merely a triggering
    # ingestion whose own clock happens to postdate both bounds.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    # Call 3: neither single-bar ingestion above can prove a gap
    # (intra-payload only). This later ingestion re-supplies both bars
    # together, forming the first intra-payload pair that reveals the gap.
    # Its own clock (20:05/20:06) must NOT become the causal timestamp.
    _ingest(
        store,
        _candles_payload(
            [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
        ),
        available_at="2026-08-05T20:05:00Z",
        ingested_at="2026-08-05T20:06:00Z",
    )

    events = _gap_events(store.database_path)
    assert len(events) == 1
    (open_at, event_type, *_rest, available_at, ingested_at, _event_id) = events[0]
    assert open_at == "2026-08-05T10:00:00+00:00"
    assert event_type == "DETECTED"
    # The causal timestamp is the UPPER bound's own first-seen (12:05/12:06):
    # neither the smallest ingested_at seen (10:06), nor the triggering
    # ingestion's own clock (20:06).
    assert available_at == "2026-08-05T12:05:00+00:00"
    assert ingested_at == "2026-08-05T12:06:00+00:00"
    # Explicitly rule out plausible wrong implementations rather than relying
    # on a single coincidental match: a bug using the triggering ingestion's
    # own clock, or one using only the lower bound's first-seen, or one using
    # min() instead of max() of the two bounds, would each produce a
    # DIFFERENT value here.
    triggering_clock = ("2026-08-05T20:05:00+00:00", "2026-08-05T20:06:00+00:00")
    lower_bound_only = ("2026-08-05T10:05:00+00:00", "2026-08-05T10:06:00+00:00")
    assert (available_at, ingested_at) != triggering_clock
    assert (available_at, ingested_at) != lower_bound_only
    assert datetime.fromisoformat(ingested_at) == max(
        datetime.fromisoformat("2026-08-05T12:06:00+00:00"),  # upper bound's first-seen
        datetime.fromisoformat("2026-08-05T10:06:00+00:00"),  # lower bound's first-seen
    )


def test_retrodated_resupply_of_a_bound_shifts_its_first_seen_into_the_causal_max(
    tmp_path, store_module
) -> None:
    """A bound's first-seen is the earliest ingested_at across ALL of its
    receipts, not frozen at the value recorded when it was first observed.
    If a later ingestion re-supplies an already-known bound with an earlier
    (retrodated) ingested_at, that earlier value legitimately becomes the
    bound's first-seen, and DETECTED.causal_at must reflect it -- proving the
    causal formula is genuinely recomputed from the store's current state,
    not cached from the original observation.
    """
    store = _store(store_module, tmp_path)
    # Bound HIGH observed once, late (18:06) -- this value must NOT survive.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T18:05:00Z",
        ingested_at="2026-08-05T18:06:00Z",
    )
    # Bound LOW observed once, early (10:06) -- stays the smaller candidate.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    # Pairing ingestion (first time the gap is provable) re-supplies HIGH
    # with a RETRODATED ingested_at=14:06: smaller than HIGH's original
    # 18:06, but still larger than LOW's 10:06.
    _ingest(
        store,
        _candles_payload(
            [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
        ),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T14:06:00Z",
    )

    events = _gap_events(store.database_path)
    assert len(events) == 1
    (open_at, event_type, *_rest, available_at, ingested_at, _event_id) = events[0]
    assert open_at == "2026-08-05T10:00:00+00:00"
    assert event_type == "DETECTED"
    # HIGH's first-seen shifted from its original 18:06 down to the
    # retrodated 14:06 (its new minimum); LOW's stays at 10:06 (14:06 does
    # not beat it). The causal max is therefore 14:06 -- neither HIGH's
    # stale original 18:06 nor LOW's 10:06 alone.
    original_high_observation = ("2026-08-05T18:05:00+00:00", "2026-08-05T18:06:00+00:00")
    lower_bound_only = ("2026-08-05T10:05:00+00:00", "2026-08-05T10:06:00+00:00")
    assert (available_at, ingested_at) != original_high_observation
    assert (available_at, ingested_at) != lower_bound_only
    assert available_at == "2026-08-05T12:05:00+00:00"
    assert ingested_at == "2026-08-05T14:06:00+00:00"


def test_retrodated_late_arrival_resolved_never_precedes_detected(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    skip_payload = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
    )
    _ingest(
        store,
        skip_payload,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    detected_before = _gap_events(store.database_path)[0]

    late_payload = _candles_payload([_bar("2026-08-05T10:00:00+00:00")])
    _ingest(
        store,
        late_payload,
        available_at="2026-08-05T11:01:00Z",
        ingested_at="2026-08-05T11:02:00Z",
    )

    events = {event[1]: event for event in _gap_events(store.database_path)}
    assert set(events) == {"DETECTED", "RESOLVED"}
    assert events["DETECTED"] == detected_before
    resolved_available_at, resolved_ingested_at = events["RESOLVED"][7], events["RESOLVED"][8]
    assert resolved_available_at == "2026-08-05T12:05:00+00:00"
    assert resolved_ingested_at == "2026-08-05T12:06:00+00:00"
    assert datetime.fromisoformat(resolved_ingested_at) >= datetime.fromisoformat(
        detected_before[8]
    )


def test_monotone_flow_causal_timestamps_match_natural_expectation(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload(
            [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
        ),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T10:00:00+00:00")]),
        available_at="2026-08-05T13:05:00Z",
        ingested_at="2026-08-05T13:06:00Z",
    )

    events = {event[1]: event for event in _gap_events(store.database_path)}
    assert events["DETECTED"][7] == "2026-08-05T12:05:00+00:00"
    assert events["DETECTED"][8] == "2026-08-05T12:06:00+00:00"
    assert events["RESOLVED"][7] == "2026-08-05T13:05:00+00:00"
    assert events["RESOLVED"][8] == "2026-08-05T13:06:00+00:00"


def test_causal_timestamps_are_identical_after_replay(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    pair_payload = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
    )
    pair_overrides = {
        "available_at": "2026-08-05T20:05:00Z",
        "ingested_at": "2026-08-05T20:06:00Z",
    }
    _ingest(store, pair_payload, **pair_overrides)
    before = _gap_events(store.database_path)

    _ingest(store, pair_payload, **pair_overrides)

    assert _gap_events(store.database_path) == before


# --- Intra-payload-only detection: no cross-payload gaps, bounded cost --


def test_gap_between_separate_payloads_is_never_detected(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T09:00:00+00:00")]),
        available_at="2026-08-05T10:05:00Z",
        ingested_at="2026-08-05T10:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    assert _gap_event_count(store.database_path) == 0


def test_discontinuous_historical_backfill_never_gets_rejected_or_saturates(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)

    # Build a substantial PRE-EXISTING stock of gap events in the SAME
    # domain, comfortably larger than what the old stock-based cap (CR2-F2)
    # would have tolerated before permanently blocking the domain. Each
    # ingestion stays individually under MAX_GAP_CANDIDATES_PER_INGESTION,
    # but the cumulative stock (8000) proves the cap no longer looks at
    # accumulated history at all.
    stock_base = datetime(2200, 1, 1, tzinfo=timezone.utc)
    for offset in (0, 5000):
        lower = stock_base + timedelta(hours=offset)
        upper = lower + timedelta(hours=4001)  # 4000 candidates, under the cap
        payload = _candles_payload([_bar(_iso(lower)), _bar(_iso(upper))])
        _ingest(
            store,
            payload,
            available_at=_iso(upper + timedelta(hours=2)),
            ingested_at=_iso(upper + timedelta(hours=2, minutes=1)),
        )
    stock_before = _gap_event_count(store.database_path)
    assert stock_before == 8000

    def contiguous_page(start_year: int):
        start = datetime(start_year, 1, 1, tzinfo=timezone.utc)
        rows = [_bar(_iso(start + timedelta(hours=i))) for i in range(300)]
        end = start + timedelta(hours=299)
        return _candles_payload(rows), end

    for year in (2024, 2023, 2020, 2010, 1990):
        payload, end = contiguous_page(year)
        result = _ingest(
            store,
            payload,
            available_at=_iso(end + timedelta(hours=2)),
            ingested_at=_iso(end + timedelta(hours=2, minutes=1)),
        )
        assert result.bar_receipts_inserted == 300

    # The pre-existing 8000-event stock never blocks these unrelated,
    # internally contiguous pages: none of them produces a new gap event.
    assert _gap_event_count(store.database_path) == stock_before
    assert _counts(store.database_path) == (7, 7, 1504)


def test_ingestion_cost_is_bounded_by_payload_not_by_domain_history(
    tmp_path, store_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    queries: list[str] = []
    original_connect = store_module.MarketDataStore._connect

    def traced_connect(self):
        connection = original_connect(self)
        connection.set_trace_callback(lambda sql: queries.append(sql))
        return connection

    monkeypatch.setattr(store_module.MarketDataStore, "_connect", traced_connect)

    def tiny_gapped_payload(start: datetime) -> bytes:
        # One missing candidate (start + 1h): exercises detection, the
        # batched existence check AND _first_seen, not just a bare insert.
        return _candles_payload([_bar(_iso(start)), _bar(_iso(start + timedelta(hours=2)))])

    def ingest_tiny(start: datetime):
        end = start + timedelta(hours=2)
        return _ingest(
            store,
            tiny_gapped_payload(start),
            available_at=_iso(end + timedelta(hours=2)),
            ingested_at=_iso(end + timedelta(hours=2, minutes=1)),
        )

    queries.clear()
    baseline_result = ingest_tiny(GAP_BASE)
    baseline_query_count = len(queries)
    assert baseline_query_count > 0
    assert baseline_result.gap_events_detected == 1

    for page in range(1, 4):
        page_start = GAP_BASE + timedelta(hours=1000 * page)
        rows = [_bar(_iso(page_start + timedelta(hours=i))) for i in range(300)]
        end = page_start + timedelta(hours=299)
        _ingest(
            store,
            _candles_payload(rows),
            available_at=_iso(end + timedelta(hours=2)),
            ingested_at=_iso(end + timedelta(hours=2, minutes=1)),
        )

    queries.clear()
    after_history_result = ingest_tiny(GAP_BASE + timedelta(hours=1000 * 10))
    after_history_query_count = len(queries)

    assert after_history_result.gap_events_detected == 1
    assert after_history_query_count == baseline_query_count


# --- CR3-F1: a bar already known in the store is never a gap -----------


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
    provider/product/timeframe the public ingestion API cannot itself
    produce (PROVIDER is a fixed module constant), so that isolation of the
    existence check can still be proven against a foreign provider."""
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
            (
                ingestion_id,
                provider,
                product_id,
                timeframe,
                bar_close_at,
                bar_close_at,
                payload_sha256,
            ),
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


def test_overlapping_payload_does_not_create_false_gap_for_known_bar(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    payload_a = _candles_payload(
        [
            _bar("2026-08-05T10:00:00+00:00"),
            _bar("2026-08-05T11:00:00+00:00"),
            _bar("2026-08-05T12:00:00+00:00"),
        ]
    )
    _ingest(
        store,
        payload_a,
        available_at="2026-08-05T13:05:00Z",
        ingested_at="2026-08-05T13:06:00Z",
    )

    payload_b = _candles_payload(
        [_bar("2026-08-05T10:00:00+00:00"), _bar("2026-08-05T12:00:00+00:00")]
    )
    result = _ingest(
        store,
        payload_b,
        available_at="2026-08-05T14:05:00Z",
        ingested_at="2026-08-05T14:06:00Z",
    )

    assert result.gap_events_detected == 0
    assert _gap_event_count(store.database_path) == 0
    with sqlite3.connect(store.database_path) as connection:
        present = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT bar_open_at FROM market_bar_receipts"
            )
        }
    assert "2026-08-05T11:00:00+00:00" in present


def test_bar_known_under_other_provider_does_not_suppress_detection(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    _seed_foreign_bar_receipt(
        store.database_path,
        provider="other_provider_rest",
        product_id="BTC-USD",
        timeframe="1h",
        bar_open_at="2026-08-05T11:00:00+00:00",
        bar_close_at="2026-08-05T12:00:00+00:00",
    )

    payload = _candles_payload(
        [_bar("2026-08-05T10:00:00+00:00"), _bar("2026-08-05T12:00:00+00:00")]
    )
    result = _ingest(
        store,
        payload,
        available_at="2026-08-05T13:05:00Z",
        ingested_at="2026-08-05T13:06:00Z",
    )

    assert result.gap_events_detected == 1
    events = _gap_events(store.database_path)
    assert len(events) == 1
    assert events[0][0] == "2026-08-05T11:00:00+00:00"


def test_bar_known_under_other_product_id_does_not_suppress_detection(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        product_id="ETH-USD",
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    payload = _candles_payload(
        [_bar("2026-08-05T10:00:00+00:00"), _bar("2026-08-05T12:00:00+00:00")]
    )
    result = _ingest(
        store,
        payload,
        available_at="2026-08-05T13:05:00Z",
        ingested_at="2026-08-05T13:06:00Z",
    )

    assert result.gap_events_detected == 1
    events = _gap_events(store.database_path)
    assert len(events) == 1
    assert events[0][0] == "2026-08-05T11:00:00+00:00"
    assert events[0][3] == "BTC-USD"


def test_bar_known_under_other_timeframe_does_not_suppress_detection(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-06T00:00:00+00:00")]),
        timeframe="1d",
        available_at="2026-08-07T00:05:00Z",
        ingested_at="2026-08-07T00:06:00Z",
    )

    payload = _candles_payload(
        [_bar("2026-08-05T23:00:00+00:00"), _bar("2026-08-06T01:00:00+00:00")]
    )
    result = _ingest(
        store,
        payload,
        timeframe="1h",
        available_at="2026-08-06T02:05:00Z",
        ingested_at="2026-08-06T02:06:00Z",
    )

    assert result.gap_events_detected == 1
    events = _gap_events(store.database_path)
    assert len(events) == 1
    assert events[0][0] == "2026-08-06T00:00:00+00:00"
    assert events[0][4] == "1h"


def test_multiple_revisions_of_a_known_bar_still_count_as_present(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    revised = _row(
        "2026-08-05T11:00:00+00:00",
        low="100.0",
        high="115.0",
        open_="105.0",
        close="112.0",
        volume="2.0",
    )
    _ingest(
        store,
        _candles_payload([revised]),
        available_at="2026-08-05T14:05:00Z",
        ingested_at="2026-08-05T14:06:00Z",
    )

    payload = _candles_payload(
        [_bar("2026-08-05T10:00:00+00:00"), _bar("2026-08-05T12:00:00+00:00")]
    )
    result = _ingest(
        store,
        payload,
        available_at="2026-08-05T15:05:00Z",
        ingested_at="2026-08-05T15:06:00Z",
    )

    assert result.gap_events_detected == 0
    assert _gap_event_count(store.database_path) == 0


def test_mixed_known_and_missing_candidates_creates_detected_only_for_missing(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    # 11:00 and 13:00 are seeded as separate single-bar ingestions so that
    # seeding itself never pairs them together and creates a gap at 12:00.
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T11:00:00+00:00")]),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    _ingest(
        store,
        _candles_payload([_bar("2026-08-05T13:00:00+00:00")]),
        available_at="2026-08-05T14:05:00Z",
        ingested_at="2026-08-05T14:06:00Z",
    )

    payload = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T14:00:00+00:00")]
    )
    result = _ingest(
        store,
        payload,
        available_at="2026-08-05T15:05:00Z",
        ingested_at="2026-08-05T15:06:00Z",
    )

    assert result.gap_events_detected == 2
    events = _gap_events(store.database_path)
    opens = sorted(event[0] for event in events)
    assert opens == ["2026-08-05T10:00:00+00:00", "2026-08-05T12:00:00+00:00"]


def test_replaying_overlapping_payload_creates_no_new_events(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    payload_a = _candles_payload(
        [
            _bar("2026-08-05T10:00:00+00:00"),
            _bar("2026-08-05T11:00:00+00:00"),
            _bar("2026-08-05T12:00:00+00:00"),
        ]
    )
    _ingest(
        store,
        payload_a,
        available_at="2026-08-05T13:05:00Z",
        ingested_at="2026-08-05T13:06:00Z",
    )
    payload_b = _candles_payload(
        [_bar("2026-08-05T10:00:00+00:00"), _bar("2026-08-05T12:00:00+00:00")]
    )
    overrides = {
        "available_at": "2026-08-05T14:05:00Z",
        "ingested_at": "2026-08-05T14:06:00Z",
    }
    _ingest(store, payload_b, **overrides)

    _ingest(store, payload_b, **overrides)

    assert _gap_event_count(store.database_path) == 0


def test_gap_candidate_limit_is_checked_before_existence_lookup(
    tmp_path, store_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    limit = store_module.MAX_GAP_CANDIDATES_PER_INGESTION
    upper = GAP_BASE + timedelta(hours=limit + 2)
    payload = _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))])

    def forbidden_lookup(*_args, **_kwargs):
        pytest.fail(
            "existence lookup ran before the cardinality preflight rejected the ingestion"
        )

    monkeypatch.setattr(store_module, "_load_known_bar_opens", forbidden_lookup)

    with pytest.raises(store_module.MarketDataStoreError, match="exceeds the maximum"):
        _ingest(
            store,
            payload,
            available_at=_iso(upper + timedelta(hours=1, minutes=5)),
            ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
        )

    assert _counts(store.database_path) == (0, 0, 0)


def test_existence_lookup_failure_rolls_back_atomically(
    tmp_path, store_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    payload = _candles_payload(
        [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
    )
    original_lookup = store_module._load_known_bar_opens

    def failing_lookup(*_args, **_kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(store_module, "_load_known_bar_opens", failing_lookup)

    with pytest.raises(store_module.MarketDataStoreError, match="persistence failed"):
        _ingest(
            store,
            payload,
            available_at="2026-08-05T12:05:00Z",
            ingested_at="2026-08-05T12:06:00Z",
        )
    assert _counts(store.database_path) == (0, 0, 0)
    assert _gap_event_count(store.database_path) == 0

    monkeypatch.setattr(store_module, "_load_known_bar_opens", original_lookup)
    result = _ingest(
        store,
        payload,
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )
    assert result.gap_events_detected == 1


def _assert_indexed_search(plan_rows: list[tuple], *, alias: str, index_name: str) -> None:
    """Assert an EXPLAIN QUERY PLAN shows an indexed SEARCH for `alias` using
    `index_name`, and no full SCAN of it.

    Deliberately robust to SQLite's plan-text format, which has changed
    across versions (e.g. "SCAN TABLE market_bar_receipts" on older SQLite
    vs. plain "SCAN r" on SQLite >= 3.36 when an alias is used) -- a naive
    substring check for one specific format silently stops catching
    regressions once the wording changes.
    """
    detail_text = " ".join(row[3] for row in plan_rows)
    full_scan = re.search(rf"\bSCAN\b(?:\s+TABLE)?\s+(?:\S+\s+AS\s+)?{re.escape(alias)}\b", detail_text)
    indexed_search = re.search(
        rf"\bSEARCH\b\s+{re.escape(alias)}\b\s+USING\s+(?:COVERING\s+)?INDEX\s+{re.escape(index_name)}\b",
        detail_text,
    )
    if full_scan is not None or indexed_search is None:
        raise AssertionError(
            f"expected an indexed SEARCH on alias {alias!r} using index "
            f"{index_name!r}, and no full SCAN of it; got plan detail: "
            f"{detail_text!r}"
        )


def test_indexed_search_guard_rejects_full_table_scan_plans() -> None:
    # A future regression (lost index, lost domain filter, SEARCH replaced by
    # SCAN) must make the guard fail. Prove it does, on synthetic plans
    # matching real SQLite plan-text formats.
    with pytest.raises(AssertionError):
        _assert_indexed_search(
            [(2, 0, 0, "SCAN r")],
            alias="r",
            index_name="market_bar_receipts_open_lookup",
        )
    with pytest.raises(AssertionError):
        _assert_indexed_search(
            [(2, 0, 0, "SCAN TABLE market_bar_receipts AS r")],
            alias="r",
            index_name="market_bar_receipts_open_lookup",
        )
    with pytest.raises(AssertionError):
        _assert_indexed_search(
            [(2, 0, 0, "SEARCH r USING INDEX some_other_index (bar_open_at=?)")],
            alias="r",
            index_name="market_bar_receipts_open_lookup",
        )
    # The matching indexed plan must NOT raise.
    _assert_indexed_search(
        [(2, 0, 0, "SEARCH r USING INDEX market_bar_receipts_open_lookup (bar_open_at=?)")],
        alias="r",
        index_name="market_bar_receipts_open_lookup",
    )


def test_existence_check_and_first_seen_use_indexed_access(
    tmp_path, store_module, monkeypatch
) -> None:
    store = _store(store_module, tmp_path)
    captured_sql: list[str] = []
    original_connect = store_module.MarketDataStore._connect

    def traced_connect(self):
        connection = original_connect(self)
        connection.set_trace_callback(captured_sql.append)
        return connection

    monkeypatch.setattr(store_module.MarketDataStore, "_connect", traced_connect)

    # A payload with one interior gap exercises both _load_known_bar_opens
    # (the batched existence check) and _first_seen (causal timestamps for
    # the bounding pair), so their REAL, production-emitted SQL is captured
    # below -- not a hand-copied approximation of it.
    _ingest(
        store,
        _candles_payload(
            [_bar("2026-08-05T09:00:00+00:00"), _bar("2026-08-05T11:00:00+00:00")]
        ),
        available_at="2026-08-05T12:05:00Z",
        ingested_at="2026-08-05T12:06:00Z",
    )

    existence_statements = [
        sql
        for sql in captured_sql
        if "FROM market_bar_receipts r" in sql and " IN (" in sql
    ]
    first_seen_statements = [
        sql
        for sql in captured_sql
        if "FROM market_bar_receipts r" in sql and "ORDER BY r.ingested_at" in sql
    ]
    assert existence_statements, (
        "_load_known_bar_opens' query was never executed by production code "
        "for this ingestion -- nothing was captured to verify"
    )
    assert first_seen_statements, (
        "_first_seen's query was never executed by production code for this "
        "ingestion -- nothing was captured to verify"
    )

    with sqlite3.connect(store.database_path) as connection:
        for sql in existence_statements:
            plan = connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            _assert_indexed_search(
                plan, alias="r", index_name="market_bar_receipts_open_lookup"
            )
        for sql in first_seen_statements:
            plan = connection.execute("EXPLAIN QUERY PLAN " + sql).fetchall()
            _assert_indexed_search(
                plan, alias="r", index_name="market_bar_receipts_open_lookup"
            )


# =========================================================================
# Phase 1C-C: write-context PRAGMAs, snapshot schema, and its structural
# verification. The snapshot MATERIALIZATION primitive itself lives in
# market_snapshots.py and is tested there; this file owns the schema.
# =========================================================================


SNAPSHOT_MANIFESTS = "market_snapshot_manifests"
SNAPSHOT_ENTRIES = "market_snapshot_entries"


def _seed_snapshot(database: Path, *, snapshot_id: str, content_sha256: str, bar_open_at: str,
                   entries_content_hash: str = "e" * 64, snapshot_request_id: str = "req-1") -> None:
    """Seed one snapshot directly (entries first, manifest last -- the order
    the sealing trigger imposes), bypassing the materialization primitive so
    the SCHEMA can be tested independently of it."""
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA recursive_triggers = ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            f"INSERT INTO {SNAPSHOT_ENTRIES} (snapshot_id, bar_open_at, content_sha256) VALUES (?, ?, ?)",
            (snapshot_id, bar_open_at, content_sha256),
        )
        connection.execute(
            f"INSERT INTO {SNAPSHOT_MANIFESTS} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot_id, snapshot_request_id, entries_content_hash,
                "trading-lab.market-snapshot.v1",
                "trading-lab.market-snapshot-selection.v1",
                "coinbase_exchange_rest", "BTC-USD", "1h",
                "2026-08-02T00:00:00+00:00", "2026-08-03T00:00:00+00:00",
                "2026-08-03T00:00:00+00:00", 1,
            ),
        )
        connection.commit()


def _one_receipt(database: Path) -> tuple[str, str]:
    with sqlite3.connect(database) as connection:
        return connection.execute(
            "SELECT content_sha256, bar_open_at FROM market_bar_receipts LIMIT 1"
        ).fetchone()


def _insert_only_tables(database: Path) -> set[str]:
    """Discover insert-only tables from the schema itself, so a table added
    later without its protections cannot silently escape this test."""
    with sqlite3.connect(database) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT tbl_name FROM sqlite_master WHERE type = 'trigger' "
                "AND name LIKE '%_no_delete'"
            )
        }


def _table_rows(database: Path, table: str) -> list[tuple]:
    with sqlite3.connect(database) as connection:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()


# --- 1C-C: write-context PRAGMAs (TM-1 / TM-2) ---------------------------


def test_store_connection_enables_foreign_keys_and_recursive_triggers(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)

    connection = store._connect()
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA recursive_triggers").fetchone()[0] == 1
    finally:
        connection.close()


def test_insert_or_replace_cannot_bypass_any_append_only_trigger(
    tmp_path, store_module
) -> None:
    """Without recursive_triggers, INSERT OR REPLACE deletes the conflicting
    row WITHOUT firing the BEFORE DELETE trigger, silently rewriting a row
    the schema calls immutable. Every insert-only table must refuse it, and
    the original row must survive byte for byte."""
    store = _store(store_module, tmp_path)
    # A gap-producing ingestion so market_data_gap_events also holds a row:
    # every insert-only table must be covered, not just the convenient ones.
    upper = GAP_BASE + timedelta(hours=3)
    _ingest(
        store,
        _candles_payload([_bar(_iso(GAP_BASE)), _bar(_iso(upper))]),
        available_at=_iso(upper + timedelta(hours=1, minutes=5)),
        ingested_at=_iso(upper + timedelta(hours=1, minutes=6)),
    )
    content_sha256, bar_open_at = _one_receipt(store.database_path)
    _seed_snapshot(
        store.database_path,
        snapshot_id="snap-replace",
        content_sha256=content_sha256,
        bar_open_at=bar_open_at,
    )

    tables = _insert_only_tables(store.database_path)
    assert {SNAPSHOT_MANIFESTS, SNAPSHOT_ENTRIES}.issubset(tables)
    assert len(tables) >= 6

    for table in sorted(tables):
        before = _table_rows(store.database_path, table)
        assert before, f"{table} needs a row for this test to mean anything"
        with sqlite3.connect(store.database_path) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA recursive_triggers = ON")
            columns = [row[1] for row in connection.execute(f"PRAGMA table_info({table})")]
            primary_key = {
                row[1] for row in connection.execute(f"PRAGMA table_info({table})") if row[5]
            }
            hijacked = []
            for column, value in zip(columns, before[0]):
                if column in primary_key:
                    hijacked.append(value)
                elif isinstance(value, int):
                    hijacked.append(value + 4242)
                elif isinstance(value, bytes):
                    hijacked.append(b"hijacked")
                else:
                    hijacked.append("hijacked")
            placeholders = ", ".join("?" for _ in columns)
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(
                    f"INSERT OR REPLACE INTO {table} VALUES ({placeholders})", hijacked
                )
            connection.rollback()

        assert _table_rows(store.database_path, table) == before, f"{table} was rewritten"


# --- 1C-C: snapshot schema ------------------------------------------------


def _table_info(database: Path, table: str) -> list[tuple]:
    with sqlite3.connect(database) as connection:
        return [
            (row[1], row[2], row[3], row[5])  # name, type, notnull, pk
            for row in connection.execute(f"PRAGMA table_info({table})")
        ]


def test_snapshot_manifest_table_has_the_exact_expected_structure(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)

    assert _table_info(store.database_path, SNAPSHOT_MANIFESTS) == [
        ("snapshot_id", "TEXT", 1, 1),
        ("snapshot_request_id", "TEXT", 1, 0),
        ("entries_content_hash", "TEXT", 1, 0),
        ("snapshot_schema_version", "TEXT", 1, 0),
        ("selection_policy_version", "TEXT", 1, 0),
        ("provider", "TEXT", 1, 0),
        ("product_id", "TEXT", 1, 0),
        ("timeframe", "TEXT", 1, 0),
        ("range_start", "TEXT", 1, 0),
        ("range_end", "TEXT", 1, 0),
        ("as_of", "TEXT", 1, 0),
        ("entry_count", "INTEGER", 1, 0),
    ]


def test_snapshot_entries_table_has_the_exact_expected_structure(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)

    assert _table_info(store.database_path, SNAPSHOT_ENTRIES) == [
        ("snapshot_id", "TEXT", 1, 1),
        ("bar_open_at", "TEXT", 1, 2),
        ("content_sha256", "TEXT", 1, 0),
    ]


def test_snapshot_entries_table_is_without_rowid(tmp_path, store_module) -> None:
    """WITHOUT ROWID removes the rowid entirely, so no ordering or identity
    can accidentally come to depend on it."""
    store = _store(store_module, tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        with pytest.raises(sqlite3.OperationalError, match="rowid"):
            connection.execute(f"SELECT rowid FROM {SNAPSHOT_ENTRIES}").fetchall()
        connection.execute("SELECT rowid FROM market_bar_receipts").fetchall()


def test_snapshot_entries_foreign_keys_are_exact(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        foreign_keys = {
            (row[2], row[3], row[4])  # target table, from column, to column
            for row in connection.execute(f"PRAGMA foreign_key_list({SNAPSHOT_ENTRIES})")
        }
        (definition,) = connection.execute(
            "SELECT sql FROM sqlite_master WHERE name = ?", (SNAPSHOT_ENTRIES,)
        ).fetchone()

    assert foreign_keys == {
        (SNAPSHOT_MANIFESTS, "snapshot_id", "snapshot_id"),
        ("market_bar_receipts", "content_sha256", "content_sha256"),
    }
    # The manifest link must be DEFERRABLE INITIALLY DEFERRED: entries are
    # inserted BEFORE their parent manifest, which seals them. PRAGMA
    # foreign_key_list exposes no deferrability column, so this is asserted on
    # the definition, and behaviorally by test_orphan_entries_are_refused_at_commit.
    assert "DEFERRABLE INITIALLY DEFERRED" in definition


def test_snapshot_manifest_has_exactly_one_expected_index(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        explicit = [
            row[1]
            for row in connection.execute(f"PRAGMA index_list({SNAPSHOT_MANIFESTS})")
            if not row[1].startswith("sqlite_autoindex")
        ]
        columns = [
            row[2]
            for row in connection.execute(
                "PRAGMA index_info(market_snapshot_manifests_request_lookup)"
            )
        ]

    assert explicit == ["market_snapshot_manifests_request_lookup"]
    assert columns == ["snapshot_request_id", "snapshot_id"]


def test_snapshot_tables_have_all_five_immutability_triggers(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        triggers = {
            row[0]: row[1]
            for row in connection.execute(
                "SELECT name, tbl_name FROM sqlite_master WHERE type = 'trigger' "
                "AND tbl_name LIKE 'market_snapshot%'"
            )
        }

    assert triggers == {
        "market_snapshot_manifests_no_update": SNAPSHOT_MANIFESTS,
        "market_snapshot_manifests_no_delete": SNAPSHOT_MANIFESTS,
        "market_snapshot_entries_no_update": SNAPSHOT_ENTRIES,
        "market_snapshot_entries_no_delete": SNAPSHOT_ENTRIES,
        "market_snapshot_entries_no_late_insert": SNAPSHOT_ENTRIES,
    }


def test_snapshot_schema_adds_no_unexpected_object(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        }

    assert tables == {
        "raw_market_payloads",
        "market_ingestions",
        "market_bar_receipts",
        "market_data_gap_events",
        SNAPSHOT_MANIFESTS,
        SNAPSHOT_ENTRIES,
    }


# --- 1C-C: sealing and immutability at the schema level -------------------


def test_manifest_seals_its_entries_against_late_insertion(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-02T10:00:00+00:00")]))
    content_sha256, bar_open_at = _one_receipt(store.database_path)
    _seed_snapshot(
        store.database_path, snapshot_id="sealed",
        content_sha256=content_sha256, bar_open_at=bar_open_at,
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.DatabaseError, match="sealed by its manifest"):
            connection.execute(
                f"INSERT INTO {SNAPSHOT_ENTRIES} VALUES (?, ?, ?)",
                ("sealed", "2026-08-02T11:00:00+00:00", content_sha256),
            )
        connection.rollback()

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(
            f"SELECT COUNT(*) FROM {SNAPSHOT_ENTRIES} WHERE snapshot_id = 'sealed'"
        ).fetchone()[0] == 1


def test_snapshot_rows_reject_update_and_delete(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-02T10:00:00+00:00")]))
    content_sha256, bar_open_at = _one_receipt(store.database_path)
    _seed_snapshot(
        store.database_path, snapshot_id="frozen",
        content_sha256=content_sha256, bar_open_at=bar_open_at,
    )

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for statement, message in (
            (f"UPDATE {SNAPSHOT_MANIFESTS} SET entry_count = 99", "manifests is insert-only"),
            (f"DELETE FROM {SNAPSHOT_MANIFESTS}", "manifests is insert-only"),
            (f"UPDATE {SNAPSHOT_ENTRIES} SET content_sha256 = 'x'", "entries is insert-only"),
            (f"DELETE FROM {SNAPSHOT_ENTRIES}", "entries is insert-only"),
        ):
            with pytest.raises(sqlite3.DatabaseError, match=message):
                connection.execute(statement)
            connection.rollback()


def test_snapshot_entry_cannot_reference_an_unknown_receipt(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(
                f"INSERT INTO {SNAPSHOT_ENTRIES} VALUES (?, ?, ?)",
                ("ghost", "2026-08-02T10:00:00+00:00", "f" * 64),
            )
        connection.rollback()


def test_orphan_entries_are_refused_at_commit(tmp_path, store_module) -> None:
    """The deferred manifest FK is what makes 'entries first' legal; it must
    still refuse a snapshot whose manifest never arrives."""
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-02T10:00:00+00:00")]))
    content_sha256, bar_open_at = _one_receipt(store.database_path)

    connection = sqlite3.connect(store.database_path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            f"INSERT INTO {SNAPSHOT_ENTRIES} VALUES (?, ?, ?)",
            ("orphan", bar_open_at, content_sha256),
        )
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.commit()
        connection.rollback()
    finally:
        connection.close()

    with sqlite3.connect(store.database_path) as connection:
        assert connection.execute(f"SELECT COUNT(*) FROM {SNAPSHOT_ENTRIES}").fetchone()[0] == 0


def test_two_entries_for_the_same_opening_are_impossible(tmp_path, store_module) -> None:
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-02T10:00:00+00:00")]))
    content_sha256, bar_open_at = _one_receipt(store.database_path)

    with sqlite3.connect(store.database_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            f"INSERT INTO {SNAPSHOT_ENTRIES} VALUES (?, ?, ?)", ("dup", bar_open_at, content_sha256)
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                f"INSERT INTO {SNAPSHOT_ENTRIES} VALUES (?, ?, ?)",
                ("dup", bar_open_at, content_sha256),
            )
        connection.rollback()


# --- 1C-C: structural schema verification (managerial amendment) ---------


def _phase_1b_database(tmp_path, store_module) -> Path:
    """A database carrying the Phase 1B schema only, built by removing the
    Phase 1C-C objects -- the exact shape a pre-1C-C deployment has."""
    store = _store(store_module, tmp_path)
    _ingest(store, _candles_payload([_bar("2026-08-02T10:00:00+00:00")]))
    with sqlite3.connect(store.database_path) as connection:
        for name in (
            "market_snapshot_entries_no_late_insert",
            "market_snapshot_entries_no_update",
            "market_snapshot_entries_no_delete",
            "market_snapshot_manifests_no_update",
            "market_snapshot_manifests_no_delete",
        ):
            connection.execute(f"DROP TRIGGER IF EXISTS {name}")
        connection.execute("DROP INDEX IF EXISTS market_snapshot_manifests_request_lookup")
        connection.execute(f"DROP TABLE IF EXISTS {SNAPSHOT_ENTRIES}")
        connection.execute(f"DROP TABLE IF EXISTS {SNAPSHOT_MANIFESTS}")
    return store.database_path


def test_existing_phase_1b_database_is_migrated_on_reopen(tmp_path, store_module) -> None:
    database = _phase_1b_database(tmp_path, store_module)
    before = _counts(database)

    store_module.MarketDataStore(database)

    with sqlite3.connect(database) as connection:
        names = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master")
        }
    assert SNAPSHOT_MANIFESTS in names and SNAPSHOT_ENTRIES in names
    assert "market_snapshot_entries_no_late_insert" in names
    assert _counts(database) == before, "historical receipts must be untouched"


def test_reopening_is_idempotent_and_creates_no_duplicate_object(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        first = sorted(row[0] for row in connection.execute("SELECT name FROM sqlite_master"))

    store_module.MarketDataStore(store.database_path)
    store_module.MarketDataStore(store.database_path)

    with sqlite3.connect(store.database_path) as connection:
        third = sorted(row[0] for row in connection.execute("SELECT name FROM sqlite_master"))
    assert first == third
    assert len(third) == len(set(third))


def test_a_wrongly_shaped_table_with_the_right_name_is_refused(
    tmp_path, store_module
) -> None:
    """CREATE TABLE IF NOT EXISTS is a no-op against an existing table with
    the same name and a different structure: initialisation must fail rather
    than silently accept it."""
    database = _phase_1b_database(tmp_path, store_module)
    with sqlite3.connect(database) as connection:
        connection.execute(
            f"CREATE TABLE {SNAPSHOT_MANIFESTS} (snapshot_id TEXT, unexpected TEXT)"
        )

    with pytest.raises(store_module.MarketDataStoreError, match="schema"):
        store_module.MarketDataStore(database)


def test_a_wrong_index_with_the_right_name_is_refused(tmp_path, store_module) -> None:
    database = _phase_1b_database(tmp_path, store_module)
    with sqlite3.connect(database) as connection:
        connection.executescript(
            f"""
            CREATE TABLE {SNAPSHOT_MANIFESTS} (
                snapshot_id TEXT PRIMARY KEY NOT NULL,
                snapshot_request_id TEXT NOT NULL,
                entries_content_hash TEXT NOT NULL,
                snapshot_schema_version TEXT NOT NULL,
                selection_policy_version TEXT NOT NULL,
                provider TEXT NOT NULL,
                product_id TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                range_start TEXT NOT NULL,
                range_end TEXT NOT NULL,
                as_of TEXT NOT NULL,
                entry_count INTEGER NOT NULL
                    CHECK (entry_count >= 0 AND entry_count <= 10000),
                CHECK (range_end > range_start)
            );
            CREATE INDEX market_snapshot_manifests_request_lookup
                ON {SNAPSHOT_MANIFESTS} (provider);
            """
        )

    with pytest.raises(store_module.MarketDataStoreError, match="schema"):
        store_module.MarketDataStore(database)


def test_an_inoperative_trigger_with_the_right_name_is_refused(
    tmp_path, store_module
) -> None:
    """A trigger carrying the expected name but raising nothing would leave
    the snapshot mutable while every name-based check still passed."""
    store = _store(store_module, tmp_path)
    with sqlite3.connect(store.database_path) as connection:
        connection.execute("DROP TRIGGER market_snapshot_entries_no_late_insert")
        connection.execute(
            f"CREATE TRIGGER market_snapshot_entries_no_late_insert "
            f"BEFORE INSERT ON {SNAPSHOT_ENTRIES} BEGIN SELECT 1; END"
        )

    with pytest.raises(store_module.MarketDataStoreError, match="schema"):
        store_module.MarketDataStore(store.database_path)


# --- journal-mode-hardening: WAL is a persistent, verified property ------


import os
import shutil
import subprocess
import sys
import textwrap
import time


def _pragma(connection, name):
    return connection.execute(f"PRAGMA {name}").fetchone()[0]


def _raw(database, *, busy_ms=250):
    connection = sqlite3.connect(database, timeout=busy_ms / 1000)
    connection.execute(f"PRAGMA busy_timeout = {busy_ms}")
    return connection


def _delete_mode_database(path):
    """A real file database left in SQLite's default journal_mode."""
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE probe (x INTEGER)")
        connection.execute("INSERT INTO probe VALUES (1)")
        connection.commit()
        assert _pragma(connection, "journal_mode") == "delete"
    finally:
        connection.close()
    return path


def test_a_delete_database_is_migrated_to_wal_and_the_mode_persists(
    tmp_path, store_module
) -> None:
    """journal_mode is a property of the FILE, not of a connection. Opening
    the store must migrate it once, and a completely independent connection
    opened afterwards must still see wal."""
    database = _delete_mode_database(tmp_path / "market-data.sqlite3")
    store = store_module.MarketDataStore(database)

    connection = store._connect()
    try:
        assert _pragma(connection, "journal_mode") == "wal"
    finally:
        connection.close()

    # Nothing of ours is open any more: read the mode back from the file.
    independent = sqlite3.connect(database)
    try:
        assert _pragma(independent, "journal_mode") == "wal"
    finally:
        independent.close()


def test_every_store_connection_carries_the_full_pragma_contract(
    tmp_path, store_module
) -> None:
    store = _store(store_module, tmp_path)
    for _ in range(2):  # the contract holds on later connections too
        connection = store._connect()
        try:
            assert _pragma(connection, "journal_mode") == "wal"
            assert _pragma(connection, "synchronous") == 2  # FULL, never lowered
            assert _pragma(connection, "busy_timeout") == 30_000
            assert _pragma(connection, "foreign_keys") == 1
            assert _pragma(connection, "recursive_triggers") == 1
            assert _pragma(connection, "wal_autocheckpoint") == 1_000
        finally:
            connection.close()


def test_connect_refuses_a_database_whose_journal_mode_was_downgraded(
    tmp_path, store_module
) -> None:
    """_connect verifies; it never repairs. A store whose file was taken back
    to DELETE behind our back is a broken deployment, not something to
    silently migrate mid-flight."""
    store = _store(store_module, tmp_path)
    saboteur = sqlite3.connect(store.database_path)
    try:
        assert saboteur.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    finally:
        saboteur.close()

    with pytest.raises(store_module.MarketDataStoreError) as excinfo:
        store._connect()
    assert "journal_mode" in str(excinfo.value)
    assert not isinstance(excinfo.value, store_module.MarketDataStoreBusy)

    # The failed verification must not have migrated anything.
    after = sqlite3.connect(store.database_path)
    try:
        assert _pragma(after, "journal_mode") == "delete"
    finally:
        after.close()


def test_connect_contains_no_journal_mode_migration_statement(store_module) -> None:
    """Guard on the split: __init__ may migrate, _connect may not. A grep is
    the cheapest way to keep a future edit from quietly re-adding it."""
    import inspect

    store = store_module.MarketDataStore
    read_path = "".join(
        inspect.getsource(member)
        for member in (store._connect, store._open, store._require_persistent_contract)
    )
    # The read path must READ the mode...
    assert re.search(r"PRAGMA\s+journal_mode(?!\s*=)", read_path), read_path
    # ...and must never issue the PRAGMA that ASSIGNS it.
    assert not re.search(r"PRAGMA\s+journal_mode\s*=", read_path), read_path
    # Exactly one place is allowed to, and it is the migration.
    migration = inspect.getsource(store._migrate_journal_mode)
    assert re.search(r"PRAGMA\s+journal_mode\s*=", migration), migration
    # Across the whole store class, exactly one statement assigns the mode.
    class_source = inspect.getsource(store)
    assert len(re.findall(r"PRAGMA\s+journal_mode\s*=", class_source)) == 1, class_source


@pytest.mark.parametrize("answer", ["delete", "memory", "truncate", "", None])
def test_a_refused_wal_migration_fails_closed(
    tmp_path, store_module, monkeypatch, answer
) -> None:
    """If SQLite answers anything but wal, there is no fallback: the store
    refuses to exist rather than run on a mode it did not ask for."""
    database = _delete_mode_database(tmp_path / "refused.sqlite3")
    real_connect = sqlite3.connect

    class _LyingConnection:
        def __init__(self, inner):
            self._inner = inner
            self.statements: list[str] = []

        def execute(self, sql, parameters=()):
            self.statements.append(sql)
            if re.match(r"\s*PRAGMA\s+journal_mode\s*=", sql, re.IGNORECASE):
                class _Answer:
                    def fetchone(self_inner):
                        return None if answer is None else (answer,)
                return _Answer()
            return self._inner.execute(sql, parameters)

        def executescript(self, sql):
            return self._inner.executescript(sql)

        def close(self):
            return self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    made: list = []

    def _fake_connect(*args, **kwargs):
        wrapper = _LyingConnection(real_connect(*args, **kwargs))
        made.append(wrapper)
        return wrapper

    monkeypatch.setattr(store_module.sqlite3, "connect", _fake_connect)
    with pytest.raises(store_module.MarketDataStoreError) as excinfo:
        store_module.MarketDataStore(database)
    monkeypatch.undo()

    assert not isinstance(excinfo.value, store_module.MarketDataStoreBusy)
    # No business schema may have run after the verdict.
    joined = " ".join(s for wrapper in made for s in wrapper.statements)
    assert "CREATE TABLE" not in joined.upper()
    survivor = sqlite3.connect(database)
    try:
        tables = {
            row[0]
            for row in survivor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert "raw_market_payloads" not in tables
    finally:
        survivor.close()


def test_a_locked_database_reports_a_retryable_busy_error(
    tmp_path, store_module
) -> None:
    """A real SQLITE_BUSY, not a simulation: a held RESERVED lock makes the
    journal-mode change fail immediately. The caller -- not this store --
    decides whether to retry."""
    database = _delete_mode_database(tmp_path / "busy.sqlite3")
    blocker = _raw(database)
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("INSERT INTO probe VALUES (2)")
    try:
        with pytest.raises(store_module.MarketDataStoreBusy) as excinfo:
            store_module.MarketDataStore(database)
        cause = excinfo.value.__cause__
        assert isinstance(cause, sqlite3.Error)
        assert cause.sqlite_errorname == "SQLITE_BUSY"
        assert isinstance(excinfo.value, store_module.MarketDataStoreError)

        # Nothing may have been half-initialised.
        inspector = _raw(database)
        try:
            tables = {
                row[0]
                for row in inspector.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
            assert "raw_market_payloads" not in tables
            assert _pragma(inspector, "journal_mode") == "delete"
        finally:
            inspector.close()
    finally:
        blocker.rollback()
        blocker.close()

    # The caller retries explicitly, and now it works.
    store = store_module.MarketDataStore(database)
    connection = store._connect()
    try:
        assert _pragma(connection, "journal_mode") == "wal"
    finally:
        connection.close()


def test_busy_is_recognised_by_sqlite_errorname_not_by_message_text(
    tmp_path, store_module, monkeypatch
) -> None:
    """An error that merely READS like a lock must not be called retryable."""
    database = _delete_mode_database(tmp_path / "impostor.sqlite3")
    impostor = sqlite3.OperationalError("database is locked")
    impostor.sqlite_errorname = "SQLITE_ERROR"
    real_connect = sqlite3.connect

    class _RaisingConnection:
        def __init__(self, inner):
            self._inner = inner

        def execute(self, sql, parameters=()):
            if re.match(r"\s*PRAGMA\s+journal_mode\s*=", sql, re.IGNORECASE):
                raise impostor
            return self._inner.execute(sql, parameters)

        def close(self):
            return self._inner.close()

        def __getattr__(self, name):
            return getattr(self._inner, name)

    monkeypatch.setattr(
        store_module.sqlite3, "connect",
        lambda *a, **k: _RaisingConnection(real_connect(*a, **k)),
    )
    with pytest.raises(store_module.MarketDataStoreError) as excinfo:
        store_module.MarketDataStore(database)
    assert not isinstance(excinfo.value, store_module.MarketDataStoreBusy), excinfo.value
    assert excinfo.value.__cause__ is impostor


@pytest.mark.parametrize(
    "target",
    [":memory:", "file::memory:", "file::memory:?cache=shared", "file:tmp?mode=memory"],
)
def test_in_memory_databases_are_refused_instead_of_silently_degraded(
    store_module, target
) -> None:
    """PRAGMA journal_mode=WAL answers 'memory' on an in-memory database.
    A store whose whole contract is persistent WAL must say so, not pretend."""
    with pytest.raises(store_module.MarketDataStoreError) as excinfo:
        store_module.MarketDataStore(target)
    message = str(excinfo.value)
    assert "memory" in message.lower()
    assert "file" in message.lower()


def test_a_writer_commits_while_a_reader_holds_an_open_transaction(
    tmp_path, store_module
) -> None:
    """The point of the whole gate. In DELETE a reader's open transaction
    blocks every writer's COMMIT; in WAL the writer commits, and the reader
    keeps the coherent snapshot it started with."""
    store = _store(store_module, tmp_path)
    seed = store._connect()
    try:
        seed.execute("BEGIN IMMEDIATE")
        seed.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("a" * 64, "coinbase_exchange_rest", b"one", 3, "2026-08-05T00:00:00Z"),
        )
        seed.commit()
    finally:
        seed.close()

    reader = store._connect()
    writer = store._connect()
    try:
        reader.execute("BEGIN")
        assert reader.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone() == (1,)

        started = time.perf_counter()
        writer.execute("BEGIN IMMEDIATE")
        writer.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("b" * 64, "coinbase_exchange_rest", b"two", 3, "2026-08-05T01:00:00Z"),
        )
        writer.commit()
        elapsed = time.perf_counter() - started
        # It committed, and it did not have to wait out a busy_timeout to do it.
        assert elapsed < 5.0, elapsed
        assert writer.in_transaction is False

        # The reader still sees its own consistent snapshot, not a hybrid.
        assert reader.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone() == (1,)
        reader.commit()

        # A NEW transaction sees the committed state.
        assert reader.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone() == (2,)
    finally:
        reader.close()
        writer.close()


def test_the_same_reader_writer_scenario_blocks_in_delete_mode(tmp_path) -> None:
    """Control: without WAL the identical sequence fails. This is what the
    gate removes."""
    database = _delete_mode_database(tmp_path / "control.sqlite3")
    reader = _raw(database)
    writer = _raw(database)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM probe").fetchone()
        writer.execute("BEGIN IMMEDIATE")
        writer.execute("INSERT INTO probe VALUES (2)")
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            writer.commit()
        assert excinfo.value.sqlite_errorname == "SQLITE_BUSY"
    finally:
        writer.rollback()
        reader.rollback()
        writer.close()
        reader.close()


def test_writers_are_still_serialized_under_wal(tmp_path, store_module) -> None:
    """WAL removes the reader->writer block. It does NOT make SQLite
    multi-writer, and this gate must not be read as claiming that."""
    store = _store(store_module, tmp_path)
    first = _raw(store.database_path)
    second = _raw(store.database_path)
    try:
        assert _pragma(first, "journal_mode") == "wal"
        first.execute("BEGIN IMMEDIATE")
        first.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("c" * 64, "coinbase_exchange_rest", b"x", 1, "2026-08-05T00:00:00Z"),
        )
        with pytest.raises(sqlite3.OperationalError) as excinfo:
            second.execute("BEGIN IMMEDIATE")
        assert excinfo.value.sqlite_errorname == "SQLITE_BUSY"
        first.commit()
        # Once the first writer is done, the second gets its turn.
        second.execute("BEGIN IMMEDIATE")
        second.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("d" * 64, "coinbase_exchange_rest", b"y", 1, "2026-08-05T01:00:00Z"),
        )
        second.commit()
        assert second.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone() == (2,)
    finally:
        first.close()
        second.close()


def test_a_killed_process_leaves_a_consistent_usable_wal_database(
    tmp_path, store_module
) -> None:
    """Process kill only -- this proves nothing about power loss, fsync or
    hardware, and the documentation says so."""
    database = tmp_path / "crash.sqlite3"
    store_module.MarketDataStore(database)
    script = textwrap.dedent(
        f"""
        import sqlite3, sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from scripts.trading_lab.market_data_store import MarketDataStore
        store = MarketDataStore({str(database)!r})
        connection = store._connect()
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("e" * 64, "coinbase_exchange_rest", b"killed", 6, "2026-08-05T00:00:00Z"),
        )
        print("READY", flush=True)
        time.sleep(30)
        """
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "READY"
        child.kill()
    finally:
        child.wait(timeout=30)

    survivor = sqlite3.connect(database, timeout=10.0)
    try:
        assert survivor.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert _pragma(survivor, "journal_mode") == "wal"
        # The uncommitted transaction is gone, and the database still works.
        assert survivor.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone() == (0,)
        survivor.execute("PRAGMA busy_timeout = 5000")
        survivor.execute("BEGIN IMMEDIATE")
        survivor.execute(
            "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
            ("f" * 64, "coinbase_exchange_rest", b"after", 5, "2026-08-05T02:00:00Z"),
        )
        survivor.commit()
        assert survivor.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone() == (1,)
    finally:
        survivor.close()


def test_wal_sidecars_are_part_of_the_live_database(tmp_path, store_module) -> None:
    """Copying the main file alone while WAL is live is NOT a backup: the
    recent commits live in the -wal until a checkpoint folds them in."""
    store = _store(store_module, tmp_path)
    connection = store._connect()
    try:
        for index in range(50):
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO raw_market_payloads VALUES (?, ?, ?, ?, ?)",
                (f"{index:064d}", "coinbase_exchange_rest", b"x", 1, "2026-08-05T00:00:00Z"),
            )
            connection.commit()

        wal = Path(f"{store.database_path}-wal")
        shm = Path(f"{store.database_path}-shm")
        assert wal.exists() and wal.stat().st_size > 0, "the -wal holds live committed data"
        assert shm.exists(), "the -shm is live shared state"

        # A naive "backup" of the main file alone, taken hot.
        naive = tmp_path / "naive-copy.sqlite3"
        shutil.copyfile(store.database_path, naive)
        copy = sqlite3.connect(naive)
        try:
            copied = copy.execute("SELECT COUNT(*) FROM raw_market_payloads").fetchone()[0]
        finally:
            copy.close()
        assert connection.execute(
            "SELECT COUNT(*) FROM raw_market_payloads"
        ).fetchone()[0] == 50
        assert copied < 50, (
            "a hot copy of the main file alone silently loses the -wal contents; "
            f"it showed {copied} of 50 rows"
        )
    finally:
        connection.close()

    # A clean close checkpoints and removes the sidecars: the file alone is
    # then a coherent offline copy.
    assert not Path(f"{store.database_path}-wal").exists()
    assert not Path(f"{store.database_path}-shm").exists()
    offline = tmp_path / "offline-copy.sqlite3"
    shutil.copyfile(store.database_path, offline)
    verified = sqlite3.connect(offline)
    try:
        assert verified.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert verified.execute(
            "SELECT COUNT(*) FROM raw_market_payloads"
        ).fetchone()[0] == 50
    finally:
        verified.close()


class _PragmaScriptedConnection:
    """A connection whose PRAGMA answers are scripted, so each guard can be
    isolated from the ones that would otherwise dominate it."""

    def __init__(self, inner, *, journal_reads=None, journal_assignment=None,
                 synchronous=None):
        self._inner = inner
        self._journal_reads = list(journal_reads or [])
        self._journal_assignment = journal_assignment
        self._synchronous = synchronous

    def execute(self, sql, parameters=()):
        stripped = sql.strip()
        if re.fullmatch(r"PRAGMA\s+journal_mode\s*=\s*\w+", stripped, re.IGNORECASE):
            if self._journal_assignment is not None:
                return _FixedAnswer(self._journal_assignment)
        elif re.fullmatch(r"PRAGMA\s+journal_mode", stripped, re.IGNORECASE):
            if self._journal_reads:
                return _FixedAnswer(self._journal_reads.pop(0))
        elif re.fullmatch(r"PRAGMA\s+synchronous", stripped, re.IGNORECASE):
            if self._synchronous is not None:
                return _FixedAnswer(self._synchronous)
        return self._inner.execute(sql, parameters)

    def close(self):
        return self._inner.close()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _FixedAnswer:
    def __init__(self, value):
        self._value = value

    def fetchone(self):
        return None if self._value is None else (self._value,)


def _scripted_store(store_module, monkeypatch, database, **script):
    real_connect = sqlite3.connect
    monkeypatch.setattr(
        store_module.sqlite3, "connect",
        lambda *a, **k: _PragmaScriptedConnection(real_connect(*a, **k), **script),
    )
    try:
        with pytest.raises(store_module.MarketDataStoreError) as excinfo:
            store_module.MarketDataStore(database)
    finally:
        monkeypatch.undo()
    return excinfo.value


def test_a_migration_answer_other_than_wal_is_refused_even_if_a_later_read_agrees(
    tmp_path, store_module, monkeypatch
) -> None:
    """The migration's OWN answer is the verdict. A build that trusted a
    later re-read instead would accept a driver that quietly refused the
    change and then reported success -- exactly the fallback this store
    forbids."""
    database = _delete_mode_database(tmp_path / "liar.sqlite3")
    error = _scripted_store(
        store_module, monkeypatch, database,
        journal_reads=["delete", "wal", "wal", "wal"],  # first read: needs migrating
        journal_assignment="delete",                     # ...which then refuses
    )
    assert "fallback" in str(error).lower(), error
    assert "'delete'" in str(error), error
    assert not isinstance(error, store_module.MarketDataStoreBusy)


def test_a_connection_reporting_degraded_synchronous_is_refused(
    tmp_path, store_module, monkeypatch
) -> None:
    """synchronous=NORMAL is the classic WAL "optimisation". This store does
    not take it, and does not accept a connection that arrives with it."""
    database = _delete_mode_database(tmp_path / "sync.sqlite3")
    error = _scripted_store(store_module, monkeypatch, database, synchronous=1)
    assert "synchronous" in str(error), error
    assert str(store_module.REQUIRED_SYNCHRONOUS) in str(error), error

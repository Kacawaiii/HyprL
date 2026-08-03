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

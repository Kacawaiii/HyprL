"""Transactional append-only persistence for raw market data and MarketBar receipts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sqlite3

from scripts.trading_lab.coinbase_candles import (
    MAX_CANDLES_PER_RESPONSE,
    MAX_RESPONSE_BYTES,
    adapt_coinbase_candles,
)
from scripts.trading_lab.market_bar import TIMEFRAME_DURATIONS, build_market_bar


INGESTION_SCHEMA_VERSION = "trading-lab.market-ingestion.v1"
GAP_EVENT_SCHEMA_VERSION = "trading-lab.market-gap-event.v1"
GAP_EVENT_CAUSE_UNKNOWN = "unknown"
MAX_GAP_CANDIDATES_PER_INGESTION = 10_000
_EXISTENCE_CHECK_CHUNK_SIZE = 500
PROVIDER = "coinbase_exchange_rest"
VENUE = "coinbase_exchange"
PRODUCT_ASSETS = {
    "BTC-USD": "BTC/USD",
    "ETH-USD": "ETH/USD",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MARKET_BAR_FIELDS = frozenset(
    {
        "schema_version",
        "bar_id",
        "bar_version_id",
        "asset",
        "venue",
        "provider",
        "timeframe",
        "bar_status",
        "bar_open_at",
        "bar_close_at",
        "available_at",
        "ingested_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "raw_payload_sha256",
        "content_sha256",
    }
)


class MarketDataStoreError(RuntimeError):
    """Raised when a market-data artifact cannot be safely persisted."""


class MarketDataConflict(MarketDataStoreError):
    """Raised when an immutable identity resolves to different content."""


@dataclass(frozen=True)
class MarketIngestionResult:
    ingestion_id: str
    raw_payloads_inserted: int
    ingestions_inserted: int
    bar_receipts_inserted: int
    exact_replays: int
    gap_events_detected: int = 0
    gap_events_resolved: int = 0


def _canonical_json(payload: dict[str, object]) -> str:
    try:
        return json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise MarketDataStoreError("market data record is invalid") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_timestamp(value: datetime | str, *, field: str) -> str:
    if isinstance(value, str) and len(value) > 64:
        raise MarketDataStoreError(f"{field} is invalid")
    try:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
    except (TypeError, ValueError) as exc:
        raise MarketDataStoreError(f"{field} is invalid") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise MarketDataStoreError(f"{field} is invalid")
    try:
        offset = parsed.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise MarketDataStoreError(f"{field} is invalid") from exc
    if offset is None:
        raise MarketDataStoreError(f"{field} is invalid")
    return parsed.astimezone(timezone.utc).isoformat()


def _validated_record(
    record: object,
    *,
    raw_payload_sha256: str,
    product_id: str,
    timeframe: str,
    available_at: str,
    ingested_at: str,
) -> tuple[str, dict[str, object]]:
    if not isinstance(record, dict) or set(record) != _MARKET_BAR_FIELDS:
        raise MarketDataStoreError("market data record is invalid")
    if (
        record.get("raw_payload_sha256") != raw_payload_sha256
        or record.get("asset") != PRODUCT_ASSETS[product_id]
        or record.get("venue") != VENUE
        or record.get("provider") != PROVIDER
        or record.get("timeframe") != timeframe
        or record.get("available_at") != available_at
        or record.get("ingested_at") != ingested_at
        or record.get("bar_status") != "complete"
    ):
        raise MarketDataStoreError("market data record is invalid")

    try:
        rebuilt = build_market_bar(
            asset=record["asset"],
            venue=record["venue"],
            provider=record["provider"],
            timeframe=record["timeframe"],
            bar_open_at=record["bar_open_at"],
            bar_close_at=record["bar_close_at"],
            available_at=record["available_at"],
            ingested_at=record["ingested_at"],
            open_price=record["open"],
            high_price=record["high"],
            low_price=record["low"],
            close_price=record["close"],
            volume=record["volume"],
            raw_payload_sha256=record["raw_payload_sha256"],
        )
    except (TypeError, ValueError) as exc:
        raise MarketDataStoreError("market data record is invalid") from exc
    if rebuilt != record:
        raise MarketDataStoreError("market data record is invalid")

    content_sha256 = record.get("content_sha256")
    if not isinstance(content_sha256, str) or not _SHA256_PATTERN.fullmatch(
        content_sha256
    ):
        raise MarketDataStoreError("market data record is invalid")
    unsigned = dict(record)
    del unsigned["content_sha256"]
    if _sha256_text(_canonical_json(unsigned)) != content_sha256:
        raise MarketDataStoreError("market data record is invalid")
    for field in ("bar_id", "bar_version_id", "bar_open_at", "bar_close_at"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise MarketDataStoreError("market data record is invalid")
    canonical = _canonical_json(record)
    return canonical, record


def _gap_event_id(
    *, product_id: str, timeframe: str, expected_bar_open_at: str, event_type: str
) -> str:
    identity = {
        "provider": PROVIDER,
        "product_id": product_id,
        "timeframe": timeframe,
        "expected_bar_open_at": expected_bar_open_at,
        "event_type": event_type,
    }
    return f"hyprl-market-gap-event-{_sha256_text(_canonical_json(identity))}"


def _record_gap_event(
    connection: sqlite3.Connection,
    *,
    product_id: str,
    timeframe: str,
    expected_bar_open_at: str,
    event_type: str,
    ingestion_id: str,
    available_at: str,
    ingested_at: str,
) -> int:
    event_id = _gap_event_id(
        product_id=product_id,
        timeframe=timeframe,
        expected_bar_open_at=expected_bar_open_at,
        event_type=event_type,
    )
    existing = connection.execute(
        "SELECT 1 FROM market_data_gap_events WHERE event_id = ?",
        (event_id,),
    ).fetchone()
    if existing is not None:
        return 0
    connection.execute(
        """
        INSERT INTO market_data_gap_events (
            event_id, schema_version, provider, product_id, timeframe,
            expected_bar_open_at, event_type, cause,
            observed_by_ingestion_id, available_at, ingested_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event_id,
            GAP_EVENT_SCHEMA_VERSION,
            PROVIDER,
            product_id,
            timeframe,
            expected_bar_open_at,
            event_type,
            GAP_EVENT_CAUSE_UNKNOWN,
            ingestion_id,
            available_at,
            ingested_at,
        ),
    )
    return 1


def _count_missing_opens(previous: datetime, current: datetime, duration) -> int:
    """Arithmetic count of interior missing opens between two aligned, present bars.

    Never enumerates timestamps: a single division proves the count even for
    spans of decades, so a cardinality preflight can reject before any loop.
    """

    delta_seconds = (current - previous).total_seconds()
    duration_seconds = duration.total_seconds()
    if delta_seconds <= 0 or delta_seconds % duration_seconds != 0:
        raise MarketDataStoreError("gap interval is not aligned to the timeframe duration")
    return int(delta_seconds // duration_seconds) - 1


def _first_seen(
    connection: sqlite3.Connection,
    *,
    product_id: str,
    timeframe: str,
    bar_open_at: str,
) -> tuple[str, str]:
    """(available_at, ingested_at) of the earliest DECLARED historical
    ingestion timestamp among known receipts for one bar_open_at.

    Contract A (adopted for MarketDataStore V1): ingested_at is a historical
    timestamp declared by the caller (collector, historical importer, or
    test fixture) -- never the wall-clock moment this store physically
    receives or persists the receipt, and never a local, immutable arrival
    order. This store has no store_received_at column, no local clock, and
    no receipt_sequence, so "earliest" here means MIN(declared ingested_at):
    intentionally independent of INSERT order. A receipt physically
    persisted later can carry an earlier declared ingested_at and
    legitimately move this value into the past -- see
    test_first_seen_returns_earliest_declared_ingested_at_not_insert_order
    and test_retrodated_resupply_of_a_bound_shifts_its_first_seen_into_the_causal_max.
    This is by design for replay/backfill from authoritative historical
    timestamps; it is not a claim about when HyprL itself first learned the
    bar, and is not proof of live knowledge without lookahead.
    """

    row = connection.execute(
        """
        SELECT r.available_at, r.ingested_at
        FROM market_bar_receipts r
        JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id
        WHERE i.provider = ? AND i.product_id = ? AND i.timeframe = ? AND r.bar_open_at = ?
        ORDER BY r.ingested_at ASC, r.content_sha256 ASC
        LIMIT 1
        """,
        (PROVIDER, product_id, timeframe, bar_open_at),
    ).fetchone()
    if row is None:
        raise MarketDataStoreError(
            "market data gap reconciliation could not locate a bounding bar"
        )
    return row[0], row[1]


def _later_timestamp(first: str, second: str) -> str:
    return first if datetime.fromisoformat(first) >= datetime.fromisoformat(second) else second


def _load_known_bar_opens(
    connection: sqlite3.Connection,
    *,
    product_id: str,
    timeframe: str,
    candidates: list[str],
) -> set[str]:
    """Which of these candidate opens already have a receipt in this domain.

    Chunked IN-list lookups against the indexed bar_open_at column, scoped by
    provider/product_id/timeframe. Cost is bounded by len(candidates) --
    itself already capped by MAX_GAP_CANDIDATES_PER_INGESTION -- never by the
    domain's total history: no SELECT DISTINCT over the full receipt table.
    """

    known: set[str] = set()
    if not candidates:
        return known
    for start in range(0, len(candidates), _EXISTENCE_CHECK_CHUNK_SIZE):
        chunk = candidates[start : start + _EXISTENCE_CHECK_CHUNK_SIZE]
        placeholders = ",".join("?" for _ in chunk)
        rows = connection.execute(
            f"""
            SELECT DISTINCT r.bar_open_at
            FROM market_bar_receipts r
            JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id
            WHERE i.provider = ? AND i.product_id = ? AND i.timeframe = ?
              AND r.bar_open_at IN ({placeholders})
            """,
            (PROVIDER, product_id, timeframe, *chunk),
        ).fetchall()
        known.update(row[0] for row in rows)
    return known


def _reconcile_gap_events(
    connection: sqlite3.Connection,
    *,
    product_id: str,
    timeframe: str,
    ingestion_id: str,
    payload_bar_open_ats: list[str],
) -> tuple[int, int]:
    """Detect gaps strictly within the current payload, resolve prior ones it fills.

    Detection never compares bars across separate ingestions: a page boundary
    is not proof that intermediate bars are missing from the provider, so only
    consecutive bars already present together in THIS payload are compared.
    This keeps structural cost bounded by the payload (<=300 bars) and by the
    number of gap candidates it implies, never by the domain's accumulated
    history, and lets discontinuous historical backfills proceed without ever
    saturating a permanent, unresolvable cap.

    A candidate missing open is only ever turned into a DETECTED event if it
    is ALSO absent from market_bar_receipts for the same
    (provider, product_id, timeframe): an ouverture already known to the
    store -- via any prior, even unrelated, ingestion -- is never a gap, even
    if it happens to be missing from the payload currently being ingested.
    This existence check is a batched/chunked lookup restricted to this
    ingestion's own candidates (see _load_known_bar_opens), never a scan of
    the domain.

    Resolution stays cross-payload by design: any bar in this payload can
    still resolve a DETECTED recorded by an earlier, unrelated ingestion, via
    a bounded point lookup by deterministic event_id (no domain scan).

    MAX_GAP_CANDIDATES_PER_INGESTION bounds the number of gap candidates this
    ingestion has to examine (arithmetic candidates between the payload's own
    interior pairs), checked before any candidate is materialized, before any
    existence lookup, and before any event is built or inserted. Causal
    timestamps are derived from the first-seen observation of the bars
    involved (Contract A, see _first_seen: the earliest DECLARED historical
    ingested_at among their receipts, not real INSERT/arrival order), never
    from the current ingestion's own clock, so DETECTED/RESOLVED stay
    causally monotone with respect to declared history even when ingestions
    arrive out of order.
    """

    duration = TIMEFRAME_DURATIONS[timeframe]
    payload_opens_text = sorted(set(payload_bar_open_ats))
    payload_opens = [datetime.fromisoformat(text) for text in payload_opens_text]

    # Step 1-2: sort the payload's own opens, compute candidates arithmetically.
    pairs: list[tuple[datetime, datetime, int]] = []
    total_candidates = 0
    for previous, current in zip(payload_opens, payload_opens[1:]):
        count = _count_missing_opens(previous, current, duration)
        pairs.append((previous, current, count))
        total_candidates += count

    # Step 3: cardinality preflight BEFORE any materialization, existence
    # lookup, or insertion.
    if total_candidates > MAX_GAP_CANDIDATES_PER_INGESTION:
        raise MarketDataStoreError(
            "market data gap reconciliation exceeds the maximum of "
            f"{MAX_GAP_CANDIDATES_PER_INGESTION} gap candidates for this ingestion"
        )

    # Materialize candidate ISO strings, grouped by their bounding pair. Safe
    # now: total_candidates is proven <= MAX_GAP_CANDIDATES_PER_INGESTION.
    all_candidates: list[str] = []
    pair_candidates: list[tuple[datetime, datetime, list[str]]] = []
    for previous, current, count in pairs:
        if count <= 0:
            continue
        opens: list[str] = []
        missing_open = previous + duration
        for _ in range(count):
            text = missing_open.isoformat()
            opens.append(text)
            all_candidates.append(text)
            missing_open += duration
        pair_candidates.append((previous, current, opens))

    # Step 4: batch-load which candidates already have a receipt in this
    # domain -- bounded by the candidate list, never a scan of the domain.
    known_opens = _load_known_bar_opens(
        connection,
        product_id=product_id,
        timeframe=timeframe,
        candidates=all_candidates,
    )

    detected = 0
    for previous, current, opens in pair_candidates:
        # Step 5: subtract opens already known to the store.
        truly_missing = [text for text in opens if text not in known_opens]
        if not truly_missing:
            continue
        lower_available_at, lower_ingested_at = _first_seen(
            connection,
            product_id=product_id,
            timeframe=timeframe,
            bar_open_at=previous.isoformat(),
        )
        upper_available_at, upper_ingested_at = _first_seen(
            connection,
            product_id=product_id,
            timeframe=timeframe,
            bar_open_at=current.isoformat(),
        )
        detected_available_at = _later_timestamp(lower_available_at, upper_available_at)
        detected_ingested_at = _later_timestamp(lower_ingested_at, upper_ingested_at)
        # Step 6: create DETECTED only for the truly missing candidates.
        for text in truly_missing:
            detected += _record_gap_event(
                connection,
                product_id=product_id,
                timeframe=timeframe,
                expected_bar_open_at=text,
                event_type="DETECTED",
                ingestion_id=ingestion_id,
                available_at=detected_available_at,
                ingested_at=detected_ingested_at,
            )

    # Step 7: resolution stays targeted at the payload's own bars.
    resolved = 0
    for bar_open_at in payload_opens_text:
        detected_id = _gap_event_id(
            product_id=product_id,
            timeframe=timeframe,
            expected_bar_open_at=bar_open_at,
            event_type="DETECTED",
        )
        detected_row = connection.execute(
            "SELECT available_at, ingested_at FROM market_data_gap_events WHERE event_id = ?",
            (detected_id,),
        ).fetchone()
        if detected_row is None:
            continue
        resolved_id = _gap_event_id(
            product_id=product_id,
            timeframe=timeframe,
            expected_bar_open_at=bar_open_at,
            event_type="RESOLVED",
        )
        already_resolved = connection.execute(
            "SELECT 1 FROM market_data_gap_events WHERE event_id = ?",
            (resolved_id,),
        ).fetchone()
        if already_resolved is not None:
            continue
        late_available_at, late_ingested_at = _first_seen(
            connection,
            product_id=product_id,
            timeframe=timeframe,
            bar_open_at=bar_open_at,
        )
        detected_available_at, detected_ingested_at = detected_row
        resolved_available_at = _later_timestamp(detected_available_at, late_available_at)
        resolved_ingested_at = _later_timestamp(detected_ingested_at, late_ingested_at)
        resolved += _record_gap_event(
            connection,
            product_id=product_id,
            timeframe=timeframe,
            expected_bar_open_at=bar_open_at,
            event_type="RESOLVED",
            ingestion_id=ingestion_id,
            available_at=resolved_available_at,
            ingested_at=resolved_ingested_at,
        )
    return detected, resolved


class MarketDataStore:
    """Store captured Coinbase pages and normalized receipts in one transaction.

    Trust boundary (Contract A, adopted for MarketDataStore V1): every
    ingested_at this store persists is a historical timestamp DECLARED by
    the caller (collector, controlled historical importer, or test fixture)
    and is trusted as-is beyond the local per-bar causal check
    `ingested_at >= available_at >= bar_close_at`. This store keeps no
    store_received_at, no wall-clock capture, and no local immutable
    receipt-arrival sequence -- callers outside that trust boundary must not
    be allowed to set ingested_at directly through this API. A future gate
    is expected to add a store-owned store_received_at and/or an immutable
    receipt_sequence, plus explicit rules for which clock governs live
    snapshots and decisions; neither exists in this V1.
    """

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            # Schema note on historical-timestamp columns (Contract A, V1):
            # raw_market_payloads.first_stored_at and every ingested_at
            # column below are populated from the CALLER-declared
            # ingested_at (see ingest_coinbase_response), never from a real
            # wall-clock capture by this store.
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS raw_market_payloads (
                    payload_sha256 TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    payload_bytes BLOB NOT NULL,
                    byte_count INTEGER NOT NULL CHECK (byte_count >= 0),
                    -- Despite its name, this is the caller-DECLARED historical
                    -- ingested_at (Contract A), not a real measurement of when
                    -- this store physically persisted the row. Historical/
                    -- legacy name, kept as-is; see MarketDataStore docstring.
                    first_stored_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS market_ingestions (
                    ingestion_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    raw_payload_sha256 TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    bar_count INTEGER NOT NULL CHECK (bar_count >= 0),
                    FOREIGN KEY (raw_payload_sha256)
                        REFERENCES raw_market_payloads(payload_sha256)
                );

                CREATE TABLE IF NOT EXISTS market_bar_receipts (
                    content_sha256 TEXT PRIMARY KEY,
                    ingestion_id TEXT NOT NULL,
                    bar_id TEXT NOT NULL,
                    bar_version_id TEXT NOT NULL,
                    bar_open_at TEXT NOT NULL,
                    bar_close_at TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    raw_payload_sha256 TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    UNIQUE (ingestion_id, bar_id),
                    FOREIGN KEY (ingestion_id)
                        REFERENCES market_ingestions(ingestion_id),
                    FOREIGN KEY (raw_payload_sha256)
                        REFERENCES raw_market_payloads(payload_sha256)
                );

                CREATE INDEX IF NOT EXISTS market_bar_receipts_lookup
                    ON market_bar_receipts (bar_id, bar_open_at, ingested_at);

                CREATE INDEX IF NOT EXISTS market_bar_receipts_open_lookup
                    ON market_bar_receipts (bar_open_at, ingested_at);

                -- Serves the Phase 1C snapshot selection, which filters by
                -- domain first (provider/product_id/timeframe resolve to a
                -- set of ingestion_id) and only then by range. Leading with
                -- ingestion_id is what keeps that read independent of how
                -- many OTHER domains happen to share the same bar_open_at
                -- values; leading with bar_open_at instead would make the
                -- cost grow with every colocated domain, and would also
                -- divert the two lookups above onto a different index.
                -- Covering on purpose: it carries every column the snapshot
                -- selection projects, so that read never touches the table.
                CREATE INDEX IF NOT EXISTS market_bar_receipts_snapshot_domain_lookup
                    ON market_bar_receipts (
                        ingestion_id,
                        bar_open_at,
                        ingested_at,
                        content_sha256,
                        bar_version_id,
                        bar_id,
                        available_at
                    );

                CREATE TABLE IF NOT EXISTS market_data_gap_events (
                    event_id TEXT PRIMARY KEY,
                    schema_version TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    expected_bar_open_at TEXT NOT NULL,
                    event_type TEXT NOT NULL CHECK (event_type IN ('DETECTED', 'RESOLVED')),
                    cause TEXT NOT NULL,
                    observed_by_ingestion_id TEXT NOT NULL,
                    available_at TEXT NOT NULL,
                    ingested_at TEXT NOT NULL,
                    FOREIGN KEY (observed_by_ingestion_id)
                        REFERENCES market_ingestions(ingestion_id)
                );

                CREATE INDEX IF NOT EXISTS market_data_gap_events_lookup
                    ON market_data_gap_events (provider, product_id, timeframe, expected_bar_open_at);

                CREATE TRIGGER IF NOT EXISTS raw_market_payloads_no_update
                BEFORE UPDATE ON raw_market_payloads
                BEGIN
                    SELECT RAISE(ABORT, 'raw_market_payloads is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS raw_market_payloads_no_delete
                BEFORE DELETE ON raw_market_payloads
                BEGIN
                    SELECT RAISE(ABORT, 'raw_market_payloads is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_ingestions_no_update
                BEFORE UPDATE ON market_ingestions
                BEGIN
                    SELECT RAISE(ABORT, 'market_ingestions is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_ingestions_no_delete
                BEFORE DELETE ON market_ingestions
                BEGIN
                    SELECT RAISE(ABORT, 'market_ingestions is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_bar_receipts_no_update
                BEFORE UPDATE ON market_bar_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'market_bar_receipts is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_bar_receipts_no_delete
                BEFORE DELETE ON market_bar_receipts
                BEGIN
                    SELECT RAISE(ABORT, 'market_bar_receipts is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_data_gap_events_no_update
                BEFORE UPDATE ON market_data_gap_events
                BEGIN
                    SELECT RAISE(ABORT, 'market_data_gap_events is insert-only');
                END;
                CREATE TRIGGER IF NOT EXISTS market_data_gap_events_no_delete
                BEFORE DELETE ON market_data_gap_events
                BEGIN
                    SELECT RAISE(ABORT, 'market_data_gap_events is insert-only');
                END;
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30.0)
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def ingest_coinbase_response(
        self,
        raw_payload: bytes,
        *,
        product_id: str,
        timeframe: str,
        available_at: datetime | str,
        ingested_at: datetime | str,
    ) -> MarketIngestionResult:
        """Persist one captured Coinbase response and its derived MarketBars.

        ingested_at (Contract A, MarketDataStore V1): a historical timestamp
        DECLARED by the caller -- not the real time this method executes and
        not this store's own clock (no store_received_at, no wall-clock
        capture, no local immutable arrival sequence exist here; see
        _first_seen). The only rule enforced is the local causal ordering
        `ingested_at >= available_at >= bar_close_at` for each bar; nothing
        here verifies the declared value is truthful or monotonic across
        calls. Callers are assumed to sit inside a trust boundary (internal
        collector, controlled historical importer, or test fixture) -- this
        method is not a validated boundary against an untrusted or
        adversarial ingested_at.
        """
        if not isinstance(raw_payload, bytes):
            raise MarketDataStoreError("market data raw payload is invalid")
        if len(raw_payload) > MAX_RESPONSE_BYTES:
            raise MarketDataStoreError("market data raw payload is too large")
        records = adapt_coinbase_candles(
            raw_payload,
            product_id=product_id,
            timeframe=timeframe,
            available_at=available_at,
            ingested_at=ingested_at,
        )
        if not isinstance(records, list) or len(records) > MAX_CANDLES_PER_RESPONSE:
            raise MarketDataStoreError("market data record set is invalid")
        canonical_available_at = _canonical_timestamp(
            available_at,
            field="available_at",
        )
        canonical_ingested_at = _canonical_timestamp(
            ingested_at,
            field="ingested_at",
        )
        if canonical_ingested_at < canonical_available_at:
            raise MarketDataStoreError("market ingestion timestamps are invalid")

        raw_payload_sha256 = hashlib.sha256(raw_payload).hexdigest()
        canonical_records: list[tuple[str, dict[str, object]]] = []
        seen_bar_ids: set[str] = set()
        for record in records:
            canonical, validated = _validated_record(
                record,
                raw_payload_sha256=raw_payload_sha256,
                product_id=product_id,
                timeframe=timeframe,
                available_at=canonical_available_at,
                ingested_at=canonical_ingested_at,
            )
            bar_id = validated["bar_id"]
            if bar_id in seen_bar_ids:
                raise MarketDataStoreError("market data record is invalid")
            seen_bar_ids.add(bar_id)
            canonical_records.append((canonical, validated))

        metadata: dict[str, object] = {
            "schema_version": INGESTION_SCHEMA_VERSION,
            "provider": PROVIDER,
            "product_id": product_id,
            "timeframe": timeframe,
            "available_at": canonical_available_at,
            "ingested_at": canonical_ingested_at,
            "raw_payload_sha256": raw_payload_sha256,
            "bar_count": len(canonical_records),
        }
        metadata_json = _canonical_json(metadata)
        ingestion_id = f"hyprl-market-ingestion-{_sha256_text(metadata_json)}"

        raw_inserted = 0
        ingestion_inserted = 0
        receipt_inserted = 0
        exact_replays = 0
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_raw = connection.execute(
                """
                SELECT provider, payload_bytes, byte_count
                FROM raw_market_payloads
                WHERE payload_sha256 = ?
                """,
                (raw_payload_sha256,),
            ).fetchone()
            if existing_raw is None:
                connection.execute(
                    """
                    INSERT INTO raw_market_payloads (
                        payload_sha256, provider, payload_bytes,
                        byte_count, first_stored_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        raw_payload_sha256,
                        PROVIDER,
                        raw_payload,
                        len(raw_payload),
                        canonical_ingested_at,
                    ),
                )
                raw_inserted = 1
            elif (
                existing_raw[0] != PROVIDER
                or bytes(existing_raw[1]) != raw_payload
                or existing_raw[2] != len(raw_payload)
            ):
                raise MarketDataConflict("conflicting immutable raw payload")

            existing_ingestion = connection.execute(
                """
                SELECT metadata_json, bar_count
                FROM market_ingestions
                WHERE ingestion_id = ?
                """,
                (ingestion_id,),
            ).fetchone()
            if existing_ingestion is not None:
                if (
                    existing_ingestion[0] != metadata_json
                    or existing_ingestion[1] != len(canonical_records)
                ):
                    raise MarketDataConflict("conflicting immutable ingestion")
                stored_payloads = {
                    row[0]
                    for row in connection.execute(
                        """
                        SELECT payload_json
                        FROM market_bar_receipts
                        WHERE ingestion_id = ?
                        """,
                        (ingestion_id,),
                    ).fetchall()
                }
                expected_payloads = {item[0] for item in canonical_records}
                if stored_payloads != expected_payloads:
                    raise MarketDataConflict("conflicting immutable ingestion")
                exact_replays = len(canonical_records)
                connection.commit()
                return MarketIngestionResult(
                    ingestion_id=ingestion_id,
                    raw_payloads_inserted=0,
                    ingestions_inserted=0,
                    bar_receipts_inserted=0,
                    exact_replays=exact_replays,
                )

            connection.execute(
                """
                INSERT INTO market_ingestions (
                    ingestion_id, schema_version, provider, product_id,
                    timeframe, available_at, ingested_at,
                    raw_payload_sha256, metadata_json, bar_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ingestion_id,
                    INGESTION_SCHEMA_VERSION,
                    PROVIDER,
                    product_id,
                    timeframe,
                    canonical_available_at,
                    canonical_ingested_at,
                    raw_payload_sha256,
                    metadata_json,
                    len(canonical_records),
                ),
            )
            ingestion_inserted = 1

            for canonical, record in canonical_records:
                content_sha256 = record["content_sha256"]
                existing_receipt = connection.execute(
                    """
                    SELECT payload_json
                    FROM market_bar_receipts
                    WHERE content_sha256 = ?
                    """,
                    (content_sha256,),
                ).fetchone()
                if existing_receipt is not None:
                    if existing_receipt[0] != canonical:
                        raise MarketDataConflict(
                            "conflicting immutable market bar receipt"
                        )
                    raise MarketDataConflict("market bar receipt belongs to another ingestion")
                occupied_slot = connection.execute(
                    """
                    SELECT content_sha256
                    FROM market_bar_receipts
                    WHERE ingestion_id = ? AND bar_id = ?
                    """,
                    (ingestion_id, record["bar_id"]),
                ).fetchone()
                if occupied_slot is not None:
                    raise MarketDataConflict("conflicting immutable market bar receipt")
                connection.execute(
                    """
                    INSERT INTO market_bar_receipts (
                        content_sha256, ingestion_id, bar_id, bar_version_id,
                        bar_open_at, bar_close_at, available_at, ingested_at,
                        raw_payload_sha256, payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        content_sha256,
                        ingestion_id,
                        record["bar_id"],
                        record["bar_version_id"],
                        record["bar_open_at"],
                        record["bar_close_at"],
                        record["available_at"],
                        record["ingested_at"],
                        raw_payload_sha256,
                        canonical,
                    ),
                )
                receipt_inserted += 1

            gap_events_detected, gap_events_resolved = _reconcile_gap_events(
                connection,
                product_id=product_id,
                timeframe=timeframe,
                ingestion_id=ingestion_id,
                payload_bar_open_ats=[
                    record["bar_open_at"] for _, record in canonical_records
                ],
            )
            connection.commit()
        except MarketDataConflict:
            connection.rollback()
            raise
        except MarketDataStoreError:
            connection.rollback()
            raise
        except Exception:
            connection.rollback()
            raise MarketDataStoreError("market data persistence failed") from None
        finally:
            connection.close()

        return MarketIngestionResult(
            ingestion_id=ingestion_id,
            raw_payloads_inserted=raw_inserted,
            ingestions_inserted=ingestion_inserted,
            bar_receipts_inserted=receipt_inserted,
            exact_replays=exact_replays,
            gap_events_detected=gap_events_detected,
            gap_events_resolved=gap_events_resolved,
        )

"""Phase 1C-A/1C-B/1C-C: causal eligibility, deterministic revision
selection, bounded cardinality, a bounded streaming read, and atomic
immutable persistence.

Historical causal snapshots based on declared ingestion time (Contract A,
see scripts/trading_lab/market_data_store.py): `as_of` is a cutoff over the
DECLARED historical `ingested_at` of already-persisted receipts, never a
live wall-clock, never a promise of lookahead-free real-time knowledge.

This module *selects*, deterministically, which already-persisted MarketBar
receipt represents the state of declared knowledge as of `as_of` for each
bar_open_at in a requested range, and (Phase 1C-C) persists that selection
as an immutable snapshot. It never modifies market_bar_receipts' rows,
never copies market data into a snapshot, and never imports network or
broker code. The write path is the internal primitive _materialize_snapshot;
Phase 1C-D adds the public read surface: load_snapshot,
list_snapshot_manifests and replay_snapshot.

Phase 1C-B adds three bounds, all fail-closed and all enforced before or
during the read rather than after it:

* every range bound must sit exactly on the timeframe's epoch-anchored UTC
  grid, and the range may span at most MAX_SNAPSHOT_RANGE_OPENS openings --
  both checked by O(1) integer arithmetic, before SQLite is touched at all;
* at most MAX_SNAPSHOT_ELIGIBLE_RECEIPTS receipts may be eligible; the SQL
  LIMIT is a detector for that condition and never a functional truncation;
* rows are consumed in bounded chunks and folded into a per-opening
  aggregate, so memory follows the number of openings, not the number of
  revisions behind them.

Phase 1C-C persists the result. The write path owns its transaction, takes
BEGIN IMMEDIATE *before* the selection so the attested rows cannot move
underneath it, writes entries before the manifest whose insertion seals
them, and proves rather than assumes idempotence: an existing snapshot_id
is verified field by field and entry by entry, never trusted.

Phase 1C-D reads it back. Reads run in two phases: one SQLite transaction
captures every value needed and commits at once, then all decoding, hashing
and validation happen on the captured copies with no transaction held --
because in journal_mode=delete a read transaction blocks every writer's
COMMIT. Nothing is ever repaired, no partial result is ever returned, and a
replay proves each bar by rebuilding it through MarketBar V1 rather than
trusting a stored hash.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3

from scripts.trading_lab.market_bar import (
    SCHEMA_VERSION as MARKET_BAR_SCHEMA_VERSION,
    TIMEFRAME_DURATIONS,
    build_market_bar,
)


SNAPSHOT_REQUEST_SCHEMA_VERSION = "trading-lab.market-snapshot-request.v1"
SNAPSHOT_SCHEMA_VERSION = "trading-lab.market-snapshot.v1"
SELECTION_POLICY_VERSION = "trading-lab.market-snapshot-selection.v1"

# Largest range a single V1 snapshot may describe, counted in timeframe
# openings: 10 000 covers ~1.14 year of 1h bars or ~27 years of 1d bars.
MAX_SNAPSHOT_RANGE_OPENS = 10_000
# Largest number of eligible receipts a single V1 selection will read. The
# range bounds the openings but never the revisions behind them, so this is
# a separate bound: without it one pathological opening could return an
# unbounded number of rows.
MAX_SNAPSHOT_ELIGIBLE_RECEIPTS = 100_000
# Single source of truth for the overflow detection: this exact value is both
# bound into the query's LIMIT and used as the row-counter threshold, so the
# detector and the threshold it detects can never drift apart. Interpolating
# a literal into the SQL instead would freeze it at import time while the
# counter kept reading the live constant.
SNAPSHOT_ELIGIBLE_RECEIPT_QUERY_LIMIT = MAX_SNAPSHOT_ELIGIBLE_RECEIPTS + 1
# Rows are consumed in chunks of this size; the chunk boundaries carry no
# semantics whatsoever (see _OpenAggregate).
SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE = 1_000

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MICROSECONDS_PER_DAY = 86_400 * 1_000_000
_MICROSECONDS_PER_SECOND = 1_000_000


class MarketSnapshotError(RuntimeError):
    """Raised when a market snapshot selection cannot be safely computed."""


class SnapshotSelectionConflict(MarketSnapshotError):
    """Raised when two contradictory MarketBar contents are both eligible
    at the exact same maximal declared ingested_at for one bar_open_at.

    Never resolved by comparing content_sha256 lexically: a hash is not an
    arbiter between two contradictory OHLCV declarations. The caller must
    resolve the contradiction upstream (e.g. by correcting the historical
    record) before a snapshot over this range/as_of can be computed.
    """

    def __init__(
        self,
        *,
        provider: str,
        product_id: str,
        timeframe: str,
        bar_open_at: str,
        ingested_at: str,
        bar_version_ids: tuple[str, ...],
    ) -> None:
        self.provider = provider
        self.product_id = product_id
        self.timeframe = timeframe
        self.bar_open_at = bar_open_at
        self.ingested_at = ingested_at
        self.bar_version_ids = bar_version_ids
        super().__init__(
            "market snapshot selection conflict: two contradictory "
            "bar_version_id values are both eligible at the same declared "
            f"ingested_at -- provider={provider!r} product_id={product_id!r} "
            f"timeframe={timeframe!r} bar_open_at={bar_open_at!r} "
            f"ingested_at={ingested_at!r} bar_version_ids={bar_version_ids!r}"
        )


class SnapshotEligibilityLimitExceeded(MarketSnapshotError):
    """Raised when more receipts are eligible than V1 agrees to read.

    Strictly fail-closed and atomic: the caller receives nothing at all. No
    selection is ever derived from the first MAX_SNAPSHOT_ELIGIBLE_RECEIPTS
    rows -- the SQL LIMIT exists only to detect this condition, never to
    truncate a result into a plausible-looking snapshot.

    V1 error precedence, deliberate and documented: overflow takes
    precedence over a SnapshotSelectionConflict located beyond the
    eligibility limit. Both branches are fail-closed and neither returns a
    partial result, so the precedence only decides which typed refusal the
    caller sees, never whether the request is refused.
    """

    def __init__(
        self,
        *,
        provider: str,
        product_id: str,
        timeframe: str,
        range_start: str,
        range_end: str,
        as_of: str,
        limit: int,
    ) -> None:
        self.provider = provider
        self.product_id = product_id
        self.timeframe = timeframe
        self.range_start = range_start
        self.range_end = range_end
        self.as_of = as_of
        self.limit = limit
        super().__init__(
            "market snapshot selection exceeds the maximum number of eligible "
            f"receipts ({limit}) -- provider={provider!r} "
            f"product_id={product_id!r} timeframe={timeframe!r} "
            f"range_start={range_start!r} range_end={range_end!r} "
            f"as_of={as_of!r}; split the range and compute the snapshot in "
            "several parts"
        )


class SnapshotPersistenceError(MarketSnapshotError):
    """Raised when persisting a snapshot fails for a storage-level reason.

    Wraps the underlying sqlite3 error (chained) so a caller never has to
    catch sqlite3 directly, while the business exceptions above keep
    propagating unchanged.
    """


class SnapshotStateCorruption(MarketSnapshotError):
    """Raised when a snapshot_id already exists but its persisted state does
    not match what the same request and content must produce.

    Never repaired silently: a manifest whose entries drifted is evidence of
    a real problem, and rewriting it would destroy that evidence -- and is
    impossible anyway, the rows being immutable.
    """


class SnapshotWriteContextError(MarketSnapshotError):
    """Raised when the connection handed to the write path cannot uphold the
    guarantees the snapshot tables depend on.

    Both required PRAGMAs are per-connection and become silent no-ops inside
    a transaction, so they are verified before anything else happens --
    before BEGIN, and before a single row is read.
    """


@dataclass(frozen=True)
class MaterializedSnapshot:
    """Outcome of one materialization. `created` distinguishes a snapshot
    this call persisted from an identical one that already existed."""

    snapshot_id: str
    snapshot_request_id: str
    entries_content_hash: str
    entry_count: int
    created: bool


@dataclass(frozen=True)
class SelectedSnapshotReceipt:
    """One deterministically selected MarketBar receipt for one bar_open_at."""

    bar_open_at: str
    content_sha256: str
    bar_id: str
    bar_version_id: str
    ingested_at: str
    available_at: str


def _canonical_timestamp(value: datetime | str, *, field: str) -> str:
    """Timezone-aware, UTC-normalized, canonical ISO timestamp.

    Duplicated (not imported) from market_data_store.py's private
    _canonical_timestamp: that module's Phase 1B schema/behavior must not be
    touched by this sub-gate, and this codebase's convention is to only
    import PUBLIC names across scripts/trading_lab modules. Equivalent
    instants expressed with different UTC offsets normalize to the same
    canonical string; naive/invalid values are rejected explicitly.
    """
    if isinstance(value, str) and len(value) > 64:
        raise MarketSnapshotError(f"{field} is invalid")
    try:
        parsed = (
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            if isinstance(value, str)
            else value
        )
    except (TypeError, ValueError) as exc:
        raise MarketSnapshotError(f"{field} is invalid") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None:
        raise MarketSnapshotError(f"{field} must be timezone-aware")
    try:
        offset = parsed.utcoffset()
    except (OverflowError, ValueError) as exc:
        raise MarketSnapshotError(f"{field} is invalid") from exc
    if offset is None:
        raise MarketSnapshotError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc).isoformat()


def _microseconds_since_epoch(canonical_timestamp: str) -> int:
    """Exact integer microseconds since 1970-01-01T00:00:00+00:00.

    Integer arithmetic only, on purpose: timedelta.total_seconds() returns a
    float, and int(float) silently truncates a residual microsecond -- which
    would let a bound one microsecond off the grid pass an alignment check.
    """
    elapsed = datetime.fromisoformat(canonical_timestamp) - _EPOCH
    return (
        elapsed.days * _MICROSECONDS_PER_DAY
        + elapsed.seconds * _MICROSECONDS_PER_SECOND
        + elapsed.microseconds
    )


def _timeframe_duration_microseconds(timeframe: str) -> int:
    """Timeframe length in exact microseconds, derived from market_bar's
    public TIMEFRAME_DURATIONS rather than redeclared here: a snapshot grid
    that disagreed with MarketBar's own notion of a timeframe could name
    positions no stored bar could ever occupy.
    """
    if not isinstance(timeframe, str) or timeframe not in TIMEFRAME_DURATIONS:
        raise MarketSnapshotError(
            f"timeframe must be one of {sorted(TIMEFRAME_DURATIONS)}"
        )
    duration = TIMEFRAME_DURATIONS[timeframe]
    return (
        duration.days * _MICROSECONDS_PER_DAY
        + duration.seconds * _MICROSECONDS_PER_SECOND
        + duration.microseconds
    )


def _canonical_range(
    range_start: datetime | str,
    range_end: datetime | str,
    *,
    timeframe: str,
) -> tuple[str, str]:
    """Validate and canonicalize a [range_start, range_end) pair against the
    timeframe's grid, and bound its cardinality -- all before any SQL runs.

    Factored out so the selector and build_snapshot_request_id can never
    diverge on what makes a range valid: both call this, so a future change
    to one automatically applies to the other, and an identity is never
    minted for a range the selector would refuse to compute.

    Alignment is checked PER BOUND against the epoch-anchored grid, never by
    the divisibility of the span: [10:30, 12:30) spans exactly two hours yet
    lies entirely off the hourly grid, so no stored bar_open_at could ever
    fall inside it. The open count is then pure integer division, so an
    absurdly wide range is refused in constant time and constant memory
    instead of materializing a grid of timestamps to count it.
    """
    duration_us = _timeframe_duration_microseconds(timeframe)
    canonical_range_start = _canonical_timestamp(range_start, field="range_start")
    canonical_range_end = _canonical_timestamp(range_end, field="range_end")
    start_us = _microseconds_since_epoch(canonical_range_start)
    end_us = _microseconds_since_epoch(canonical_range_end)
    if end_us <= start_us:
        raise MarketSnapshotError("range_end must be strictly after range_start")
    for field, canonical, value_us in (
        ("range_start", canonical_range_start, start_us),
        ("range_end", canonical_range_end, end_us),
    ):
        if value_us % duration_us != 0:
            raise MarketSnapshotError(
                f"{field} ({canonical!r}) is not aligned on the {timeframe} "
                f"grid anchored at {_EPOCH.isoformat()}"
            )
    expected_open_count = (end_us - start_us) // duration_us
    if expected_open_count > MAX_SNAPSHOT_RANGE_OPENS:
        raise MarketSnapshotError(
            f"snapshot range spans {expected_open_count} {timeframe} openings, "
            f"which exceeds the maximum of {MAX_SNAPSHOT_RANGE_OPENS}"
        )
    return canonical_range_start, canonical_range_end


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MarketSnapshotError("market snapshot identity payload is invalid") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_ELIGIBLE_RECEIPTS_SQL = """
        SELECT r.bar_open_at, r.ingested_at, r.content_sha256,
               r.bar_id, r.bar_version_id, r.available_at
        FROM market_bar_receipts r
        WHERE r.ingestion_id IN (
                SELECT i.ingestion_id
                FROM market_ingestions i
                WHERE i.provider = ? AND i.product_id = ? AND i.timeframe = ?
              )
          AND r.bar_open_at >= ? AND r.bar_open_at < ?
          AND r.ingested_at <= ?
        LIMIT ?
        """


class _OpenAggregate:
    """Streaming aggregate for one bar_open_at.

    Holds only what the final decision needs -- the largest declared
    ingested_at seen so far, the distinct bar_version_id values sitting at
    it, and the current canonical row -- never the rows behind them. This is
    what keeps memory proportional to the number of openings instead of the
    number of revisions.

    Makes no assumption about the order rows arrive in, and therefore none
    about chunk boundaries: an opening split across chunks, or a group of
    rows sharing the maximal ingested_at split across chunks, folds to the
    same state as any other arrival order. A contradiction observed at some
    ingested_at is simply forgotten when a strictly more recent declaration
    supersedes it, which is why conflicts are only decided once every row
    has been folded in.
    """

    __slots__ = ("max_ingested_at", "bar_version_ids", "winner")

    def __init__(self, row: tuple) -> None:
        self.max_ingested_at = row[1]
        self.bar_version_ids = {row[4]}
        self.winner = row

    def observe(self, row: tuple) -> None:
        ingested_at = row[1]
        if ingested_at > self.max_ingested_at:
            self.max_ingested_at = ingested_at
            self.bar_version_ids = {row[4]}
            self.winner = row
        elif ingested_at == self.max_ingested_at:
            self.bar_version_ids.add(row[4])
            if row[2] > self.winner[2]:
                self.winner = row


def _select_snapshot_receipts(
    connection: sqlite3.Connection,
    *,
    provider: str,
    product_id: str,
    timeframe: str,
    range_start: datetime | str,
    range_end: datetime | str,
    as_of: datetime | str,
) -> tuple[SelectedSnapshotReceipt, ...]:
    """Deterministically select one receipt per bar_open_at in
    [range_start, range_end) representing the state of DECLARED historical
    knowledge as of `as_of` (Contract A).

    All timestamps are validated and normalized to canonical UTC ISO form,
    both range bounds are checked against the timeframe grid, and the range
    cardinality is bounded BEFORE any query is issued: an unaligned,
    inverted, naive or oversized range fails closed without ever touching
    `connection`.

    For each bar_open_at, among receipts with ingested_at <= as_of, the one
    (or ones) with the largest ingested_at are the candidates; if they carry
    more than one distinct bar_version_id, this is a genuine contradiction
    in the declared history and SnapshotSelectionConflict is raised --
    content_sha256 is only ever used to break a tie among candidates sharing
    the SAME bar_version_id, never to arbitrate between different OHLCV
    contents. Determinism never depends on rowid, INSERT order, SQL's
    returned row order, or dict/set iteration order: the query carries no
    ORDER BY at all, each opening's state is folded from the rows in
    whatever order they arrive, and bar_open_at groups are visited in
    explicit sorted() order at the end.

    Rows are consumed in bounded chunks and never accumulated: at most
    MAX_SNAPSHOT_ELIGIBLE_RECEIPTS receipts may be eligible, and one row
    beyond that refuses the whole request rather than truncating it.
    """

    canonical_range_start, canonical_range_end = _canonical_range(
        range_start, range_end, timeframe=timeframe
    )
    canonical_as_of = _canonical_timestamp(as_of, field="as_of")

    aggregates: dict[str, _OpenAggregate] = {}
    eligible_row_count = 0
    # Read once, then used for BOTH the SQL bound and the counter threshold:
    # the query can never be allowed to return more rows than the counter is
    # willing to refuse, nor fewer than it is willing to accept.
    query_limit = SNAPSHOT_ELIGIBLE_RECEIPT_QUERY_LIMIT
    # Same close policy as every other cursor in this module: a secondary
    # close failure must never displace the verdict the caller needs -- an
    # eligibility overflow above all -- and no raw sqlite3 error may escape a
    # function whose error surface is MarketSnapshotError.
    try:
        with _closing_cursor(
            connection.execute(
                _ELIGIBLE_RECEIPTS_SQL,
                (
                    provider,
                    product_id,
                    timeframe,
                    canonical_range_start,
                    canonical_range_end,
                    canonical_as_of,
                    query_limit,
                ),
            )
        ) as cursor:
            while True:
                chunk = cursor.fetchmany(SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE)
                if not chunk:
                    break
                eligible_row_count += len(chunk)
                if eligible_row_count >= query_limit:
                    # Refuse the whole request here, before folding this chunk
                    # in and before any entry is built: a snapshot must never
                    # be derived from a prefix of a result set cut short.
                    raise SnapshotEligibilityLimitExceeded(
                        provider=provider,
                        product_id=product_id,
                        timeframe=timeframe,
                        range_start=canonical_range_start,
                        range_end=canonical_range_end,
                        as_of=canonical_as_of,
                        limit=query_limit - 1,
                    )
                for row in chunk:
                    aggregate = aggregates.get(row[0])
                    if aggregate is None:
                        aggregates[row[0]] = _OpenAggregate(row)
                    else:
                        aggregate.observe(row)
    except sqlite3.Error as exc:
        raise SnapshotReadError(
            f"snapshot selection failed for provider={provider!r} "
            f"product_id={product_id!r} timeframe={timeframe!r}"
        ) from exc

    selected: list[SelectedSnapshotReceipt] = []
    for bar_open_at in sorted(aggregates):
        aggregate = aggregates[bar_open_at]
        max_ingested_at = aggregate.max_ingested_at
        distinct_versions = sorted(aggregate.bar_version_ids)
        if len(distinct_versions) > 1:
            raise SnapshotSelectionConflict(
                provider=provider,
                product_id=product_id,
                timeframe=timeframe,
                bar_open_at=bar_open_at,
                ingested_at=max_ingested_at,
                bar_version_ids=tuple(distinct_versions),
            )
        canonical_winner = aggregate.winner
        winner_available_at = canonical_winner[5]
        # Defensive re-check on data already fetched (no extra query): the
        # build-time invariant (bar_close_at <= available_at <= ingested_at)
        # already guarantees this, but a snapshot must never silently trust
        # that invariant without verifying it on the exact row selected.
        # Routed through _canonical_timestamp (not a bare datetime.fromisoformat)
        # so a corrupted/unparseable value surfaces as MarketSnapshotError,
        # never a raw ValueError.
        canonical_winner_available_at = _canonical_timestamp(
            winner_available_at, field="available_at"
        )
        if datetime.fromisoformat(canonical_winner_available_at) > datetime.fromisoformat(
            canonical_as_of
        ):
            raise MarketSnapshotError(
                "internal invariant violated: selected receipt's available_at "
                f"({canonical_winner_available_at!r}) is after as_of "
                f"({canonical_as_of!r})"
            )
        selected.append(
            SelectedSnapshotReceipt(
                bar_open_at=canonical_winner[0],
                ingested_at=canonical_winner[1],
                content_sha256=canonical_winner[2],
                bar_id=canonical_winner[3],
                bar_version_id=canonical_winner[4],
                available_at=winner_available_at,
            )
        )

    return tuple(selected)


_REQUIRED_WRITE_PRAGMAS = ("foreign_keys", "recursive_triggers")

_INSERT_ENTRY_SQL = """
        INSERT INTO market_snapshot_entries (snapshot_id, bar_open_at, content_sha256)
        VALUES (?, ?, ?)
        """

_INSERT_MANIFEST_SQL = """
        INSERT INTO market_snapshot_manifests (
            snapshot_id, snapshot_request_id, entries_content_hash,
            snapshot_schema_version, selection_policy_version,
            provider, product_id, timeframe, range_start, range_end, as_of,
            entry_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

_SELECT_MANIFEST_SQL = """
        SELECT snapshot_request_id, entries_content_hash, snapshot_schema_version,
               selection_policy_version, provider, product_id, timeframe,
               range_start, range_end, as_of, entry_count
        FROM market_snapshot_manifests
        WHERE snapshot_id = ?
        """

_SELECT_ENTRIES_SQL = """
        SELECT bar_open_at, content_sha256
        FROM market_snapshot_entries
        WHERE snapshot_id = ?
        ORDER BY bar_open_at ASC
        """

# Confirms, for the rows just written, that every entry still points at a
# receipt of the right domain, at the right opening, declared no later than
# as_of. Bounded by the number of entries, and never a replay of the causal
# selection: BEGIN IMMEDIATE already froze the view it ran against.
_VALIDATE_ENTRIES_SQL = """
        SELECT COUNT(*)
        FROM market_snapshot_entries e
        JOIN market_bar_receipts r ON r.content_sha256 = e.content_sha256
        JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id
        WHERE e.snapshot_id = ?
          AND r.bar_open_at = e.bar_open_at
          AND i.provider = ? AND i.product_id = ? AND i.timeframe = ?
          AND r.ingested_at <= ? AND r.available_at <= ?
        """


def _require_write_pragmas(connection: sqlite3.Connection) -> None:
    """Fail closed unless the connection enforces foreign keys and recursive
    triggers.

    Both are per-connection and OFF by default on a bare sqlite3.connect().
    Without foreign_keys the deferred manifest link stops rejecting orphan
    entries; without recursive_triggers INSERT OR REPLACE silently rewrites
    rows the schema calls immutable. Neither can be turned on from inside a
    transaction (SQLite makes the PRAGMA a no-op there), so this refuses
    rather than trying to fix the caller's connection.
    """
    for pragma in _REQUIRED_WRITE_PRAGMAS:
        try:
            enabled = connection.execute(f"PRAGMA {pragma}").fetchone()[0]
        except sqlite3.Error as exc:
            raise SnapshotWriteContextError(
                f"snapshot write context could not be verified: PRAGMA {pragma}"
            ) from exc
        if enabled != 1:
            raise SnapshotWriteContextError(
                f"snapshot write context requires PRAGMA {pragma} = ON; "
                "open the connection through MarketDataStore"
            )


def _verify_existing_snapshot(
    connection: sqlite3.Connection,
    *,
    snapshot_id: str,
    manifest_row: tuple,
    expected_manifest: tuple,
    expected_entries: tuple[tuple[str, str], ...],
    expected_entries_content_hash: str,
) -> None:
    """Verify a persisted snapshot matches, exactly, what this request and
    this content must produce.

    Never `INSERT OR IGNORE` followed by an assumption of success: the whole
    point of an idempotent path is that it proves the existing state is the
    state it would have written. The entries are reloaded and their content
    hash recomputed from what is actually stored, so a manifest whose hash
    no longer describes its own entries is caught rather than trusted.
    """
    if manifest_row != expected_manifest:
        raise SnapshotStateCorruption(
            f"persisted snapshot {snapshot_id!r} does not match the request that "
            "produced its identity"
        )
    # Bounded and explicitly closed like every other snapshot-range read:
    # iterating the cursor would materialize whatever a sabotaged database
    # holds before the cardinality check below ever gets to refuse it. The
    # bound is deliberately NOT derived from manifest_row[-1] -- that column
    # is the very persisted value this function exists to distrust.
    with _closing_cursor(
        connection.execute(_SELECT_ENTRIES_SQL, (snapshot_id,))
    ) as entries_cursor:
        persisted_entries = tuple(
            (row[0], row[1])
            for row in _fetch_all_bounded(
                entries_cursor,
                fetch_size=SNAPSHOT_LOAD_FETCH_SIZE,
                max_rows=MAX_SNAPSHOT_RANGE_OPENS,
            )
        )
    if len(persisted_entries) != manifest_row[-1]:
        raise SnapshotStateCorruption(
            f"persisted snapshot {snapshot_id!r} declares {manifest_row[-1]} entries "
            f"but stores {len(persisted_entries)}"
        )
    if persisted_entries != expected_entries:
        raise SnapshotStateCorruption(
            f"persisted snapshot {snapshot_id!r} entries differ from the selected content"
        )
    recomputed = build_entries_content_hash(
        tuple(
            SelectedSnapshotReceipt(
                bar_open_at=bar_open_at,
                content_sha256=content_sha256,
                bar_id="",
                bar_version_id="",
                ingested_at="",
                available_at="",
            )
            for bar_open_at, content_sha256 in persisted_entries
        )
    )
    if recomputed != expected_entries_content_hash:
        raise SnapshotStateCorruption(
            f"persisted snapshot {snapshot_id!r} entries_content_hash does not "
            "describe its own entries"
        )


def _materialize_snapshot(
    connection: sqlite3.Connection,
    *,
    provider: str,
    product_id: str,
    timeframe: str,
    range_start: datetime | str,
    range_end: datetime | str,
    as_of: datetime | str,
) -> MaterializedSnapshot:
    """Persist, atomically and immutably, the causal selection for one
    request -- or prove an identical snapshot already exists.

    Owns its transaction, exclusively. BEGIN IMMEDIATE is taken BEFORE the
    causal selection so that the rows the manifest attests to cannot change
    between being selected and being referenced: a concurrent backfill
    landing in that window would otherwise produce a manifest describing a
    state that never existed. Entries are written before the manifest, whose
    insertion seals the snapshot for good.

    Any failure rolls the whole thing back: a manifest without its entries,
    or entries without their manifest, are states this primitive never
    leaves behind. SnapshotSelectionConflict and
    SnapshotEligibilityLimitExceeded propagate unchanged after the rollback.
    """
    _require_write_pragmas(connection)
    if connection.in_transaction:
        raise SnapshotWriteContextError(
            "snapshot materialization owns its transaction and cannot join an "
            "open one; commit or roll back before calling it"
        )

    request = {
        "provider": provider,
        "product_id": product_id,
        "timeframe": timeframe,
        "range_start": range_start,
        "range_end": range_end,
        "as_of": as_of,
    }
    snapshot_request_id = ""
    connection.execute("BEGIN IMMEDIATE")
    try:
        selected = _select_snapshot_receipts(connection, **request)
        snapshot_request_id = build_snapshot_request_id(**request)
        entries_content_hash = build_entries_content_hash(selected)
        snapshot_id = build_snapshot_id(
            snapshot_request_id=snapshot_request_id,
            entries_content_hash=entries_content_hash,
        )
        canonical_range_start, canonical_range_end = _canonical_range(
            range_start, range_end, timeframe=timeframe
        )
        canonical_as_of = _canonical_timestamp(as_of, field="as_of")
        expected_manifest = (
            snapshot_request_id,
            entries_content_hash,
            SNAPSHOT_SCHEMA_VERSION,
            SELECTION_POLICY_VERSION,
            provider,
            product_id,
            timeframe,
            canonical_range_start,
            canonical_range_end,
            canonical_as_of,
            len(selected),
        )
        expected_entries = tuple(
            (entry.bar_open_at, entry.content_sha256) for entry in selected
        )

        manifest_row = connection.execute(
            _SELECT_MANIFEST_SQL, (snapshot_id,)
        ).fetchone()
        if manifest_row is not None:
            _verify_existing_snapshot(
                connection,
                snapshot_id=snapshot_id,
                manifest_row=tuple(manifest_row),
                expected_manifest=expected_manifest,
                expected_entries=expected_entries,
                expected_entries_content_hash=entries_content_hash,
            )
            connection.rollback()
            return MaterializedSnapshot(
                snapshot_id=snapshot_id,
                snapshot_request_id=snapshot_request_id,
                entries_content_hash=entries_content_hash,
                entry_count=len(selected),
                created=False,
            )

        connection.executemany(
            _INSERT_ENTRY_SQL,
            [
                (snapshot_id, bar_open_at, content_sha256)
                for bar_open_at, content_sha256 in expected_entries
            ],
        )
        connection.execute(_INSERT_MANIFEST_SQL, (snapshot_id, *expected_manifest))

        validated = connection.execute(
            _VALIDATE_ENTRIES_SQL,
            (snapshot_id, provider, product_id, timeframe, canonical_as_of, canonical_as_of),
        ).fetchone()[0]
        if validated != len(selected):
            raise SnapshotPersistenceError(
                f"snapshot {snapshot_id!r} references {len(selected)} receipts but only "
                f"{validated} satisfy the domain and causality checks"
            )
        _verify_existing_snapshot(
            connection,
            snapshot_id=snapshot_id,
            manifest_row=tuple(
                connection.execute(_SELECT_MANIFEST_SQL, (snapshot_id,)).fetchone()
            ),
            expected_manifest=expected_manifest,
            expected_entries=expected_entries,
            expected_entries_content_hash=entries_content_hash,
        )
        connection.commit()
    except MarketSnapshotError:
        _rollback_quietly(connection)
        raise
    except sqlite3.Error as exc:
        _rollback_quietly(connection)
        # Identifiers and parameters only: never a payload, never the entry
        # list, however large it was.
        raise SnapshotPersistenceError(
            "snapshot persistence failed -- "
            f"snapshot_request_id={snapshot_request_id!r} provider={provider!r} "
            f"product_id={product_id!r} timeframe={timeframe!r}"
        ) from exc
    except BaseException:
        # BEGIN IMMEDIATE holds the write lock outright, so leaking it on a
        # KeyboardInterrupt or SystemExit is strictly worse than leaking a
        # read view: every other writer is locked out until this connection
        # dies. `except Exception` never sees either of them.
        _rollback_quietly(connection)
        raise

    return MaterializedSnapshot(
        snapshot_id=snapshot_id,
        snapshot_request_id=snapshot_request_id,
        entries_content_hash=entries_content_hash,
        entry_count=len(selected),
        created=True,
    )


def build_snapshot_request_id(
    *,
    provider: str,
    product_id: str,
    timeframe: str,
    range_start: datetime | str,
    range_end: datetime | str,
    as_of: datetime | str,
    selection_policy_version: str = SELECTION_POLICY_VERSION,
) -> str:
    """Deterministic identity of a snapshot REQUEST (parameters only, never
    the computed result). Two requests with equivalent parameters -- even if
    as_of/range_start/range_end are expressed with different UTC offsets for
    the same instant -- produce the same id. A different
    selection_policy_version always produces a different id, so a future
    change of selection rule can never silently reuse an incompatible id.
    """
    canonical_range_start, canonical_range_end = _canonical_range(
        range_start, range_end, timeframe=timeframe
    )
    identity = {
        "kind": "market_snapshot_request",
        "schema_version": SNAPSHOT_REQUEST_SCHEMA_VERSION,
        "provider": provider,
        "product_id": product_id,
        "timeframe": timeframe,
        "range_start": canonical_range_start,
        "range_end": canonical_range_end,
        "as_of": _canonical_timestamp(as_of, field="as_of"),
        "selection_policy_version": selection_policy_version,
    }
    return f"hyprl-market-snapshot-request-{_sha256_text(_canonical_json(identity))}"


def build_entries_content_hash(entries: tuple[SelectedSnapshotReceipt, ...]) -> str:
    """Deterministic identity of a snapshot RESULT.

    Independent of the order entries are passed in (always re-sorted by
    canonical bar_open_at before hashing). Each bar_open_at is canonicalized
    (UTC, timezone-aware) before comparison and hashing, so this helper
    protects its own contract instead of trusting that callers -- who may
    invoke it independently of _select_snapshot_receipts -- already pass
    canonical strings.

    Rejects, before any hash is computed, an entry set containing more than
    one entry for the same bar_open_at (whether their content_sha256 agree
    or conflict): such a set is structurally impossible for a real
    selection (one receipt per bar_open_at) and this helper must never mint
    a stable-looking identity for it.
    """
    canonical_entries: list[tuple[str, str]] = []
    seen_opens: set[str] = set()
    for entry in entries:
        canonical_bar_open_at = _canonical_timestamp(entry.bar_open_at, field="bar_open_at")
        if canonical_bar_open_at in seen_opens:
            raise MarketSnapshotError(
                "market snapshot entries are structurally invalid: duplicate "
                f"bar_open_at={canonical_bar_open_at!r}"
            )
        seen_opens.add(canonical_bar_open_at)
        canonical_entries.append((canonical_bar_open_at, entry.content_sha256))

    canonical_entries.sort(key=lambda item: item[0])
    envelope = {
        "kind": "market_snapshot_entries",
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "entries": [[bar_open_at, content_sha256] for bar_open_at, content_sha256 in canonical_entries],
    }
    return _sha256_text(_canonical_json(envelope))


def build_snapshot_id(*, snapshot_request_id: str, entries_content_hash: str) -> str:
    """Deterministic identity of one immutable MATERIALIZATION: combines the
    request identity and the result identity. The same request that yields a
    different result (e.g. after a retrodated backfill) produces a different
    snapshot_id -- the earlier materialization is never silently overwritten
    or reused.
    """
    identity = {
        "kind": "market_snapshot",
        "snapshot_request_id": snapshot_request_id,
        "entries_content_hash": entries_content_hash,
    }
    return f"hyprl-market-snapshot-{_sha256_text(_canonical_json(identity))}"


# =========================================================================
# Phase 1C-D: verified loading, bounded listing and offline replay.
#
# Reads are split in two phases on purpose. Phase A captures every value
# needed under ONE SQLite transaction and commits immediately; phase B does
# all decoding, hashing and validation on the captured copies, with no
# transaction open. In journal_mode=delete a read transaction blocks every
# writer's COMMIT, so holding it across JSON decoding would stall writers
# for the whole replay instead of just the capture. The split is safe
# because manifests, entries, receipts and ingestions are all insert-only
# and sealed: the captured values cannot become stale in a way that matters.
# =========================================================================

DEFAULT_SNAPSHOT_LIST_LIMIT = 100
MAX_SNAPSHOT_LIST_LIMIT = 1_000
SNAPSHOT_LOAD_FETCH_SIZE = 1_000
# Defence in depth against a sabotaged database, counted in UTF-8 bytes of
# payload_json. NOT a RAM limit: decoded Python objects cost several times
# their JSON size, so the peak sits well above this number.
MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES = 32_000_000

_SNAPSHOT_ID_PATTERN = re.compile(r"^hyprl-market-snapshot-[0-9a-f]{64}$")
_SNAPSHOT_REQUEST_ID_PATTERN = re.compile(r"^hyprl-market-snapshot-request-[0-9a-f]{64}$")
_REQUIRED_READ_PRAGMAS = ("foreign_keys", "recursive_triggers")


class SnapshotInputError(MarketSnapshotError):
    """Raised when a public read argument is malformed.

    Rejected before any transaction and before any business query, and never
    normalized silently: an identifier that differs by case or whitespace is
    a different string, not the same one written carelessly.
    """


class SnapshotNotFound(MarketSnapshotError):
    """Raised when no manifest carries the requested snapshot_id.

    Strictly distinct from SnapshotStateCorruption: absent is a legitimate
    answer, incomplete never is.
    """


class SnapshotUnsupportedVersion(MarketSnapshotError):
    """Raised when a stored version string is not one this build understands.

    Never upcast, never converted. An old proof stays readable only by a
    build that declares it can read it; anything else is a refusal.
    """


class SnapshotReadContextError(MarketSnapshotError):
    """Raised when the connection handed to a read cannot be used as-is."""


class SnapshotReplayLimitExceeded(MarketSnapshotError):
    """Raised when the stored bytes a replay would transfer exceed the budget."""


class SnapshotReadError(MarketSnapshotError):
    """Raised when SQLite fails during the capture phase."""


@dataclass(frozen=True)
class SnapshotManifest:
    """The immutable header of one materialized snapshot."""

    snapshot_id: str
    snapshot_request_id: str
    entries_content_hash: str
    snapshot_schema_version: str
    selection_policy_version: str
    provider: str
    product_id: str
    timeframe: str
    range_start: str
    range_end: str
    as_of: str
    entry_count: int


@dataclass(frozen=True)
class SnapshotEntryRef:
    """One selected opening and the receipt it points at.

    Deliberately minimal: bar_id, bar_version_id, ingested_at, available_at,
    the receipt's domain and its payload are all verified during loading but
    never published here. Exposing them would duplicate the domain across up
    to 10 000 objects and weld this API to Phase 1B's row shape.
    """

    bar_open_at: str
    content_sha256: str


@dataclass(frozen=True)
class LoadedSnapshot:
    manifest: SnapshotManifest
    entries: tuple[SnapshotEntryRef, ...]


@dataclass(frozen=True)
class SnapshotManifestPage:
    items: tuple[SnapshotManifest, ...]
    next_after_snapshot_id: str | None


@dataclass(frozen=True)
class _CapturedSnapshot:
    """Raw values copied out of one SQLite view, before any validation."""

    snapshot_id: str
    manifest_row: tuple
    metadata_rows: tuple[tuple, ...]
    payload_rows: tuple[tuple, ...] | None


def _require_snapshot_identifier(value: object, *, field: str, pattern: re.Pattern[str]) -> str:
    if type(value) is not str or pattern.fullmatch(value) is None:
        raise SnapshotInputError(
            f"{field} must be a canonical identifier matching {pattern.pattern}"
        )
    return value


def _require_list_limit(limit: object) -> int:
    # bool is an int subclass; True would otherwise pass as limit=1.
    if type(limit) is not int or not (1 <= limit <= MAX_SNAPSHOT_LIST_LIMIT):
        raise SnapshotInputError(
            f"limit must be an int between 1 and {MAX_SNAPSHOT_LIST_LIMIT}"
        )
    return limit


def _payload_byte_length(payload_json: str) -> int:
    """UTF-8 byte length, never len(str): a character count would under-count
    every non-ASCII payload and quietly widen the budget."""
    return len(payload_json.encode("utf-8"))


def _require_read_context(connection: sqlite3.Connection) -> None:
    """Refuse a connection that cannot host a clean read.

    The read path writes nothing, so the PRAGMAs are not needed for its own
    correctness; they are required as a health check that this connection
    came from MarketDataStore, so an audit never runs against a handle that
    could not uphold the store's guarantees.
    """
    if connection.in_transaction:
        raise SnapshotReadContextError(
            "snapshot reads own their transaction and cannot join an open one"
        )
    for pragma in _REQUIRED_READ_PRAGMAS:
        try:
            enabled = connection.execute(f"PRAGMA {pragma}").fetchone()[0]
        except sqlite3.Error as exc:
            raise SnapshotReadContextError(
                f"snapshot read context could not be verified: PRAGMA {pragma}"
            ) from exc
        if enabled != 1:
            raise SnapshotReadContextError(
                f"snapshot read context requires PRAGMA {pragma} = ON; "
                "open the connection through MarketDataStore"
            )


_SELECT_SNAPSHOT_MANIFEST_SQL = """
        SELECT snapshot_id, snapshot_request_id, entries_content_hash,
               snapshot_schema_version, selection_policy_version,
               provider, product_id, timeframe, range_start, range_end, as_of,
               entry_count
        FROM market_snapshot_manifests
        WHERE snapshot_id = ?
        """

# LEFT JOIN, never INNER: a missing receipt or ingestion must show up as a
# NULL to be reported, not vanish from the result set.
_SELECT_SNAPSHOT_METADATA_SQL = """
        SELECT e.bar_open_at, e.content_sha256,
               r.content_sha256, r.bar_open_at, r.bar_id, r.bar_version_id,
               r.ingested_at, r.available_at,
               i.ingestion_id, i.provider, i.product_id, i.timeframe
        FROM market_snapshot_entries e
        LEFT JOIN market_bar_receipts r ON r.content_sha256 = e.content_sha256
        LEFT JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id
        WHERE e.snapshot_id = ?
        ORDER BY e.bar_open_at ASC
        """

# Same shape plus the stored byte length, measured by SQLite so the payloads
# themselves are never transferred just to be counted.
_SELECT_SNAPSHOT_METADATA_WITH_BYTES_SQL = """
        SELECT e.bar_open_at, e.content_sha256,
               r.content_sha256, r.bar_open_at, r.bar_id, r.bar_version_id,
               r.ingested_at, r.available_at,
               i.ingestion_id, i.provider, i.product_id, i.timeframe,
               length(CAST(r.payload_json AS BLOB))
        FROM market_snapshot_entries e
        LEFT JOIN market_bar_receipts r ON r.content_sha256 = e.content_sha256
        LEFT JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id
        WHERE e.snapshot_id = ?
        ORDER BY e.bar_open_at ASC
        """

_SELECT_SNAPSHOT_PAYLOADS_SQL = """
        SELECT e.bar_open_at, e.content_sha256, r.content_sha256, r.payload_json
        FROM market_snapshot_entries e
        LEFT JOIN market_bar_receipts r ON r.content_sha256 = e.content_sha256
        WHERE e.snapshot_id = ?
        ORDER BY e.bar_open_at ASC
        """

_LIST_MANIFESTS_SQL = """
        SELECT snapshot_id, snapshot_request_id, entries_content_hash,
               snapshot_schema_version, selection_policy_version,
               provider, product_id, timeframe, range_start, range_end, as_of,
               entry_count
        FROM market_snapshot_manifests
        WHERE snapshot_request_id = ?
        ORDER BY snapshot_id ASC
        LIMIT ?
        """

_LIST_MANIFESTS_AFTER_SQL = """
        SELECT snapshot_id, snapshot_request_id, entries_content_hash,
               snapshot_schema_version, selection_policy_version,
               provider, product_id, timeframe, range_start, range_end, as_of,
               entry_count
        FROM market_snapshot_manifests
        WHERE snapshot_request_id = ? AND snapshot_id > ?
        ORDER BY snapshot_id ASC
        LIMIT ?
        """


def _require_supported_versions(manifest_row: tuple, *, snapshot_id: str) -> None:
    for label, found, expected in (
        ("snapshot_schema_version", manifest_row[3], SNAPSHOT_SCHEMA_VERSION),
        ("selection_policy_version", manifest_row[4], SELECTION_POLICY_VERSION),
    ):
        if found != expected:
            raise SnapshotUnsupportedVersion(
                f"snapshot {snapshot_id!r} declares {label}={found!r}; this build "
                f"only reads {expected!r}"
            )


@contextmanager
def _closing_cursor(cursor):
    """Close a cursor on every path, without ever masking a primary error.

    The read path must not rely on CPython's refcount, on __del__ or on the
    connection being closed later: a cursor left open holds SQLite resources
    for as long as the object survives. When the body already failed, a
    failure from close() is dropped -- the original error is what the caller
    needs to see. When the body succeeded, a failure from close() surfaces,
    because a capture whose cursor could not be released has not succeeded.
    """
    try:
        yield cursor
    except BaseException:
        try:
            cursor.close()
        except Exception:
            pass
        raise
    else:
        cursor.close()


def _rollback_quietly(connection) -> None:
    """Roll back without ever displacing the error being propagated.

    Same policy as `_closing_cursor`: cleanup runs on the way out of a
    failure, so a failure OF the cleanup must not become the verdict the
    caller sees -- the original error is what explains what happened. A
    rollback that cannot run means the connection is already broken, and
    hiding the real cause behind that fact helps nobody.
    """
    try:
        connection.rollback()
    except Exception:
        pass


def _fetch_all_bounded(cursor, *, fetch_size: int, max_rows: int) -> tuple[tuple, ...]:
    """Drain a cursor in bounded chunks, refusing more than `max_rows` rows.

    The bound is enforced DURING the drain, not after it: a cursor that keeps
    producing rows -- a corrupted database, or a duck-typed object that never
    empties -- is stopped instead of being allowed to fill memory before any
    validation runs.

    Requests shrink as the limit approaches: every `fetchmany` is sized so it
    can never reach past the sentinel position `max_rows + 1`. A conforming
    DB-API cursor therefore DELIVERS at most `max_rows + 1` rows before the
    refusal, and no row past the sentinel is ever delivered or consumed. The
    arithmetic sum of the requested sizes may still exceed `max_rows + 1`: a
    cursor that runs out early answers short, and the following request is a
    terminal probe that comes back empty. That probe delivers nothing, so the
    memory bound is untouched. Detection is immediate -- the cursor is not
    drained further once the overflow is seen.

    A cursor may of course ignore the requested size and return a larger
    chunk. That chunk is rejected on arrival and never accumulated, but this
    helper cannot prevent a foreign object from having allocated it already;
    it bounds its own accumulation, not someone else's.

    `fetch_size` and `max_rows` are internal call-site constants: passing an
    invalid one is a programming error (ValueError), never a data condition.
    """
    for name, value, minimum in (("fetch_size", fetch_size, 1), ("max_rows", max_rows, 0)):
        # bool is an int subclass; True must not read as 1 here.
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")

    rows: list[tuple] = []
    while True:
        remaining_with_sentinel = max_rows + 1 - len(rows)
        request_size = min(fetch_size, remaining_with_sentinel)
        chunk = cursor.fetchmany(request_size)
        if not chunk:
            break
        if len(chunk) > request_size:
            raise SnapshotStateCorruption(
                f"cursor returned {len(chunk)} rows for a request of {request_size}"
            )
        if len(rows) + len(chunk) > max_rows:
            raise SnapshotStateCorruption(
                f"read produced more than the {max_rows} rows it allows"
            )
        rows.extend(chunk)
    return tuple(rows)


def _capture_snapshot(
    connection: sqlite3.Connection,
    *,
    snapshot_id: str,
    include_payloads: bool,
) -> _CapturedSnapshot:
    """Copy everything one snapshot needs out of a single SQLite view.

    Owns its transaction exclusively, uses a plain BEGIN (never IMMEDIATE:
    a read must not take the write lock), and commits as soon as the values
    are in memory so the CPU-bound validation happens with no transaction
    held. All reads share one view, so a manifest can never be paired with
    entries written after it.
    """
    _require_read_context(connection)
    connection.execute("BEGIN")
    try:
        # One manifest per snapshot_id is the whole point of the primary key:
        # a second row is a corrupted state to refuse, not rows to gather.
        with _closing_cursor(
            connection.execute(_SELECT_SNAPSHOT_MANIFEST_SQL, (snapshot_id,))
        ) as manifest_cursor:
            manifest_rows = _fetch_all_bounded(
                manifest_cursor, fetch_size=SNAPSHOT_LOAD_FETCH_SIZE, max_rows=1
            )
            # Decided while the cursor context is still open, on purpose: the
            # verdict must become the active exception BEFORE the cursor is
            # closed, so that a failing close() can never displace it.
            if not manifest_rows:
                raise SnapshotNotFound(
                    f"no snapshot manifest for snapshot_id={snapshot_id!r}"
                )
            if len(manifest_rows) > 1:
                raise SnapshotStateCorruption(
                    f"snapshot_id={snapshot_id!r} matches {len(manifest_rows)} manifests"
                )
        manifest_row = tuple(manifest_rows[0])
        _require_supported_versions(manifest_row, snapshot_id=snapshot_id)

        metadata_sql = (
            _SELECT_SNAPSHOT_METADATA_WITH_BYTES_SQL
            if include_payloads
            else _SELECT_SNAPSHOT_METADATA_SQL
        )
        # Bounded independently of manifest.entry_count: that column is
        # persisted data and may itself be corrupted, so deriving the bound
        # from it could hide the very extra entry validation must catch.
        with _closing_cursor(
            connection.execute(metadata_sql, (snapshot_id,))
        ) as metadata_cursor:
            metadata_rows = tuple(
                tuple(row)
                for row in _fetch_all_bounded(
                    metadata_cursor,
                    fetch_size=SNAPSHOT_LOAD_FETCH_SIZE,
                    max_rows=MAX_SNAPSHOT_RANGE_OPENS,
                )
            )

        payload_rows: tuple[tuple, ...] | None = None
        if include_payloads:
            total_bytes = 0
            for row in metadata_rows:
                stored_bytes = row[12]
                if not isinstance(stored_bytes, int) or stored_bytes < 0:
                    raise SnapshotStateCorruption(
                        f"snapshot {snapshot_id!r} has an unreadable payload length for "
                        f"bar_open_at={row[0]!r}"
                    )
                total_bytes += stored_bytes
                if total_bytes > MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES:
                    # Refused from the lengths alone: not one payload has been
                    # transferred to Python at this point.
                    raise SnapshotReplayLimitExceeded(
                        f"snapshot {snapshot_id!r} stores at least {total_bytes} bytes, "
                        f"over the budget of {MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES}"
                    )
            with _closing_cursor(
                connection.execute(_SELECT_SNAPSHOT_PAYLOADS_SQL, (snapshot_id,))
            ) as payload_cursor:
                payload_rows = tuple(
                    tuple(row)
                    for row in _fetch_all_bounded(
                        payload_cursor,
                        fetch_size=SNAPSHOT_LOAD_FETCH_SIZE,
                        max_rows=MAX_SNAPSHOT_RANGE_OPENS,
                    )
                )
        connection.commit()
    except MarketSnapshotError:
        _rollback_quietly(connection)
        raise
    except sqlite3.Error as exc:
        _rollback_quietly(connection)
        raise SnapshotReadError(
            f"snapshot capture failed for snapshot_id={snapshot_id!r}"
        ) from exc
    except BaseException:
        # BaseException, not Exception: a KeyboardInterrupt or SystemExit
        # would otherwise skip the rollback and leave this BEGIN open, and in
        # journal_mode=delete an open read view blocks every writer's COMMIT
        # for as long as the connection lives.
        _rollback_quietly(connection)
        raise

    return _CapturedSnapshot(
        snapshot_id=snapshot_id,
        manifest_row=manifest_row,
        metadata_rows=metadata_rows,
        payload_rows=payload_rows,
    )


def _validate_captured_snapshot(
    captured: _CapturedSnapshot,
) -> tuple[SnapshotManifest, tuple[SnapshotEntryRef, ...], tuple[str, ...]]:
    """Validate a captured snapshot end to end, with no transaction open.

    Returns the public manifest, the public entry references, and the
    internal bar_version_id list the replay needs. Every failure is
    fail-closed and nothing partial is ever handed back.
    """
    row = captured.manifest_row
    snapshot_id = captured.snapshot_id
    manifest = SnapshotManifest(
        snapshot_id=row[0], snapshot_request_id=row[1], entries_content_hash=row[2],
        snapshot_schema_version=row[3], selection_policy_version=row[4],
        provider=row[5], product_id=row[6], timeframe=row[7],
        range_start=row[8], range_end=row[9], as_of=row[10], entry_count=row[11],
    )
    if not isinstance(manifest.entry_count, int) or not (
        0 <= manifest.entry_count <= MAX_SNAPSHOT_RANGE_OPENS
    ):
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} declares entry_count={manifest.entry_count!r}"
        )
    if len(captured.metadata_rows) != manifest.entry_count:
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} declares {manifest.entry_count} entries but "
            f"stores {len(captured.metadata_rows)}"
        )

    selected: list[SelectedSnapshotReceipt] = []
    entries: list[SnapshotEntryRef] = []
    previous_open: str | None = None
    for meta in captured.metadata_rows:
        (bar_open_at, content_sha256, receipt_sha, receipt_open, bar_id, bar_version_id,
         ingested_at, available_at, ingestion_id, provider, product_id, timeframe) = meta[:12]
        if previous_open is not None and not bar_open_at > previous_open:
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} entries are not strictly increasing at "
                f"bar_open_at={bar_open_at!r}"
            )
        previous_open = bar_open_at
        if receipt_sha is None:
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} references a missing receipt at "
                f"bar_open_at={bar_open_at!r}"
            )
        if ingestion_id is None:
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} references a receipt whose ingestion is "
                f"missing at bar_open_at={bar_open_at!r}"
            )
        if receipt_sha != content_sha256 or receipt_open != bar_open_at:
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} entry at bar_open_at={bar_open_at!r} does not "
                "match the receipt it points at"
            )
        if (provider, product_id, timeframe) != (
            manifest.provider, manifest.product_id, manifest.timeframe
        ):
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} references a receipt from another domain at "
                f"bar_open_at={bar_open_at!r}"
            )
        canonical_as_of = _canonical_timestamp(manifest.as_of, field="as_of")
        for label, value in (("ingested_at", ingested_at), ("available_at", available_at)):
            if _canonical_timestamp(value, field=label) > canonical_as_of:
                raise SnapshotStateCorruption(
                    f"snapshot {snapshot_id!r} references a receipt whose {label} is after "
                    f"as_of at bar_open_at={bar_open_at!r}"
                )
        entries.append(SnapshotEntryRef(bar_open_at=bar_open_at, content_sha256=content_sha256))
        selected.append(
            SelectedSnapshotReceipt(
                bar_open_at=bar_open_at, content_sha256=content_sha256, bar_id=bar_id,
                bar_version_id=bar_version_id, ingested_at=ingested_at,
                available_at=available_at,
            )
        )

    # Identities are recomputed with the module's own builders, never
    # reimplemented, and never trusted from the stored columns.
    try:
        entries_content_hash = build_entries_content_hash(tuple(selected))
        snapshot_request_id = build_snapshot_request_id(
            provider=manifest.provider, product_id=manifest.product_id,
            timeframe=manifest.timeframe, range_start=manifest.range_start,
            range_end=manifest.range_end, as_of=manifest.as_of,
            selection_policy_version=manifest.selection_policy_version,
        )
        recomputed_snapshot_id = build_snapshot_id(
            snapshot_request_id=snapshot_request_id,
            entries_content_hash=entries_content_hash,
        )
    except MarketSnapshotError as exc:
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} stores parameters its own identities cannot be "
            "recomputed from"
        ) from exc
    if entries_content_hash != manifest.entries_content_hash:
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} entries_content_hash does not describe its entries"
        )
    if snapshot_request_id != manifest.snapshot_request_id:
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} snapshot_request_id does not match its parameters"
        )
    if recomputed_snapshot_id != manifest.snapshot_id or recomputed_snapshot_id != snapshot_id:
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} identity does not match its own content"
        )
    return manifest, tuple(entries), tuple(item.content_sha256 for item in selected)


def _decode_market_bar(payload_json: str, *, expected_content_sha256: str) -> dict[str, object]:
    """Decode one stored MarketBar and prove it is the bar it claims to be.

    Rebuilt through market_bar's own builder rather than field-checked here:
    a payload whose hash was recomputed after tampering would satisfy a
    hash-only check, but cannot survive MarketBar V1 rebuilding it from its
    own declared fields.
    """
    try:
        record = json.loads(payload_json)
    except (TypeError, ValueError) as exc:
        raise SnapshotStateCorruption(
            f"stored payload for content_sha256={expected_content_sha256!r} is not valid JSON"
        ) from exc
    if not isinstance(record, dict):
        raise SnapshotStateCorruption(
            f"stored payload for content_sha256={expected_content_sha256!r} is not an object"
        )
    declared_version = record.get("schema_version")
    if declared_version != MARKET_BAR_SCHEMA_VERSION:
        raise SnapshotUnsupportedVersion(
            f"stored bar declares schema_version={declared_version!r}; this build only "
            f"reads {MARKET_BAR_SCHEMA_VERSION!r}"
        )
    unsigned = {key: value for key, value in record.items() if key != "content_sha256"}
    if _sha256_text(_canonical_json(unsigned)) != record.get("content_sha256"):
        raise SnapshotStateCorruption(
            f"stored payload does not hash to its own content_sha256 "
            f"({expected_content_sha256!r})"
        )
    if record.get("content_sha256") != expected_content_sha256:
        raise SnapshotStateCorruption(
            f"stored payload hashes to {record.get('content_sha256')!r}, not the referenced "
            f"{expected_content_sha256!r}"
        )
    try:
        rebuilt = build_market_bar(
            asset=record["asset"], venue=record["venue"], provider=record["provider"],
            timeframe=record["timeframe"], bar_open_at=record["bar_open_at"],
            bar_close_at=record["bar_close_at"], available_at=record["available_at"],
            ingested_at=record["ingested_at"], open_price=record["open"],
            high_price=record["high"], low_price=record["low"], close_price=record["close"],
            volume=record["volume"], raw_payload_sha256=record["raw_payload_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotStateCorruption(
            f"stored payload for content_sha256={expected_content_sha256!r} is not a valid "
            "MarketBar V1 record"
        ) from exc
    if rebuilt != record:
        raise SnapshotStateCorruption(
            f"stored payload for content_sha256={expected_content_sha256!r} does not match "
            "the MarketBar its own fields rebuild"
        )
    return record


def load_snapshot(connection: sqlite3.Connection, *, snapshot_id: str) -> LoadedSnapshot:
    """Load one snapshot's manifest and entry references, fully verified.

    Reads no payload at all -- a load is a proof of structure, not a replay.
    Absent, empty and corrupted are three different answers: SnapshotNotFound,
    an empty entries tuple, and SnapshotStateCorruption respectively. Nothing
    is ever repaired, and nothing is written.
    """
    _require_snapshot_identifier(snapshot_id, field="snapshot_id", pattern=_SNAPSHOT_ID_PATTERN)
    captured = _capture_snapshot(connection, snapshot_id=snapshot_id, include_payloads=False)
    manifest, entries, _ = _validate_captured_snapshot(captured)
    return LoadedSnapshot(manifest=manifest, entries=entries)


def list_snapshot_manifests(
    connection: sqlite3.Connection,
    *,
    snapshot_request_id: str,
    after_snapshot_id: str | None = None,
    limit: int = DEFAULT_SNAPSHOT_LIST_LIMIT,
) -> SnapshotManifestPage:
    """List the materializations of one request, one keyset page at a time.

    Ordered lexicographically by snapshot_id -- never chronologically, since
    no local clock exists and every materialization of a request shares the
    same as_of. A page is a consistent, ordered extract; paginating across
    several calls does NOT guarantee exhaustiveness while new snapshots are
    being written, because snapshot_id is a hash and therefore not monotonic:
    a new materialization sorting below the cursor already consumed will not
    be seen. No OFFSET, no rowid, no notion of latest.
    """
    _require_snapshot_identifier(
        snapshot_request_id, field="snapshot_request_id",
        pattern=_SNAPSHOT_REQUEST_ID_PATTERN,
    )
    if after_snapshot_id is not None:
        _require_snapshot_identifier(
            after_snapshot_id, field="after_snapshot_id", pattern=_SNAPSHOT_ID_PATTERN
        )
    bounded_limit = _require_list_limit(limit)
    _require_read_context(connection)

    if after_snapshot_id is None:
        sql, parameters = _LIST_MANIFESTS_SQL, (snapshot_request_id, bounded_limit + 1)
    else:
        sql, parameters = (
            _LIST_MANIFESTS_AFTER_SQL,
            (snapshot_request_id, after_snapshot_id, bounded_limit + 1),
        )
    # The listing drain obeys the same close policy as the capture path: a
    # secondary close failure must never displace the primary verdict, and no
    # raw sqlite3 error may escape a function whose error surface is
    # MarketSnapshotError. A bare `finally: cursor.close()` would do both.
    try:
        with _closing_cursor(connection.execute(sql, parameters)) as cursor:
            rows = _fetch_all_bounded(
                cursor, fetch_size=SNAPSHOT_LOAD_FETCH_SIZE, max_rows=bounded_limit + 1
            )
    except sqlite3.Error as exc:
        raise SnapshotReadError(
            f"snapshot listing failed for snapshot_request_id={snapshot_request_id!r}"
        ) from exc

    has_more = len(rows) > bounded_limit
    page_rows = rows[:bounded_limit]
    items = tuple(
        SnapshotManifest(
            snapshot_id=row[0], snapshot_request_id=row[1], entries_content_hash=row[2],
            snapshot_schema_version=row[3], selection_policy_version=row[4],
            provider=row[5], product_id=row[6], timeframe=row[7],
            range_start=row[8], range_end=row[9], as_of=row[10], entry_count=row[11],
        )
        for row in page_rows
    )
    return SnapshotManifestPage(
        items=items,
        next_after_snapshot_id=items[-1].snapshot_id if has_more and items else None,
    )


def replay_snapshot(
    connection: sqlite3.Connection, *, snapshot_id: str
) -> tuple[dict[str, object], ...]:
    """Rebuild the exact MarketBars a snapshot attests to, entirely offline.

    Nothing is contacted: no network, no provider, no broker. Every bar is
    decoded from its stored payload and proven against the receipt it is
    referenced by. Returns only once the LAST bar has been validated -- no
    generator, no partial tuple, no callback ever sees an unverified bar.
    """
    _require_snapshot_identifier(snapshot_id, field="snapshot_id", pattern=_SNAPSHOT_ID_PATTERN)
    captured = _capture_snapshot(connection, snapshot_id=snapshot_id, include_payloads=True)
    manifest, entries, _ = _validate_captured_snapshot(captured)

    payload_rows = captured.payload_rows or ()
    if len(payload_rows) != len(entries):
        raise SnapshotStateCorruption(
            f"snapshot {snapshot_id!r} returned {len(payload_rows)} payload rows for "
            f"{len(entries)} entries"
        )
    bars: list[dict[str, object]] = []
    observed_bytes = 0
    for entry, payload_row in zip(entries, payload_rows):
        bar_open_at, content_sha256, receipt_sha, payload_json = payload_row
        if bar_open_at != entry.bar_open_at or content_sha256 != entry.content_sha256:
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} payload rows do not align with its entries"
            )
        if receipt_sha is None or payload_json is None:
            raise SnapshotStateCorruption(
                f"snapshot {snapshot_id!r} has no stored payload at "
                f"bar_open_at={bar_open_at!r}"
            )
        observed_bytes += _payload_byte_length(payload_json)
        if observed_bytes > MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES:
            raise SnapshotReplayLimitExceeded(
                f"snapshot {snapshot_id!r} transferred {observed_bytes} bytes, over the "
                f"budget of {MAX_SNAPSHOT_REPLAY_PAYLOAD_BYTES}"
            )
        bars.append(_decode_market_bar(payload_json, expected_content_sha256=entry.content_sha256))
    return tuple(bars)

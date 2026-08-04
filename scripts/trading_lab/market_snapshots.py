"""Phase 1C-A/1C-B: causal eligibility, deterministic revision selection,
bounded cardinality and a bounded streaming read.

Historical causal snapshots based on declared ingestion time (Contract A,
see scripts/trading_lab/market_data_store.py): `as_of` is a cutoff over the
DECLARED historical `ingested_at` of already-persisted receipts, never a
live wall-clock, never a promise of lookahead-free real-time knowledge.

This module only *selects*, deterministically and without side effects,
which already-persisted MarketBar receipt represents the state of declared
knowledge as of `as_of` for each bar_open_at in a requested range. It never
persists a snapshot manifest, never touches market_bar_receipts' rows, and
never imports network or broker code.

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
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sqlite3

from scripts.trading_lab.market_bar import TIMEFRAME_DURATIONS


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
    cursor = connection.execute(
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
    try:
        while True:
            chunk = cursor.fetchmany(SNAPSHOT_RECEIPT_FETCH_CHUNK_SIZE)
            if not chunk:
                break
            eligible_row_count += len(chunk)
            if eligible_row_count >= query_limit:
                # Refuse the whole request here, before folding this chunk in
                # and before any entry is built: a snapshot must never be
                # derived from a prefix of a result set that was cut short.
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
    finally:
        cursor.close()

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

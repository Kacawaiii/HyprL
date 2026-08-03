"""Phase 1C-A: causal eligibility and deterministic revision selection.

Historical causal snapshots based on declared ingestion time (Contract A,
see scripts/trading_lab/market_data_store.py): `as_of` is a cutoff over the
DECLARED historical `ingested_at` of already-persisted receipts, never a
live wall-clock, never a promise of lookahead-free real-time knowledge.

This module only *selects*, deterministically and without side effects,
which already-persisted MarketBar receipt represents the state of declared
knowledge as of `as_of` for each bar_open_at in a requested range. It never
persists a snapshot manifest, never touches market_bar_receipts' rows, and
never imports network or broker code.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import sqlite3


SNAPSHOT_REQUEST_SCHEMA_VERSION = "trading-lab.market-snapshot-request.v1"
SNAPSHOT_SCHEMA_VERSION = "trading-lab.market-snapshot.v1"
SELECTION_POLICY_VERSION = "trading-lab.market-snapshot-selection.v1"


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


def _canonical_range(
    range_start: datetime | str, range_end: datetime | str
) -> tuple[str, str]:
    """Validate and canonicalize a [range_start, range_end) pair.

    Factored out so the selector and build_snapshot_request_id can never
    diverge on what makes a range valid: both call this, so a future change
    to one automatically applies to the other.
    """
    canonical_range_start = _canonical_timestamp(range_start, field="range_start")
    canonical_range_end = _canonical_timestamp(range_end, field="range_end")
    if canonical_range_end <= canonical_range_start:
        raise MarketSnapshotError("range_end must be strictly after range_start")
    return canonical_range_start, canonical_range_end


def _canonical_json(payload: object) -> str:
    try:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise MarketSnapshotError("market snapshot identity payload is invalid") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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

    All timestamps are validated and normalized to canonical UTC ISO form
    BEFORE any query is issued: an invalid range or a naive timestamp fails
    closed without ever touching `connection`.

    For each bar_open_at, among receipts with ingested_at <= as_of, the one
    (or ones) with the largest ingested_at are the candidates; if they carry
    more than one distinct bar_version_id, this is a genuine contradiction
    in the declared history and SnapshotSelectionConflict is raised --
    content_sha256 is only ever used to break a tie among candidates sharing
    the SAME bar_version_id, never to arbitrate between different OHLCV
    contents. Determinism never depends on rowid, INSERT order, SQL's
    returned row order, or dict/set iteration order: the winning ingested_at
    and the winning content_sha256 are each computed with an explicit max()
    over the fetched rows, and bar_open_at groups are visited in explicit
    sorted() order.
    """

    canonical_range_start, canonical_range_end = _canonical_range(range_start, range_end)
    canonical_as_of = _canonical_timestamp(as_of, field="as_of")

    rows = connection.execute(
        """
        SELECT r.bar_open_at, r.ingested_at, r.content_sha256,
               r.bar_id, r.bar_version_id, r.available_at
        FROM market_bar_receipts r
        JOIN market_ingestions i ON i.ingestion_id = r.ingestion_id
        WHERE i.provider = ? AND i.product_id = ? AND i.timeframe = ?
          AND r.bar_open_at >= ? AND r.bar_open_at < ?
          AND r.ingested_at <= ?
        ORDER BY r.bar_open_at ASC, r.ingested_at DESC, r.content_sha256 DESC
        """,
        (
            provider,
            product_id,
            timeframe,
            canonical_range_start,
            canonical_range_end,
            canonical_as_of,
        ),
    ).fetchall()

    by_open: dict[str, list[tuple]] = {}
    for row in rows:
        by_open.setdefault(row[0], []).append(row)

    selected: list[SelectedSnapshotReceipt] = []
    for bar_open_at in sorted(by_open):
        candidates = by_open[bar_open_at]
        max_ingested_at = max(row[1] for row in candidates)
        winners = [row for row in candidates if row[1] == max_ingested_at]
        distinct_versions = sorted({row[4] for row in winners})
        if len(distinct_versions) > 1:
            raise SnapshotSelectionConflict(
                provider=provider,
                product_id=product_id,
                timeframe=timeframe,
                bar_open_at=bar_open_at,
                ingested_at=max_ingested_at,
                bar_version_ids=tuple(distinct_versions),
            )
        canonical_winner = max(winners, key=lambda row: row[2])
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
    canonical_range_start, canonical_range_end = _canonical_range(range_start, range_end)
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

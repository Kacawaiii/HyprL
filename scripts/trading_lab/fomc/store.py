"""The authoritative FOMC EventStore: one SQLite file (WAL, synchronous=FULL, serialized writers)
plus content-addressed raw bodies (durable_ordering, raw_policy, storage_policy of design rev4).

Every write is one transaction with one `commit_seq` (STORE_COMMIT_SEQUENCE_V1). A transaction may
carry several rows (a processing outcome and the candidates it creates commit together); rows are
append-only and never updated. Uniqueness that the spec makes durable is enforced by partial unique
indexes, so a duplicate insert aborts the whole transaction.
"""

from __future__ import annotations

from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Callable, Iterable

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.fomc.clock import iso

UNIQUE_KINDS = (
    "ATTEMPT_OUTCOME",  # one outcome per TRANSPORT_INVOKED (attempt_outcome_fence)
    "EPISODE_OPEN",  # at most one episode per key, ever
    "EPISODE_TERMINAL",  # the first terminal record is final
    "ACQUISITION",  # one LIVE_ACQUISITION per source_item_id
    "CANDIDATE",
    "UNIDENTIFIABLE",
    "REVISION",  # (source item, content hash)
    "PROCESSING_OUTCOME",  # first terminal processing outcome is authoritative
    "LINK",
    "CYCLE_CONCLUSION",
    "DIAGNOSTIC_ONCE",  # GUID / title diagnostics raised once per value
    "INTEGRITY_DIAGNOSTIC",  # one per record, committed with its source-health result
    "GRANT",  # the FIX15 grant journal: one row per consumed grant, keyed (epoch, order)
)


class StoreBusy(RuntimeError):
    """The store lock was not obtained within the calling thread's bound: another operation holds it
    (possibly a stalled COMMIT). Nothing was read or written."""


# STORE_OPENING_RULE: a store records the schema version and spec hash it was written with. An existing
# store is inspected through a read-only connection before anything else; unless both match this code
# it is rejected (StoreRejected) before any write, pragma, DDL, epoch or request. There is no automatic
# migration: rows are append-only and never rewritten, so an older layout (LINK `mode`, REVISION
# without the normalized fields, no source-health rows, no schema version) cannot be upgraded in
# place without inventing history. Open such a store with the code that wrote it, or start a new one.
SCHEMA_VERSION = "fomc-store-v5"  # spec revision 25: content identity V2, FIX15 grant journal


class StoreRejected(RuntimeError):
    """An existing store this code must not open; nothing was written to it."""


def _admit_existing(db: Path) -> None:
    if not db.exists():
        return
    wal = db.with_name(db.name + "-wal")
    # read-only; without WAL frames the file alone is the store, so `immutable` creates no side file
    immutable = "" if wal.exists() and wal.stat().st_size else "&immutable=1"
    conn = sqlite3.connect(f"file:{db}?mode=ro{immutable}", uri=True)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not tables:
            return  # an empty file: a new store
        meta = dict(conn.execute("SELECT name, value FROM meta")) if "meta" in tables else {}
    finally:
        conn.close()
    version = meta.get("schema_version")
    if version != SCHEMA_VERSION:
        raise StoreRejected(f"store {db.parent} has schema {version or 'unversioned (written before ' + SCHEMA_VERSION + ')'}; "
                            f"this code opens only {SCHEMA_VERSION}. No migration: open it with the code that wrote it "
                            "or start a new store")
    if meta.get("spec_hash") != spec.SPEC_HASH:
        raise StoreRejected(f"store {db.parent} is bound to spec {meta.get('spec_hash')}, this code implements {spec.SPEC_HASH}")


class Rejected(RuntimeError):
    """An atomic predicate failed; nothing was committed."""


class RawCorrupt(RuntimeError):
    """Persisted raw bytes do not match their recorded SHA-256 (fail closed, never repaired)."""


@dataclass(frozen=True)
class Row:
    seq: int
    kind: str
    key: str | None
    body: dict


class FomcStore:
    def __init__(self, root: Path, *, wall_clock: Callable[[], datetime], mono: Callable[[], float] | None = None):
        self.root = Path(root)
        _admit_existing(self.root / "fomc.sqlite3")  # before any write (STORE_OPENING_RULE)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)
        self._wall = wall_clock
        self._mono = mono or time.monotonic
        self._lock = threading.RLock()
        # storage_incident.detection: markers of the operation in progress, written by the thread that
        # runs it and read by stalled() without the lock, so detection works while the store is stalled
        self._depth = 0
        self._op: tuple[str, float] | None = None  # (operation, started on the monotonic clock)
        self._raw_ops: dict[int, float] = {}  # thread ident -> start of a raw-body write in progress
        self._waits = threading.local()  # per-thread bound on waiting for the lock (None: unbounded)
        self.fault = None  # test hook: fault(operation) runs inside a marked store operation (a stall)
        # read accounting (cost regression tests): SQLite queries and rows decoded, and the in-memory
        # work of derivations, counted as the rows a view hands out (every derivation iterates them)
        self.reads = {"queries": 0, "rows": 0, "view_rows": 0}
        self._mirror = _Mirror()
        self._conn = sqlite3.connect(self.root / "fomc.sqlite3", isolation_level=None, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        mode = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
        sync = self._conn.execute("PRAGMA synchronous").fetchone()[0]
        if mode != "wal" or sync != 2:  # store_assumption: WAL + synchronous=FULL
            raise RuntimeError(f"FOMC store requires WAL/FULL, got {mode}/{sync}")
        with self.locked("open"):
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS txn (commit_seq INTEGER PRIMARY KEY AUTOINCREMENT, "
                    "kind TEXT NOT NULL, wall_at_commit TEXT NOT NULL)"
                )
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS rec (commit_seq INTEGER NOT NULL REFERENCES txn(commit_seq), "
                    "ord INTEGER NOT NULL, kind TEXT NOT NULL, key TEXT, body TEXT NOT NULL, PRIMARY KEY (commit_seq, ord))"
                )
                self._conn.execute("CREATE INDEX IF NOT EXISTS i_kind ON rec(kind, key)")
                for kind in UNIQUE_KINDS:
                    self._conn.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS u_{kind.lower()} ON rec(key) WHERE kind = '{kind}'")
                self._conn.execute(
                    "CREATE TABLE IF NOT EXISTS meta (name TEXT PRIMARY KEY, value TEXT NOT NULL)"
                )
                self._conn.execute("INSERT OR IGNORE INTO meta VALUES ('spec_hash', ?)", (spec.SPEC_HASH,))
                self._conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))
                bound = self._conn.execute("SELECT value FROM meta WHERE name = 'spec_hash'").fetchone()[0]
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        if bound != spec.SPEC_HASH:
            raise RuntimeError(f"store bound to spec {bound}, code implements {spec.SPEC_HASH}")

    def close(self) -> None:
        with self.locked("close"):
            self._conn.close()

    # ---- the lock, its operation marker and bounded waits -----------------------------------------
    @contextmanager
    def locked(self, operation: str):
        """Hold the store for one operation. The outermost holder publishes (operation, start) for
        stalled(); a thread with a wait bound (the service owner) gets StoreBusy instead of waiting
        behind a stalled operation."""
        bound = getattr(self._waits, "bound", None)
        if not (self._lock.acquire() if bound is None else self._lock.acquire(timeout=bound)):
            raise StoreBusy(f"store busy: {self._op[0] if self._op else 'another operation'} in progress")
        try:
            self._depth += 1
            if self._depth == 1:
                self._op = (operation, self._mono())
            yield
        finally:
            self._depth -= 1
            if self._depth == 0:
                self._op = None
            self._lock.release()

    def set_wait_bound(self, seconds: float | None) -> None:
        """Bound how long the calling thread waits for the store (None: wait without bound)."""
        self._waits.bound = seconds

    def use_clock(self, mono: Callable[[], float]) -> None:
        self._mono = mono

    def stalled(self) -> tuple[str, float] | None:
        """The store operation in progress for the longest time and its age in seconds, read without
        the lock (storage_incident.detection); None when nothing is in progress."""
        now = self._mono()
        ops = [(f"raw_write", now - start) for start in list(self._raw_ops.values())]
        op = self._op
        if op is not None:
            ops.append((op[0], now - op[1]))
        return max(ops, key=lambda o: o[1]) if ops else None

    # ---- writes -------------------------------------------------------------------------------
    def append(self, txn_kind: str, rows: Iterable[tuple[str, str | None, dict]],
               check: Callable[["FomcStore"], None] | None = None, *, wall_at_commit: str | None = None) -> int:
        """Commit one transaction. `check` runs under the write lock and may raise Rejected. A caller
        that records the commit's wall reading inside a row passes it as `wall_at_commit`, so the row
        repeats the transaction's own durable provenance exactly."""
        rows = list(rows)
        with self.locked(f"transaction {txn_kind}"):
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if check is not None:
                    check(self)
                cur = self._conn.execute(
                    "INSERT INTO txn (kind, wall_at_commit) VALUES (?, ?)", (txn_kind, wall_at_commit or iso(self._wall()))
                )
                seq = cur.lastrowid
                for ordinal, (kind, key, body) in enumerate(rows):
                    self._conn.execute(
                        "INSERT INTO rec (commit_seq, ord, kind, key, body) VALUES (?, ?, ?, ?, ?)",
                        (seq, ordinal, kind, key, spec.canonical_bytes(body).decode("utf-8")),
                    )
                if self.fault is not None:
                    self.fault(f"commit {txn_kind}")  # after the check and the insertions: the COMMIT itself stalls
                self._conn.execute("COMMIT")
                return seq
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise Rejected(f"uniqueness: {exc}") from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def wall_iso(self) -> str:
        """The local wall-clock reading, as provenance only (never a causal input)."""
        return iso(self._wall())

    # ---- reads (inside or outside a write transaction) ------------------------------------------
    def horizon(self) -> int:
        with self.locked("read"):
            self.reads["queries"] += 1
            return self._conn.execute("SELECT COALESCE(MAX(commit_seq), 0) FROM txn").fetchone()[0]

    def rows(self, kind: str | None = None, *, key: str | None = None, upto: int | None = None,
             after: int | None = None) -> list[Row]:
        sql, args = "SELECT commit_seq, kind, key, body FROM rec WHERE 1=1", []
        if after is not None:
            sql += " AND commit_seq > ?"; args.append(after)
        if kind is not None:
            sql += " AND kind = ?"; args.append(kind)
        if key is not None:
            sql += " AND key = ?"; args.append(key)
        if upto is not None:
            sql += " AND commit_seq <= ?"; args.append(upto)
        sql += " ORDER BY commit_seq, ord"
        with self.locked("read"):
            out = [Row(s, k, kk, json.loads(b)) for s, k, kk, b in self._conn.execute(sql, args)]
            self.reads["queries"] += 1
            self.reads["rows"] += len(out)
            return out

    def select(self, kind: str, field: str, value, *, upto: int | None = None) -> list[Row]:
        """Rows of `kind` whose body[field] == value, in commit order."""
        return self.view(upto).select(kind, field, value)

    def row_at(self, kind: str, seq: int) -> Row | None:
        """The row of `kind` committed by transaction `seq` (per-response and outcome records have their own)."""
        return self.view().row_at(kind, seq)

    def txns(self, *, upto: int | None = None, after: int = 0) -> list[tuple[int, str, str]]:
        with self.locked("read"):
            out = list(self._conn.execute(
                "SELECT commit_seq, kind, wall_at_commit FROM txn WHERE commit_seq > ? AND commit_seq <= ? ORDER BY commit_seq",
                (after, upto if upto is not None else 2**62),
            ))
            self.reads["queries"] += 1
            self.reads["rows"] += len(out)
            return out

    def view(self, upto: int | None = None) -> "StoreView":
        """A consistent read-only view up to min(upto, current horizon). The in-memory mirror reads
        only the rows committed since its last refresh, so repeated derivations never re-read the
        store (inside a write transaction the refresh sees exactly the committed state)."""
        with self.locked("read"):
            horizon = self.horizon()
            if horizon > self._mirror.upto:
                self._mirror.add(self.rows(after=self._mirror.upto, upto=horizon),
                                 self.txns(after=self._mirror.upto, upto=horizon), horizon)
            return StoreView(self, self._mirror, horizon if upto is None else min(upto, horizon))

    # ---- raw content-addressed bodies ---------------------------------------------------------
    def _raw_path(self, digest: str) -> Path:
        return self.root / "raw" / digest[:2] / digest

    def put_raw(self, data: bytes) -> str:
        """Durably write body bytes before the record that references them commits (raw-first).

        Raw is immutable: a digest path is published once with os.link (atomic create-if-absent) and
        never overwritten. If the path already holds other bytes, that corruption is reported
        (RawCorrupt) and left untouched - it is never repaired by a rewrite."""
        ident = threading.get_ident()
        self._raw_ops[ident] = self._mono()  # a raw write in progress (storage_incident.detection)
        try:
            return self._put_raw(data)
        finally:
            self._raw_ops.pop(ident, None)

    def _put_raw(self, data: bytes) -> str:
        digest = spec.sha256_bytes(data)
        path = self._raw_path(digest)
        if self.fault is not None:
            self.fault("raw_write")
        if os.path.lexists(path):
            self._verify_existing(path, digest)
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{digest}.tmp{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            self._verify_existing(path, digest)  # a concurrent writer won the race
        finally:
            os.unlink(tmp)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return digest

    @staticmethod
    def _verify_existing(path: Path, digest: str) -> None:
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise RawCorrupt(f"raw {digest} exists but is unreadable") from exc
        if spec.sha256_bytes(existing) != digest:
            raise RawCorrupt(f"raw {digest} exists with different bytes; it is never overwritten")

    def read_raw(self, digest: str) -> bytes:
        """RAW_INTEGRITY_EVERYWHERE_V1: every consumer re-verifies the digest before use."""
        try:
            data = self._raw_path(digest).read_bytes()
        except OSError as exc:
            raise RawCorrupt(f"raw {digest} missing") from exc
        if spec.sha256_bytes(data) != digest:
            raise RawCorrupt(f"raw {digest} digest mismatch")
        return data



class _Mirror:
    """Append-only in-memory copy of the committed rows, indexed by kind, (kind, key) and, lazily, by
    one body field. Rows with commit_seq <= its horizon never change, so it is only ever extended."""

    def __init__(self):
        self.upto = 0
        self.all: tuple[list, list] = ([], [])
        self.txns: tuple[list, list] = ([], [])
        self.by_kind: dict[str, tuple[list, list]] = {}
        self.by_key: dict[tuple, tuple[list, list]] = {}
        self.by_field: dict[tuple, dict] = {}
        self.aggregates: dict[str, object] = {}  # incremental derived state, fed row by row

    def aggregate(self, name: str, factory):
        """An incremental aggregate (an object with add(row)), built once from every mirrored row and
        then extended with each new row, so derivations never rescan for it."""
        agg = self.aggregates.get(name)
        if agg is None:
            agg = self.aggregates[name] = factory()
            for row in self.all[0]:
                agg.add(row)
        return agg

    @staticmethod
    def _push(indexed: tuple[list, list], row: Row) -> None:
        indexed[0].append(row)
        indexed[1].append(row.seq)

    @staticmethod
    def _value(v):
        return tuple(v) if isinstance(v, list) else v

    def add(self, rows: list[Row], txns: list, upto: int) -> None:
        for row in rows:
            self._push(self.all, row)
            self._push(self.by_kind.setdefault(row.kind, ([], [])), row)
            self._push(self.by_key.setdefault((row.kind, row.key), ([], [])), row)
            for (kind, field), index in self.by_field.items():
                if row.kind == kind:
                    self._push(index.setdefault(self._value(row.body.get(field)), ([], [])), row)
            for agg in self.aggregates.values():
                agg.add(row)
        for txn in txns:
            self.txns[0].append(txn)
            self.txns[1].append(txn[0])
        self.upto = upto

    def field_index(self, kind: str, field: str) -> dict:
        index = self.by_field.get((kind, field))
        if index is None:
            index = {}
            for row in self.by_kind.get(kind, ([], []))[0]:
                self._push(index.setdefault(self._value(row.body.get(field)), ([], [])), row)
            self.by_field[(kind, field)] = index  # published complete
        return index


class StoreView:
    """A read-only view of the store bounded by a horizon, served from the mirror: every derivation
    over it equals the same derivation over the store bounded by that horizon."""

    _EMPTY: tuple[list, list] = ([], [])

    def __init__(self, store: FomcStore, mirror: _Mirror, upto: int):
        self.store, self._mirror, self.upto = store, mirror, upto
        self._memo: dict = {}

    def at_head(self) -> bool:
        """True while no row newer than this view's horizon has been mirrored: aggregates that
        describe the mirror's head then describe this view exactly."""
        return self.upto == self._mirror.upto

    def read_aggregate(self, name: str, factory, read, *, head_only: bool = False):
        """read(aggregate) under the store lock, so a concurrent mirror refresh (another thread's
        commit) never shows a half-updated aggregate. With head_only, None unless this view is at the
        head. `read` must return a value that does not alias the aggregate's mutable state."""
        with self.store.locked("read"):
            if head_only and not self.at_head():
                return None
            return read(self._mirror.aggregate(name, factory))

    def memo(self, key, compute):
        """A derivation computed once per view (a view never changes)."""
        if key not in self._memo:
            self._memo[key] = compute()
        return self._memo[key]

    def _cut(self, indexed: tuple[list, list], upto: int | None) -> list:
        limit = self.upto if upto is None else min(upto, self.upto)
        out = indexed[0][:bisect_right(indexed[1], limit)]
        self.store.reads["view_rows"] += len(out)
        return out

    def horizon(self) -> int:
        return self.upto

    def view(self, upto: int | None = None) -> "StoreView":
        return self if upto is None or upto >= self.upto else StoreView(self.store, self._mirror, upto)

    def rows(self, kind: str | None = None, *, key: str | None = None, upto: int | None = None) -> list[Row]:
        if kind is None:
            return self._cut(self._mirror.all, upto)
        if key is None:
            return self._cut(self._mirror.by_kind.get(kind, self._EMPTY), upto)
        return self._cut(self._mirror.by_key.get((kind, key), self._EMPTY), upto)

    def select(self, kind: str, field: str, value, *, upto: int | None = None) -> list[Row]:
        index = self._mirror.by_field.get((kind, field))
        if index is None:  # built once, under the lock: a concurrent refresh extends every index
            with self.store.locked("read"):
                index = self._mirror.field_index(kind, field)
        return self._cut(index.get(_Mirror._value(value), self._EMPTY), upto)

    def txns(self, *, upto: int | None = None) -> list[tuple[int, str, str]]:
        return self._cut(self._mirror.txns, upto)

    def row_at(self, kind: str, seq: int) -> Row | None:
        rows, seqs = self._mirror.by_kind.get(kind, self._EMPTY)
        i = bisect_right(seqs, min(seq, self.upto)) - 1
        self.store.reads["view_rows"] += 1
        return rows[i] if i >= 0 and seqs[i] == seq else None

    def read_raw(self, digest: str) -> bytes:
        return self.store.read_raw(digest)

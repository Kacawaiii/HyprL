"""The authoritative FOMC EventStore: one SQLite file (WAL, synchronous=FULL, serialized writers)
plus content-addressed raw bodies (durable_ordering, raw_policy, storage_policy of design rev4).

Every write is one transaction with one `commit_seq` (STORE_COMMIT_SEQUENCE_V1). A transaction may
carry several rows (a processing outcome and the candidates it creates commit together); rows are
append-only and never updated. Uniqueness that the spec makes durable is enforced by partial unique
indexes, so a duplicate insert aborts the whole transaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import threading
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
)


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
    def __init__(self, root: Path, *, wall_clock: Callable[[], datetime]):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "raw").mkdir(exist_ok=True)
        self._wall = wall_clock
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.root / "fomc.sqlite3", isolation_level=None, check_same_thread=False, timeout=30)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        mode = self._conn.execute("PRAGMA journal_mode").fetchone()[0]
        sync = self._conn.execute("PRAGMA synchronous").fetchone()[0]
        if mode != "wal" or sync != 2:  # store_assumption: WAL + synchronous=FULL
            raise RuntimeError(f"FOMC store requires WAL/FULL, got {mode}/{sync}")
        with self._lock:
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
                bound = self._conn.execute("SELECT value FROM meta WHERE name = 'spec_hash'").fetchone()[0]
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
        if bound != spec.SPEC_HASH:
            raise RuntimeError(f"store bound to spec {bound}, code implements {spec.SPEC_HASH}")

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ---- writes -------------------------------------------------------------------------------
    def append(self, txn_kind: str, rows: Iterable[tuple[str, str | None, dict]],
               check: Callable[["FomcStore"], None] | None = None) -> int:
        """Commit one transaction. `check` runs under the write lock and may raise Rejected."""
        rows = list(rows)
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                if check is not None:
                    check(self)
                cur = self._conn.execute(
                    "INSERT INTO txn (kind, wall_at_commit) VALUES (?, ?)", (txn_kind, iso(self._wall()))
                )
                seq = cur.lastrowid
                for ordinal, (kind, key, body) in enumerate(rows):
                    self._conn.execute(
                        "INSERT INTO rec (commit_seq, ord, kind, key, body) VALUES (?, ?, ?, ?, ?)",
                        (seq, ordinal, kind, key, spec.canonical_bytes(body).decode("utf-8")),
                    )
                self._conn.execute("COMMIT")
                return seq
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise Rejected(f"uniqueness: {exc}") from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    # ---- reads (inside or outside a write transaction) ------------------------------------------
    def horizon(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COALESCE(MAX(commit_seq), 0) FROM txn").fetchone()[0]

    def rows(self, kind: str | None = None, *, key: str | None = None, upto: int | None = None) -> list[Row]:
        sql, args = "SELECT commit_seq, kind, key, body FROM rec WHERE 1=1", []
        if kind is not None:
            sql += " AND kind = ?"; args.append(kind)
        if key is not None:
            sql += " AND key = ?"; args.append(key)
        if upto is not None:
            sql += " AND commit_seq <= ?"; args.append(upto)
        sql += " ORDER BY commit_seq, ord"
        with self._lock:
            return [Row(s, k, kk, json.loads(b)) for s, k, kk, b in self._conn.execute(sql, args)]

    def txns(self, *, upto: int | None = None) -> list[tuple[int, str, str]]:
        with self._lock:
            return list(self._conn.execute(
                "SELECT commit_seq, kind, wall_at_commit FROM txn WHERE commit_seq <= ? ORDER BY commit_seq",
                (upto if upto is not None else 2**62,),
            ))

    # ---- raw content-addressed bodies ---------------------------------------------------------
    def _raw_path(self, digest: str) -> Path:
        return self.root / "raw" / digest[:2] / digest

    def put_raw(self, data: bytes) -> str:
        """Durably write body bytes before the record that references them commits (raw-first)."""
        digest = spec.sha256_bytes(data)
        path = self._raw_path(digest)
        if path.exists() and spec.sha256_bytes(path.read_bytes()) == digest:
            return digest
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        return digest

    def read_raw(self, digest: str) -> bytes:
        """RAW_INTEGRITY_EVERYWHERE_V1: every consumer re-verifies the digest before use."""
        try:
            data = self._raw_path(digest).read_bytes()
        except OSError as exc:
            raise RawCorrupt(f"raw {digest} missing") from exc
        if spec.sha256_bytes(data) != digest:
            raise RawCorrupt(f"raw {digest} digest mismatch")
        return data

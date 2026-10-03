"""The authoritative FOMC EventStore: one SQLite file (WAL, synchronous=FULL, serialized writers)
plus content-addressed raw bodies (durable_ordering, raw_policy, storage_policy of design rev4).

Every write is one transaction with one `commit_seq` (STORE_COMMIT_SEQUENCE_V1). A transaction may
carry several rows (a processing outcome and the candidates it creates commit together); rows are
append-only and never updated. Uniqueness that the spec makes durable is enforced by partial unique
indexes, so a duplicate insert aborts the whole transaction.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.sources.store import (  # noqa: F401 - the FOMC store's public names
    RawCorrupt, RecordStore, Rejected, Row, StoreBusy, StoreRejected, StoreView, admit_existing,
)

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


# STORE_OPENING_RULE: a store records the schema version and spec hash it was written with. An existing
# store is inspected through a read-only connection before anything else; unless both match this code
# it is rejected (StoreRejected) before any write, pragma, DDL, epoch or request. There is no automatic
# migration: rows are append-only and never rewritten, so an older layout (LINK `mode`, REVISION
# without the normalized fields, no source-health rows, no schema version) cannot be upgraded in
# place without inventing history. Open such a store with the code that wrote it, or start a new one.
SCHEMA_VERSION = "fomc-store-v5"  # spec revision 25: content identity V2, FIX15 grant journal


def _admit_existing(db: Path) -> None:
    admit_existing(db, schema_version=SCHEMA_VERSION, spec_hash=spec.SPEC_HASH)


class FomcStore(RecordStore):
    DB_NAME = "fomc.sqlite3"
    LABEL = "FOMC"

    def __init__(self, root: Path, *, wall_clock: Callable[[], datetime] | None, mono: Callable[[], float] | None = None,
                 read_only: bool = False):
        super().__init__(root, wall_clock=wall_clock, mono=mono, schema_version=SCHEMA_VERSION, spec_hash=spec.SPEC_HASH,
                         unique_kinds=UNIQUE_KINDS, read_only=read_only)

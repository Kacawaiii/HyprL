"""The EDGAR store: the shared append-only record store (sources.store) bound to the EDGAR spec."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable

from scripts.trading_lab.edgar import spec
from scripts.trading_lab.sources.store import RecordStore

UNIQUE_KINDS = (
    "MANIFEST",  # the watchlist, given once
    "ATTEMPT_OUTCOME",  # one outcome per attempt
    "PROCESSING_OUTCOME",  # one terminal processing outcome per listing record
    "FILING_REVISION",  # (source item, filing metadata identity)
    "FILING_OBSERVATION",  # (record, accession)
    "FILING_ABSENCE",  # (record, accession)
)


class EdgarStore(RecordStore):
    DB_NAME = "edgar.sqlite3"
    LABEL = "EDGAR"

    def __init__(self, root: Path, *, wall_clock: Callable[[], datetime] | None, mono: Callable[[], float] | None = None,
                 read_only: bool = False):
        super().__init__(root, wall_clock=wall_clock, mono=mono, schema_version=spec.SCHEMA_VERSION,
                         spec_hash=spec.SPEC_HASH, unique_kinds=UNIQUE_KINDS, read_only=read_only)

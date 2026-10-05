"""Reading a research archive must never write SQLite WAL side files there."""
import hashlib

import pytest

from scripts.trading_lab.research.store import IntegrityError, ResearchStore
from tests.research.conftest import issue


def fingerprint(root):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}


def test_closed_archive_files_unchanged_and_schema_stays_bound(tmp_path):
    writer = ResearchStore(tmp_path)
    prediction, _ = issue(writer)
    before = fingerprint(tmp_path)
    reader = ResearchStore(tmp_path, read_only=True)
    assert reader.get(prediction.identity)["payload"] == prediction.to_dict()
    assert reader.verify()["verified"]
    assert fingerprint(tmp_path) == before
    with writer.connect() as db:
        db.execute("UPDATE metadata SET schema='synthetic-wrong-schema'")
    before = fingerprint(tmp_path)
    with pytest.raises(IntegrityError, match="schema mismatch"):
        ResearchStore(tmp_path, read_only=True)
    assert fingerprint(tmp_path) == before


def test_live_archive_committed_wal_is_visible_without_touching_source_files(tmp_path):
    writer = ResearchStore(tmp_path)
    with writer.connect():
        first, _ = issue(writer)
        before = fingerprint(tmp_path)
        reader = ResearchStore(tmp_path, read_only=True)
        assert reader.get(first.identity)["payload"] == first.to_dict()
        assert fingerprint(tmp_path) == before
        second, _ = issue(writer, identifier="synthetic-second")
        before = fingerprint(tmp_path)
        assert reader.get(second.identity)["payload"] == second.to_dict()
        assert fingerprint(tmp_path) == before

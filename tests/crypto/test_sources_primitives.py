"""Primitives shared by the event sources: the record store's read-only opening (no byte written, every
write refused, incompatible stores rejected), the parametrized rolling limiter and the causal prefix."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3
import hashlib
import shutil

import pytest

from scripts.trading_lab.fomc.store import FomcStore
from scripts.trading_lab.sources import causal
from scripts.trading_lab.sources.limiter import RollingLimiter
from scripts.trading_lab.sources.store import StoreRejected

WALL = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)


def _fingerprint(root):
    return sorted((p.relative_to(root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                  for p in root.rglob("*"))


def _closed_store(root):
    store = FomcStore(root, wall_clock=lambda: WALL)
    digest = store.put_raw(b"bytes")
    store.append("T", [("EPOCH", "e1", {"token": "e1", "raw": digest})])
    store.close()
    with sqlite3.connect(root / "fomc.sqlite3") as conn:  # fold the WAL in, as a closure copy is
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.execute("PRAGMA journal_mode=DELETE")
    return digest


def test_a_read_only_store_writes_nothing_and_refuses_every_write(tmp_path):
    digest = _closed_store(tmp_path / "s")
    before = _fingerprint(tmp_path / "s")
    store = FomcStore(tmp_path / "s", wall_clock=None, read_only=True)
    assert [r.key for r in store.rows("EPOCH")] == ["e1"] and store.read_raw(digest) == b"bytes"
    assert store.view().horizon() == 1
    with pytest.raises(PermissionError):
        store.append("T", [("EPOCH", "e2", {})])
    with pytest.raises(PermissionError):
        store.put_raw(b"other")
    store.close()
    assert _fingerprint(tmp_path / "s") == before  # no -wal, no -shm, no meta row, no raw


def test_store_paths_are_encoded_before_use_as_sqlite_uris(tmp_path):
    root = tmp_path / "store?#% space"
    digest = _closed_store(root)
    before = _fingerprint(root)
    reader = FomcStore(root, wall_clock=None, read_only=True)
    try:
        assert reader.horizon() == 1 and reader.read_raw(digest) == b"bytes"
    finally:
        reader.close()
    assert _fingerprint(root) == before
    writer = FomcStore(root, wall_clock=lambda: WALL)
    writer.close()


def test_read_only_rejects_a_missing_or_incompatible_store(tmp_path):
    with pytest.raises(StoreRejected):
        FomcStore(tmp_path / "none", wall_clock=None, read_only=True)
    assert not (tmp_path / "none").exists()
    _closed_store(tmp_path / "s")
    with sqlite3.connect(tmp_path / "s" / "fomc.sqlite3") as conn:
        conn.execute("UPDATE meta SET value = 'fomc-store-v4' WHERE name = 'schema_version'")
    before = _fingerprint(tmp_path / "s")
    with pytest.raises(StoreRejected):
        FomcStore(tmp_path / "s", wall_clock=None, read_only=True)
    assert _fingerprint(tmp_path / "s") == before


def test_a_read_only_reader_of_a_live_store_sees_new_commits(tmp_path):
    writer = FomcStore(tmp_path / "s", wall_clock=lambda: WALL)
    writer.append("T", [("EPOCH", "e1", {})])
    reader = FomcStore(tmp_path / "s", wall_clock=None, read_only=True)  # WAL present: not immutable
    assert reader.view().horizon() == 1
    writer.append("T", [("EPOCH", "e2", {})])
    assert reader.view().horizon() == 2 and [r.key for r in reader.rows("EPOCH")] == ["e1", "e2"]
    reader.close()
    writer.close()


def _bytes_of(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def test_read_only_opening_does_not_change_a_live_stores_shared_memory(tmp_path):
    root = tmp_path / "live"
    writer = FomcStore(root, wall_clock=lambda: WALL)
    writer.append("T", [("EPOCH", "e1", {})])
    before = _bytes_of(root), _fingerprint(root)
    reader = FomcStore(root, wall_clock=None, read_only=True)
    try:
        assert reader.view().horizon() == 1
    finally:
        reader.close()
    assert (_bytes_of(root), _fingerprint(root)) == before
    writer.close()


def test_read_only_opening_a_wal_without_shared_memory_creates_no_source_file(tmp_path):
    writer = FomcStore(tmp_path / "live", wall_clock=lambda: WALL)
    writer.append("T", [("EPOCH", "e1", {})])
    root = tmp_path / "copy"
    root.mkdir()
    for name in ("fomc.sqlite3", "fomc.sqlite3-wal"):
        shutil.copyfile(writer.root / name, root / name)
    before = _bytes_of(root), _fingerprint(root)
    reader = FomcStore(root, wall_clock=None, read_only=True)
    try:
        assert reader.view().horizon() == 1
    finally:
        reader.close()
    assert (_bytes_of(root), _fingerprint(root)) == before
    writer.close()


def test_a_reader_opened_before_a_writer_restarts_sees_the_new_wal(tmp_path):
    root = tmp_path / "closed"
    _closed_store(root)
    reader = FomcStore(root, wall_clock=None, read_only=True)
    assert reader.view().horizon() == 1
    writer = FomcStore(root, wall_clock=lambda: WALL)
    writer.append("T", [("EPOCH", "e2", {})])
    try:
        assert reader.view().horizon() == 2
    finally:
        reader.close()
        writer.close()


def test_a_checkpoint_during_read_only_copy_retries_the_whole_snapshot(tmp_path, monkeypatch):
    root = tmp_path / "live"
    writer = FomcStore(root, wall_clock=lambda: WALL)
    writer.append("T", [("EPOCH", "e1", {})])
    original = shutil.copyfile
    injected = False

    def copy_then_checkpoint(source, target):
        nonlocal injected
        result = original(source, target)
        if source == root / "fomc.sqlite3" and not injected:
            injected = True
            writer.append("T", [("EPOCH", "e2", {})])
            writer._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return result

    monkeypatch.setattr(shutil, "copyfile", copy_then_checkpoint)
    reader = FomcStore(root, wall_clock=None, read_only=True)
    try:
        assert injected and reader.view().horizon() == 2
        assert [r.key for r in reader.rows("EPOCH")] == ["e1", "e2"]
    finally:
        reader.close()
        writer.close()


def test_the_rolling_limiter_takes_its_parameters():
    now = [0.0]
    limiter = RollingLimiter(lambda: now[0], lambda s: now.__setitem__(0, now[0] + s),
                             spacing_s=2, window_s=10, window_max=3, embargo_s=5)
    starts = [limiter.grant() for _ in range(5)]
    assert starts == [5.0, 7.0, 9.0, 15.0, 17.0]  # embargo, spacing, then the 3-per-10 s window


def test_the_causal_prefix_resolves_only_with_a_later_resolved_transaction():
    t0 = WALL
    table = [causal.Avail(1, True, t0), causal.Avail(2, True, t0 + timedelta(seconds=60)), causal.Avail(3, False, None)]
    assert causal.prefix(table, t0 + timedelta(seconds=30)) == (True, 1)
    assert causal.prefix(table, t0 + timedelta(seconds=90)) == (False, 2)  # the next one is unresolved
    assert causal.prefix(table[:2], t0 + timedelta(seconds=90)) == (False, 2)  # nothing beyond: unresolved


def test_timestamp_parsing_requires_an_instant_and_preserves_dst_offsets():
    from scripts.trading_lab.sources.httpclock import parse_iso
    with pytest.raises(ValueError, match="offset"):
        parse_iso("2026-11-01T01:30:00")
    # The repeated local hour names different instants on either side of the DST change.
    first = parse_iso("2026-11-01T01:30:00-04:00")
    second = parse_iso("2026-11-01T01:30:00-05:00")
    assert second - first == timedelta(hours=1)
    assert parse_iso("2026-11-01T05:30:00Z") == first


@pytest.mark.parametrize("date_line, age_lines, expected", [
    ("Fri, 02 Oct 2026 12:00:00 GMT\n", [], None),
    ("Fri, 02 Oct 2026 12:00:00 GMT", ["9" * 5000], None),
    ("Fri, 02 Oct 2026 12:00:00 GMT", ["0" * 5000], WALL),
    ("Fri, 31 Dec 9999 23:59:59 GMT", ["1"], None),
])
def test_unusable_http_clock_headers_do_not_crash_or_attest(date_line, age_lines, expected):
    from scripts.trading_lab.sources.httpclock import server_time
    assert server_time([date_line], age_lines, age_max=2147483647, age_cap_s=86400) == expected

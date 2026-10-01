"""STORE_OPENING_RULE: an existing store whose schema version or spec hash differs from this code is
rejected before any write (no pragma, DDL, epoch, lock or request); there is no automatic migration."""

from __future__ import annotations

import hashlib
import sqlite3

import pytest

from scripts.trading_lab.fomc import spec
from scripts.trading_lab.fomc import synthetic as syn
from scripts.trading_lab.fomc.collector import Collector
from scripts.trading_lab.fomc.store import SCHEMA_VERSION, FomcStore, StoreRejected

from tests.crypto.fomc_support import Env, statement_item


def _fingerprint(root):
    """Every byte of the store directory except SQLite's shared-memory index."""
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not p.name.endswith("-shm")}


@pytest.fixture
def closed_store(tmp_path):
    env = Env(tmp_path)
    env.feed([statement_item()])
    env.provider.routes[syn.statement_path("20260617")] = syn.page_response()
    env.drive(200)
    env.collector.close()
    env.store.close()
    env.provider.close()
    return env.root, env.clock


def _edit_meta(root, sql, *args):
    with sqlite3.connect(root / "fomc.sqlite3") as conn:
        conn.execute(sql, args)
    conn.close()


@pytest.mark.parametrize("sql, args, message", [
    ("DELETE FROM meta WHERE name = 'schema_version'", (), "unversioned"),  # a store of an early checkpoint
    ("UPDATE meta SET value = ? WHERE name = 'schema_version'", ("fomc-store-v2",), "fomc-store-v2"),  # an older schema
    ("UPDATE meta SET value = ? WHERE name = 'schema_version'", ("fomc-store-v3",), "fomc-store-v3"),  # the rev23 pilot's
    ("UPDATE meta SET value = ? WHERE name = 'schema_version'", ("fomc-store-v5",), "fomc-store-v5"),  # a later one
    ("UPDATE meta SET value = ? WHERE name = 'spec_hash'", (spec.SUPERSEDED_SPEC_HASH,), "bound to spec"),  # the previous revision
    ("UPDATE meta SET value = ? WHERE name = 'spec_hash'", ("0" * 64,), "bound to spec"),
])
def test_an_incompatible_store_is_rejected_before_any_activity(closed_store, sql, args, message):
    root, clock = closed_store
    _edit_meta(root, sql, *args)
    before = _fingerprint(root)
    with pytest.raises(StoreRejected, match=message):
        FomcStore(root, wall_clock=clock.wall)
    assert _fingerprint(root) == before  # not one byte written: no pragma, DDL, epoch or lock file
    assert not (root / "owner.lock").read_bytes()  # no collector ever took ownership


def test_the_current_schema_reopens_and_a_new_store_is_versioned(closed_store, tmp_path):
    root, clock = closed_store
    store = FomcStore(root, wall_clock=clock.wall)
    try:
        assert dict(store._conn.execute("SELECT name, value FROM meta"))["schema_version"] == SCHEMA_VERSION
        provider = syn.LocalProvider(clock)
        collector = Collector(store, provider.connector(), clock, boot_id="boot-2")  # activity starts only now
        collector.close()
        provider.close()
    finally:
        store.close()
    fresh = FomcStore(tmp_path / "new", wall_clock=clock.wall)
    try:
        assert dict(fresh._conn.execute("SELECT name, value FROM meta"))["schema_version"] == SCHEMA_VERSION
    finally:
        fresh.close()


def test_an_unversioned_store_with_pending_wal_frames_is_rejected_without_writing_it(tmp_path):
    env = Env(tmp_path)  # still open: its last transactions may live only in the WAL
    env.feed([])
    env.drive(120)
    env.store._conn.execute("DELETE FROM meta WHERE name = 'schema_version'")
    before = _fingerprint(env.root)
    try:
        with pytest.raises(StoreRejected, match="unversioned"):
            FomcStore(env.root, wall_clock=env.clock.wall)
        assert _fingerprint(env.root) == before
    finally:
        env.collector.close()
        env.store.close()
        env.provider.close()

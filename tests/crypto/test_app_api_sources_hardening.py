"""Offline regressions for source API refusals, bounds and concurrent readers."""

from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from scripts.trading_lab.app_api.contracts import ConflictError
from scripts.trading_lab.app_api.sources import EdgarViews, FomcViews
from scripts.trading_lab.edgar.store import EdgarStore
from scripts.trading_lab.fomc.store import FomcStore

WALL = datetime(2026, 10, 2, 12, tzinfo=timezone.utc)


@pytest.mark.parametrize("views_type, store_type", [(FomcViews, FomcStore), (EdgarViews, EdgarStore)])
def test_source_status_and_refusals_do_not_disclose_the_configured_path(tmp_path, views_type, store_type):
    root = tmp_path / "private-source-location"
    store = store_type(root, wall_clock=lambda: WALL)
    store.close()
    views = views_type(root)
    assert views.status()["status"] == "AVAILABLE"
    assert str(root) not in json.dumps(views.status())
    with sqlite3.connect(root / store_type.DB_NAME) as conn:
        conn.execute("UPDATE meta SET value = 'other-schema' WHERE name = 'schema_version'")
    status = views.status()
    assert status["status"] == "REJECTED" and str(root) not in json.dumps(status)
    with pytest.raises(ConflictError) as refused:
        views.snapshot(as_of=WALL.isoformat())
    assert str(root) not in str(refused.value)


@pytest.mark.parametrize("views_type, store_type", [(FomcViews, FomcStore), (EdgarViews, EdgarStore)])
def test_an_invalid_database_is_a_named_refusal(tmp_path, views_type, store_type):
    root = tmp_path / "invalid-store"
    root.mkdir()
    (root / store_type.DB_NAME).write_bytes(b"synthetic non-database bytes")
    status = views_type(root).status()
    assert status["status"] == "REJECTED" and str(root) not in json.dumps(status)
    with pytest.raises(ConflictError):
        views_type(root).snapshot(as_of=WALL.isoformat())


@pytest.mark.parametrize("views_type, store_type", [(FomcViews, FomcStore), (EdgarViews, EdgarStore)])
def test_status_attestation_uses_the_same_horizon_as_its_counts(tmp_path, monkeypatch, views_type, store_type):
    root = tmp_path / "live"
    writer = store_type(root, wall_clock=lambda: WALL)
    writer.append("T", [("EPOCH", "e1", {})])
    original = store_type.view
    injected = False

    def view_then_commit(store, *args, **kwargs):
        nonlocal injected
        view = original(store, *args, **kwargs)
        if store.read_only and not injected:
            injected = True
            writer.append("RESPONSE", [("RESPONSE", None, {"verdict": "CLOCK_VERIFIED", "late_evidence": False,
                                                          "observed_at": (WALL + timedelta(seconds=300)).isoformat()})])
        return view

    monkeypatch.setattr(store_type, "view", view_then_commit)
    try:
        status = views_type(root).status()
        assert writer.horizon() == 2 and status["horizon"] == 1
        assert status["counts"]["responses"] == 0 and status["suggested_as_of"] is None
    finally:
        writer.close()


def test_edgar_detail_verifies_raws_of_older_revisions_too(tmp_path):
    from tests.crypto.test_edgar_slice import Env, ACC1
    from scripts.trading_lab.edgar import synthetic as syn
    env = Env(tmp_path)
    try:
        middle = None
        for items in ("first", "middle", "first"):
            env.serve(syn.filing(ACC1, items=items))
            result = env.poll()
            if items == "middle":
                middle = env.store.row_at("RESPONSE", result["record"]).body["raw_sha"]
        snap = env.settle()
        raw = env.root / "raw" / middle[:2] / middle
        raw.write_bytes(b"synthetic corruption")
        views = EdgarViews(env.root)
        assert views.snapshot(as_of=snap["T"], horizon=snap["H"])["snapshot"]["read_state"] == "EDGAR_RESOLVED"
        with pytest.raises(ConflictError, match="integrity"):
            views.filing(ACC1, as_of=snap["T"], horizon=snap["H"])
    finally:
        env.collector.close()
        env.store.close()


def test_fomc_detail_verifies_raws_of_older_revisions_too(tmp_path):
    from tests.crypto.fomc_support import Env, P1, SID1, statement_item
    from scripts.trading_lab.fomc import synthetic as syn
    env = Env(tmp_path)
    try:
        env.feed([statement_item()])
        env.provider.routes[P1] = syn.page_response(body="first synthetic revision")
        env.drive(240)
        old_raw = env.store.rows("REVISION")[0].body["first_raw_sha256"]
        env.provider.routes[P1] = syn.page_response(body="second synthetic revision")
        env.drive(1000, idle=60)
        raw = env.root / "raw" / old_raw[:2] / old_raw
        raw.write_bytes(b"synthetic corruption")
        views = FomcViews(env.root)
        as_of = views.status()["suggested_as_of"]
        assert views.snapshot(as_of=as_of)["snapshot"]["read_state"] == "FOMC_RESOLVED"
        with pytest.raises(ConflictError, match="integrity"):
            views.item(SID1, as_of=as_of)
    finally:
        env.collector.close()
        env.store.close()
        env.provider.close()

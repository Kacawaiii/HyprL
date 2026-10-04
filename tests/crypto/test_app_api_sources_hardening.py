"""Offline regressions for source API refusals, bounds and concurrent readers."""

from datetime import datetime, timezone
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

"""Pytest configuration for offline API tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
project_root_str = str(PROJECT_ROOT)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

import pytest  # noqa: E402

from api import db  # noqa: E402

os.environ.setdefault("HYPRL_DEV_TOKEN", "hyprl_dev_token_123")


@pytest.fixture()
def sqlite_session(tmp_path, monkeypatch):
    """Provide a fresh SQLite database per test."""
    db_path = tmp_path / "hyprl_test.db"
    db_url = f"sqlite:///{db_path}"
    monkeypatch.setenv("HYPRL_DB_URL", db_url)
    monkeypatch.setenv("HYPRL_ADMIN_TOKEN", "tok_admin.hyprl_admin_test")
    db.configure_engine(db_url, force=True)
    db.init_db()
    from api.auth import bootstrap_admin

    bootstrap_admin("tok_admin.hyprl_admin_test")
    yield db.SessionLocal

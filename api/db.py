"""Database utilities for HyprL V2 API."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DEFAULT_DB_URL = "sqlite:///data/hyprl_v2.db"


class Base(DeclarativeBase):
    """Declarative base class shared across models."""


DATABASE_URL = os.getenv("HYPRL_DB_URL", DEFAULT_DB_URL)
_engine: Engine | None = None
SessionLocal: sessionmaker[Session]


def _ensure_sqlite_parent(db_url: str) -> None:
    if db_url.startswith("sqlite:///"):
        db_path = db_url.replace("sqlite:///", "", 1)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _create_engine(db_url: str) -> Engine:
    _ensure_sqlite_parent(db_url)
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    return create_engine(db_url, connect_args=connect_args, future=True)


def configure_engine(db_url: Optional[str] = None, *, force: bool = False) -> None:
    """Reconfigure the global engine/session factory (useful for tests)."""
    global _engine, SessionLocal, DATABASE_URL
    target_url = db_url or os.getenv("HYPRL_DB_URL", DEFAULT_DB_URL)
    if _engine is not None and not force and target_url == DATABASE_URL:
        return
    DATABASE_URL = target_url
    if _engine is not None:
        _engine.dispose()
    _engine = _create_engine(DATABASE_URL)
    SessionLocal = sessionmaker(
        bind=_engine,
        autocommit=False,
        autoflush=False,
        expire_on_commit=False,
        future=True,
    )


def get_engine() -> Engine:
    if _engine is None:
        configure_engine()
    assert _engine is not None  # for mypy
    return _engine


def init_db() -> None:
    """Create tables if missing."""
    from api import models  # noqa: F401  (ensures models are registered)

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency returning a database session."""
    if _engine is None:
        configure_engine()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Context manager for transactional operations."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# initialize engine/session on import
configure_engine(DATABASE_URL, force=True)

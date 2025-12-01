"""Database helpers for the HyprL public API."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from hyprl_api.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for API persistence."""


_engine: Engine | None = None
SessionLocal: sessionmaker[Session]


def _default_url() -> str:
    return get_settings().database_url


def _ensure_parent_dir(url: str) -> None:
    if url.startswith("sqlite"):
        if url.startswith("sqlite:///"):
            db_path = url.replace("sqlite:///", "", 1)
        elif url.startswith("sqlite+pysqlite:///"):
            db_path = url.replace("sqlite+pysqlite:///", "", 1)
        else:
            return
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def _create_engine(url: str) -> Engine:
    _ensure_parent_dir(url)
    connect_args: dict[str, object] = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, future=True, connect_args=connect_args)


def configure_engine(url: str | None = None, *, force: bool = False) -> None:
    """Configure the global SQLAlchemy engine/session factory."""

    global _engine, SessionLocal
    target_url = url or _default_url()
    if _engine is not None and not force:
        return
    _engine = _create_engine(target_url)
    SessionLocal = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True,
    )


def get_engine() -> Engine:
    if _engine is None:
        configure_engine()
    assert _engine is not None
    return _engine


def init_db() -> None:
    """Create DB schema if missing."""

    from hyprl_api import models  # noqa: F401 - ensure metadata registration

    engine = get_engine()
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    if _engine is None:
        configure_engine()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


configure_engine()

"""SQLAlchemy engine/session for the CycleZero Postgres DB.

Sync (not async) on purpose: FastAPI runs sync route handlers in a threadpool,
and sync SQLAlchemy avoids the session/greenlet sharp edges of the async ORM.
The connection URL comes from ``CYCLEZERO_DATABASE_URL`` (in .env.secrets).
"""
from __future__ import annotations

from typing import Iterator, Optional

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


_engine: Optional[Engine] = None
_Session: Optional[sessionmaker] = None


def _database_url() -> str:
    url = getattr(settings, "CYCLEZERO_DATABASE_URL", None)
    if not url:
        raise RuntimeError(
            "CYCLEZERO_DATABASE_URL is not configured (set it in .env.secrets)"
        )
    return url


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = create_engine(_database_url(), pool_pre_ping=True, future=True)
    return _engine


def get_session_factory() -> sessionmaker:
    global _Session
    if _Session is None:
        _Session = sessionmaker(
            bind=get_engine(), expire_on_commit=False, future=True, class_=Session
        )
    return _Session


def init_models() -> None:
    """Create tables if missing (no-op when they already exist). Idempotent."""
    from . import models  # noqa: F401 — register mappers

    Base.metadata.create_all(get_engine(), checkfirst=True)


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yields a session and always closes it."""
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()

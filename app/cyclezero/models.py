"""ORM models for the CycleZero design graph (generic node + edge).

Column types are portable: native ``UUID``/``JSONB`` on Postgres, and plain
``CHAR``/``JSON`` on SQLite — so the same models run under pytest (in-memory
SQLite) and in production (Postgres). IDs are generated Python-side
(``uuid.uuid4``) so inserts work identically on both backends.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import JSON, DateTime, ForeignKey, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base

# JSON on SQLite, JSONB on Postgres.
JSONType = JSON().with_variant(JSONB, "postgresql")


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Game(Base):
    __tablename__ = "games"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slug: Mapped[str] = mapped_column(Text, unique=True, index=True)  # the resource URI
    title: Mapped[str] = mapped_column(Text)
    owner_id: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(Text, default="draft")
    data: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Entity(Base):
    """A generic node, typed by ``layer`` (scene|character|system|story|…)."""

    __tablename__ = "entities"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    game_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("games.id", ondelete="CASCADE"), index=True
    )
    layer: Mapped[str] = mapped_column(Text)
    key: Mapped[str] = mapped_column(Text)  # unique within (game, layer)
    name: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )


class Relation(Base):
    """A generic edge between two entities (APPEARS_IN|PART_OF|…)."""

    __tablename__ = "relations"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    game_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("games.id", ondelete="CASCADE"), index=True
    )
    src_entity: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    dst_entity: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("entities.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(Text)
    data: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class AssetJob(Base):
    """A generation trigger; ``result`` holds S3 links + the Mongo run id."""

    __tablename__ = "asset_jobs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    game_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("games.id", ondelete="CASCADE"), index=True
    )
    entity_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        Uuid, ForeignKey("entities.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(Text)  # image|3d|rig|…
    status: Mapped[str] = mapped_column(Text, default="queued", index=True)
    params: Mapped[dict] = mapped_column(JSONType, default=dict)
    result: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_now, onupdate=_now
    )

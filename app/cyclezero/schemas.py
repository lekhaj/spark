"""Pydantic request/response schemas for the CycleZero API."""
from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(text: str) -> str:
    s = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return s or "game"


# ── games ────────────────────────────────────────────────────────────────────
class GameCreate(BaseModel):
    title: str
    slug: Optional[str] = None
    owner_id: Optional[str] = None
    status: str = "draft"
    data: Dict[str, Any] = Field(default_factory=dict)


class GameUpdate(BaseModel):
    title: Optional[str] = None
    status: Optional[str] = None
    data: Optional[Dict[str, Any]] = None


class GameOut(BaseModel):
    id: uuid.UUID
    slug: str
    title: str
    owner_id: Optional[str]
    status: str
    data: Dict[str, Any]
    uri: str
    created_at: datetime
    updated_at: datetime


# ── entities ─────────────────────────────────────────────────────────────────
class EntityCreate(BaseModel):
    layer: str
    name: str
    key: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)
    spec_stage: Optional[str] = None  # defaults from the layer metamodel if omitted


class EntityUpdate(BaseModel):
    name: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    spec_stage: Optional[str] = None


class EntityOut(BaseModel):
    id: uuid.UUID
    game_id: uuid.UUID
    layer: str
    key: str
    name: str
    data: Dict[str, Any]
    spec_stage: Optional[str] = None
    accepted_spec_run_id: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ── relations ────────────────────────────────────────────────────────────────
class RelationCreate(BaseModel):
    src: str = Field(description="source entity key")
    dst: str = Field(description="destination entity key")
    kind: str
    data: Dict[str, Any] = Field(default_factory=dict)


class RelationOut(BaseModel):
    id: uuid.UUID
    game_id: uuid.UUID
    src_entity: uuid.UUID
    dst_entity: uuid.UUID
    kind: str
    data: Dict[str, Any]
    created_at: datetime


# ── asset jobs ───────────────────────────────────────────────────────────────
class JobCreate(BaseModel):
    kind: str = Field(description="image|3d|rig|…")
    params: Dict[str, Any] = Field(default_factory=dict)


class JobOut(BaseModel):
    id: uuid.UUID
    game_id: uuid.UUID
    entity_id: Optional[uuid.UUID]
    kind: str
    status: str
    params: Dict[str, Any]
    result: Dict[str, Any]
    error: Optional[str]
    created_at: datetime
    updated_at: datetime


# ── matching ─────────────────────────────────────────────────────────────────
class MatchRequest(BaseModel):
    contract: Dict[str, Any] = Field(description="the built/deployed scene contract JSON")

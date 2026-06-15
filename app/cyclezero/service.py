"""CRUD service for the CycleZero design graph. Pure data access — no FastAPI
types here, so it is unit-testable against any SQLAlchemy session."""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from . import models, schemas


# ── games ────────────────────────────────────────────────────────────────────
def create_game(db: Session, body: schemas.GameCreate) -> models.Game:
    slug = body.slug or schemas.slugify(body.title)
    game = models.Game(
        slug=slug,
        title=body.title,
        owner_id=body.owner_id,
        status=body.status,
        data=body.data or {},
    )
    db.add(game)
    db.commit()
    db.refresh(game)
    return game


def list_games(db: Session) -> List[models.Game]:
    return list(db.scalars(select(models.Game).order_by(models.Game.created_at.desc())))


def get_game(db: Session, slug: str) -> Optional[models.Game]:
    return db.scalar(select(models.Game).where(models.Game.slug == slug))


def update_game(db: Session, game: models.Game, body: schemas.GameUpdate) -> models.Game:
    if body.title is not None:
        game.title = body.title
    if body.status is not None:
        game.status = body.status
    if body.data is not None:
        game.data = body.data
    db.commit()
    db.refresh(game)
    return game


def delete_game(db: Session, game: models.Game) -> None:
    db.delete(game)
    db.commit()


# ── entities ─────────────────────────────────────────────────────────────────
def create_entity(
    db: Session,
    game: models.Game,
    body: schemas.EntityCreate,
    spec_stage: Optional[str] = None,
) -> models.Entity:
    key = body.key or schemas.slugify(body.name)
    entity = models.Entity(
        game_id=game.id,
        layer=body.layer,
        key=key,
        name=body.name,
        data=body.data or {},
        # explicit body value wins; else the route passes the layer's bound stage
        spec_stage=body.spec_stage or spec_stage,
    )
    db.add(entity)
    db.commit()
    db.refresh(entity)
    return entity


def list_entities(
    db: Session, game: models.Game, layer: Optional[str] = None
) -> List[models.Entity]:
    stmt = select(models.Entity).where(models.Entity.game_id == game.id)
    if layer:
        stmt = stmt.where(models.Entity.layer == layer)
    return list(db.scalars(stmt.order_by(models.Entity.layer, models.Entity.key)))


def get_entity(db: Session, game: models.Game, key: str) -> Optional[models.Entity]:
    return db.scalar(
        select(models.Entity).where(
            models.Entity.game_id == game.id, models.Entity.key == key
        )
    )


def update_entity(
    db: Session, entity: models.Entity, body: schemas.EntityUpdate
) -> models.Entity:
    if body.name is not None:
        entity.name = body.name
    if body.data is not None:
        entity.data = body.data
    if body.spec_stage is not None:
        entity.spec_stage = body.spec_stage
    db.commit()
    db.refresh(entity)
    return entity


def delete_entity(db: Session, entity: models.Entity) -> None:
    db.delete(entity)
    db.commit()


# ── S1 bridge: link a node to its accepted Mongo spec body ────────────────────
def get_entity_by_key(db: Session, game_id: uuid.UUID, key: str) -> Optional[models.Entity]:
    return db.scalar(
        select(models.Entity).where(
            models.Entity.game_id == game_id, models.Entity.key == key
        )
    )


def set_accepted_spec(
    db: Session, entity: models.Entity, run_id: Optional[str]
) -> models.Entity:
    """Stamp (or clear) the accepted spec-run pointer on a node."""
    entity.accepted_spec_run_id = run_id
    db.commit()
    db.refresh(entity)
    return entity


def resolve_body(mongo_db, entity: models.Entity) -> Optional[Dict[str, Any]]:
    """Fetch the validated JSON body for a node from Mongo ``spec_gen_runs``.

    Returns the accepted run's ``output`` (the schema-validated content), or None
    when the node has no accepted spec yet. ``mongo_db`` is a pymongo Database.
    """
    if not entity.accepted_spec_run_id:
        return None
    doc = mongo_db["spec_gen_runs"].find_one({"run_id": entity.accepted_spec_run_id})
    return doc.get("output") if doc else None


# ── relations ────────────────────────────────────────────────────────────────
def create_relation(
    db: Session,
    game: models.Game,
    body: schemas.RelationCreate,
    metamodel: Optional[Dict[str, Any]] = None,
) -> models.Relation:
    src = get_entity(db, game, body.src)
    dst = get_entity(db, game, body.dst)
    if src is None or dst is None:
        missing = body.src if src is None else body.dst
        raise ValueError(f"entity not found: {missing}")
    # S2: when a metamodel is supplied, the edge must obey the relation contract
    # (legal layers + cardinality). Omitting it keeps create_relation usable as a
    # raw graph op (e.g. in pure-Postgres tests).
    if metamodel is not None:
        _validate_new_edge(db, game, src, dst, body.kind, metamodel)
    rel = models.Relation(
        game_id=game.id,
        src_entity=src.id,
        dst_entity=dst.id,
        kind=body.kind,
        data=body.data or {},
    )
    db.add(rel)
    db.commit()
    db.refresh(rel)
    return rel


def _validate_new_edge(
    db: Session,
    game: models.Game,
    src: models.Entity,
    dst: models.Entity,
    kind: str,
    metamodel: Dict[str, Any],
) -> None:
    """Raise ValueError if a proposed edge breaks the relation contract."""
    rt = metamodel.get("relation_types", {}).get(kind)
    if rt is None:
        raise ValueError(f"unknown relation kind: {kind}")
    if rt.get("src_layers") and src.layer not in rt["src_layers"]:
        raise ValueError(f"src layer '{src.layer}' not allowed for {kind}")
    if rt.get("dst_layers") and dst.layer not in rt["dst_layers"]:
        raise ValueError(f"dst layer '{dst.layer}' not allowed for {kind}")
    existing = list_relations(db, game)
    if rt.get("src_cardinality") == "one" and any(
        r.kind == kind and r.src_entity == src.id for r in existing
    ):
        raise ValueError(f"{kind} is one-per-source; '{src.key}' already has one")
    if rt.get("dst_cardinality") == "one" and any(
        r.kind == kind and r.dst_entity == dst.id for r in existing
    ):
        raise ValueError(f"{kind} is one-per-target; '{dst.key}' already has one")


def list_relations(db: Session, game: models.Game) -> List[models.Relation]:
    return list(
        db.scalars(select(models.Relation).where(models.Relation.game_id == game.id))
    )


def delete_relation(db: Session, game: models.Game, rel_id: uuid.UUID) -> bool:
    rel = db.scalar(
        select(models.Relation).where(
            models.Relation.id == rel_id, models.Relation.game_id == game.id
        )
    )
    if rel is None:
        return False
    db.delete(rel)
    db.commit()
    return True


# ── asset jobs ───────────────────────────────────────────────────────────────
def create_job(
    db: Session,
    game: models.Game,
    entity: Optional[models.Entity],
    body: schemas.JobCreate,
) -> models.AssetJob:
    job = models.AssetJob(
        game_id=game.id,
        entity_id=entity.id if entity else None,
        kind=body.kind,
        status="queued",
        params=body.params or {},
        result={},
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def list_jobs(db: Session, game: models.Game) -> List[models.AssetJob]:
    return list(
        db.scalars(
            select(models.AssetJob)
            .where(models.AssetJob.game_id == game.id)
            .order_by(models.AssetJob.created_at.desc())
        )
    )


def get_job(db: Session, game: models.Game, job_id: uuid.UUID) -> Optional[models.AssetJob]:
    return db.scalar(
        select(models.AssetJob).where(
            models.AssetJob.id == job_id, models.AssetJob.game_id == game.id
        )
    )


def set_job_status(
    db: Session,
    job: models.AssetJob,
    status: str,
    result: Optional[dict] = None,
    error: Optional[str] = None,
) -> models.AssetJob:
    job.status = status
    if result is not None:
        job.result = result
    if error is not None:
        job.error = error
    db.commit()
    db.refresh(job)
    return job

"""CycleZero game-authoring API. Mounted at ``/cyclezero`` (see app/main.py).

Surface:
  POST   /cyclezero/games                         create a game (→ URI)
  GET    /cyclezero/games                          list games
  GET    /cyclezero/games/{slug}                   fetch a game
  PATCH  /cyclezero/games/{slug}                   update a game
  DELETE /cyclezero/games/{slug}                   delete a game (cascades)

  POST   /cyclezero/games/{slug}/entities          create an entity (any layer)
  GET    /cyclezero/games/{slug}/entities[?layer=] list entities
  GET    /cyclezero/games/{slug}/entities/{key}    fetch an entity
  PATCH  /cyclezero/games/{slug}/entities/{key}    update an entity
  DELETE /cyclezero/games/{slug}/entities/{key}    delete an entity

  POST   /cyclezero/games/{slug}/relations         link two entities
  GET    /cyclezero/games/{slug}/relations         list relations
  DELETE /cyclezero/games/{slug}/relations/{id}    remove a relation

  POST   /cyclezero/games/{slug}/entities/{key}/generate   trigger asset gen
  GET    /cyclezero/games/{slug}/jobs               list asset jobs
  GET    /cyclezero/games/{slug}/jobs/{id}          fetch a job

  GET    /cyclezero/games/{slug}/contract           build the scene contract (P6)
  POST   /cyclezero/games/{slug}/match              coverage vs a contract (P7)
"""
from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import contract as contract_builder
from . import generation, matching, schemas, service
from .db import get_db

router = APIRouter()


def _game_uri(slug: str) -> str:
    return f"/cyclezero/games/{slug}"


def _game_out(game) -> schemas.GameOut:
    return schemas.GameOut(
        id=game.id,
        slug=game.slug,
        title=game.title,
        owner_id=game.owner_id,
        status=game.status,
        data=game.data,
        uri=_game_uri(game.slug),
        created_at=game.created_at,
        updated_at=game.updated_at,
    )


def _require_game(db: Session, slug: str):
    game = service.get_game(db, slug)
    if game is None:
        raise HTTPException(404, f"game not found: {slug}")
    return game


def _require_entity(db: Session, game, key: str):
    entity = service.get_entity(db, game, key)
    if entity is None:
        raise HTTPException(404, f"entity not found: {key}")
    return entity


# ── games ────────────────────────────────────────────────────────────────────
@router.post("/games", response_model=schemas.GameOut, status_code=201)
def create_game(body: schemas.GameCreate, db: Session = Depends(get_db)):
    try:
        game = service.create_game(db, body)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "a game with that slug already exists")
    return _game_out(game)


@router.get("/games", response_model=List[schemas.GameOut])
def list_games(db: Session = Depends(get_db)):
    return [_game_out(g) for g in service.list_games(db)]


@router.get("/games/{slug}", response_model=schemas.GameOut)
def get_game(slug: str, db: Session = Depends(get_db)):
    return _game_out(_require_game(db, slug))


@router.patch("/games/{slug}", response_model=schemas.GameOut)
def update_game(slug: str, body: schemas.GameUpdate, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    return _game_out(service.update_game(db, game, body))


@router.delete("/games/{slug}", status_code=204)
def delete_game(slug: str, db: Session = Depends(get_db)):
    service.delete_game(db, _require_game(db, slug))


# ── entities ─────────────────────────────────────────────────────────────────
@router.post("/games/{slug}/entities", response_model=schemas.EntityOut, status_code=201)
def create_entity(slug: str, body: schemas.EntityCreate, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    try:
        return service.create_entity(db, game, body)
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, f"entity key already exists in layer '{body.layer}'")


@router.get("/games/{slug}/entities", response_model=List[schemas.EntityOut])
def list_entities(
    slug: str, layer: Optional[str] = Query(None), db: Session = Depends(get_db)
):
    return service.list_entities(db, _require_game(db, slug), layer)


@router.get("/games/{slug}/entities/{key}", response_model=schemas.EntityOut)
def get_entity(slug: str, key: str, db: Session = Depends(get_db)):
    return _require_entity(db, _require_game(db, slug), key)


@router.patch("/games/{slug}/entities/{key}", response_model=schemas.EntityOut)
def update_entity(
    slug: str, key: str, body: schemas.EntityUpdate, db: Session = Depends(get_db)
):
    game = _require_game(db, slug)
    return service.update_entity(db, _require_entity(db, game, key), body)


@router.delete("/games/{slug}/entities/{key}", status_code=204)
def delete_entity(slug: str, key: str, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    service.delete_entity(db, _require_entity(db, game, key))


# ── relations ────────────────────────────────────────────────────────────────
@router.post("/games/{slug}/relations", response_model=schemas.RelationOut, status_code=201)
def create_relation(slug: str, body: schemas.RelationCreate, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    try:
        return service.create_relation(db, game, body)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except IntegrityError:
        db.rollback()
        raise HTTPException(409, "that relation already exists")


@router.get("/games/{slug}/relations", response_model=List[schemas.RelationOut])
def list_relations(slug: str, db: Session = Depends(get_db)):
    return service.list_relations(db, _require_game(db, slug))


@router.delete("/games/{slug}/relations/{rel_id}", status_code=204)
def delete_relation(slug: str, rel_id: uuid.UUID, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    if not service.delete_relation(db, game, rel_id):
        raise HTTPException(404, "relation not found")


# ── asset generation ─────────────────────────────────────────────────────────
@router.post(
    "/games/{slug}/entities/{key}/generate",
    response_model=schemas.JobOut,
    status_code=202,
)
def generate(slug: str, key: str, body: schemas.JobCreate, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    entity = _require_entity(db, game, key)
    job = service.create_job(db, game, entity, body)
    outcome = generation.submit(job, entity)
    job = service.set_job_status(db, job, "queued", result={**job.result, **outcome})
    return job


@router.get("/games/{slug}/jobs", response_model=List[schemas.JobOut])
def list_jobs(slug: str, db: Session = Depends(get_db)):
    return service.list_jobs(db, _require_game(db, slug))


@router.get("/games/{slug}/jobs/{job_id}", response_model=schemas.JobOut)
def get_job(slug: str, job_id: uuid.UUID, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    job = service.get_job(db, game, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


# ── contract (P6) + matching (P7) ────────────────────────────────────────────
@router.get("/games/{slug}/contract")
def get_contract(slug: str, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    entities = [
        {"layer": e.layer, "key": e.key, "name": e.name, "data": e.data}
        for e in service.list_entities(db, game)
    ]
    return contract_builder.build_contract(
        {"slug": game.slug, "title": game.title}, entities
    )


@router.post("/games/{slug}/match")
def match_contract(slug: str, body: schemas.MatchRequest, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    entities = [
        {"layer": e.layer, "key": e.key, "name": e.name, "data": e.data}
        for e in service.list_entities(db, game)
    ]
    return matching.match(entities, body.contract)

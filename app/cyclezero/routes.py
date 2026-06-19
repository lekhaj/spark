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

  GET    /cyclezero/games/{slug}/contract           build the scene contract (P6/S6)
  POST   /cyclezero/games/{slug}/match              coverage vs a contract (P7)

  GET    /cyclezero/metamodel/layers               list layer→schema_key (S0)
  POST   /cyclezero/metamodel/layers               upsert a layer
  GET    /cyclezero/metamodel/relation-types       list relation contracts (S0)
  POST   /cyclezero/metamodel/relation-types       upsert a relation type

  GET    /cyclezero/games/{slug}/graph/validate    structural validation (S2)
  GET    /cyclezero/games/{slug}/graph/order       generation/spec order (S3)
  GET    /cyclezero/games/{slug}/graph/ripple?entity=  downstream of a node (S3)
  GET    /cyclezero/games/{slug}/entities/{key}/packet  graph-aware LLM packet (S5)

  POST   /cyclezero/games/{slug}/releases          cut a release snapshot (S7)
  GET    /cyclezero/games/{slug}/releases          list releases (newest first)
  GET    /cyclezero/games/{slug}/releases/{version}  fetch a release manifest
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from . import contract as contract_builder
from . import capability_store, compile_agent, compile_tools, propose_agent, validate_agent
from . import generation, graph, matching, metamodel, outcome, schemas, service
from .db import get_db

router = APIRouter()


def _mongo():
    """Authoring Mongo DB (spec schemas/runs + the metamodel). Overridable in
    tests via monkeypatch, mirroring schema_routes._db."""
    from worker.lib import manual_gen_schema as mgs

    return mgs.get_db()


def _load_metamodel():
    return metamodel.load_metamodel(_mongo())


def _entity_dicts(db: Session, game) -> List[dict]:
    """Entities as plain dicts for the pure graph/contract/matching helpers."""
    return [
        {
            "layer": e.layer,
            "key": e.key,
            "name": e.name,
            "data": e.data,
            "spec_stage": e.spec_stage,
            "accepted_spec_run_id": e.accepted_spec_run_id,
        }
        for e in service.list_entities(db, game)
    ]


def _relation_dicts(db: Session, game) -> List[dict]:
    """Relations as ``{src, dst, kind}`` keyed by entity *key* for the graph algos."""
    id2key = {e.id: e.key for e in service.list_entities(db, game)}
    return [
        {"src": id2key.get(r.src_entity), "dst": id2key.get(r.dst_entity), "kind": r.kind}
        for r in service.list_relations(db, game)
    ]


def _entity_data_dicts(db: Session, game) -> List[dict]:
    """Entities with inline ``data`` for the outcome model (factors carry kind/
    min/max/default in ``data``)."""
    return [
        {
            "layer": e.layer,
            "key": e.key,
            "name": e.name,
            "data": e.data,
            "accepted_spec_run_id": e.accepted_spec_run_id,
        }
        for e in service.list_entities(db, game)
    ]


def _relation_data_dicts(db: Session, game) -> List[dict]:
    """Relations keyed by entity *key*, **including edge ``data``** — needed by the
    outcome model (AFFECTS deltas) where ``_relation_dicts`` drops it."""
    id2key = {e.id: e.key for e in service.list_entities(db, game)}
    return [
        {
            "src": id2key.get(r.src_entity),
            "dst": id2key.get(r.dst_entity),
            "kind": r.kind,
            "data": r.data,
        }
        for r in service.list_relations(db, game)
    ]


def _entity_dict(e) -> dict:
    """Serialize an Entity ORM row to the same shape as ``schemas.EntityOut`` for
    endpoints that return a custom (non-``response_model``) structure."""
    return {
        "id": str(e.id),
        "game_id": str(e.game_id),
        "layer": e.layer,
        "key": e.key,
        "name": e.name,
        "data": e.data,
        "spec_stage": e.spec_stage,
        "accepted_spec_run_id": e.accepted_spec_run_id,
        "created_at": e.created_at,
        "updated_at": e.updated_at,
    }


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
    # Default the node's spec_stage from the layer metamodel (best-effort: if the
    # metamodel/Mongo is unavailable, the node is still created without a stage).
    stage = None
    try:
        layer_def = metamodel.get_layer(_mongo(), body.layer)
        stage = (layer_def or {}).get("schema_key")
    except Exception:  # noqa: BLE001 — never block authoring on the metamodel
        stage = None
    try:
        return service.create_entity(db, game, body, spec_stage=stage)
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
    # Validate the edge against the relation contract when the metamodel is
    # reachable; degrade to a raw edge create if Mongo is down.
    try:
        mm = _load_metamodel()
    except Exception:  # noqa: BLE001
        mm = None
    try:
        return service.create_relation(db, game, body, metamodel=mm)
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


# ── contract (P6) + matching (P7), now sourced from accepted specs (S6) ───────
def _contract_entity_dicts(db: Session, game) -> List[dict]:
    """Entities for the contract builder, with ``data`` taken from each node's
    accepted spec body when one exists (falling back to inline ``entity.data``).
    This is the S6 shift: the contract compiles from *validated* content."""
    try:
        mongo = _mongo()
    except Exception:  # noqa: BLE001 — no Mongo → fall back to inline data
        mongo = None
    out: List[dict] = []
    for e in service.list_entities(db, game):
        body = None
        if mongo is not None and e.accepted_spec_run_id:
            try:
                body = service.resolve_body(mongo, e)
            except Exception:  # noqa: BLE001
                body = None
        out.append(
            {"layer": e.layer, "key": e.key, "name": e.name, "data": body or e.data}
        )
    return out


@router.get("/games/{slug}/contract")
def get_contract(slug: str, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    return contract_builder.build_contract(
        {"slug": game.slug, "title": game.title}, _contract_entity_dicts(db, game)
    )


@router.post("/games/{slug}/match")
def match_contract(slug: str, body: schemas.MatchRequest, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    return matching.match(_contract_entity_dicts(db, game), body.contract)


# ── compile / churn-out (U4) ──────────────────────────────────────────────────
def _schemas_by_layer(mongo, mm, layers: List[str]) -> Dict[str, Any]:
    """Active JSON Schema for each layer in scope, keyed by layer name (the compile
    tools look up by layer and by ``{layer}_spec``)."""
    out: Dict[str, Any] = {}
    if mongo is None:
        return out
    for layer in layers:
        schema_key = (mm["layers"].get(layer, {}) or {}).get("schema_key") or f"{layer}_spec"
        doc = mongo["spec_schemas"].find_one({"schema_key": schema_key, "active": True})
        if doc:
            out[layer] = doc.get("json_schema", {})
    return out


_CAP_COLLECTION = "cz_capabilities"
_BUILDLOG_COLLECTION = "cz_build_log"


def _load_ledger(mongo, engine: str):
    """Living capability ledger for an engine (per engine — the runtime repo is
    shared across games). None when Mongo is unavailable."""
    if mongo is None:
        return None
    return mongo[_CAP_COLLECTION].find_one({"engine": engine}, {"_id": 0})


def _save_ledger(mongo, ledger: dict) -> None:
    mongo[_CAP_COLLECTION].update_one(
        {"engine": ledger["engine"]}, {"$set": ledger}, upsert=True
    )


@router.get("/games/{slug}/capabilities")
def get_capabilities(slug: str, engine: str = Query("babylon"), db: Session = Depends(get_db)):
    """The engine Capability Registry — base seed + the **living ledger** of what
    Claude Code has built — so the compiler knows what NOT to re-implement."""
    _require_game(db, slug)
    try:
        ledger = _load_ledger(_mongo(), engine)
    except Exception:  # noqa: BLE001
        ledger = None
    return compile_tools.get_capability_registry(engine, ledger=ledger)


@router.get("/games/{slug}/capabilities/log")
def get_build_log(slug: str, db: Session = Depends(get_db)):
    """Per-game build log — the status sheet of compiles + what Claude Code reported."""
    game = _require_game(db, slug)
    try:
        mongo = _mongo()
        rows = list(mongo[_BUILDLOG_COLLECTION].find({"slug": game.slug}, {"_id": 0}).sort("at", -1).limit(50))
    except Exception:  # noqa: BLE001
        rows = []
    return {"slug": game.slug, "entries": rows}


@router.post("/games/{slug}/capabilities/parse")
def parse_done_note(slug: str, body: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Structure a freeform Claude Code done-note into {systems, consumes, repo,
    commit, files, summary} for review. Deterministic extraction first; an optional
    Bedrock (AWS-credit) pass enriches it. Writes nothing — the creator confirms then
    calls /ingest."""
    _require_game(db, slug)
    engine = body.get("engine", "babylon")
    note = body.get("note", "")
    base = compile_tools.get_base_registry(engine)
    try:
        mm = _load_metamodel()
        known_layers = list(mm.get("layers", {}).keys())
    except Exception:  # noqa: BLE001
        known_layers = []
    suggestion = capability_store.extract_from_note(note, known_layers, base.get("systems", []))
    return {"suggestion": suggestion, "source": "deterministic"}


@router.post("/games/{slug}/validate")
def validate_game(slug: str, body: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    """Validate the authored game (U5). Deterministic static gate first (graph/contract/
    outcome); the optional semantic check (Bedrock Tier-B, AWS credits) only runs when
    both ``acceptance`` and ``done_note`` are supplied. Returns verdict + Fix Packet."""
    game = _require_game(db, slug)
    try:
        mm = _load_metamodel()
    except Exception:  # noqa: BLE001
        mm = {"layers": {}, "relation_types": {}}
    entities = _entity_data_dicts(db, game)
    relations = _relation_data_dicts(db, game)

    acceptance = body.get("acceptance")
    done_note = body.get("done_note")
    semantic_fn = None
    if acceptance and done_note:
        try:
            semantic_fn = validate_agent.make_bedrock_semantic()
        except Exception:  # noqa: BLE001 — no LLM → static only
            semantic_fn = None

    return validate_agent.validate(
        entities, relations, mm,
        game={"slug": game.slug, "title": game.title},
        acceptance=acceptance, done_note=done_note, semantic_fn=semantic_fn,
    )


@router.post("/games/{slug}/capabilities/ingest")
def ingest_report(slug: str, body: Dict[str, Any] = Body(...), db: Session = Depends(get_db)):
    """Record what Claude Code built and **merge it into the capability ledger** so the
    next compile reflects the new engine state. Body: {engine?, systems?, consumes?,
    repo?, commit?, files?, summary?}. Returns the updated merged registry."""
    game = _require_game(db, slug)
    engine = body.get("engine", "babylon")
    try:
        mongo = _mongo()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"capability store unavailable: {exc}")
    ledger = _load_ledger(mongo, engine) or capability_store.empty_ledger(engine)
    ledger = capability_store.apply_report(ledger, body)
    _save_ledger(mongo, ledger)
    # status sheet entry
    mongo[_BUILDLOG_COLLECTION].insert_one({
        "slug": game.slug, "engine": engine,
        "at": ledger["entries"][-1]["at"],
        "report": ledger["entries"][-1],
    })
    return {
        "ok": True,
        "registry": compile_tools.get_capability_registry(engine, ledger=ledger),
        "entry": ledger["entries"][-1],
    }


@router.post("/games/{slug}/compile")
def compile_game(slug: str, body: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)):
    """Churn out a code-gen prompt for a scope (whole game / scene / entities).

    Body: {scope?, target?, output?, acceptance?, stitch?}. ``stitch`` (default true)
    enables the Tier-C LLM implementation-plan seam when Bedrock is configured; the
    deterministic skeleton always renders even without it (tools-first)."""
    game = _require_game(db, slug)
    try:
        mongo = _mongo()
        mm = metamodel.load_metamodel(mongo)
    except Exception:  # noqa: BLE001 — no Mongo → empty schemas/metamodel, still compiles
        mongo, mm = None, {"layers": {}, "relation_types": {}}

    entities = _entity_data_dicts(db, game)
    relations = _relation_data_dicts(db, game)
    scope = body.get("scope") or {"kind": "game"}

    # Resolve active schemas only for the layers actually in scope.
    bundle = compile_tools.gather_scope(entities, relations, scope)
    schemas_by_layer = _schemas_by_layer(mongo, mm, bundle["layers"])

    target = body.get("target", "babylon")
    ledger = _load_ledger(mongo, target)  # living capability state → shrinks gaps

    stitch_fn = None
    if body.get("stitch", True):
        try:
            stitch_fn = compile_agent.make_bedrock_stitcher()
        except Exception:  # noqa: BLE001 — no LLM configured → deterministic only
            stitch_fn = None

    return compile_agent.compile_prompt(
        entities, relations, mm, schemas_by_layer,
        scope=scope,
        target=target,
        output=body.get("output", "build_packet"),
        acceptance=body.get("acceptance"),
        ledger=ledger,
        stitch_fn=stitch_fn,
    )


# ── system proposal + bulk install (U6) ───────────────────────────────────────
@router.post("/metamodel/propose-systems")
def propose_systems(body: Dict[str, Any] = Body(...)):
    """Describe a game → propose layers + relations (Bedrock Tier-C, AWS credits),
    deterministically linted. Body: {description, feedback?, prior?}. Global (operates
    on the shared metamodel); writes nothing — the creator reviews then installs."""
    description = (body.get("description") or "").strip()
    if not description:
        raise HTTPException(422, "description is required")
    try:
        mm = _load_metamodel()
        known = list(mm.get("layers", {}).keys())
    except Exception:  # noqa: BLE001
        known = []
    try:
        propose_fn = propose_agent.make_bedrock_proposer(known)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"proposer unavailable (no LLM configured): {exc}")
    return propose_agent.propose_systems(
        description, known,
        feedback=body.get("feedback"), prior=body.get("prior"), propose_fn=propose_fn,
    )


@router.post("/metamodel/install")
def install_systems(body: Dict[str, Any] = Body(...)):
    """Bulk-create proposed systems: for each layer create+activate a schema and
    register the layer; for each relation upsert the relation type. One request →
    avoids many slow round-trips on the cold single-worker box. Idempotent-ish
    (re-installing a layer just adds a schema version)."""
    from datetime import datetime, timezone
    mongo = _mongo()
    col = mongo["spec_schemas"]
    created_layers: List[str] = []
    created_relations: List[str] = []

    for l in body.get("layers") or []:
        layer = l.get("layer")
        schema = l.get("schema") or {"type": "object", "properties": {}}
        if not layer:
            continue
        lint = compile_tools.lint_schema(schema)
        if not lint["ok"]:
            raise HTTPException(422, f"layer '{layer}' schema invalid: {'; '.join(lint['errors'])}")
        schema_key = f"{layer}_spec"
        latest = col.find_one({"schema_key": schema_key}, sort=[("version", -1)])
        version = (latest["version"] + 1) if latest else 1
        col.update_many({"schema_key": schema_key, "active": True}, {"$set": {"active": False}})
        col.insert_one({
            "schema_key": schema_key, "version": version, "title": l.get("title") or layer,
            "engine_bound": False, "json_schema": schema, "changelog": "proposed via U6",
            "created_at": datetime.now(timezone.utc), "active": True,
        })
        metamodel.upsert_layer(mongo, layer, schema_key, l.get("title") or layer)
        created_layers.append(layer)

    for r in body.get("relations") or []:
        if not r.get("kind"):
            continue
        try:
            metamodel.upsert_relation_type(mongo, r)
            created_relations.append(r["kind"])
        except ValueError as exc:
            raise HTTPException(422, f"relation '{r.get('kind')}' invalid: {exc}")

    return {"ok": True, "created_layers": created_layers, "created_relations": created_relations}


# ── metamodel (S0) ───────────────────────────────────────────────────────────
@router.get("/metamodel/layers")
def get_layers():
    return metamodel.list_layers(_mongo())


@router.post("/metamodel/layers")
def post_layer(body: Dict[str, Any] = Body(...)):
    layer = body.get("layer")
    schema_key = body.get("schema_key")
    if not layer or not schema_key:
        raise HTTPException(422, "layer and schema_key are required")
    return metamodel.upsert_layer(_mongo(), layer, schema_key, body.get("title"))


@router.get("/metamodel/relation-types")
def get_relation_types():
    return metamodel.list_relation_types(_mongo())


@router.post("/metamodel/relation-types")
def post_relation_type(body: Dict[str, Any] = Body(...)):
    try:
        return metamodel.upsert_relation_type(_mongo(), body)
    except ValueError as exc:
        raise HTTPException(422, str(exc))


# ── graph (S2/S3) ─────────────────────────────────────────────────────────────
@router.get("/games/{slug}/graph/validate")
def graph_validate(slug: str, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    return graph.validate_graph(
        _entity_dicts(db, game), _relation_dicts(db, game), _load_metamodel()
    )


@router.get("/games/{slug}/graph/order")
def graph_order(slug: str, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    return graph.topo_order(
        _entity_dicts(db, game), _relation_dicts(db, game), _load_metamodel()
    )


@router.get("/games/{slug}/graph/ripple")
def graph_ripple(slug: str, entity: str = Query(...), db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    _require_entity(db, game, entity)  # 404 if the key is unknown
    return graph.ripple(
        entity, _entity_dicts(db, game), _relation_dicts(db, game), _load_metamodel()
    )


# ── graph-aware packet (S5) ───────────────────────────────────────────────────
@router.get("/games/{slug}/entities/{key}/packet")
def entity_packet(slug: str, key: str, db: Session = Depends(get_db)):
    """Assemble the Generation Packet input for a node: its bound schema plus its
    *graph neighborhood* (upstream accepted specs + outgoing relation context) so
    an external LLM / the refiner sees the connected subgraph, not an island.
    Shape mirrors spark_studio packet.ts ``PacketInput``."""
    game = _require_game(db, slug)
    entity = _require_entity(db, game, key)
    mongo = _mongo()
    mm = metamodel.load_metamodel(mongo)
    rtypes = mm["relation_types"]

    id2entity = {e.id: e for e in service.list_entities(db, game)}
    key2entity = {e.key: e for e in id2entity.values()}
    relations = service.list_relations(db, game)

    # Outgoing edges from this node → relation context + dependency upstreams.
    references: List[dict] = []
    upstream: List[dict] = []
    for r in relations:
        if r.src_entity != entity.id:
            continue
        dst = id2entity.get(r.dst_entity)
        if dst is None:
            continue
        references.append({"kind": r.kind, "dst": dst.key, "dst_layer": dst.layer})
        if rtypes.get(r.kind, {}).get("dependency"):
            upstream.append(
                {
                    "key": dst.key,
                    "layer": dst.layer,
                    "kind": r.kind,
                    "accepted_body": service.resolve_body(mongo, dst),
                }
            )

    stage = entity.spec_stage or (mm["layers"].get(entity.layer, {}) or {}).get("schema_key")
    schema_doc = None
    if stage:
        schema_doc = mongo["spec_schemas"].find_one({"schema_key": stage, "active": True})

    return {
        "projectId": game.slug,
        "entityId": entity.key,
        "stage": stage,
        "schemaTitle": (schema_doc or {}).get("title", stage),
        "schemaVersion": (schema_doc or {}).get("version", 0),
        "jsonSchema": (schema_doc or {}).get("json_schema", {}),
        "inputSpec": {
            "node": {"key": entity.key, "layer": entity.layer, "name": entity.name},
            "upstream": upstream,
            "references": references,
        },
        "userIntent": "",
    }


# ── scene hub (X2) + outcome model (X5) ───────────────────────────────────────
@router.get("/games/{slug}/scenes/{key}/hub")
def scene_hub(slug: str, key: str, db: Session = Depends(get_db)):
    """Everything a scene CONTAINS, grouped by layer, plus the **inherited
    globals** (systems scoped ``global`` or not contained by any scene, and all
    factors — factors are world-level). Powers the Scene-hub dashboard."""
    game = _require_game(db, slug)
    scene = _require_entity(db, game, key)
    if scene.layer != "scene":
        raise HTTPException(400, f"entity '{key}' is not a scene (layer={scene.layer})")

    entities = service.list_entities(db, game)
    relations = service.list_relations(db, game)
    id2e = {e.id: e for e in entities}

    members: Dict[str, List[dict]] = defaultdict(list)
    contained_anywhere: set = set()
    for r in relations:
        if r.kind != "CONTAINS":
            continue
        contained_anywhere.add(r.dst_entity)
        if r.src_entity == scene.id:
            dst = id2e.get(r.dst_entity)
            if dst is not None:
                members[dst.layer].append(_entity_dict(dst))

    inherited_systems: List[dict] = []
    inherited_factors: List[dict] = []
    for e in entities:
        if e.layer == "system":
            scope = (e.data or {}).get("scope")
            if scope == "global" or e.id not in contained_anywhere:
                inherited_systems.append(_entity_dict(e))
        elif e.layer == "factor":
            inherited_factors.append(_entity_dict(e))

    return {
        "scene": _entity_dict(scene),
        "members": dict(members),
        "inherited": {"systems": inherited_systems, "factors": inherited_factors},
    }


@router.get("/games/{slug}/factors/{key}/contributors")
def factor_contributors(slug: str, key: str, db: Session = Depends(get_db)):
    """Ranked AFFECTS-in edges for a factor (who pushes it, by magnitude)."""
    game = _require_game(db, slug)
    entity = _require_entity(db, game, key)
    if entity.layer != "factor":
        raise HTTPException(400, f"entity '{key}' is not a factor (layer={entity.layer})")
    return outcome.contributors(
        key, _entity_data_dicts(db, game), _relation_data_dicts(db, game)
    )


@router.post("/games/{slug}/outcome/project")
def outcome_project(
    slug: str, body: Dict[str, Any] = Body(default={}), db: Session = Depends(get_db)
):
    """Project the factor end-state from a baseline (optionally with what-if
    ``overrides``), then run the outcome resolver → ending + per-rule trace. The
    UI's live "if the game ended now" panel."""
    game = _require_game(db, slug)
    entities = _entity_data_dicts(db, game)
    relations = _relation_data_dicts(db, game)

    state = outcome.project(entities, relations)
    overrides = (body or {}).get("overrides") or {}
    if isinstance(overrides, dict):
        state.update(overrides)

    outcome_nodes = [e for e in entities if e["layer"] == "outcome"]
    od = (outcome_nodes[0].get("data") or {}) if outcome_nodes else {}
    res = outcome.resolve(state, od.get("rules") or [], od.get("default_ending"))

    return {"factor_state": state, **res}


# ── releases (S7) ─────────────────────────────────────────────────────────────
def _spec_version(mongo, run_id: Optional[str]) -> Optional[str]:
    """Resolve a spec run's ``major.minor`` for the release manifest (best-effort)."""
    if not run_id or mongo is None:
        return None
    try:
        doc = mongo["spec_gen_runs"].find_one({"run_id": run_id})
    except Exception:  # noqa: BLE001
        return None
    if not doc:
        return None
    return f"{doc.get('major')}.{doc.get('minor')}"


def _build_manifest(db: Session, game) -> Dict[str, Any]:
    """Freeze the game's current authored state: entities (with the spec version
    each points at), relations, the compiled contract, and the validation report."""
    try:
        mongo = _mongo()
    except Exception:  # noqa: BLE001
        mongo = None
    entities = service.list_entities(db, game)
    manifest_entities = [
        {
            "key": e.key,
            "layer": e.layer,
            "name": e.name,
            "spec_stage": e.spec_stage,
            "accepted_spec_run_id": e.accepted_spec_run_id,
            "spec_version": _spec_version(mongo, e.accepted_spec_run_id),
        }
        for e in entities
    ]
    relations = _relation_dicts(db, game)
    contract = contract_builder.build_contract(
        {"slug": game.slug, "title": game.title}, _contract_entity_dicts(db, game)
    )
    try:
        validation = graph.validate_graph(_entity_dicts(db, game), relations, _load_metamodel())
    except Exception:  # noqa: BLE001 — release should still cut without the metamodel
        validation = {"ok": None, "complete": False}
    return {
        "entities": manifest_entities,
        "relations": relations,
        "contract": contract,
        "validation": validation,
    }


@router.post("/games/{slug}/releases", response_model=schemas.ReleaseOut, status_code=201)
def create_release(slug: str, body: schemas.ReleaseCreate, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    manifest = _build_manifest(db, game)
    complete = bool(manifest.get("validation", {}).get("complete"))
    return service.create_release(db, game, manifest, complete, body.label, body.notes)


@router.get("/games/{slug}/releases", response_model=List[schemas.ReleaseSummary])
def list_releases(slug: str, db: Session = Depends(get_db)):
    return service.list_releases(db, _require_game(db, slug))


@router.get("/games/{slug}/releases/{version}", response_model=schemas.ReleaseOut)
def get_release(slug: str, version: int, db: Session = Depends(get_db)):
    game = _require_game(db, slug)
    rel = service.get_release(db, game, version)
    if rel is None:
        raise HTTPException(404, f"release v{version} not found")
    return rel

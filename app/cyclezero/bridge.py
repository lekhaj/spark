"""S1/S4 bridge: react to spec-gen accepts on the Mongo side by updating the
Postgres graph (stamp the node's accepted spec) and, when the accept supersedes
a prior version, compute the downstream ripple and surface it on the journey feed.

Everything here is **best-effort and never raises** — a spec accept must succeed
even if the cyclezero Postgres DB is unconfigured (e.g. in spec-gen unit tests)
or the node doesn't exist in the graph yet.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

log = logging.getLogger("cyclezero.bridge")


def stamp_accepted(project_id: str, entity_id: str, run_id: str) -> Optional[bool]:
    """Point the matching graph node (game.slug=project_id, entity.key=entity_id)
    at the just-accepted spec run. Returns True on stamp, False if no node, None
    if the graph DB is unavailable."""
    try:
        from .db import get_session_factory
        from . import service
        from .models import Game
        from sqlalchemy import select
    except Exception:  # noqa: BLE001
        return None
    try:
        session = get_session_factory()()
    except Exception as exc:  # noqa: BLE001 — CYCLEZERO_DATABASE_URL unset, etc.
        log.debug("cyclezero bridge: no graph DB (%s)", exc)
        return None
    try:
        game = session.scalar(select(Game).where(Game.slug == project_id))
        if game is None:
            return False
        entity = service.get_entity_by_key(session, game.id, entity_id)
        if entity is None:
            return False
        service.set_accepted_spec(session, entity, run_id)
        return True
    except Exception:  # noqa: BLE001
        log.exception("cyclezero bridge: stamp failed for %s/%s", project_id, entity_id)
        return None
    finally:
        session.close()


def compute_ripple(project_id: str, entity_id: str) -> Optional[Dict[str, Any]]:
    """Return the graph ripple report for a changed node (downstream dependents),
    or None if the graph DB is unavailable / the game is unknown."""
    try:
        from .db import get_session_factory
        from . import graph, metamodel, service
        from .models import Game
        from .routes import _mongo, _entity_dicts, _relation_dicts
        from sqlalchemy import select
    except Exception:  # noqa: BLE001
        return None
    try:
        session = get_session_factory()()
    except Exception:  # noqa: BLE001
        return None
    try:
        game = session.scalar(select(Game).where(Game.slug == project_id))
        if game is None:
            return None
        mm = metamodel.load_metamodel(_mongo())
        return graph.ripple(entity_id, _entity_dicts(session, game), _relation_dicts(session, game), mm)
    except Exception:  # noqa: BLE001
        log.exception("cyclezero bridge: ripple failed for %s/%s", project_id, entity_id)
        return None
    finally:
        session.close()


def ripple_to_impact_items(
    db_mongo, project_id: str, entity_id: str, downstream: List[str]
) -> Optional[str]:
    """S4: turn a computed downstream set into a journey + impact feed on the
    existing Mongo journeys surface, so a spec change shows *computed* ripple
    instead of the refiner's guess. Returns the journey_id, or None if nothing to
    surface. Best-effort."""
    if not downstream:
        return None
    try:
        import uuid
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc)
        journey_id = uuid.uuid4().hex
        journey = {
            "journey_id": journey_id,
            "project_id": project_id,
            "kind": "data",
            "title": f"Re-validate downstream of {entity_id}",
            "user_intent": "",
            "rail": [
                {"label": "Change accepted", "sub": entity_id, "state": "done"},
                {"label": "Review ripple", "sub": f"{len(downstream)} node(s)", "state": "active"},
            ],
            "status": "active",
            "created_at": now,
            "completed_at": None,
            "origin": "graph_ripple",
        }
        db_mongo["journeys"].insert_one(journey)
        items = []
        for dep in downstream:
            items.append({
                "item_id": uuid.uuid4().hex,
                "journey_id": journey_id,
                "project_id": project_id,
                "impact": "ripple",
                "icon": "↻",
                "title": f"{dep} depends on {entity_id}",
                "body": f"{entity_id} changed; re-validate {dep}.",
                "target": {"entity_id": dep, "stage": ""},
                "workspace": "revalidate",
                "suggested_intent": "",
                "status": "open",
                "resolution_note": None,
                "resolved_at": None,
                "origin": "graph_ripple",
            })
        if items:
            db_mongo["impact_items"].insert_many(items)
        return journey_id
    except Exception:  # noqa: BLE001
        log.exception("cyclezero bridge: ripple_to_impact_items failed")
        return None


def on_spec_accepted(db_mongo, project_id: str, entity_id: str, run_id: str, superseded: bool) -> None:
    """Single entry point the spec-gen accept handler calls. Stamps the node and,
    when this accept *supersedes* a prior version (i.e. a real change), surfaces
    the computed ripple on the journey feed."""
    stamp_accepted(project_id, entity_id, run_id)
    if not superseded:
        return
    rip = compute_ripple(project_id, entity_id)
    if rip and rip.get("downstream"):
        ripple_to_impact_items(db_mongo, project_id, entity_id, rip["downstream"])

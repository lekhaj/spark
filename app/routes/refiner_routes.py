"""
refiner_routes.py — refiner chat proxy (CycleZero U02, supersedes T07)
======================================================================

Keeps LLM keys server-side. The studio composer posts the conversation +
project context; the reply either converses or ends with ONE fenced
```journey_plan block the studio parses into a journey (U01).

Routes mounted at ``/refiner`` (set in app/main.py).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.lib import refiner_providers

log = logging.getLogger("refiner_routes")

router = APIRouter()

SYSTEM_TEMPLATE = """You are the Spark Studio refiner — the chat brain of a game-spec pipeline.
The user describes a change to their game ("add a stamina system", "make Kael
older"). Converse briefly until the change is clear, then emit EXACTLY ONE
fenced code block tagged journey_plan containing JSON of this shape:

```journey_plan
{{
  "kind": "system | asset | data | mixed",
  "title": "Stamina system",
  "rail": [{{"label": "...", "sub": "..."}}],
  "items": [{{"impact": "direct|high|review|ripple", "icon": "⚙",
             "title": "...", "body": "...",
             "target": {{"entity_id": "...", "stage": "..."}},
             "workspace": "spec_code|asset|story_review|revalidate|none",
             "suggested_intent": "..."}}]
}}
```

Rules:
- target.stage MUST be one of the schema keys listed below. Items pointing at
  unknown stages will be dropped.
- impact reflects how directly the target changes: direct = the thing being
  changed; high = must be revalidated; review = a human should look; ripple =
  informational only (workspace "none").
- Every non-ripple item names a workspace. Data-only stages (engine_bound
  false) never get spec_code engine steps — use story_review or revalidate.
- Keep the rail to 3-5 steps. Do not emit the block until the user's intent
  is clear; ask at most one clarifying question first.

Project: {project_id}
Available schema stages: {schemas}
Known entities: {entities}
"""


class ChatContext(BaseModel):
    project_id: str = "cyclezero"
    schemas: List[str] = []
    entities: List[str] = []


class ChatBody(BaseModel):
    provider_id: Optional[str] = None
    messages: List[Dict[str, str]] = Field(min_length=1)
    context: ChatContext = ChatContext()


@router.get("/providers")
def list_providers() -> List[Dict[str, str]]:
    return [
        {"id": p.id, "model": p.model}
        for p in refiner_providers.get_providers().values()
    ]


@router.post("/chat")
def chat(body: ChatBody) -> Dict[str, Any]:
    providers = refiner_providers.get_providers()
    if not providers:
        raise HTTPException(503, "no refiner providers configured — set ANTHROPIC_API_KEY or REFINER_OPENAI_BASE")
    if body.provider_id:
        provider = providers.get(body.provider_id)
        if not provider:
            raise HTTPException(404, f"unknown provider: {body.provider_id}")
    else:
        provider = next(iter(providers.values()))

    system = SYSTEM_TEMPLATE.format(
        project_id=body.context.project_id,
        schemas=json.dumps(body.context.schemas),
        entities=json.dumps(body.context.entities),
    )
    try:
        reply = provider.chat(system, body.messages)
    except HTTPException:
        raise
    except Exception as e:
        log.exception("refiner provider %s failed", provider.id)
        raise HTTPException(502, f"provider {provider.id} failed: {e}")
    return {"reply": reply, "provider_id": provider.id}

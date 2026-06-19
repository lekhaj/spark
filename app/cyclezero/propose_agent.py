"""U6 — the system-proposal agent: describe a game → propose layers + relations.

The entry point to the creator loop. The user types what they want in plain language;
the reasoning model (Tier-C, Bedrock = AWS credits) proposes a **system set** (layers
with JSON Schemas + relation types). Tools-first: the LLM proposes, deterministic
linting (``compile_tools.lint_schema``) gates it, and install is a deterministic bulk
write — the LLM never touches persistence. The user reviews, gives feedback, re-proposes,
then installs.

Pure except the injected ``propose_fn`` (the single LLM seam) — unit-testable offline.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional

from . import compile_tools


PROPOSE_SYSTEM = """You are the Spark Studio system designer. The user describes a game.
Propose the **systems** it needs as STRICT JSON (no prose outside the JSON):

{
  "layers": [
    {"layer": "snake_case_name", "title": "Human Title",
     "schema": {"type": "object",
                "properties": {"field": {"type": "integer|string|number|boolean|array",
                                          "enum": ["..."]?, "x-ref": "other_layer"?}},
                "required": ["..."]}}
  ],
  "relations": [
    {"kind": "UPPER_SNAKE", "src_layers": ["a"], "dst_layers": ["b"],
     "src_cardinality": "one|many", "dst_cardinality": "one|many"}
  ],
  "notes": "one short paragraph on the design"
}

Rules:
- Reuse these built-in layers when they fit (do NOT redefine them): {known_layers}.
- Only propose NEW layers the game actually needs. Keep schemas tight (3-7 fields).
- Use x-ref to point a field at another layer; enum for fixed choices.
- relations must reference layers that are either built-in or in your proposed list.
- Output ONLY the JSON object (optionally in a ```json fence). No commentary."""


def _parse_json(text: str) -> Dict[str, Any]:
    """Extract the JSON object from an LLM reply (tolerates a ```json fence/prose)."""
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = m.group(1) if m else None
    if blob is None:
        # fall back to the first {...last}
        i, j = text.find("{"), text.rfind("}")
        blob = text[i:j + 1] if i >= 0 and j > i else "{}"
    try:
        return json.loads(blob)
    except Exception:  # noqa: BLE001
        return {}


def validate_proposal(
    proposal: Dict[str, Any], known_layers: List[str]
) -> Dict[str, Any]:
    """Deterministically lint a proposal: each layer schema valid? relations reference
    real layers? Returns {layers, relations, notes, warnings} (never raises)."""
    warnings: List[str] = []
    layers_in = proposal.get("layers") or []
    relations_in = proposal.get("relations") or []

    clean_layers: List[dict] = []
    proposed_names = set(known_layers)
    for l in layers_in:
        name = l.get("layer")
        schema = l.get("schema") or {}
        if not name:
            warnings.append("a proposed layer has no name — skipped")
            continue
        lint = compile_tools.lint_schema(schema)
        if not lint["ok"]:
            warnings.append(f"layer '{name}' schema issues: {'; '.join(lint['errors'])}")
        clean_layers.append({"layer": name, "title": l.get("title") or name, "schema": schema})
        proposed_names.add(name)

    clean_relations: List[dict] = []
    for r in relations_in:
        kind = r.get("kind")
        if not kind:
            warnings.append("a proposed relation has no kind — skipped")
            continue
        for side in ("src_layers", "dst_layers"):
            for lyr in r.get(side) or []:
                if lyr not in proposed_names:
                    warnings.append(f"relation '{kind}' {side} references unknown layer '{lyr}'")
        clean_relations.append({
            "kind": kind,
            "src_layers": r.get("src_layers") or [],
            "dst_layers": r.get("dst_layers") or [],
            "src_cardinality": r.get("src_cardinality") or "many",
            "dst_cardinality": r.get("dst_cardinality") or "many",
        })

    return {
        "layers": clean_layers,
        "relations": clean_relations,
        "notes": proposal.get("notes") or "",
        "warnings": warnings,
    }


def propose_systems(
    description: str,
    known_layers: List[str],
    *,
    feedback: Optional[str] = None,
    prior: Optional[Dict[str, Any]] = None,
    propose_fn: Callable[[str], str],
) -> Dict[str, Any]:
    """Run one proposal round. ``propose_fn`` is the LLM seam (gets the full user
    message, returns raw text). Iteration: pass ``feedback`` + ``prior`` to refine."""
    parts = [f"Game description:\n{description}"]
    if prior:
        parts.append("Current proposed systems (JSON):\n" + json.dumps(prior, default=str))
    if feedback:
        parts.append(f"Revise per this feedback:\n{feedback}")
    raw = propose_fn("\n\n".join(parts))
    parsed = _parse_json(raw)
    return validate_proposal(parsed, known_layers)


def make_bedrock_proposer(known_layers: List[str]) -> Callable[[str], str]:
    """Tier-C (reasoning) proposer via Bedrock Converse (AWS credits)."""
    from app.lib import refiner_providers

    provider = refiner_providers.get_tier_provider("C")
    system = PROPOSE_SYSTEM.format(known_layers=", ".join(known_layers) or "(none)")

    def _propose(user: str) -> str:
        return provider.chat(system, [{"role": "user", "content": user}])

    return _propose

"""U4-BE1 — pure deterministic tools for the compile/validate agents.

Tools-first architecture (`spark_studio/docs/inference-architecture.md`): these
functions do the heavy lifting deterministically so the LLM only stitches the last
seam. Everything here is **pure logic over plain dicts** (no Mongo, no SQLAlchemy,
no LLM) — trivially unit-testable, exactly like ``graph.py``/``outcome.py``.

Shapes (resolved by the route before calling in), matching graph.py:
  entity   = {"layer", "key", "name", "data", "accepted_spec_run_id"?}
  relation = {"src", "dst", "kind"}          # src/dst are entity *keys*
  metamodel= {"layers": {layer: {...}}, "relation_types": {kind: {...}}}

The agent calls, in order: ``gather_scope`` → ``get_capability_registry`` →
``diff_capabilities`` → ``assemble_prompt_skeleton``; the LLM then writes only the
reasoning prose, and ``contract.build_contract`` / ``graph.validate_graph`` gate it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional


# ── Capability registry ───────────────────────────────────────────────────────
# What each engine runtime ALREADY provides, so the compiler can tell the code-gen
# LLM "do not re-implement these". Seeded from cyclezero/src/contract/types.ts (v1)
# + src/systems/*. ``consumes`` = layers the engine already turns into runtime
# behaviour via the Scene Contract; everything else needs new code.
_REGISTRY: Dict[str, Dict[str, Any]] = {
    "babylon": {
        "engine": "babylon",
        "contract_version": 1,
        "systems": [
            "Engine", "IsoCamera", "PickToMove", "SceneLoader", "applyColliders",
            "glbLoader", "havok", "quality", "BehaviorTracker", "Rng", "triggers",
        ],
        "contract_fields": [
            "camera", "quality", "environment", "player", "entities",
            "scatter", "triggers", "assets",
        ],
        # Layers the contract compiler already consumes (build_contract).
        "consumes": ["scene", "character", "collider", "prop", "trigger", "environment"],
    },
}


def get_capability_registry(engine: str = "babylon") -> Dict[str, Any]:
    """Return what the named engine runtime already provides. Unknown engine →
    an empty/generic registry (everything is a gap), never an error."""
    reg = _REGISTRY.get(engine)
    if reg is None:
        return {"engine": engine, "systems": [], "contract_fields": [],
                "consumes": [], "unknown_engine": True}
    return dict(reg)


# ── Scope gathering ───────────────────────────────────────────────────────────
def gather_scope(
    entities: List[dict],
    relations: List[dict],
    scope: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Filter the full graph to a scope, returning a self-contained bundle.

    scope.kind:
      "game"      → everything (default)
      "scene"     → a scene + everything it CONTAINS (and their relations)
      "entities"  → an explicit list of keys (scope.keys) + their relations

    The returned ``entities``/``relations`` are closed under the chosen keys:
    a relation is included only when BOTH endpoints are in scope.
    """
    scope = scope or {"kind": "game"}
    kind = scope.get("kind", "game")

    if kind == "game":
        keys = {e["key"] for e in entities}
    elif kind == "scene":
        root = scope.get("key")
        keys = {root} if root else set()
        # Pull direct CONTAINS members of the scene.
        for r in relations:
            if r.get("kind") == "CONTAINS" and r.get("src") == root:
                keys.add(r.get("dst"))
    elif kind == "entities":
        keys = set(scope.get("keys", []))
    else:
        keys = {e["key"] for e in entities}

    scoped_entities = [e for e in entities if e["key"] in keys]
    scoped_relations = [
        r for r in relations if r.get("src") in keys and r.get("dst") in keys
    ]
    layers = sorted({e.get("layer") for e in scoped_entities if e.get("layer")})

    return {
        "scope": {"kind": kind, **{k: v for k, v in scope.items() if k != "kind"}},
        "entities": scoped_entities,
        "relations": scoped_relations,
        "layers": layers,
        "counts": {"entities": len(scoped_entities), "relations": len(scoped_relations)},
    }


# ── Capability diff ───────────────────────────────────────────────────────────
def diff_capabilities(
    required_layers: List[str], registry: Dict[str, Any]
) -> Dict[str, Any]:
    """Compare the layers a scope uses against what the engine already consumes.

    Returns ``provided`` (engine handles it today) and ``gaps`` (needs new code).
    The gaps list is exactly what the code-gen prompt must ask for."""
    consumed = set(registry.get("consumes", []))
    required = sorted(set(required_layers))
    provided = [l for l in required if l in consumed]
    gaps = [l for l in required if l not in consumed]
    return {
        "engine": registry.get("engine"),
        "provided": provided,
        "gaps": gaps,
        "fully_covered": not gaps,
    }


# ── Prompt skeleton (everything except the reasoning prose) ────────────────────
def assemble_prompt_skeleton(
    bundle: Dict[str, Any],
    schemas: Dict[str, Any],
    registry: Dict[str, Any],
    capability_diff: Dict[str, Any],
    target: str = "babylon",
    output: str = "build_packet",
    acceptance: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build the deterministic structure of the code-gen prompt. The LLM stitch
    fills only ``plan_prose`` afterwards; this provides every fact it reasons over.

    ``schemas`` maps layer → active JSON Schema. ``output`` is "build_packet"
    (code-gen) or "design_brief" (content). Returns sections + a rendered text the
    agent can ship verbatim if the LLM is unavailable (graceful degradation)."""
    entities = bundle.get("entities", [])
    relations = bundle.get("relations", [])
    layers = bundle.get("layers", [])

    sections = {
        "target": target,
        "output": output,
        "scope": bundle.get("scope", {}),
        "systems_defined": [
            {"layer": l, "schema": schemas.get(l) or schemas.get(f"{l}_spec")}
            for l in layers
        ],
        "relations": _summarize_relations(relations),
        "world_data": [
            {"key": e["key"], "layer": e.get("layer"), "name": e.get("name"),
             "data": e.get("data", {})}
            for e in entities
        ],
        "engine_capabilities": {
            "engine": registry.get("engine"),
            "already_implemented": registry.get("systems", []),
            "contract_fields": registry.get("contract_fields", []),
            "do_not_reimplement": registry.get("consumes", []),
        },
        "gaps": capability_diff.get("gaps", []),
        "acceptance_criteria": acceptance or [],
    }
    return {
        "sections": sections,
        "rendered": _render_skeleton(sections),
        "needs_llm_stitch": bool(sections["gaps"]) and output == "build_packet",
    }


def _summarize_relations(relations: List[dict]) -> List[dict]:
    by_kind: Dict[str, List[dict]] = defaultdict(list)
    for r in relations:
        by_kind[r.get("kind")].append({"src": r.get("src"), "dst": r.get("dst"),
                                       "data": r.get("data")})
    return [{"kind": k, "edges": v} for k, v in sorted(by_kind.items())]


def _render_skeleton(s: Dict[str, Any]) -> str:
    """Deterministic markdown the agent can ship as-is (LLM stitch optional)."""
    lines: List[str] = []
    out = "Build Packet (code-gen)" if s["output"] == "build_packet" else "Design Brief (content)"
    lines.append(f"# {out} — target: {s['target']}")
    lines.append("")
    lines.append("## Systems defined")
    for sd in s["systems_defined"]:
        lines.append(f"- **{sd['layer']}** schema: {_compact(sd['schema'])}")
    lines.append("")
    lines.append("## Relations")
    for rk in s["relations"]:
        lines.append(f"- {rk['kind']}: {len(rk['edges'])} edge(s)")
    lines.append("")
    lines.append("## World data (in scope)")
    for wd in s["world_data"]:
        lines.append(f"- [{wd['layer']}] {wd['key']} — {wd['name']}")
    lines.append("")
    cap = s["engine_capabilities"]
    lines.append(f"## Engine capabilities ({cap['engine']})")
    lines.append(f"Already implemented (DO NOT re-implement): {', '.join(cap['already_implemented'])}")
    lines.append(f"Layers already consumed: {', '.join(cap['do_not_reimplement'])}")
    lines.append("")
    lines.append("## Gaps to implement")
    for g in s["gaps"]:
        lines.append(f"- {g}")
    if s["acceptance_criteria"]:
        lines.append("")
        lines.append("## Acceptance criteria")
        for a in s["acceptance_criteria"]:
            lines.append(f"- {a}")
    return "\n".join(lines)


def _compact(schema: Optional[dict]) -> str:
    if not schema:
        return "(none)"
    props = (schema.get("properties") or {}) if isinstance(schema, dict) else {}
    if not props:
        return "{}"
    return "{" + ", ".join(sorted(props.keys())) + "}"


# ── Linters (gate LLM-drafted schemas / rules) ────────────────────────────────
_SUPPORTED_TYPES = {"object", "string", "number", "integer", "boolean", "array", "null"}


def lint_schema(schema: Any) -> Dict[str, Any]:
    """Validate a JSON Schema against the supported subset the studio renders.
    Returns {ok, errors[]} — used to gate LLM-drafted schemas before they're saved."""
    errors: List[str] = []
    if not isinstance(schema, dict):
        return {"ok": False, "errors": ["schema must be an object"]}
    t = schema.get("type")
    if t is None:
        errors.append("missing top-level 'type'")
    elif t not in _SUPPORTED_TYPES:
        errors.append(f"unsupported type '{t}'")
    props = schema.get("properties")
    if t == "object" and props is not None:
        if not isinstance(props, dict):
            errors.append("'properties' must be an object")
        else:
            for name, spec in props.items():
                if not isinstance(spec, dict):
                    errors.append(f"property '{name}' must be an object")
                    continue
                pt = spec.get("type")
                # x-ref/x-formula/x-asset are studio custom keywords; type optional then.
                custom = any(k in spec for k in ("x-ref", "x-formula", "x-asset"))
                if pt is None and not custom:
                    errors.append(f"property '{name}' missing 'type'")
                elif pt is not None and pt not in _SUPPORTED_TYPES:
                    errors.append(f"property '{name}' has unsupported type '{pt}'")
    req = schema.get("required")
    if req is not None:
        if not isinstance(req, list):
            errors.append("'required' must be a list")
        elif isinstance(props, dict):
            for r in req:
                if r not in props:
                    errors.append(f"required field '{r}' not in properties")
    return {"ok": not errors, "errors": errors}


_RULE_OPS = {"==", "!=", ">", ">=", "<", "<=", "in", "not_in"}
_RULE_EFFECTS = {"set", "add", "grant", "spawn", "transition"}


def lint_rule(rule: Any) -> Dict[str, Any]:
    """Validate a when/then rule card. Returns {ok, errors[]}.

    rule = {name, when:[{factor, op, value}], then:[{effect, target, value?}],
            priority?, formula?, notes?}"""
    errors: List[str] = []
    if not isinstance(rule, dict):
        return {"ok": False, "errors": ["rule must be an object"]}
    if not rule.get("name"):
        errors.append("missing 'name'")
    when = rule.get("when", [])
    if not isinstance(when, list):
        errors.append("'when' must be a list")
    else:
        for i, c in enumerate(when):
            if not isinstance(c, dict):
                errors.append(f"when[{i}] must be an object")
                continue
            if not c.get("factor"):
                errors.append(f"when[{i}] missing 'factor'")
            if c.get("op") not in _RULE_OPS:
                errors.append(f"when[{i}] op '{c.get('op')}' not in {sorted(_RULE_OPS)}")
            if "value" not in c:
                errors.append(f"when[{i}] missing 'value'")
    then = rule.get("then", [])
    if not isinstance(then, list):
        errors.append("'then' must be a list")
    else:
        for i, eff in enumerate(then):
            if not isinstance(eff, dict):
                errors.append(f"then[{i}] must be an object")
                continue
            if eff.get("effect") not in _RULE_EFFECTS:
                errors.append(f"then[{i}] effect '{eff.get('effect')}' not in {sorted(_RULE_EFFECTS)}")
            if not eff.get("target"):
                errors.append(f"then[{i}] missing 'target'")
    if "priority" in rule and not isinstance(rule["priority"], (int, float)):
        errors.append("'priority' must be numeric")
    return {"ok": not errors, "errors": errors}

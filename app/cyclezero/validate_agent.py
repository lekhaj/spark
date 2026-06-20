"""U5 — the validate agent: deterministic static gate first, LLM only on the residue.

Tools-first: ~80% of validation is deterministic (graph structure, contract compiles,
outcome resolves). The single LLM seam (Tier-B, Bedrock = AWS credits) only does the
*semantic* check — does Claude Code's done-note actually satisfy the acceptance
criteria — and only when both are supplied. A Fix Packet is emitted on warn/fail.

Pure except for the injected ``semantic_fn`` (the LLM seam), so it's unit-testable
without Bedrock.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import contract as contract_builder
from . import graph as graph_algos
from . import outcome as outcome_model


SEMANTIC_SYSTEM = """You are the Spark Studio validator. You are given acceptance
criteria and a developer's done-note describing what was built. Decide whether the
done-note plausibly satisfies EACH criterion. Output a short Markdown list: for each
criterion, "PASS" / "WARN" / "FAIL" + a one-line reason. End with one line:
VERDICT: pass | warn | fail (fail if any criterion fails; warn if any warns).
"""


def static_validate(
    entities: List[dict], relations: List[dict], metamodel: Dict[str, Any],
    game: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Deterministic checks: graph structure, contract compiles, outcome resolves.
    Returns {ok, checks:[{name, ok, detail}], issues:[...]}."""
    checks: List[dict] = []
    issues: List[str] = []

    gv = graph_algos.validate_graph(entities, relations, metamodel)
    checks.append({"name": "graph_structure", "ok": gv["ok"], "detail": gv["counts"]})
    if not gv["ok"]:
        for e in gv["illegal_edges"]:
            issues.append(f"illegal edge {e.get('kind')} {e.get('src')}→{e.get('dst')}: {e.get('reason')}")
        for c in gv["cardinality_violations"]:
            issues.append(f"cardinality: {c.get('kind')} {c.get('endpoint')} {c.get('key')} x{c.get('count')}")
        for m in gv["missing_required_edges"]:
            issues.append(f"missing required edge {m.get('kind')} on {m.get('key')}")

    # contract compiles?
    try:
        c = contract_builder.build_contract(
            {"slug": (game or {}).get("slug", "game"), "title": (game or {}).get("title", "Game")},
            entities,
        )
        ok_c = bool(c.get("contractVersion"))
        checks.append({"name": "contract_compiles", "ok": ok_c, "detail": {"entities": len(c.get("entities", []))}})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "contract_compiles", "ok": False, "detail": str(exc)})
        issues.append(f"contract failed to compile: {exc}")

    # outcome resolves (if there's an outcome node with rules)?
    try:
        state = outcome_model.project(entities, relations)
        outcome_node = next((e for e in entities if e.get("layer") == "outcome"), None)
        if outcome_node:
            rules = (outcome_node.get("data") or {}).get("rules", [])
            default_ending = (outcome_node.get("data") or {}).get("default_ending")
            res = outcome_model.resolve(state, rules, default_ending)
            checks.append({"name": "outcome_resolves", "ok": True, "detail": {"ending": res.get("ending")}})
        else:
            checks.append({"name": "outcome_resolves", "ok": True, "detail": "no outcome node"})
    except Exception as exc:  # noqa: BLE001
        checks.append({"name": "outcome_resolves", "ok": False, "detail": str(exc)})
        issues.append(f"outcome failed to resolve: {exc}")

    ok = all(c["ok"] for c in checks)
    return {"ok": ok, "checks": checks, "issues": issues}


def build_fix_packet(static: Dict[str, Any], semantic: Optional[str]) -> str:
    """A one-paste Fix Packet for Claude Code when validation isn't clean."""
    lines = ["# Fix Packet", ""]
    if static["issues"]:
        lines.append("## Static issues (deterministic)")
        lines += [f"- {i}" for i in static["issues"]]
        lines.append("")
    if semantic:
        lines.append("## Semantic review (acceptance vs done-note)")
        lines.append(semantic)
    if not static["issues"] and not semantic:
        lines.append("No issues — nothing to fix.")
    return "\n".join(lines)


def validate(
    entities: List[dict], relations: List[dict], metamodel: Dict[str, Any],
    *,
    game: Optional[Dict[str, Any]] = None,
    acceptance: Optional[List[str]] = None,
    done_note: Optional[str] = None,
    semantic_fn: Optional[Callable[[List[str], str], str]] = None,
) -> Dict[str, Any]:
    """Run the static gate, then (only if acceptance + done_note + an LLM are present)
    the semantic check. Verdict = worst of the two. Always returns a Fix Packet."""
    static = static_validate(entities, relations, metamodel, game)
    semantic_text: Optional[str] = None
    semantic_verdict = "pass"

    if semantic_fn and acceptance and done_note:
        try:
            semantic_text = semantic_fn(acceptance, done_note)
            low = (semantic_text or "").lower()
            if "verdict: fail" in low:
                semantic_verdict = "fail"
            elif "verdict: warn" in low:
                semantic_verdict = "warn"
        except Exception as exc:  # noqa: BLE001 — degrade to static only
            semantic_text = f"(semantic check unavailable: {exc})"

    if not static["ok"]:
        verdict = "fail"
    elif semantic_verdict == "fail":
        verdict = "fail"
    elif semantic_verdict == "warn":
        verdict = "warn"
    else:
        verdict = "pass"

    return {
        "verdict": verdict,
        "static": static,
        "semantic": semantic_text,
        "fix_packet": build_fix_packet(static, semantic_text) if verdict != "pass" else "",
    }


def make_bedrock_semantic() -> Callable[[List[str], str], str]:
    """Tier-B (structured) semantic checker via Bedrock Converse (AWS credits)."""
    from app.lib import refiner_providers

    provider = refiner_providers.get_tier_provider("B")
    from app.lib import usage_recorder

    def _check(acceptance: List[str], done_note: str) -> str:
        user = (
            "Acceptance criteria:\n"
            + "\n".join(f"- {a}" for a in acceptance)
            + "\n\nDeveloper done-note:\n"
            + done_note
        )
        with usage_recorder.attribution(agent="validate"):
            return provider.chat(SEMANTIC_SYSTEM, [{"role": "user", "content": user}])

    return _check

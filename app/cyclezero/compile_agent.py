"""U4-BE2 — the compile agent: tools → bundle → (optional) LLM stitch → gate.

Tools-first (`spark_studio/docs/inference-architecture.md`): ~85% of a compile is the
deterministic `compile_tools` assembly; the LLM only stitches the implementation-plan
prose, and a deterministic graph check gates the result. The stitch is **injected**
(``stitch_fn``) so this is fully unit-testable without Bedrock, and degrades
gracefully — with no stitch_fn the rendered skeleton ships as-is.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import compile_tools as ct
from . import graph as graph_algos


# System prompt for the Tier-C stitch. The model receives the fully-assembled,
# deterministic sections and writes ONLY the reasoning prose — it never invents data.
STITCH_SYSTEM = """You are the Spark Studio compile agent. You are given a fully
assembled Build Packet (systems, schemas, relations, world data, the engine's existing
capabilities, and the gaps to implement). Everything factual is already provided —
do NOT invent layers, fields, or data.

Write ONLY the implementation plan prose: a numbered, dependency-ordered plan for a
coding agent to close the listed gaps. Weigh multiple aspects explicitly: reuse the
engine capabilities listed under "already implemented" (never re-implement them), the
build order implied by the relations, edge cases, and how each new system maps onto the
Scene Contract. Be concrete and concise. Output Markdown, starting at "1.".
"""


def compile_prompt(
    entities: List[dict],
    relations: List[dict],
    metamodel: Dict[str, Any],
    schemas_by_layer: Dict[str, Any],
    *,
    scope: Optional[Dict[str, Any]] = None,
    target: str = "babylon",
    output: str = "build_packet",
    acceptance: Optional[List[str]] = None,
    ledger: Optional[Dict[str, Any]] = None,
    stitch_fn: Optional[Callable[[Dict[str, Any]], str]] = None,
) -> Dict[str, Any]:
    """Assemble a code-gen prompt for a scope. Pure except for the injected
    ``stitch_fn`` (the single LLM seam). ``ledger`` (the living capability ledger of
    what Claude Code has already built) is merged into the registry so gaps shrink as
    the engine grows. Returns the prompt + provenance for the UI."""
    # 1. deterministic tools
    bundle = ct.gather_scope(entities, relations, scope)
    registry = ct.get_capability_registry(target, ledger=ledger)
    diff = ct.diff_capabilities(bundle["layers"], registry)
    skeleton = ct.assemble_prompt_skeleton(
        bundle, schemas_by_layer, registry, diff, target, output, acceptance
    )

    # 2. deterministic gate: is the scoped graph structurally sound?
    validation = graph_algos.validate_graph(
        bundle["entities"], bundle["relations"], metamodel
    )

    prompt = skeleton["rendered"]
    plan_prose: Optional[str] = None
    stitched = False

    # 3. single LLM seam — only when there are gaps to plan AND a stitcher is wired
    if stitch_fn is not None and skeleton["needs_llm_stitch"]:
        try:
            plan_prose = stitch_fn(skeleton["sections"])
            if plan_prose:
                prompt = f"{prompt}\n\n## Implementation plan\n{plan_prose}"
                stitched = True
        except Exception as exc:  # noqa: BLE001 — degrade to deterministic prompt
            plan_prose = None
            prompt = f"{prompt}\n\n<!-- LLM stitch unavailable: {exc} -->"

    return {
        "prompt": prompt,
        "stitched": stitched,
        "plan_prose": plan_prose,
        "gaps": diff["gaps"],
        "fully_covered": diff["fully_covered"],
        "scope": bundle["scope"],
        "counts": bundle["counts"],
        "gate": {"ok": validation["ok"], "counts": validation["counts"]},
        "needs_llm_stitch": skeleton["needs_llm_stitch"],
    }


def make_bedrock_stitcher() -> Callable[[Dict[str, Any]], str]:
    """Build the Tier-C (reasoning) stitch function backed by Bedrock Converse.
    Imported lazily so the agent stays usable/testable without the LLM stack."""
    import json

    from app.lib import refiner_providers

    provider = refiner_providers.get_tier_provider("C")

    def _stitch(sections: Dict[str, Any]) -> str:
        user = (
            "Here is the assembled Build Packet (JSON). Write the implementation "
            "plan prose only.\n\n```json\n"
            + json.dumps(sections, indent=2, default=str)
            + "\n```"
        )
        return provider.chat(STITCH_SYSTEM, [{"role": "user", "content": user}])

    return _stitch

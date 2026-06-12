"""
code_prompt_template.py — engine change-request prompt rendering (CycleZero T10)
================================================================================

Code never enters Spark Studio — **prompts do**. When an engine-bound schema
bumps or a system_behavior_spec needs engine work, the studio produces a final
*code-change prompt* the user pastes into Claude Code. These render functions
are the single source of that prompt text; the drafts they fill are stored as
ordinary ``/spec-gen`` runs with stage ``code_change_prompt``.
"""

from __future__ import annotations

import json
from typing import Any, Dict

PROMPT_FRAME = """# ENGINE CHANGE REQUEST — {title}
Target repo: cyclezero. Run `npm test` and loop until green.

## WHY
{why}

## CONTRACT
{contract}

## ACCEPTANCE TESTS (must exist and pass)
{acceptance}

## RULES
Only touch files under the module path(s) named above plus their tests.
Do not modify other systems, the engine layer, or schemas.
"""

# JSON Schema for the code_change_prompt artifact itself (engine_bound: false —
# a prompt is data; applying it is what changes the engine). Auto-seeded the
# first time a draft is created so the runs API can validate against it.
CODE_CHANGE_PROMPT_SCHEMA: Dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Code Change Prompt",
    "type": "object",
    "required": ["prompt_id", "title", "target_repo", "source_kind", "source_ref", "prompt_text", "applied"],
    "properties": {
        "prompt_id":      {"type": "string", "minLength": 1},
        "title":          {"type": "string", "minLength": 1},
        "target_repo":    {"type": "string", "minLength": 1},
        "source_kind":    {"enum": ["system_behavior_spec", "schema_bump", "manual"]},
        "source_ref":     {"type": "string", "minLength": 1},
        "prompt_text":    {"type": "string", "minLength": 1},
        "applied":        {"type": "boolean"},
        "applied_commit": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}


def _pretty(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True)


def render_schema_bump_prompt(
    *,
    schema_key: str,
    title: str,
    old_version: int,
    old_schema: Dict[str, Any],
    new_version: int,
    new_schema: Dict[str, Any],
    changelog: str = "",
) -> str:
    """Prompt for an engine-bound schema bump: old vs new contract."""
    why = (
        f"Schema `{schema_key}` bumped v{old_version} → v{new_version} and is engine-bound: "
        f"the engine's loaders/types must match the new contract.\n"
        f"Changelog: {changelog or '(none provided)'}"
    )
    contract = (
        f"### OLD — {schema_key} v{old_version}\n```json\n{_pretty(old_schema)}\n```\n\n"
        f"### NEW — {schema_key} v{new_version}\n```json\n{_pretty(new_schema)}\n```"
    )
    acceptance = (
        "Update the engine-side types/loaders so artifacts valid under the NEW schema "
        "load correctly; add or update vitest cases covering every changed field."
    )
    return PROMPT_FRAME.format(title=f"{title} schema v{new_version}", why=why, contract=contract, acceptance=acceptance)


def render_system_spec_prompt(spec: Dict[str, Any]) -> str:
    """Prompt for implementing a system_behavior_spec in the engine."""
    rules = "\n".join(
        f"- WHEN {r.get('when', '?')} THEN {r.get('then', '?')}"
        for r in spec.get("behavior_rules", [])
    )
    why = (
        f"System `{spec.get('system_id', '?')}` was specified/updated: {spec.get('summary', '')}\n"
        f"Behavior rules:\n{rules}"
    )
    contract = (
        f"- module_path: `{spec.get('module_path', '?')}`\n"
        f"- interface_name: `{spec.get('interface_name', '?')}`\n"
        f"- config:\n```json\n{_pretty(spec.get('config', {}))}\n```"
    )
    tests = spec.get("acceptance_tests", [])
    if tests:
        rows = "\n".join(
            f"| {t.get('name', '?')} | {t.get('given', '?')} | {t.get('expect', '?')} |" for t in tests
        )
        acceptance = (
            "Implement each row below as a vitest case.\n\n"
            "| test name | given | expect |\n|---|---|---|\n" + rows
        )
    else:
        acceptance = "No acceptance tests in the spec — derive vitest cases from the behavior rules."
    return PROMPT_FRAME.format(
        title=spec.get("system_id", "system change"), why=why, contract=contract, acceptance=acceptance
    )

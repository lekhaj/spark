"""X5 — the Factors + Outcome model: pure logic over a game's authored graph.

Like ``graph.py``/``contract.py``/``matching.py`` this is pure functions over
plain dicts (no Mongo, no SQLAlchemy), so it is trivially unit-testable.

The outcome model is a **Factors + branch-graph hybrid**:

  - A *factor* node (layer ``factor``) is a tracked variable — ``numeric`` (with
    optional ``min``/``max``/``default``) or a boolean ``flag``.
  - Story beats / missions / interactions / environments push the factors via
    ``AFFECTS`` edges whose ``data`` carries the delta:
    ``{"op": "add"|"set", "value": <number|bool>, "when"?: <label>}``.
  - An *outcome* node (layer ``outcome``) holds ordered guard ``rules`` mapping a
    factor end-state to an ending.

Shapes (resolved by the route before calling in):
  entity   = {"layer", "key", "name", "data", ...}
  relation = {"src", "dst", "kind", "data"}     # src/dst are entity *keys*
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


# ── helpers ───────────────────────────────────────────────────────────────────
def _is_number(v: Any) -> bool:
    """True for real numbers — explicitly excluding bool (a Python int subclass)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _factor_default(entity: dict) -> Any:
    data = entity.get("data") or {}
    if data.get("default") is not None:
        return data["default"]
    return False if data.get("kind") == "flag" else 0


def _clamp(value: Any, data: dict) -> Any:
    """Clamp a numeric factor to its ``min``/``max``; pass flags through."""
    if not _is_number(value):
        return value
    lo, hi = data.get("min"), data.get("max")
    if _is_number(lo):
        value = max(value, lo)
    if _is_number(hi):
        value = min(value, hi)
    return value


_OPS = {
    ">=": lambda a, b: a >= b,
    ">": lambda a, b: a > b,
    "<=": lambda a, b: a <= b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
    "!=": lambda a, b: a != b,
}


def _cmp(actual: Any, op: str, expected: Any) -> bool:
    fn = _OPS.get(op)
    if fn is None:
        return False
    try:
        return bool(fn(actual, expected))
    except TypeError:
        return False


# ── projection: walk AFFECTS deltas from each factor's baseline ────────────────
def project(entities: List[dict], relations: List[dict]) -> Dict[str, Any]:
    """Compute the projected end-state of every factor: start each at its default,
    then apply every ``AFFECTS`` edge's delta (``add`` accumulates, ``set``
    overrides), clamping numeric factors to their range."""
    factors = {e["key"]: e for e in entities if e.get("layer") == "factor"}
    state: Dict[str, Any] = {k: _factor_default(e) for k, e in factors.items()}

    for r in relations:
        if r.get("kind") != "AFFECTS":
            continue
        dst = r.get("dst")
        if dst not in state:
            continue
        data = r.get("data") or {}
        op = data.get("op", "add")
        val = data.get("value")
        if val is None:
            continue
        cur = state[dst]
        if op == "set":
            state[dst] = val
        elif op == "add":
            if isinstance(cur, bool):
                state[dst] = bool(cur or val)
            elif _is_number(cur) and _is_number(val):
                state[dst] = cur + val
            else:  # incompatible types — treat add as set so authors aren't stuck
                state[dst] = val
        state[dst] = _clamp(state[dst], factors[dst].get("data") or {})

    return state


# ── resolver: ordered guard rules → ending ─────────────────────────────────────
def resolve(
    factor_state: Dict[str, Any],
    rules: List[dict],
    default_ending: Optional[str] = None,
) -> Dict[str, Any]:
    """First-match resolver. Rules are evaluated by ``priority`` desc then declared
    order; the first whose ``when[]`` predicates all hold wins. An empty ``when``
    is a catch-all. Returns the ending plus a per-rule trace for the UI."""
    indexed = list(enumerate(rules or []))
    ordered = sorted(indexed, key=lambda iv: (-(iv[1].get("priority") or 0), iv[0]))

    trace: List[dict] = []
    matched_idx: Optional[int] = None
    matched_ending: Optional[str] = None

    for idx, rule in ordered:
        checks: List[dict] = []
        all_ok = True
        for cond in rule.get("when", []) or []:
            factor, op, expected = cond.get("factor"), cond.get("op"), cond.get("value")
            actual = factor_state.get(factor)
            ok = actual is not None and _cmp(actual, op, expected)
            checks.append({"factor": factor, "op": op, "value": expected,
                           "actual": actual, "ok": ok})
            if not ok:
                all_ok = False
        if all_ok and matched_idx is None:
            matched_idx = idx
            matched_ending = rule.get("ending")
        trace.append({
            "rule": idx,
            "ending": rule.get("ending"),
            "priority": rule.get("priority") or 0,
            "ok": all_ok,
            "checks": checks,
            "matched": all_ok and matched_idx == idx,
        })

    return {
        "ending": matched_ending if matched_idx is not None else default_ending,
        "matched_rule": matched_idx,
        "trace": trace,
    }


# ── contributors: who pushes a factor, ranked by magnitude ─────────────────────
def contributors(
    factor_key: str, entities: List[dict], relations: List[dict]
) -> List[dict]:
    """All ``AFFECTS`` edges that target ``factor_key``, ranked by ``abs(value)``
    desc — the Factor inspector's "all contributors" list."""
    by_key = {e["key"]: e for e in entities}
    out: List[dict] = []
    for r in relations:
        if r.get("kind") != "AFFECTS" or r.get("dst") != factor_key:
            continue
        data = r.get("data") or {}
        src = by_key.get(r.get("src")) or {}
        out.append({
            "src_key": r.get("src"),
            "src_layer": src.get("layer"),
            "src_name": src.get("name"),
            "op": data.get("op", "add"),
            "value": data.get("value"),
            "when": data.get("when"),
        })
    out.sort(
        key=lambda c: abs(c["value"]) if _is_number(c["value"]) else 0,
        reverse=True,
    )
    return out

"""S2/S3 — pure graph algorithms over a game's authored entities + relations.

Everything here is pure logic over plain dicts (no Mongo, no SQLAlchemy) so it is
trivially unit-testable, exactly like ``contract.py``/``matching.py``.

Shapes (resolved by the route before calling in):
  entity   = {"layer", "key", "name", "data", "accepted_spec_run_id"?}
  relation = {"src", "dst", "kind"}          # src/dst are entity *keys*
  metamodel= {"layers": {layer: {...}}, "relation_types": {kind: {...}}}

Dependency convention: a relation type flagged ``dependency: True`` means
*src depends on dst* — dst must be authored/accepted before src. That single
convention drives both topological order and ripple.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, List


# ── helpers ───────────────────────────────────────────────────────────────────
def _key_to_layer(entities: List[dict]) -> Dict[str, str]:
    return {e["key"]: e.get("layer") for e in entities}


def _dependency_edges(relations: List[dict], metamodel: Dict[str, Any]) -> List[dict]:
    rtypes = metamodel.get("relation_types", {})
    return [r for r in relations if rtypes.get(r.get("kind"), {}).get("dependency")]


# ── S2: structural validation ─────────────────────────────────────────────────
def validate_graph(
    entities: List[dict], relations: List[dict], metamodel: Dict[str, Any]
) -> Dict[str, Any]:
    """Check every edge against the metamodel and report structural problems.

    ``ok`` is structural legality only (no illegal / cardinality / required-edge
    violations). ``complete`` additionally requires every node to carry an
    accepted spec — i.e. the game is fully stitched *and* every contract is filled.
    """
    k2l = _key_to_layer(entities)
    rtypes = metamodel.get("relation_types", {})

    illegal_edges: List[dict] = []
    cardinality_violations: List[dict] = []
    missing_required_edges: List[dict] = []

    # Tally edges per (kind) for cardinality + required checks.
    out_count: Dict[tuple, int] = defaultdict(int)  # (kind, src) -> n
    in_count: Dict[tuple, int] = defaultdict(int)   # (kind, dst) -> n
    src_has_kind: Dict[tuple, bool] = defaultdict(bool)  # (kind, src) -> seen

    for r in relations:
        kind, src, dst = r.get("kind"), r.get("src"), r.get("dst")
        rt = rtypes.get(kind)
        if rt is None:
            illegal_edges.append({"kind": kind, "src": src, "dst": dst, "reason": "unknown relation kind"})
            continue
        if src not in k2l or dst not in k2l:
            missing = src if src not in k2l else dst
            illegal_edges.append({"kind": kind, "src": src, "dst": dst, "reason": f"endpoint not found: {missing}"})
            continue
        if rt.get("src_layers") and k2l[src] not in rt["src_layers"]:
            illegal_edges.append({"kind": kind, "src": src, "dst": dst,
                                  "reason": f"src layer '{k2l[src]}' not allowed for {kind}"})
            continue
        if rt.get("dst_layers") and k2l[dst] not in rt["dst_layers"]:
            illegal_edges.append({"kind": kind, "src": src, "dst": dst,
                                  "reason": f"dst layer '{k2l[dst]}' not allowed for {kind}"})
            continue
        out_count[(kind, src)] += 1
        in_count[(kind, dst)] += 1
        src_has_kind[(kind, src)] = True

    # Cardinality: "one" caps the corresponding endpoint at a single edge.
    for (kind, src), n in out_count.items():
        if rtypes[kind].get("src_cardinality") == "one" and n > 1:
            cardinality_violations.append({"kind": kind, "endpoint": "src", "key": src, "count": n})
    for (kind, dst), n in in_count.items():
        if rtypes[kind].get("dst_cardinality") == "one" and n > 1:
            cardinality_violations.append({"kind": kind, "endpoint": "dst", "key": dst, "count": n})

    # Required: every node whose layer is a valid src for a required kind must
    # participate in at least one such edge.
    for kind, rt in rtypes.items():
        if not rt.get("required"):
            continue
        allowed = set(rt.get("src_layers", []))
        for e in entities:
            if e.get("layer") in allowed and not src_has_kind[(kind, e["key"])]:
                missing_required_edges.append({"kind": kind, "key": e["key"], "layer": e["layer"]})

    nodes_without_accepted_spec = [
        e["key"] for e in entities if not e.get("accepted_spec_run_id")
    ]

    ok = not (illegal_edges or cardinality_violations or missing_required_edges)
    return {
        "ok": ok,
        "complete": ok and not nodes_without_accepted_spec,
        "counts": {
            "entities": len(entities),
            "relations": len(relations),
            "illegal": len(illegal_edges),
            "cardinality": len(cardinality_violations),
            "missing_required": len(missing_required_edges),
            "without_spec": len(nodes_without_accepted_spec),
        },
        "illegal_edges": illegal_edges,
        "cardinality_violations": cardinality_violations,
        "missing_required_edges": missing_required_edges,
        "nodes_without_accepted_spec": nodes_without_accepted_spec,
    }


# ── S3: ordering, cycles, ripple ──────────────────────────────────────────────
def find_cycles(
    entities: List[dict], relations: List[dict], metamodel: Dict[str, Any]
) -> List[List[str]]:
    """Return dependency cycles (each a list of keys). DFS three-colour walk over
    the prerequisite graph (dst → src)."""
    k2l = _key_to_layer(entities)
    adj: Dict[str, List[str]] = defaultdict(list)
    for r in _dependency_edges(relations, metamodel):
        if r["src"] in k2l and r["dst"] in k2l:
            adj[r["dst"]].append(r["src"])  # prerequisite dst → dependent src

    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {k: WHITE for k in k2l}
    cycles: List[List[str]] = []
    stack: List[str] = []

    def dfs(u: str) -> None:
        color[u] = GRAY
        stack.append(u)
        for v in adj[u]:
            if color[v] == GRAY:
                # back-edge: extract the cycle slice from the stack
                cycles.append(stack[stack.index(v):] + [v])
            elif color[v] == WHITE:
                dfs(v)
        stack.pop()
        color[u] = BLACK

    for k in k2l:
        if color[k] == WHITE:
            dfs(k)
    return cycles


def topo_order(
    entities: List[dict], relations: List[dict], metamodel: Dict[str, Any]
) -> Dict[str, Any]:
    """Kahn's algorithm over the prerequisite graph → generation/spec order
    (prerequisites first). If a cycle blocks completion, ``order`` is the partial
    prefix and ``cycle`` lists the nodes that could not be ordered."""
    k2l = _key_to_layer(entities)
    adj: Dict[str, List[str]] = defaultdict(list)  # dst -> [src...]
    indeg: Dict[str, int] = {k: 0 for k in k2l}
    for r in _dependency_edges(relations, metamodel):
        if r["src"] in k2l and r["dst"] in k2l:
            adj[r["dst"]].append(r["src"])
            indeg[r["src"]] += 1

    # Stable order: process zero-indegree nodes alphabetically for determinism.
    queue = deque(sorted(k for k, d in indeg.items() if d == 0))
    order: List[str] = []
    while queue:
        u = queue.popleft()
        order.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                queue.append(v)

    blocked = [k for k in k2l if k not in set(order)]
    return {"order": order, "cycle": sorted(blocked), "has_cycle": bool(blocked)}


def ripple(
    changed_key: str,
    entities: List[dict],
    relations: List[dict],
    metamodel: Dict[str, Any],
) -> Dict[str, Any]:
    """Reverse-reachability BFS: everything that (transitively) depends on
    ``changed_key`` and therefore needs re-validation when it changes."""
    k2l = _key_to_layer(entities)
    dependents: Dict[str, List[str]] = defaultdict(list)  # dst -> [src depends on dst]
    for r in _dependency_edges(relations, metamodel):
        if r["src"] in k2l and r["dst"] in k2l:
            dependents[r["dst"]].append(r["src"])

    seen: List[str] = []
    seen_set = {changed_key}
    q = deque([changed_key])
    while q:
        cur = q.popleft()
        for dep in dependents.get(cur, []):
            if dep not in seen_set:
                seen_set.add(dep)
                seen.append(dep)
                q.append(dep)

    return {
        "changed": changed_key,
        "downstream": seen,
        "count": len(seen),
    }

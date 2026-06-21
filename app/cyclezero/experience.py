"""Experience scorer — the deterministic STRUCTURAL channel of the Critic.

Pure logic over a game's authored entities + relations (no Mongo, no SQLAlchemy),
exactly like ``graph.py``/``contract.py`` — so it is trivially unit-testable and the
SCORE is fully deterministic. The LLM (added in a later step) only writes prose ON TOP
of these numbers; it can never move them. That is the trust contract of the Critic.

It scores 7 axes (0-100), backed by the measurement framework (PENS/GEQ/PXI/MDA +
Sid Meier interesting-decisions / Crawford illusion-of-winnability / Koster mastery):

  CHOICE      interesting decisions  — verbs reach distinct outcomes; real decision points
  MASTERY     learning + progression — a rising chain of gates; mechanical novelty
  AUTONOMY    chose, un-forced       — low railroad ratio (few single-path nodes)
  FEEL        acting has a result    — every verb produces a visible consequence
  TENSION     stakes + hope          — a failure outcome exists AND a comeback stays reachable
  IMMERSION   a world to be in       — scene + player + narrative density, few orphans
  DISCOVERY   emergence              — outcomes reached by chaining systems, not 1-hop authored

This is the STRUCTURAL channel only (readable before anyone plays). The FELT channel
(telemetry + micro-probes) and the Tier-2 rigor techniques (RPE/SDT/JND/Wundt) attach to
the same axes later, once the engine emits play-signals.

Shapes (resolved by the route before calling in) — identical to ``graph.py``:
  entity   = {"layer", "key", "name", "data", ...}
  relation = {"src", "dst", "kind"}          # src/dst are entity *keys*
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── semantic vocabulary (stable relation/layer groups the metrics reason over) ──
ACTION_LAYERS = frozenset({"system", "gameplay_loop", "item"})   # things the player does/uses
STATE_LAYERS = frozenset({"outcome", "factor"})                  # things that change
WORLD_LAYERS = frozenset({"scene", "character", "story", "quest", "prop"})

CONSEQUENCE_EDGES = frozenset({"AFFECTS", "MODIFIES", "REWARDS", "TRIGGERS", "GATES"})
PROGRESSION_EDGES = frozenset({"LEADS_TO", "GATES"})
COST_EDGES = frozenset({"REQUIRES", "READS"})

# tokens that mark an outcome as a *failure / loss* state (stakes signal)
_FAILURE_TOKENS = ("lose", "lost", "loss", "death", "die", "dead", "fail", "defeat",
                   "game over", "gameover", "perish", "ko")

# axis weights for the headline roll-up — CHOICE highest (it is the thesis).
WEIGHTS: Dict[str, float] = {
    "choice": 0.25, "mastery": 0.15, "autonomy": 0.15, "feel": 0.10,
    "tension": 0.15, "immersion": 0.10, "discovery": 0.10,
}


@dataclass
class Axis:
    score: int                              # 0-100 (deterministic)
    evidence: str                           # why — always cites graph facts
    pitfalls: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"score": self.score, "evidence": self.evidence, "pitfalls": self.pitfalls}


@dataclass
class Scorecard:
    headline: int
    axes: Dict[str, Axis]
    weakest: str
    suggestion: str
    pitfalls: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "headline": self.headline,
            "axes": {k: v.as_dict() for k, v in self.axes.items()},
            "weakest": self.weakest,
            "suggestion": self.suggestion,
            "pitfalls": self.pitfalls,
        }


def _clamp(x: float) -> int:
    return max(0, min(100, int(round(x))))


def _is_failure(ent: dict) -> bool:
    name = (ent.get("name") or "").lower()
    data = ent.get("data") or {}
    if any(tok in name for tok in _FAILURE_TOKENS):
        return True
    kind = str(data.get("kind") or data.get("polarity") or "").lower()
    return kind in ("loss", "fail", "failure", "defeat", "negative")


# ── the scorer ─────────────────────────────────────────────────────────────────
def score_structural(
    entities: List[dict],
    relations: List[dict],
    metamodel: Optional[Dict[str, Any]] = None,
) -> Scorecard:
    """Compute the 7-axis structural scorecard from the authored graph. Pure +
    deterministic: same graph in → same numbers out. Robust to an empty graph."""
    layer_of = {e["key"]: e.get("layer") for e in entities}
    by_layer: Dict[str, List[dict]] = defaultdict(list)
    for e in entities:
        by_layer[e.get("layer")].append(e)

    actions = [e for e in entities if e.get("layer") in ACTION_LAYERS]
    states = [e for e in entities if e.get("layer") in STATE_LAYERS]

    # adjacency over consequence edges (a source produces a change in a target)
    out_edges: Dict[str, List[dict]] = defaultdict(list)
    in_sources: Dict[str, set] = defaultdict(set)   # target_key -> {distinct source keys}
    for r in relations:
        src, dst, kind = r.get("src"), r.get("dst"), r.get("kind")
        if not src or not dst:
            continue
        out_edges[src].append(r)
        if kind in CONSEQUENCE_EDGES:
            in_sources[dst].add(src)

    # decision points = state nodes pushed by >=2 distinct sources (a real choice converges here)
    decision_points = [k for k, srcs in in_sources.items()
                       if layer_of.get(k) in STATE_LAYERS and len(srcs) >= 2]

    axes: Dict[str, Axis] = {}

    # 1 CHOICE — do verbs reach distinct outcomes, and are there competing options?
    verbs_with_outcome = sum(
        1 for a in actions
        if any(e.get("kind") in CONSEQUENCE_EDGES and layer_of.get(e.get("dst")) in STATE_LAYERS
               for e in out_edges.get(a["key"], []))
    )
    if not actions:
        axes["choice"] = Axis(0, "No action/system entities — nothing to decide between.",
                              ["EmptySandbox"])
    else:
        reach = verbs_with_outcome / len(actions)
        # reward both coverage (verbs that matter) and the existence of decision points
        score = 60 * reach + min(40, 20 * len(decision_points))
        pf = []
        if not decision_points:
            pf.append("SolvedGame")  # no convergent choice → likely one dominant line
        axes["choice"] = Axis(
            _clamp(score),
            f"{len(actions)} actions, {verbs_with_outcome} reach a distinct outcome, "
            f"{len(decision_points)} decision point(s) (outcomes with ≥2 competing sources).",
            pf,
        )

    # 2 MASTERY — a progression of rising gates + mechanical novelty
    prog_edges = [r for r in relations if r.get("kind") in PROGRESSION_EDGES]
    n_systems = len(by_layer.get("system", []))
    if not prog_edges and n_systems < 2:
        axes["mastery"] = Axis(0, "No progression edges (LEADS_TO/GATES) and <2 systems — "
                                  "nothing to learn or climb.", ["FlatCurve"])
    else:
        score = min(60, 20 * len(prog_edges)) + min(40, 13 * n_systems)
        pf = ["FlatCurve"] if not prog_edges else []
        axes["mastery"] = Axis(
            _clamp(score),
            f"{len(prog_edges)} progression edge(s), {n_systems} distinct system(s) "
            f"(novelty to master).", pf,
        )

    # 3 AUTONOMY — railroad ratio: fraction of forward-nodes with exactly one path out
    fwd_out: Dict[str, int] = defaultdict(int)
    for r in relations:
        if r.get("kind") in PROGRESSION_EDGES and r.get("src"):
            fwd_out[r["src"]] += 1
    nodes_with_fwd = [k for k, n in fwd_out.items() if n >= 1]
    if not nodes_with_fwd:
        # no progression authored yet — neutral, not penalised (too early to tell)
        axes["autonomy"] = Axis(50, "No progression authored yet — autonomy undetermined.", [])
    else:
        single = sum(1 for k in nodes_with_fwd if fwd_out[k] == 1)
        railroad = single / len(nodes_with_fwd)
        pf = ["Railroad"] if railroad >= 0.8 else []
        axes["autonomy"] = Axis(
            _clamp(100 * (1 - railroad)),
            f"railroad ratio {railroad:.2f} ({single}/{len(nodes_with_fwd)} forward nodes have "
            f"a single path) — lower is freer.", pf,
        )

    # 4 FEEL — every verb should produce a visible consequence (no dead verbs)
    if not actions:
        axes["feel"] = Axis(0, "No actions to feel.", [])
    else:
        dead = [a["key"] for a in actions if not any(
            e.get("kind") in CONSEQUENCE_EDGES for e in out_edges.get(a["key"], []))]
        score = 100 * (1 - len(dead) / len(actions))
        pf = ["VerbWithoutConsequence"] if dead else []
        ev = f"{len(actions) - len(dead)}/{len(actions)} actions produce a consequence."
        if dead:
            ev += f" Dead verbs: {', '.join(dead[:5])}."
        axes["feel"] = Axis(_clamp(score), ev, pf)

    # 5 TENSION / HOPE — a failure outcome must exist AND a comeback stay reachable
    failures = [e for e in entities if e.get("layer") in STATE_LAYERS and _is_failure(e)]
    if not failures:
        axes["tension"] = Axis(15, "No failure/loss outcome — no stakes, so no drama or hope.",
                               ["NoStakes"])
    else:
        # comeback = at least one failure state is NOT a terminal sink (has a forward/reward edge)
        comeback = any(any(e.get("kind") in (PROGRESSION_EDGES | {"REWARDS", "TRIGGERS"})
                           for e in out_edges.get(f["key"], [])) for f in failures)
        if comeback:
            axes["tension"] = Axis(85, f"{len(failures)} failure outcome(s) with a reachable "
                                       f"comeback path — stakes plus hope.", [])
        else:
            axes["tension"] = Axis(45, f"{len(failures)} failure outcome(s) but each is a "
                                       f"terminal sink — losing is a death spiral.", ["NoComeback"])

    # 6 IMMERSION — a world to be in: scene + player char + narrative density, few orphans
    has_scene = bool(by_layer.get("scene"))
    has_player = any((e.get("data") or {}).get("role") == "player"
                     for e in by_layer.get("character", []))
    world_n = sum(len(by_layer.get(l, [])) for l in WORLD_LAYERS)
    linked_keys = {r.get("src") for r in relations} | {r.get("dst") for r in relations}
    orphans = [e["key"] for e in entities if e["key"] not in linked_keys]
    orphan_ratio = (len(orphans) / len(entities)) if entities else 1.0
    score = (35 if has_scene else 0) + (25 if has_player else 0) + min(20, 4 * world_n) \
        + 20 * (1 - orphan_ratio)
    axes["immersion"] = Axis(
        _clamp(score),
        f"scene={'yes' if has_scene else 'no'}, player={'yes' if has_player else 'no'}, "
        f"{world_n} world entities, orphan ratio {orphan_ratio:.2f}.", [])

    # 7 DISCOVERY — emergence: states reached by chaining (system→state→state), plus convergence
    chained = [r for r in relations
               if r.get("kind") in CONSEQUENCE_EDGES
               and layer_of.get(r.get("src")) in STATE_LAYERS
               and layer_of.get(r.get("dst")) in STATE_LAYERS]
    emergent_signal = len(chained) + len(decision_points)
    if not states:
        axes["discovery"] = Axis(0, "No state/outcome entities — nothing to discover.", [])
    elif emergent_signal == 0:
        axes["discovery"] = Axis(10, "Every outcome is a direct 1-hop authored edge — no systemic "
                                     "interaction, so no emergent discovery.", ["AllAuthored"])
    else:
        axes["discovery"] = Axis(
            _clamp(25 * emergent_signal),
            f"{len(chained)} chained state→state interaction(s), {len(decision_points)} "
            f"convergence point(s) — emergent surface.", [])

    # headline roll-up + weakest-axis suggestion
    headline = _clamp(sum(WEIGHTS[k] * axes[k].score for k in WEIGHTS))
    weakest = min(axes, key=lambda k: axes[k].score)
    pitfalls = sorted({p for ax in axes.values() for p in ax.pitfalls})
    return Scorecard(headline, axes, weakest,
                     _suggest(weakest, axes[weakest]), pitfalls)


# the smallest concrete next step per weak axis — phrased in the graph's own vocabulary
_SUGGESTIONS: Dict[str, str] = {
    "choice": "Add an outcome reached by two different actions so the player has a real "
              "decision (e.g. two abilities that compete for the same resource).",
    "mastery": "Add a LEADS_TO/GATES progression so challenge rises, and a second system to "
               "learn — right now there's little to climb.",
    "autonomy": "Give at least one progression node a second forward path — the route is a "
                "corridor, which reads as forced.",
    "feel": "Wire every action to a consequence (AFFECTS/MODIFIES/REWARDS); a verb that "
            "changes nothing feels dead.",
    "tension": "Add a failure outcome (a way to lose) and keep a comeback edge from it — "
               "no stakes means no hope.",
    "immersion": "Add a scene and a character with data.role='player' so there's a world to "
                 "be in, and connect orphan entities.",
    "discovery": "Let two systems affect a shared factor/outcome so interactions emerge "
                 "instead of every result being directly authored.",
}


def _suggest(weakest: str, ax: Axis) -> str:
    return _SUGGESTIONS.get(weakest, "Strengthen the weakest axis.")

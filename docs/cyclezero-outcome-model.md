# CycleZero — Factors + Outcome Model (X5)

How a CycleZero game's **ending** is derived from play. The model is a
**Factors + branch-graph hybrid**: authored set-pieces (story branches) *and*
emergent, system-driven shifts (factors moved by deltas) both feed an ordered
resolver that maps the factor end-state to an ending.

Pure logic lives in `spark/app/cyclezero/outcome.py` (no Mongo, no SQLAlchemy —
unit-testable like `graph.py`). Routes in `spark/app/cyclezero/routes.py`. Tests in
`spark/tests/test_outcome.py`.

---

## Concepts

- **Factor** (`layer: factor`) — a tracked variable. `data`:
  `{ kind: "numeric" | "flag", min?, max?, default?, description? }`.
- **AFFECTS edge** — story / mission / interaction / environment → factor. Edge
  `data` carries the delta: `{ op: "add" | "set", value: <number|bool>, when?: <label> }`.
  `add` accumulates; `set` overrides. Numeric factors clamp to `min`/`max`.
- **Outcome** (`layer: outcome`) — holds ordered guard `rules`. `data`:
  ```json
  {
    "rules": [
      { "when": [{ "factor": "trust", "op": ">=", "value": 8 }],
        "ending": "hero", "priority": 5 }
    ],
    "default_ending": "neutral"
  }
  ```
  Ops: `>= > <= < == !=`. An empty `when` is a catch-all.

These shapes match the seeded JSON Schemas in `schema_seeds.py` (`factor_spec`,
`outcome_spec`) and the relation metamodel in `metamodel.py` (`AFFECTS`, `READS`,
`LEADS_TO`).

---

## `outcome.py` functions

- `project(entities, relations) -> {factor_key: value}` — seed each factor at its
  `default`, apply every `AFFECTS` delta, clamp numeric factors.
- `resolve(factor_state, rules, default_ending) -> {ending, matched_rule, trace}` —
  rules evaluated by `priority` desc then declared order; first all-true rule wins;
  `trace` records each rule's pass/fail per condition. Falls back to `default_ending`.
- `contributors(factor_key, entities, relations) -> [...]` — all `AFFECTS`-in edges
  for a factor, ranked by `abs(value)` desc.

---

## Endpoints

| Method · Path | Purpose |
|---------------|---------|
| `GET  /cyclezero/games/{slug}/scenes/{key}/hub` | Scene-hub aggregation: CONTAINS members by layer + inherited globals (X2) |
| `GET  /cyclezero/games/{slug}/factors/{key}/contributors` | Ranked AFFECTS-in for a factor |
| `POST /cyclezero/games/{slug}/outcome/project` | Project factor end-state (optional `{overrides}`), run resolver → `{factor_state, ending, matched_rule, trace}` |

All are **additive** read/compute endpoints over the existing tables — no schema
change. They reuse `service.list_entities` / `service.list_relations`; the route
helpers `_entity_data_dicts` / `_relation_data_dicts` carry inline `data` (the
existing `_relation_dicts` drops edge `data`, which the outcome model needs).

---

## Verification

`cd spark && pytest tests/test_outcome.py tests/test_cyclezero_graph.py`
(SQLite + mongomock). Covers resolve precedence/default, project add/set/clamp,
contributors ranking, scene-hub inheritance, and the outcome-project route with
what-if overrides. **30 passed** as of this build.

## Deploy

Additive; deploy with the standard CPU ritual on explicit go-ahead:
`git pull && sudo systemctl restart fastapi_app.service` on `18.207.13.85`.
No migration required (everything rides `entities.data` / `relations.data`).

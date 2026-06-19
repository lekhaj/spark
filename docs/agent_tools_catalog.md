# Spark Studio — Agent Tool Catalog (pre-implementation plan)

**Status:** planning only. No agentic flow implemented yet — this is the *tool surface*
we want to exist before wiring any orchestration. Companion to the CycleZero Explore
plan (`peaceful-cuddling-piglet.md`) and the X-series tasks.

## Framing

An "agent" here is an LLM that authors / explores / refines a game by **calling tools**.
We are *not* designing the loop yet — we are cataloguing every tool the loop could call,
so the loop is just "pick tools + reason." Three design rules:

1. **Every authoring tool is a thin wrapper over the generic graph** (S0–S7). New game
   concepts are *registry rows + JSON schemas*, never migrations. Tools that write
   content ultimately call `service.create_entity / create_relation / set_accepted_spec`.
2. **Read tools are cheap and composable; write tools are gated.** The agent can query,
   research, simulate, and project freely; anything that mutates the graph, spends GPU,
   sends mail, or publishes goes through a confirm/checkpoint path.
3. **Grounding before generation.** The model should research (web + GDD + the game's own
   graph) and cite, *then* generate. "Search all over the internet" and "reason from the
   GDD" are first-class tool tiers, not afterthoughts.

Backing modules already in the repo are named per tool so we know what's a wrapper vs. new
build. Key existing surfaces: `spark/app/cyclezero/{service,graph,contract,matching,bridge,
generation,metamodel,schemas}.py`, `spark_studio/src/lib/{cyclezeroApi,journeyApi,
schemaApi,specGenApi}.ts`, worker (`flux_concept_generator.py`, rig/mesh workers,
generators `trellis|pixal3d|hunyuan3d`).

---

## Part 1 — Use cases (jobs-to-be-done the tools must cover)

These are the end-to-end things a creator (or investor demo) should be able to ask for.
Each maps to a chain of tools from Part 2.

**Authoring from intent**
- U1. "Here's my GDD (pdf/doc/link) — build the game." → ingest → research gaps →
  propose graph (scenes/characters/systems/factors/outcomes) → confirm → generate specs →
  generate assets.
- U2. "Make a roguelike merchant town with 3 vendor NPCs and a reputation system." →
  research genre conventions → scaffold scene hub + NPCs + a `reputation` factor +
  AFFECTS wiring → assets.
- U3. "Add a betrayal storyline that can flip the ending." → story beats + `LEADS_TO`
  branches + `AFFECTS` a `trust` factor + outcome rule.

**Grounding / research**
- U4. "What monetization models fit a cozy farming sim?" → web research brief (cited).
- U5. "Tear down Hades' core loop and tell me what we're missing." → competitor teardown
  vs. our graph → gap list.
- U6. "Find reference art for a bioluminescent swamp." → image/style research → moodboard
  attached to an `environment` node.

**Asset production**
- U7. "Generate the merchant: concept → 3D candidates → pick one → rig → place in scene."
  → flux concept → trellis/pixal/hunyuan fan-out → `chooseModel3d` → rig → CONTAINS edge.
- U8. "Re-skin all forest props in autumn palette." → batch image edit → 3D refresh.

**Explore / understand / balance**
- U9. "Why does my game always end badly?" → outcome projection + factor contributors
  (which beats/missions push `trust` down).
- U10. "Is my economy balanced?" → balance report over `system` numbers + sim playthroughs.
- U11. "What's incomplete?" → coverage report (orphan nodes, dead branches, factors with
  no contributors, beats with no factor link).
- U12. "Show me everything in Level 2." → scene hub aggregation incl. inherited globals.

**Lifecycle**
- U13. "Snapshot this as v3 and diff against v2." → release snapshot + diff.
- U14. "I changed the trust system — what breaks downstream?" → ripple analysis →
  impact feed.

---

## Part 2 — The tool catalog

Legend — **Build:** `wrap` (existing endpoint/module), `new-be` (new backend, mostly pure
functions over existing tables), `new-ext` (external/3rd-party or GPU), `new-fe` (studio
UI surface). **Write?** = mutates state / spends money → gated.

### Tier A — Research & grounding ("search the whole internet" + GDD)

| Tool | Does | In → Out | Build | Write? |
|---|---|---|---|---|
| `web_search` | Query the open web for mechanics, lore, competitors, art refs | query, recency, domain filter → ranked hits w/ snippets+urls | new-ext (WebSearch / SerpAPI/Brave/Tavily) | no |
| `web_fetch` | Pull & clean one page/wiki/article to text | url → markdown text + metadata | new-ext (WebFetch) | no |
| `deep_research` | Multi-source synthesis into a **cited brief** (genre conventions, audience, monetization, difficulty norms) | topic, scope → structured brief w/ citations | new-ext (orchestrates web_search+fetch+LLM) | no |
| `image_reference_search` | Find reference imagery / moodboard for art direction | style prompt → image urls + tags | new-ext | no |
| `competitor_teardown` | Analyze a named game's loop/systems/economy/monetization | game name → structured loop + systems map | new-ext + LLM | no |
| `gdd_ingest` | Parse a GDD (pdf/docx/md/gdoc) into structured intent | file/url → sections, entities mentioned, goals, constraints | new-be (reuse docx/pdf skills + LLM) | no |
| `gdd_to_graph_proposal` | Map parsed GDD → **proposed** layers/entities/relations (dry-run, no write) | parsed GDD + current graph → proposed diff | new-be (LLM + metamodel/schemas) | no |
| `reference_library_add` / `_query` | RAG store of a game's own docs/briefs/refs, scoped per game | doc → id / query → passages | new-be (Mongo + embeddings) | add=yes |
| `gap_analysis` | Compare GDD intent vs. current graph → what's missing/contradictory | gdd + graph → gap list | new-be (LLM over graph) | no |

> The GDD bridge is the centerpiece of "reason from GDD": **ingest → propose (dry-run) →
> human confirm → commit via Tier B**. The proposal is always a diff the user approves;
> the agent never silently writes a whole game from a document.

### Tier B — Graph authoring (CRUD over the generic model, S0–S7)

| Tool | Does | Backing | Write? |
|---|---|---|---|
| `list_games` / `get_game` / `create_game` | game CRUD | `cyclezeroApi` / `service` | create=yes |
| `list_entities` / `get_entity` | read nodes (layer-filtered) | `service.list/get_entity` | no |
| `create_entity` / `update_entity` / `delete_entity` | node CRUD, layer-aware | `service.*_entity` | yes |
| `create_relation` / `delete_relation` | typed edges w/ `data` payload (AFFECTS delta, MODIFIES patch, GATES, LEADS_TO label…) | `service.create_relation` + `_validate_new_edge` | yes |
| `query_graph` | scoped neighborhood (1–2 hops) | `graph` + `cyclezeroApi` | no |
| `validate_graph` | completeness/legality/cycles | `service` + `graph` (`validateGraph`,`graphOrder`) | no |
| `graph_ripple` | downstream impact of a node change | `bridge.compute_ripple` (`graphRipple`) | no |
| `register_layer` / `register_relation_type` | metamodel rows (no migration) | `metamodel.py` `/metamodel/*` | yes (registry) |
| `author_schema` / `get_contract` | per-layer JSON Schema (the "contract") | `schemas.py` / `schemaApi` / `contract.build_contract` | author=yes |
| `spec_gen` | generate a node's JSON spec **body** per its layer schema (S1 bridge) | `specGenApi` + Mongo spec-gen | yes (staged) |
| `diff_spec` / `accept_spec` | stage → review → accept a spec run | `service.set_accepted_spec`, `bridge.on_spec_accepted` | accept=yes |
| `compile_contract` / `match` | compile contract + match entities → readiness | `contract.build_contract`, `matching.match` | no |

### Tier C — Per-layer authoring helpers (bespoke generators → Tier B writes)

Each emits a **proposed spec/edges** the user confirms; commit goes through Tier B.

- **Story:** `generate_story_beats`, `design_branches` (LEADS_TO + choice labels),
  `link_factor_to_beat` (AFFECTS ±delta), `outcome_contribution` (which endings a beat pushes).
- **Character / NPC:** `generate_bio`, `generate_relationships`, `design_behavior_schedule`
  (NPC), `design_dialogue_hooks`.
- **Mission:** `generate_objectives` (visible **+ hidden**), `design_conditions` (layered),
  `design_carry_forward`, `set_completion_timer`, `design_variants`, `set_rewards`.
- **System:** `propose_system_numbers`, `set_scope` (global/local), `add_env_modifier`
  (MODIFIES patch w/ effective-value preview).
- **Environment / area theme:** `generate_area_theme` (palette/lighting/scatter/ambient),
  `apply_system_modifiers`.
- **Interaction:** `design_interaction` (trigger → conditions → effects on factor/system/story).
- **Factor / Outcome:** `define_factor` (numeric|flag, min/max/default), `wire_affects`,
  `author_outcome_rules` (ordered guards), `validate_outcome` (reachability of endings).

### Tier D — Asset pipeline (flux → 3D → rig; existing GPU stack)

| Tool | Does | Backing | Write? |
|---|---|---|---|
| `generate_concept_image` | flux / Union-Pro concept (per asset_spec morphology) | worker `flux_concept_generator.py`, `generation.submit` | yes (GPU $) |
| `image_edit` / `variations` / `inpaint` | edit/restyle concepts (e.g. autumn re-skin) | worker | yes (GPU $) |
| `generate_3d_candidates` | fan-out **trellis + pixal3d + hunyuan3d** | `journeyApi.createAssetRun`, GENERATORS | yes (GPU $) |
| `choose_model3d` | pick the candidate used in-game | `journeyApi.chooseModel3d` | yes |
| `rig_model` | all-7 rig + dual FBX/GLB export | worker rig (`run_rig_worker.py`) | yes (GPU $) |
| `generate_texture_pbr` | PBR/material pass | worker | yes (GPU $) |
| `asset_status` | poll long-running GPU jobs | `journeyApi.getAssetRun`, `service.get_job` | no |
| `generate_audio` *(future)* | sfx / music / ambient / VO | new-ext | yes |

### Tier E — Analysis, validation, simulation

| Tool | Does | Backing | Write? |
|---|---|---|---|
| `outcome_resolve` | run resolver: factor end-state → ending + per-rule trace | **new `outcome.py` `resolve()`** | no |
| `outcome_project` | "if the game ended now" over a chosen/derived factor state | `outcome.py` `project()` (walk AFFECTS) | no |
| `factor_contributors` | ranked AFFECTS-in for a factor + projected range | new-be (graph query) | no |
| `scene_hub` | aggregate everything a scene CONTAINS + inherited globals | new-be endpoint `/scenes/{key}/hub` | no |
| `playtest_sim` | simulate N playthroughs over branch+factor model → outcome distribution | new-be (uses `outcome.py`) | no |
| `balance_report` | economy/difficulty curves from system numbers + sims | new-be | no |
| `coverage_report` | orphans, dead branches, factor-less beats, unreachable endings | new-be over `validate` | no |
| `design_critique` | LLM critique of graph vs. GDD goals + genre best practices | new-be (LLM + Tier A briefs) | no |

### Tier F — Lifecycle, memory, human-in-the-loop (loop infra — tools, not the loop)

| Tool | Does | Backing | Write? |
|---|---|---|---|
| `release_snapshot` / `list_releases` / `release_diff` | versioned game cycles + diff | `service.create_release` (S7) | snapshot=yes |
| `ripple_to_impact` | turn a change into journey impact-feed items | `bridge.ripple_to_impact_items` (S4) | yes |
| `game_memory_write` / `_read` | per-game agent working memory (decisions, locked choices) | new-be (Mongo) | write=yes |
| `request_decision` | surface a choice to the human (the AskUser gate) | studio UI | no |
| `agent_match` | the existing matching/readiness loop signal | `matching.match` | no |
| `checkpoint` | lightweight save before a risky batch | new-be | yes |

---

## Part 3 — How tools compose (preview of flows — not built yet)

Two illustrative chains (read-then-confirm-then-write throughout):

**U1 "Build from my GDD":**
`gdd_ingest` → `deep_research`(fill gaps) → `gap_analysis` → `gdd_to_graph_proposal`
→ **request_decision** (approve diff) → Tier B writes (`register_layer?` →
`create_entity`×N → `create_relation`×N → `spec_gen` → `accept_spec`) → Tier C per-layer
fill → `validate_graph` → **request_decision** → Tier D assets → `release_snapshot`.

**U9 "Why does it end badly?":**
`outcome_resolve`(current) → `factor_contributors`(the low factor) →
`graph_ripple`/`scene_hub` to locate the beats → `design_critique` → propose edits
(Tier C) → `outcome_project` to confirm the fix flips the ending → confirm → write.

**Gating model (which tools need confirmation):** all Tier B/C/F writes, all Tier D
(GPU spend), `reference_library_add`, and anything outward-facing. Read/research/sim
tools (A read-side, all of E, query/validate in B) run freely.

---

## Part 4 — Build phasing (T-series, after X-series Explore lands)

- **T0 — Tool runtime + registry.** A single schema for tool defs + dispatcher that maps
  to the wrappers above; no agent loop yet. Wire the pure read tools first
  (`query_graph`, `validate_graph`, `scene_hub`, `outcome_*`, `web_search`, `web_fetch`).
- **T1 — Grounding tools.** `gdd_ingest`, `deep_research`, `gap_analysis`,
  `reference_library`. These are high-leverage and read-only/safe.
- **T2 — Write tools behind confirm.** Tier B CRUD + `spec_gen`/`accept_spec` as tools,
  each returning a *proposed diff* the UI confirms.
- **T3 — Per-layer generators (Tier C)** on top of the bespoke inspectors (X3–X6).
- **T4 — Asset tools (Tier D)** wrapping the existing GPU pipeline.
- **T5 — Analysis/sim (Tier E):** `playtest_sim`, `balance_report`, `coverage_report`,
  `design_critique`.
- **T6 — Only now: the agent loop** (plan → call tools → reflect), reusing `matching.py`
  for readiness and Tier F for memory/checkpoints/decisions.

## Open decisions (for the user)

1. **Web search provider** — Tavily / Brave / SerpAPI / Bing? (cost + citation quality).
   Recommend Tavily or Brave for cited, agent-friendly results.
2. **Where the agent runs** — CPU box (orchestration, cheap) calling GPU only for Tier D?
   Recommend yes: agent on CPU, GPU strictly for asset jobs.
3. **Embeddings for `reference_library`** — local vs. API.
4. **Confirm granularity** — per-write vs. per-batch-diff. Recommend per-batch-diff
   (one approval for a proposed graph change set) to avoid approval fatigue.
5. **Scope of T6 autonomy** — supervised (every write confirmed) vs. semi-autonomous
   (read/research/sim free, batched write approvals). Recommend supervised for the
   investor demo, semi-autonomous later.

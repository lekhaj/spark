# Asset Generation — End-to-End Architecture

*How a character/prop in Spark Studio becomes a rigged, playable 3D asset.*
Last verified live: 2026-06-23 (diablo-2 `blacksmith-gareth`, on-demand g7e Blackwell).

---

## 1. The flow at a glance

```
┌─────────────┐   POST generate    ┌────────────────────┐
│ Spark Studio│ ─────────────────▶ │  FastAPI (CPU box) │
│  (browser)  │ ◀───── poll job ── │  18.207.13.85:8000 │
└─────────────┘                    └─────────┬──────────┘
   ▲  progressive UI                         │ creates
   │  (image-first, then 3D, then rig)       ▼
   │                              ┌────────────────────────┐
   │                              │ asset_run (World_builder│  poll-driven
   │                              │   Mongo) — state machine│  state machine
   │                              └─────────┬──────────────┘
   │                                        │ enqueues stage tasks
   │                                        ▼
   │                              ┌────────────────────────┐
   │                              │ Redis queue            │
   │                              │ manual_gen_tasks_spot  │
   │                              └─────────┬──────────────┘
   │                                        │ BLPOP (stage-affinity)
   │                                        ▼
   │                              ┌────────────────────────┐
   │                              │ manual_gen_worker      │  GPU box
   │                              │ (g7e Blackwell, 95 GB) │  (on-demand / spot)
   │                              └─────────┬──────────────┘
   │                       flux_pose → trellis ┃ pixal3d ┃ hunyuan3d → rig
   │                                        ▼
   │                              ┌────────────────────────┐
   │                              │ S3  sparkassets-us     │
   │                              │ chars/{slug}__{key}/…  │
   │                              └─────────┬──────────────┘
   │      reconciler (20s) + poll _refresh  │ writes URLs back
   └────────────────────────────────────────┘
                                            ▼
                              entity.data.glb / .fbx  (Postgres)
                                            │
                              GET /contract │ (any entity with data.glb → assets[])
                                            ▼
                                   Babylon play app (cyclezero)
```

**One sentence:** the browser triggers a job, the backend turns it into a *poll-driven
asset_run* that fans stage tasks onto a Redis queue, a single GPU worker drains them
warm-batched into S3, and a reconciler folds the per-stage results back into the job (for
the UI) and into the entity (for the playable scene contract).

---

## 2. Where it goes, stage by stage (what each step serves)

| # | Step | Where it runs | Serves | Persistence |
|---|------|---------------|--------|-------------|
| 1 | `POST /cyclezero/games/{slug}/entities/{key}/generate` | FastAPI (CPU) | creates an `AssetJob` (Postgres) + `asset_request` record (Mongo `cyclezero`) | returns `job.id`, `asset_run_id` |
| 2 | `generation.submit()` | FastAPI (CPU) | synthesizes an accepted `asset_spec` + creates an `asset_run` (Mongo `World_builder`) | the run is the real state machine |
| 3 | asset_run `_refresh` / reconciler | FastAPI (CPU) | advances stages, **enqueues** the next GPU task to Redis | poll- **and** timer-driven (20 s) |
| 4 | `flux_pose` (image) | GPU worker | A-pose concept image (ControlNet-Union-Pro) | `chars/{slug}__{key}/v…/…png` in S3 |
| 5 | `trellis` + `pixal3d` + `hunyuan3d` (3D fan-out) | GPU worker | **three** candidate meshes (all run, full quality) | `…/{gen}.glb` each in S3 |
| 6 | choose | asset_run | picks first `done` mesh as `model3d_chosen` | field on the run |
| 7 | `rig` | GPU worker | skeletoned GLB **+** FBX (auto-rig, manual fallback) | `…_rigged.glb` / `.fbx` in S3 |
| 8 | reconcile write-back | FastAPI (CPU) | patches `entity.data.glb` / `.fbx` (Postgres) | closes the loop |
| 9 | `GET /contract` | FastAPI (CPU) | any entity with `data.glb` → an `assets[]` entry; characters with a spawn → placement | served live to the play app |
| 10 | Babylon load + runtime anims | play app | loads the GLB, retargets shared CC0 humanoid clips | client-side |

**Segmentation invariant (parallel games never collide):**
`asset_id = char_label = "{slug}__{entity_key}"`, S3 prefix `chars/{slug}__{key}/v{M.N}/`.
A second game (different slug) shares **no** Mongo doc, S3 prefix, or Postgres row.

---

## 3. The self-describing job (what the UI polls)

A single `GET /cyclezero/games/{slug}/jobs/{id}` returns everything the progressive UI
needs — derived from the `asset_run` doc by `generation.derive_phase()`:

```
phase:    queued → gpu_warming → image → model3d → rigging → complete | failed
progress: 0.0 … 1.0
stages: {
  image:   { url },
  model3d: { trellis:{status,url}, pixal3d:{…}, hunyuan3d:{…} },
  model3d_chosen: "trellis" | …,
  rigged:  { status, url, fbx_url, rig_status:"auto"|"manual" }
}
```

- `gpu_warming` is surfaced explicitly so the UI shows **"starting GPU…"** instead of a
  stall during the ~12 min cold prewarm + boot.
- The **image is shown the moment it lands** while the three 3D meshes are still running.
- `GET …/jobs?status=active` lists in-flight jobs → used for **resume on reopen** and the
  **app-shell watcher** that toasts completion even if the user navigated away.

Frontend pieces: `useAssetJob` (poll, auto-stop on terminal) → `AssetProgress`
(image-first → candidates → rig → viewer + GLB/FBX downloads) → `useJobWatcher` +
`jobNotify` (deduped global completion toast).

---

## 4. Failure scenarios & how they're handled

| Scenario | Behavior | Mechanism |
|----------|----------|-----------|
| **GPU stopped at submit** | Job stays `queued`/`gpu_warming`; pipeline resumes when worker drains | poll-driven run never raises; `submit()` is best-effort (`submitted:false` on defer) |
| **Cold box (lazy EBS)** | ~12.4 min prewarm before first task; UI shows `gpu_warming` | `spark-prewarm.service` reads ~150 GB into page cache; worker ordered `After=` it |
| **Worker didn't auto-start on boot** | **Fixed** — was the AL2023 `network-online.target` hang | units now `After=network.target` only + `Restart=always` + Redis-wait loop |
| **One 3D generator fails** | Pipeline proceeds with the remaining meshes; failed tile shown in UI | `_refresh` auto-picks first `done`; per-candidate status surfaced |
| **Auto-rig fails** | Falls back to manual rig path; `rig_status:"manual"` shown | rig stage tolerant; UI badge |
| **Spot reclaim mid-task** | In-flight task re-queued to **front**; on-demand/relaunch retries first | SIGTERM handler `lpush` in `manual_gen_worker.run()` |
| **No spot capacity** | Falls back to on-demand | orchestrator `ensure_gpu_ready` spot-first → on-demand |
| **Browser closed / navigated away** | Server reconciler still advances the run; toast fires on return | background `asset_reconciler` (20 s) + `useJobWatcher` |
| **Transient poll error** | Ignored, next poll recovers | swallowed in hook + watcher |
| **Mixed-content (HTTPS page → HTTP backend)** | Routed through `/_api` Cloudflare Pages Function | `apiBase.ts` resolution order |
| **Two games, same entity name** | No collision | `{slug}__{key}` segmentation everywhere |

---

## 5. Cost optimizations implemented

Principle (locked with user): **never reduce content/quality** — all three 3D generators
always run at full quality. Every optimization is *waste removal*, not content cutting.

| Optimization | What it saves | Status |
|--------------|---------------|--------|
| **Prewarm page-cache** (`spark-prewarm.sh`) | Avoids ~90 min first-flux EBS lazy-load; pays a bounded ~12.4 min instead | live |
| **Stage-affinity scheduler** (P1, `stage_affinity.pop_next_task`) | Reorders the queue **stage-major** (all trellis, then all pixal3d, then all hunyuan3d) so a model loads once per *batch*, not once per *character* | live + unit-tested |
| **Lookahead-evict** (P2-2a, `peek_has_stage`) | Trellis (in-process) stays resident across consecutive same-stage tasks; evicts only at the group boundary | live + unit-tested |
| **Persistent subprocess servers** (P2-2b) | hunyuan3d/pixal3d would load weights once per batch instead of per task (the biggest single win) | **planned** — hunyuan3d in-repo doable; pixal3d blocked (its `inference.py` is box-local, no-edit rule) |
| **Auto-shutdown** (15 min idle) | Stops paying for an idle GPU | live ⚠️ see note |
| **Spot-first → on-demand failover** | Cheaper spot when available, reliable on-demand fallback | live |
| **Background reconciler** | One cheap sweep advances all runs; no per-client polling load | live |
| **Prompt caching + history trim** (LLM) | Caches stable tool schemas; trims chat window to 6 turns → fewer Haiku tokens | live |

### ⚠️ Auto-shutdown vs. prewarm (action item)
Measured cold start: **prewarm ≈ 12.4 min** for ~150 GB. The idle-shutdown threshold is
**15 min**. So a box that boots for a job and then sits idle self-terminates only ~2.6 min
after becoming ready — too tight (risks killing the box mid-queue-gap, or right before a
follow-up request). **Recommendation:** make the idle timer **start only after the prewarm
sentinel is published** (don't count warmup as idle), and/or raise the threshold to
~25–30 min. This is the "configure autoshutdown with the prewarm logic" item.

### Cold-start budget (for autoshutdown tuning)
- Boot + driver: ~1 min
- Prewarm (~150 GB @ ~205 MB/s EBS): **~12.4 min**
- First task model load (lazy, from warm cache): seconds–1 min
- → **time-to-first-asset on a cold box ≈ 13–14 min**; on a warm box, seconds.

---

## 6. Key components (file map)

**Backend (`spark`)**
- `app/cyclezero/routes.py` — generate + jobs endpoints, `reconcile_job`
- `app/cyclezero/generation.py` — `submit`, `derive_phase`, `reconcile`
- `app/services/asset_reconciler.py` — 20 s background sweep
- `app/routes/asset_run_routes.py` — the asset_run state machine (`_refresh`)
- `worker/workers/manual_gen_worker.py` — queue loop + per-stage handlers
- `worker/lib/stage_affinity.py` — stage-major scheduling + lookahead peek
- `worker/gpu_setup/spark-prewarm.{service,sh}`, `deploy/systemd/manual_gen_worker.service`
- `deploy/gpu/{apply_units,ensure_access}.sh` — GPU box install/access

**Frontend (`spark_studio`)**
- `src/lib/cyclezeroApi.ts` — `generateEntity`/`getAssetJob`/`listAssetJobs` + types
- `src/hooks/useAssetJob.ts`, `src/hooks/useJobWatcher.ts`
- `src/components/asset/AssetProgress.tsx`, `EntityAssetPanel.tsx`
- `src/lib/jobNotify.ts`, `src/lib/apiBase.ts`

**Data stores**
- Postgres — graph entities (`game_id`-scoped), `AssetJob`
- Mongo `cyclezero` — `asset_requests`; Mongo `World_builder` — `asset_runs`, `spec_gen_runs`, `manual_gen_*`
- Redis — `manual_gen_tasks_spot` queue + `gpu:active_instance_id`, `prewarm:ready:*`
- S3 `sparkassets-us` — `chars/{slug}__{key}/v{M.N}/…`

**Infra**
- CPU box `18.207.13.85` (Ubuntu) — FastAPI + reconciler + orchestrator + result_consumer
- GPU on-demand `i-05e8570023728c112` (EIP 52.91.128.47) / spot `i-09fca0acb4cc429f7` — g7e.2xlarge, Amazon Linux 2023, `ec2-user`, RTX PRO 6000 Blackwell 95 GB

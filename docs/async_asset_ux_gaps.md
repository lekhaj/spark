# Async asset generation UX — gap analysis + plan

Desired flow: a user in Spark Studio triggers generation and **waits with live feedback** —
job status + per-stage progress, the **image shown as soon as it's ready while 3D is still
running**, then 3D candidates, then rig, and a **notification when it's done and viewable** in
the UI. Implemented cleanly, with edge cases handled.

## Two flows today (very uneven)

- **Flow A — `/asset-runs` + `AssetWorkspace.tsx`** ✅ already does this well: 5s poll on
  `getAssetRun`, renders `stages.image.url` the moment it exists, 3D candidate tiles with status
  + "Use this", rig tile, then manifest + GLB/FBX downloads; resumes on reopen via `listAssetRuns`.
  Reached through the *journey/impact-item* resolve path.
- **Flow B — cyclezero game/chat** (`POST /games/{slug}/entities/{key}/generate` → `asset_jobs`):
  the backend pipeline + (new) background reconciler exist, but the **UI/async surface is missing**.
  This is the flow a normal user driving a game (the diablo-2 path) actually uses.

## Gaps (Flow B)

| # | Gap | Detail |
|---|-----|--------|
| **G1** | No frontend trigger/poll client | `cyclezeroApi.ts` has full CRUD but **no `generateEntity`, `getJob`, `listJobs`**, and nothing in the UI polls cyclezero jobs. |
| **G2** | Job payload too thin for progressive display | `get_job`/`generation.reconcile` put only `asset_run_status` + final `glb/fbx` in `job.result`. The per-stage **image URL, the 3 model3d candidate URLs, model3d_chosen, rig status** live in the `asset_run` doc but aren't surfaced via the job — so "show image while 3D runs" is impossible from the job alone. |
| **G3** | Chat trigger is record-only | `creator_agent.generate_asset` calls `service.create_job(...)` **record-only ("no GPU spend; batched later")** — it does NOT start the pipeline. Only the REST endpoint (`generation.submit`) does. So "generate the elder" in chat records a job but produces nothing. |
| **G4** | No completion notification | `notify()` exists (Sonner + desktop fallback) but isn't wired to a cyclezero job/asset-run completion. |
| **G5** | No post-gen asset viewer on the entity | The inspector/chat card has no image + GLB preview + downloads bound to the entity once `entity.data.glb` is written. |
| **G6** | No global/background watcher | Polls are component-scoped (unmount → poll stops). If the user navigates away, no "done" notice. Need an app-shell watcher (read-only; the server reconciler already advances runs). |
| **G7** | Thin status vocabulary / partial-failure UX | Job status is queued/running/done. No surfaced **phase** (gpu_warming / image / model3d / rigging / complete / failed), no per-3D-generator failure tolerance shown, no rig manual-fallback surfaced in this flow. |
| **G8** | No "GPU warming" signal | While `ensure_gpu_ready` is starting/failing-over (or the worker is booting), the user sees a stall, not "starting GPU…". |

## Status — ALL PHASES IMPLEMENTED 2026-06-23
- **P1** ✅ `generation.derive_phase` + enriched `reconcile`/`reconcile_job` (job.result now carries
  `phase`/`progress`/full `stages`); `GET .../jobs?status=active`. Test: `test_derive_phase_progression` (16/16 pass).
- **P2** ✅ chat `generate_asset` starts the real pipeline via `generation.submit` (env `CYCLEZERO_RECORD_ONLY=1` to defer).
- **P3** ✅ `cyclezeroApi`: `generateEntity`/`getAssetJob`/`listAssetJobs` + types; `useAssetJob` polling hook.
- **P4** ✅ `components/asset/AssetProgress.tsx` (image-first → 3D candidates → rig → viewer+downloads) +
  `EntityAssetPanel.tsx`, wired into Character/NPC/Prop inspectors.
- **P5** ✅ `useJobWatcher` mounted in `StudioShell` (app-shell completion toast regardless of screen) +
  `lib/jobNotify.ts` global dedup; `EntityAssetPanel` resumes in-flight jobs on reopen.
- **P6** ✅ gpu_warming label+hint, per-candidate failure shown, rig manual-fallback, transient-error swallow,
  dedup. Frontend `tsc -b` clean.
- Remaining: one live GPU pass to watch it end-to-end through the studio UI.

## Plan (phased; each phase shippable + verifiable on its own)

### P1 — Backend: make the job self-describing (additive, unit-testable, no UI dep) ⭐ start here
- Enrich `reconcile`/`get_job` so `job.result` carries the full `stages` block (image, the 3
  model3d candidates, `model3d_chosen`, rigged + `rig_status`) **plus a derived `phase`**
  (`queued | gpu_warming | image | model3d | rigging | complete | failed`) and a 0–1 `progress`.
  Source everything from the `asset_run` doc (it already has it). One poll → full progressive state.
- Add `GET /games/{slug}/jobs?status=active` (list in-flight jobs for resume + the global watcher).
- Fold a GPU-readiness hint (from `gpu:active_instance_id` / `ensure_gpu_ready` state) into `phase`
  so G8 is covered.
- Tests: phase derivation from each stage combo; enriched result shape; active-jobs filter.

### P2 — Backend: unify the trigger (fix G3)
- Make `creator_agent.generate_asset` call the **same** real pipeline as the REST endpoint
  (`generation.submit`) — keep the dims/descriptor capture, but actually start the run and return
  the `asset_run_id` so the UI can watch it. (Or have the tool POST the REST generate path.)
- Test: chat generate_asset → creates a real asset_run (mongomock/sqlite, patched submit).

### P3 — Frontend: generate + progressive watcher client (fixes G1)
- `cyclezeroApi.ts`: `generateEntity(slug,key)`, `getJob(slug,jobId)`, `listJobs(slug, active?)`,
  typed to the enriched `JobOut` (stages + phase + progress).
- `useAssetJob(slug, jobId)` hook (TanStack Query, `refetchInterval` while phase not terminal).

### P4 — Frontend: progressive UI + viewer (fixes G2-display, G5, G7)
- A reusable `AssetProgress` component (lift AssetWorkspace's stage renderer): image-as-soon-as-ready,
  3D candidate tiles streaming with status + "Use this", rig tile, manual-fallback note, then a
  **viewer** (image + GLB `<model-viewer>`/Babylon thumbnail + GLB/FBX download). Bind it to the
  character/prop inspector and the chat "generating…" card.
- Fire `notify('success', '<name> ready', …)` on phase→complete.

### P5 — Global watcher + resume (fixes G4 fully, G6)
- App-shell watcher polls `listJobs(active)` for the current game/user; on a job→complete it fires the
  notification + desktop fallback regardless of the current screen, and the inspector resumes an
  in-flight job on open (mirrors AssetWorkspace's `listAssetRuns` resume).

### P6 — Robustness / edge cases (G7, G8)
- Show `gpu_warming` ("starting GPU…"); tolerate a single 3D generator failing (proceed with the rest
  — `_refresh` already auto-picks the first `done`); surface rig manual-fallback; handle job
  expiry/timeout, transient poll errors (ignore), and idempotent resume. No partial/blank states.

## Verification
- Backend: unit tests for phase derivation + enriched result + active-jobs (no GPU).
- Frontend: smoke the generate→poll→stages-render→notify path against a mock job.
- One live GPU pass: run the diablo-2 generate **through the studio UI** end-to-end — image shows
  while 3D runs, candidates stream, rig lands, toast fires, asset viewable. Then idle-stop.

## Non-negotiables carried in
- All 3 generators always run at full quality (no content reduction — see
  `memory/cost_optimization_constraint.md`).
- Reuse Flow A's proven renderer rather than building a second one; keep one progressive component.
- Best-effort/never-500 on the poll path (matches existing `get_job`/reconcile contract).

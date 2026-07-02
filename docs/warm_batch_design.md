# Design: same-stage batching to kill redundant model reloads

Goal (per user): **remove the wasteful model reloads, keep ALL output** — every character
still runs all three 3D generators (trellis + pixal3d + hunyuan3d) at full quality. This is
pure waste-removal: same meshes, fewer model loads.

## The waste, precisely (from the live diablo-2 run)

The worker pulls the queue FIFO (`ManualGenWorker.run` → `r.blpop`). The orchestrator fans
out per character, so the queue interleaves stages:

```
trellis_A, pixal3d_A, hunyuan3d_A, trellis_B, pixal3d_B, hunyuan3d_B
```

Two different reload mechanisms make this expensive (`worker/workers/manual_gen_worker.py`):

1. **In-process models** (`trellis`, `flux_cn`, `sd`) — managed by `ModelManager`. The
   trellis handler **self-evicts after each task** (`self._mgr.evict("trellis")`, line ~459),
   and the subprocess handlers **fully unload all in-process models** before spawning
   (line ~483, to free VRAM). So with the interleaved order, trellis **loads twice** (once per
   character) instead of once.
2. **Subprocess models** (`pixal3d`, `hunyuan3d`, `model_family=None`) — each task spawns a
   **cold subprocess** (`Pixal3D/inference.py`, `worker/models/hunyuan3d_infer.py`) that loads
   multi-GB weights, runs once, and exits. So these reload their model on **every single task**,
   regardless of order. This is the dominant cost (hunyuan3d ≈ 9 min/char, much of it load).

Reordering to a **stage-major** sequence is what we want:

```
trellis_A, trellis_B, pixal3d_A, pixal3d_B, hunyuan3d_A, hunyuan3d_B
```

…but ordering alone isn't enough — the handlers still evict/cold-spawn. So the fix is
**(a) group same-stage tasks, then (b) make execution warm within a group.**

Correctness note: all three 3D stages only consume the already-done `flux_pose` image and are
independent of each other; `rig` runs on a *different* queue after `choose`. So every task
sitting in `manual_gen_tasks_spot` at any moment is runnable — reordering among them is safe.

## Phased plan

### Phase 1 — Stage-affinity scheduler (worker, in spark repo, unit-testable, SAFE) ✅ DONE 2026-06-23
Implemented in `worker/lib/stage_affinity.py` (`pop_next_task`), wired into
`ManualGenWorker.run()` (tracks `last_stage`, resets on idle). Unit-tested by
`tests/test_stage_affinity.py` (5 tests: grouping, FIFO head, fallback, empty, 3-char
once-per-model). No VRAM behavior changed; live-applies on next worker deploy.
Replace the plain FIFO pull with a "prefer the current stage" pull:
- Track `last_stage`. In `run()`, `_next_task(r, last_stage)`:
  1. If `last_stage` set: `LRANGE` the queue, find the first task with `stage == last_stage`,
     `LREM` it (count=1) and return it. (Single worker → no pop race.)
  2. Else `BLPOP` the head (timeout=30), preserving FIFO + idle/auto-shutdown behavior.
- Result: tasks execute stage-major (all trellis, then all pixal3d, then all hunyuan3d).
- This is the scheduling **foundation**; on its own it reorders but doesn't yet skip reloads
  (handlers still evict). It is independently correct and safe, and unit-testable with a fake
  Redis (assert grouping). No VRAM behavior changes → no GPU needed to verify.

### Phase 2 — Warm execution within a group (VRAM-sensitive, needs a GPU verify session)
Convert the grouping into actual saved reloads. Two parts, both gated on live GPU testing
(do them in ONE batched GPU session to avoid repeated paid boots):

- **2a · In-process lookahead evict** — stop the trellis/flux handlers from unconditionally
  evicting; centralize eviction in `process_task` with a `next_stage` hint: keep the family
  resident if the next task reuses it, evict only when the stage actually changes. Saves the
  trellis/flux reloads inside a group. Small, but must be VRAM-verified (no OOM when switching
  into a subprocess stage).
- **2b · Persistent subprocess servers (the big win)** — replace cold per-task subprocesses for
  `pixal3d`/`hunyuan3d` with a **long-lived server subprocess** that loads weights once and
  serves consecutive requests (image→glb) over a simple stdin/JSON or domain-socket protocol.
  The worker keeps the server alive while same-stage tasks remain (Phase 1 groups them), then
  tears it down when the stage changes (mirrors VRAM hygiene — only one big model warm at a
  time). Wraps the existing `inference.py` / `hunyuan3d_infer.py` in a request loop; touches the
  companion `Pixal3D` (box-local) + `worker/models/hunyuan3d_infer.py`. For N characters, each
  big model loads **once per batch** instead of N times — the headline saving.

### VRAM strategy (invariant for Phase 2)
Only one heavy model resident at a time. Within a group the warm model/server stays; on stage
change, fully release before the next group loads. This is the same hygiene the handlers do
today — we just defer it to the group boundary instead of every task.

### Verification
- Phase 1: unit test the scheduler (fake Redis: mixed queue → assert stage-major pop order;
  empty/last-stage-absent → falls back to FIFO head). No GPU.
- Phase 2: one GPU session — run a 3+ character batch, confirm each heavy model loads once per
  group (grep worker journal for load/evict counts), wall-clock drops vs the interleaved
  baseline, and **all three GLBs per character still produced at full quality** (the hard
  requirement). Then idle-stop.

## Out of scope (explicit)
No dropping generators, no fewer steps/lower texture on the final bake, no skipped stages.
Output is identical; only the load/unload churn is removed. See `cost_optimization_constraint`.

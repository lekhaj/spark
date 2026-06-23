# Asset pipeline — issue map, normal user flow, and cost-optimization plan

Written 2026-06-22 after the first real end-to-end run (diablo-2: elder-marcus +
zombie-shambler, image→3D→rig→play). Covers (1) every issue we hit and its fix,
(2) what the normal user flow *should* be, (3) a prioritized plan to cut GPU and
Haiku-token cost.

---

## 1. Issue map — what broke, why, and the fix

| # | Symptom | Root cause | Fix / status |
|---|---------|-----------|--------------|
| 1 | Orchestrator couldn't start the GPU | CPU role `ec2_s3` lacked `ec2:StartInstances` etc. | **FIXED** — `AmazonEC2FullAccess` attached. `ensure_gpu_ready` now works. |
| 2 | Spot start always failed | `InsufficientInstanceCapacity` for g7e spot | **Mitigated** — spot-first → on-demand failover (built). Spot capacity still scarce. |
| 3 | GPU up but queue never drained | After a stop→start, `manual_gen_worker.service` does NOT auto-start despite `enabled` | **Worked around** (manual `systemctl start`). **TODO**: orchestrator should start/verify the worker — see §3. |
| 4 | Could not SSH the GPU box | Box is **Amazon Linux 2023** → user is `ec2-user`, not `ubuntu`; docs/code said `ubuntu` | **FIXED** — LOCAL_INFRA.md + `pipeline_dashboard.py` corrected; user is always `ec2-user`. |
| 5 | Two GPU boxes need different keys | on-demand created with `KeyName=us_cpu_key`; spot with `KeyName=None` | **Partially fixed** — `deploy/gpu/ensure_access.sh` adds both pubkeys to either box; create future boxes with `--key-name us_cpu_key`. Spot still needs the script run once it's reachable. |
| 6 | Stale infra docs | LOCAL_INFRA/CLAUDE predated the g7e migration (listed the decommissioned L40S, `spark-bootstrap`) | **FIXED** — refreshed with current g7e IDs, services, gotchas. |
| 7 | `asset_run` stuck in "generating" | State machine is **poll-driven** (`_refresh` only advances on GET); nothing was polling | **Worked around** with a CPU-side poller loop. **TODO**: a small background reconciler — see §3. |
| 8 | Play app: "Failed to fetch" contract | No `/_api` dev proxy → cross-origin CORS block | **FIXED** — `vite.config.ts` proxy (`CYCLEZERO_API`). |
| 9 | Couldn't verify render via console | Babylon bundled as ESM (no `window.BABYLON`); iso ortho cam hard to reframe | **FIXED** — `?debug=1` exposes `window.__cz`; use perspective cam for inspection shots. |
| 10 | hunyuan3d failed on an old run | "missing input_front URL" (Jun-14 run) | Not reproduced this run — all 3 generators succeeded. Watch if it recurs. |

---

## 2. Normal user flow (the happy path)

This is what a user (or a second game like cyclezero) should experience — no manual GPU/SSH steps.

1. **Author content** in Spark Studio (or a per-user game repo seed script) → entities land in
   the game-segmented graph (`slug`-keyed). Descriptions can be template-authored (zero LLM).
2. **Generate** an asset: `POST /cyclezero/games/{slug}/entities/{key}/generate`.
   - Backend synthesizes an accepted `asset_spec`, creates an `asset_run`, and queues the image stage.
   - `ensure_gpu_ready` auto-starts a GPU (spot→on-demand), attaches EIP, **starts the worker** (TODO #3),
     points the active queue.
3. **Pipeline runs** (segmented S3 `chars/{slug}__{key}/v…`): Flux Union Pro → trellis+pixal3d+hunyuan3d
   → auto-choose (or artist picks) → Auto-Rig Pro → `rigged.glb`+`.fbx`.
4. **Reconcile**: polling `GET …/jobs/{id}` advances the run and writes `entity.data.glb/fbx/lod`.
   (A background reconciler should do this automatically — TODO #3.)
5. **GPU idle-stops** after the queue drains (15-min autoshutdown).
6. **Play**: open `https://…/?game={slug}` — the contract serves rigged GLBs in `assets[]`/`actors[]`;
   the engine loads them (capsule fallback for ungenerated actors) and animates via the shared clip library.

The two manual hops to eliminate so this is hands-off: **#3 worker auto-start** and **#7 auto-reconcile**.

---

## 3. Cost-optimization plan

### Implemented 2026-06-22 (this pass)
- **A1 — worker auto-start on boot** ✅ — `manual_gen_worker.service` + `spark-prewarm.service`
  no longer gate on `network-online.target` (the AL2023 stop→start hang); worker is `Restart=always`;
  `start_manual_worker.sh` waits (bounded) for Redis. Live-apply on next boot via
  `deploy/gpu/apply_units.sh`.
- **A2 — auto-reconcile** ✅ — `app/services/asset_reconciler.py` sweeps in-flight asset jobs every
  `RECONCILE_INTERVAL_S` (20s), advancing runs + writing GLBs back with no human poller (started in
  `main.py` lifespan; reuses `routes.reconcile_job`). GPU *stop* still owned by the worker's
  AutoShutdown — tighten via `IDLE_SHUTDOWN_MINUTES` (config, not a competing CPU stop).
- **B3 — chat history trim** ✅ — `HISTORY_FOR_LLM` 12→6 (env `REFINER_HISTORY_TURNS`); durable
  memory is the graph + cached `facts_json`, so this roughly halves uncached per-call history tokens.
- **B1 — prompt caching** ✅ — was already on for the system block; extended `chat_tools` to also
  cache the (large, stable) **tool schemas**. Verify hits via the usage admin page (`cache_read>0`).

### Remaining (planned, not yet done)

### A. GPU running time (biggest $ lever — g7e.2xlarge ≈ on-demand)

Priority order (impact × ease):

1. **Make the worker auto-start on boot** *(eliminates idle GPU + manual SSH; HIGH impact, LOW effort)*
   After a stop→start the box currently bills while idle because the worker never picks up work. Fix one of:
   - have `ensure_gpu_ready` SSH/SSM `systemctl start manual_gen_worker` after the instance is `running`, and
     block on the `prewarm:ready:<iid>` Redis signal the worker already publishes; **or**
   - fix the systemd unit so it reliably starts on boot (investigate why `enabled` didn't fire — likely the
     `After=spark-prewarm` ordering + a failing condition).
2. **Auto-reconcile + don't idle needlessly** *(HIGH, LOW)* — a background reconciler advances runs without a
   human poller (done). The worker's existing AutoShutdown still owns the stop (single clock, no race); just
   make sure it isn't left idle longer than necessary via `IDLE_SHUTDOWN_MINUTES`. **Not aggressive** — the box
   stops when genuinely done, not mid-work.
3. **Batch all characters into one GPU session** *(HIGH, MED)* — one GPU start should drain *every* pending
   character, not one at a time. Queue all first, then start once. Pure efficiency — same output, far less
   boot/idle overhead amortized across the batch.
4. **Avoid redundant model reloads** *(MED, MED)* — across the 3-generator fan-out and across characters we
   evict/reload models per stage. Keep each model resident while consecutive same-stage tasks are pending
   (e.g. run all `trellis` tasks back-to-back, then all `pixal3d`, then all `hunyuan3d`) so each heavy model
   loads once per batch instead of once per character. **All 3 generators still run at full quality** — this
   only removes load/unload churn.
5. **Keep spot-first working** *(MED, ongoing)* — spot g7e is ~60–70% cheaper for the *same* work. Capacity is
   scarce now; keep the failover and consider an alternate AZ / capacity reservation for predictable runs.
6. **Faster boot / warm model cache** *(LOW, MED)* — boot→worker-ready is minutes (lazy load on first task).
   Pre-pull/warm the model cache on the persistent root volume so the GPU spends its paid minutes generating,
   not booting. Same output, less wasted wall-clock.

> **Out of scope (per user):** do NOT reduce generated content or quality to save cost — always fan out all
> three generators (trellis + pixal3d + hunyuan3d) at full settings; no single-generator default, no lowering
> the final bake's steps/texture. Optimization = remove waste/inefficiency, never produce less. See
> `memory/cost_optimization_constraint.md`.

### B. Haiku token usage (LLM cost)

Current state: 3-tier provider (`refiner_providers.py`) is **Haiku-everywhere** (A/B/C). Call sites:
creator (A, chat), validate (B), propose/compile (C). `creator_agent.HISTORY_FOR_LLM=12` replays 12 turns/call.

Priority order:

1. **Enable Bedrock prompt caching** *(HIGH, LOW)* — the system prompt + tool/schema definitions are large
   and identical across calls. Cache them so repeated structured calls (validate/compile/propose) only pay
   for the variable suffix. Typically 50–80% input-token reduction on repeated calls.
2. **Template-first routing, expand it** *(HIGH, MED)* — `info.py` already answers some chat with hand-written
   templates (zero tokens); the diablo-2 descriptions were 100% template-authored. Route deterministic
   intents (CRUD, lookups, canned answers) to templates and only fall through to Haiku for genuine NL.
3. **Trim conversation history** *(HIGH, LOW)* — `HISTORY_FOR_LLM=12` is the dominant input cost in chat.
   Drop to ~4–6 turns + a rolling summary of older turns (summarize once, reuse) instead of replaying raw.
4. **Cap output + use low effort** *(MED, LOW)* — set tight `max_tokens` per tier and terse-output system
   instructions; structured tiers (B) rarely need long generations.
5. **Cap validator iterations** *(MED, LOW)* — the validate→iterate loop (U5) can re-call Haiku repeatedly;
   cap at N rounds and short-circuit when the validator returns no actionable diffs.
6. **Tier downgrade per the locked plan** *(MED, MED)* — the inference-architecture plan is "start
   Haiku-everywhere, then drop A→Llama 4 (cheap chat) and keep Sonnet only for C reasoning." Move Tier A
   chat off Haiku to a cheaper model once quality is confirmed; reserve the strongest model for compile (C).
7. **Batch validation** *(LOW, MED)* — validate a whole proposed bundle in one call rather than per-item.

### Guiding principle
Optimization here means **removing waste and fixing inefficiencies fluidly** — not aggressive cost-cutting and
never reducing generated content or quality. Every item above keeps the same output (all 3 generators, full
settings, full Haiku capability) and only eliminates idle time, redundant reloads, replayed/uncached tokens,
or pointless re-calls. Done first (this pass): **A1 worker auto-start, A2 auto-reconcile, B1 prompt caching,
B3 history trim** — all pure waste-removal, no behavior change.

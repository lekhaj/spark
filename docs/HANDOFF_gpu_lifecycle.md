# Handoff — Asset-pipeline reliability + GPU cost-efficiency hardening

**Date:** 2026-06-23  **Branch:** `main`  **Commits:** `6bc7d9f`, `75eeaa2`
**Repo:** https://github.com/lekhaj/spark  (origin)

This file is self-contained: copy it to any machine to resume. It documents the
incident, the fix design, every file touched, how it's deployed, how to verify,
and exactly what is still pending.

---

## 1. Why this work exists (the incident)

A live diablo-2 `blacksmith-gareth` asset generation **stalled and wasted GPU
time**. Three root causes:

1. **GPU stopped mid-pipeline.** Idle-shutdown judged idleness from the GPU's
   *local Redis queue depth* + a coarse `_active` flag — blind to (a) long 3D
   subprocesses (`pixal3d`/`hunyuan3d`, ~5–7 min) that run with an **empty
   queue**, and (b) an in-flight `asset_run` between stages (CPU about to enqueue
   the next stage). Two idle clocks (GPU-side `auto_shutdown.py` + CPU
   `orchestrator_service.py`) compounded it. Result: box auto-stopped at ~15 min
   while pixal3d was running → forced restart (~8 min second prewarm + redo two
   3D stages) → ~2× wall-clock.
2. **A lost GPU task froze the run forever.** When the box died mid-pixal3d the
   `manual_gen_stage_runs` doc stuck at `status:"running"`. The asset-run state
   machine (`_refresh`) only auto-picks a generator once **all three** are
   terminal, so a stuck `running` blocked choose→rig indefinitely. No
   stuck-stage recovery existed; the worker's SIGTERM-requeue is unreliable on a
   stop-initiated SIGKILL.
3. **Duplicate fan-out.** The queue held two `hunyuan3d` tasks — a race where the
   20s reconciler and a `GET` both ran `_refresh` before `_model3d_queued`
   persisted.

(The earlier `session_id` confusion during manual recovery was operator error —
re-enqueuing by hand-copying a task payload. Recovery code now only re-enqueues
through the `queue_*` helpers, which keep the run doc `_id` == task `session_id`.)

---

## 2. The design (what changed conceptually)

**One busy signal, one stop authority, self-healing runs.**

- **GPU busy heartbeat** (`worker/lib/gpu_heartbeat.py`) — the single "GPU is
  doing work right now" signal. The worker stamps `gpu:last_activity[:iid]` on
  task pop and every ~30s while processing (covers long subprocesses with an
  empty queue). Readers fail safe: a Redis error reads as *busy*.
- **CPU orchestrator = the single, pipeline-aware stop authority.** It stops the
  active GPU only when there is genuinely no work, judged by `_pipeline_has_work`:
  queued GPU tasks **OR** a fresh heartbeat **OR** an in-flight asset_run that
  still owes GPU work. Idle-stop window: **22 min** (`infra.IDLE_STOP_SECONDS`).
- **`_run_has_gpu_work(doc)`** in `app/routes/asset_run_routes.py` is the **single
  source of truth** the orchestrator and the state machine share, so the stop
  decision can never drift from the pipeline. It returns **False** once every GPU
  response is in (success *or* failure) and nothing more will be enqueued (so a
  hard-failed run can't hold the GPU forever), and **True** in healthy
  between-stage gaps (about to fan out / about to rig) so the box is never
  stopped mid-pipeline.
- **GPU-side `auto_shutdown.py` → heartbeat-aware ~45 min safety net** only
  (catches a dead CPU orchestrator); never the thing that ends a normal run.
- **Stuck-stage recovery** in `_refresh`: a stage stuck `queued`/`running` past
  its per-stage timeout (`infra.STAGE_TIMEOUT_SECONDS`) **with no fresh
  heartbeat** is re-enqueued via the proper `queue_*` helper; retry-capped
  (`STAGE_MAX_RETRIES=2`); after the cap a 3D generator is left `error` (run can
  finish with another) and image/rig failure fails the run cleanly. Doubles as
  resume-on-restart.
- **Idempotent fan-out / rig** — atomic Mongo claim (`_model3d_queued` /
  `_rig_queued` flip with `$ne` guard) so concurrent reconciler+GET refreshes
  enqueue exactly once.
- **Terminal-always**: `_refresh` now fails the run on a hard image/rig error so
  a run always reaches `complete|failed` and stops holding the GPU.

---

## 3. Files changed

**New**
- `worker/lib/gpu_heartbeat.py` — `touch()`, `seconds_since()`, `is_busy()` (pure; redis-in, fail-safe).
- `tests/conftest.py` — sets test-env defaults so the suite runs without infra.
- `tests/test_gpu_heartbeat.py`, `tests/test_orchestrator_idle.py`, `tests/test_asset_run_recovery.py`.

**Modified**
- `worker/workers/manual_gen_worker.py` — heartbeat touch on pop + daemon thread every 30s while processing.
- `worker/workers/auto_shutdown.py` — heartbeat-aware "busy"; default threshold 15→45 min (safety net).
- `worker/gpu_main.py` — watch the real queue (`resolve_active_queue()`), not the literal `manual_gen_tasks`.
- `app/services/orchestrator_service.py` — `_pipeline_has_work()`, `_has_inflight_asset_run()` (uses `_run_has_gpu_work`); pipeline-aware 22-min idle-stop; `_honor_stop_requests` gated on work.
- `app/routes/asset_run_routes.py` — `_run_has_gpu_work()`, `_recover_stuck_stages()`, atomic fan-out/rig claims, image/rig hard-fail → run failed, `_retries` field.
- `app/infra.py` — `IDLE_STOP_SECONDS=1320`, `HEARTBEAT_FRESH_SECONDS=120`, `STAGE_TIMEOUT_SECONDS`, `STAGE_TIMEOUT_DEFAULT`, `STAGE_MAX_RETRIES`.

**Key Redis keys**
- `gpu:last_activity`, `gpu:last_activity:<iid>` — heartbeat (TTL 1800s).
- `autoshutdown:enabled[:<iid>]` (`1`/`0`), `autoshutdown:idle_minutes[:<iid>]` — GPU safety-net config.
- `prewarm:ready:<iid>`, `gpu:active_instance_id`, `manual_gen_tasks_spot` (active queue).

---

## 4. Infra / hosts (authoritative: `spark/LOCAL_INFRA.md`)

| Role | name | instance-id | addr | user | path |
|---|---|---|---|---|---|
| CPU (FastAPI/Redis/Mongo/orchestrator) | spark_cpu_1 | i-0f5a6edd3ce343281 | 18.207.13.85 | ubuntu | /home/ubuntu/spark |
| GPU on-demand (g7e/Blackwell) | spark_gpu_high | i-05e8570023728c112 | 52.91.128.47 (EIP) | ec2-user | /home/ec2-user/spark | **normally STOPPED** |
| GPU spot (g7e) | spark_gpu_spot_high | i-09fca0acb4cc429f7 | EIP on start | ec2-user | /home/ec2-user/spark | STOPPED; capacity scarce |

- SSH key: `/Users/lekhaj/Documents/us_cpu_key.pem` (works for CPU + both GPU boxes).
- Redis at `172.31.26.6:6379` (password in CPU `.env*` → `REDIS_PASSWORD`).
- Mongo `World_builder` (asset_runs / manual_gen_stage_runs); `cyclezero`.
- **g7e boxes have NO `spark-bootstrap`** — the repo is updated manually
  (git pull or scp) and units restarted by hand. **Never edit code on the live
  GPU box** (would be overwritten / drift).

---

## 5. Deploy (how this is shipped)

```bash
# 1. Push (already done): origin = github.com/lekhaj/spark, branch main
git push origin main

# 2. CPU box — pull + restart FastAPI (orchestrator/reconciler/result_consumer)
ssh -i ~/Documents/us_cpu_key.pem ubuntu@18.207.13.85 \
  'cd ~/spark && git pull origin main && sudo systemctl restart fastapi_app.service'

# 3. GPU box — its git tree is divergent/dirty, so scp the 4 worker files
#    (start the box first; it is normally stopped):
aws ec2 start-instances --instance-ids i-05e8570023728c112 --region us-east-1
KEY=~/Documents/us_cpu_key.pem; H=ec2-user@52.91.128.47
for f in worker/lib/gpu_heartbeat.py worker/workers/manual_gen_worker.py \
         worker/workers/auto_shutdown.py worker/gpu_main.py; do
  scp -i $KEY spark/$f $H:/home/ec2-user/spark/$f
done
ssh -i $KEY $H 'sudo systemctl restart --no-block manual_gen_worker.service'

# 4. (Re)set autoshutdown config in Redis (run on CPU box):
PW=$(grep -hoE 'REDIS_PASSWORD=.*' ~/spark/.env* | head -1 | cut -d= -f2- | tr -d '"')
RC="redis-cli -h 172.31.26.6 -a $PW --no-auth-warning"
$RC SET autoshutdown:enabled 1
$RC SET autoshutdown:enabled:i-05e8570023728c112 1
$RC SET autoshutdown:idle_minutes 45            # GPU safety net
$RC SET autoshutdown:idle_minutes:i-05e8570023728c112 45
$RC DEL autoshutdown:stop_requested:i-05e8570023728c112
```

CPU primary idle-stop (22 min) comes from `infra.IDLE_STOP_SECONDS` (code, not Redis).

---

## 6. Verify

```bash
# Unit (local venv .venv-test, or the box venv):
cd spark && ./.venv-test/bin/python -m pytest \
  tests/test_gpu_heartbeat.py tests/test_orchestrator_idle.py \
  tests/test_asset_run_recovery.py tests/test_stage_affinity.py \
  tests/test_cyclezero_generation.py -q          # expect: 30 passed (+ new edge tests)

# Live end-to-end (costs GPU + ~25 min; needs a fresh character):
#   POST /cyclezero/games/diablo-2/entities/<key>/generate   (e.g. zombie-brute)
#   watch: gpu:last_activity ticks through the 3D subprocess (box stays up),
#          run completes image→3D→choose→rig→write-back hands-free.
# Recovery drill: mid-3D `aws ec2 stop-instances …` the box → orchestrator
#   restarts it (work outstanding) and _refresh re-enqueues the lost stage.
# Idle drill: after completion, box stops ~22 min later (orchestrator logs the
#   idle timer starting only after _pipeline_has_work() goes false).
```

Heartbeat live check:
```bash
$RC GET gpu:last_activity   # epoch; should advance every ~30s during a task
```

---

## 7. STATUS at handoff

- ✅ Code committed + pushed: `6bc7d9f` (core) + `75eeaa2` (edge-case fix).
- ✅ Unit tests: **31 passing** (heartbeat, orchestrator truth table,
  recovery/retry-cap/idempotency, `_run_has_gpu_work` truth table,
  image/rig hard-fail; + existing stage-affinity & cyclezero suites).
- ✅ CPU deployed (`75eeaa2`), FastAPI restarted clean.
- ✅ GPU worker files scp'd + byte-compiled on the box; autoshutdown re-enabled
  (45 min net / 22 min CPU primary), incident overrides cleared.
- ⏳ **Full live end-to-end + recovery + idle drills NOT yet run** — they require
  generating a fresh asset (GPU $ + ~25 min). The logic is covered by the 31
  unit tests; run the live drills (§6) on the next real generation to confirm
  end-to-end on hardware. `zombie-brute` is a clean (glb=False) candidate.

---

## 8. Still pending / follow-ups

- **#65 Warm-batch P2-2b** (persistent inference servers for hunyuan3d/pixal3d) —
  the biggest remaining latency win (each 3D stage currently reloads its conda
  env + weights per task, ~5–7 min). hunyuan3d server is in-repo doable; pixal3d
  is blocked (its `inference.py` is box-local under `~/Pixal3D`, and we never
  edit code on the live GPU box).
- Run the live drills (§6) once and record results here.
- Consider lowering per-stage timeouts once P2-2b lands (warm stages finish in
  seconds, so "lost" can be detected sooner).

---

## 9. Hard constraints (do not violate)

- Bedrock-only for any LLM; never call the Anthropic API directly.
- Never commit/print `.env.secrets` / `.env.cloudflare` / `.env.gpu`.
- Never edit code on the live GPU box (bootstrap absent on g7e → manual scp + restart).
- Never reduce content/quality for cost: always run all 3 generators
  (trellis + pixal3d + hunyuan3d). Recovery re-runs a lost generator rather than
  skipping; it only gives up on a single generator after the retry cap.
- Git commit footer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

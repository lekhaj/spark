# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- `uvicorn app.main:app --host 0.0.0.0 --port 8000` — FastAPI backend (prod: `fastapi_app.service`).
- `python app/gradio/web_app.py` — operator Gradio UI (prod: `gradio_app.service`; unit files snapshotted in `infra/systemd/cpu/`).
- `pytest` — run the suite (config in `pytest.ini`, tests in `tests/`; legacy-pipeline tests live in `legacy/tests/` and are not collected).
- `bash scripts/smoke_manual_gen_api.sh` — smoke the `/manual-gen/*` REST surface.
- `bash scripts/start_cpu_worker.sh` — start the CPU Celery worker (activates `conda env txt23d`, loads `.env.cpu`).
- `bash quick_start_gpu_worker.sh` / `setup_gpu_environment.sh` — GPU-side worker bootstrap helpers (run on the GPU host, not locally).

Env files: source as `set -a && source .env.cpu && source .env.secrets && set +a` (CPU) or `.env.gpu` (GPU). `.env.cpu` and `.env.gpu` are tracked; `.env.secrets` is gitignored.

## Deploy & infra (READ BEFORE SSH)

See `LOCAL_INFRA.md` for the authoritative SSH/host table (CPU `18.207.13.85`, GPU on-demand `spark_gpu` `52.91.128.47` `i-089872baa8e109ca3`, normally **stopped** — start before a GPU run; key at `/Users/lekhaj/Documents/us_cpu_key.pem`). `deploy/README.md` documents the **bootstrap** workflow — the git repo is the only source of truth on the on-demand GPU instance; **never edit code on the live instance**. (The GPU was a spot box historically; it is now on-demand. The bootstrap mechanics are unchanged — only the lifecycle differs.)

**Adding a new GPU stage** is a 7-step workflow (model → handler → schema → UI → installer → spec.yaml → push). The full sequence is in `deploy/README.md` — follow it exactly, including adding the stage name to `worker/lib/manual_gen_schema.py:STAGE_NAMES` and writing `deploy/install/<stage>.sh`. The bootstrap service auto-installs on next boot.

CPU deploy: `git pull && sudo systemctl restart gradio_app.service fastapi_app.service`. GPU deploy: usually automatic via `spark-bootstrap.service`; manual override `sudo systemctl restart spark-bootstrap`.

## Architecture

Two-tier system across two EC2 instances, glued by Redis (queue) and MongoDB (state).

### CPU tier (`app/`)
FastAPI + Gradio. Owns orchestration, persistence, REST surface; no GPU work.

- **`app/main.py`** — FastAPI app factory + lifespan. Mounts routers from `app/routes/`.
- **`app/routes/manual_gen_routes.py`** — public REST surface consumed by `spark_studio` (the frontend repo at `../spark_studio`). Endpoints under `/manual-gen/*`. **The frontend's `src/lib/api.ts` types mirror the Pydantic models here — keep them in sync.**
- **`app/routes/`** — also: `aws_routes`, `mongo_routes`, `orchestrator_router`, `gpu_orchestrator_routes`.
- **`app/services/`** — `orchestrator_service` (job orchestration loop), `result_consumer` (drains GPU results back into Mongo), `aws_service` / `spot_gpu_service` (EC2 lifecycle, autoshutdown), `mongo_service`, `redis_service`.
- **`app/gradio/pages/`** — operator UI (biome inspector, generation studio, decimation, rigging, pipeline dashboard).
- **`app/src_biome_gen/`** — biome generator + DB module used by the Gradio biome flows.

### GPU tier (`worker/`)
Pulls tasks from Redis queues, runs models, writes results back. Runs on the on-demand GPU instance (`spark_gpu`).

- **`worker/run_manual_worker.py`** — main entry point for the manual-gen worker (started by `manual_gen_worker.service`).
- **`worker/workers/manual_gen_worker.py`** — dispatch by stage to model handlers.
- **`worker/models/`** — per-model wrappers (SD1.5+ControlNet, Hunyuan3D-2.0, TRELLIS, flux_pose, rigging, etc.). The heaviest workers (TRELLIS, Hunyuan3D) live in the companion repo `../spark_gpu`.
- **`worker/lib/manual_gen_queue.py`** — **pure** Mongo-write + Redis-push functions, no Gradio imports. Shared between the Gradio handlers, the FastAPI routes, and CLI/cron callers. All three call into the same queue layer.
- **`worker/lib/manual_gen_schema.py`** — stage names and Mongo doc shapes (canonical).
- **`worker/lib/autoshutdown_ctl.py`** — GPU autoshutdown; **must** use IMDS-resolved instance-id (never hardcoded — see commit `d889b0e`).
- **`worker/lib/gpu_launcher.py`** — CPU-side starter for the GPU spot instance (retries on spot capacity errors — see `1e902f0`).

### Stage pipeline

`flux → normalize → flux_pose → sd_tpose → trellis → pixal3d → hunyuan3d → mesh_lod → rig`. Each `char_label` accumulates versioned runs (`major.minor`) per stage. Mongo collections: `manual_gen_characters`, `manual_gen_stage_runs`, plus `biomes` and `character_specs`. Redis queues are per-target (e.g. `manual_gen_tasks_spot` for the L40S spot box — see `GPU_INSTANCE_MAP` in `.env.cpu`).

### flux_pose specifics

A bundled canonical OpenPose T-pose/A-pose lives in `worker/controlnet_refs/` (committed). UI exposes an explicit mode selector: **bundled / preset / source**. Bundled = use the committed skeleton. Don't reintroduce the auto-detection logic that was removed in the `flux_pose` cleanup commits.

## Conventions

- **Don't add gradio imports under `worker/lib/`.** That layer is shared with FastAPI/CLI and must stay UI-free.
- **Schema source of truth = `worker/lib/manual_gen_schema.py`.** Frontend types in `../spark_studio/src/lib/api.ts` mirror it; if you change shapes here, update there too.
- **`.env.cpu` / `.env.gpu` are tracked** for non-secret config (instance IDs, queue maps, hosts). Secrets go in `.env.secrets` (gitignored). Quote values that contain spaces (see `01c4b2e`).
- **No code edits on the live GPU instance** — the bootstrap will overwrite on next boot.
- All historical/prototype code is archived under `legacy/` (old `src/` Gradio pipeline, root experiment scripts, old docs, old tests). Nothing imports it — never add new imports against it. New work goes in `app/`, `worker/`, `scripts/`, `tests/`.
- `Hunyuan3D-2.1/` and `3D_pipeline_compress/` are untracked local-only dirs (gitignored) — the GPU box clones Hunyuan3D from GitHub at bootstrap.

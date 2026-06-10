# Spark — generation pipeline backend

Two-tier asset-generation system across two EC2 instances, glued by Redis
(task queues) and MongoDB (state). The frontend is the companion repo
[`spark_studio`](../spark_studio); the heavy GPU model repos live in
[`spark_gpu`](../spark_gpu).

```
spark_studio (React)  ──REST──▶  app/ FastAPI  ──Redis──▶  worker/ (GPU spot)
Gradio operator UI   ──────────▶  (same queue layer)        │
        ▲                              │                    ▼
        └────────── MongoDB ◀── result_consumer ◀───── results queue
```

## CPU tier (`app/`) — orchestration box

- `app/main.py` — FastAPI app (`fastapi_app.service`). Routers from
  `app/routes/`; lifespan starts the orchestrator loop + result consumer.
- `app/routes/manual_gen_routes.py` — REST surface consumed by spark_studio
  (`/manual-gen/*`). Frontend types in `../spark_studio/src/lib/api.ts`
  mirror these models — keep in sync.
- `app/gradio/web_app.py` — operator Gradio UI (`gradio_app.service`):
  generation studio, biome tools, decimation, rigging, pipeline dashboard
  (pages under `app/gradio/pages/`).
- `app/services/` — orchestrator, result consumer, AWS/spot lifecycle,
  Mongo/Redis helpers.

## GPU tier (`worker/`) — spot instance

- `worker/run_manual_worker.py` — manual-gen worker entry
  (`manual_gen_worker.service`).
- `worker/lib/` — **UI-free** queue/schema layer shared by Gradio, FastAPI
  and CLI. `manual_gen_schema.py` is the canonical stage list.
- Stage pipeline: `flux → normalize → flux_pose → sd_tpose → trellis →
  pixal3d → hunyuan3d → mesh_lod → rig` (versioned runs per `char_label`).

## Run

```bash
# FastAPI (local)
set -a && source .env.cpu && source .env.secrets && set +a
uvicorn app.main:app --reload

# Operator Gradio UI
python -m app.gradio.web_app

# Tests
pytest
```

## Deploy

- CPU box: `git pull && sudo systemctl restart gradio_app.service fastapi_app.service`
- GPU spot: automatic via `spark-bootstrap.service` on boot — see
  `deploy/README.md` (git is the only source of truth on the instance).
- Host/SSH table: `LOCAL_INFRA.md` (untracked).

## Repo layout

| Path | What |
|------|------|
| `app/` | FastAPI + operator Gradio UI (live) |
| `worker/` | GPU workers + pure queue/schema lib (live) |
| `deploy/` | Spot bootstrap + per-stage installers (live) |
| `scripts/`, `infra/`, `tools/`, `AWS_Scripts/` | ops helpers |
| `tests/` | live-code smoke tests |
| `legacy/` | archived prototype (old Gradio pipeline) — see `legacy/README.md` |

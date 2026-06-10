# legacy/

Archived code from the original single-box prototype (SDXL-Turbo + Gradio
`merged_gradio_app`, Celery tasks, root-level experiment scripts). **Nothing in
this folder is imported by the live system** — kept only for reference.

The live system is:

- `app/` — FastAPI (`fastapi_app.service`) + operator Gradio UI
  (`app/gradio/web_app.py`, `gradio_app.service`)
- `worker/` — GPU-side workers (spot instance, `spark-bootstrap.service`)
- `deploy/`, `scripts/`, `infra/` — deploy & ops

Contents:

- `src/` — old merged Gradio pipeline (superseded by `app/gradio/`)
- `main.py`, `viewer.py` — old launchers
- `tests/` — tests for the old `src/` pipeline (none covered live code)
- `app_pages/` — old duplicate of `app/gradio/pages/`
- `orchestrate_biome_img_3d.py` — superseded by `app/routes/orchestrator_router.py`
- `docs_old/` — docs describing the old pipeline
- root `test_*.py`, one-off scripts, `my_scripts/`, `examples/` — experiments

Safe to delete this entire folder at any time; git history preserves it.

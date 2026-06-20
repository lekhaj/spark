import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import redis
import uuid
import asyncio
from app.routes import mongo_routes 
from app.services.orchestrator_service import orchestrator_main
from app.services.result_consumer import result_consumer_main
from app.services.mongo_service import get_db
from app.config import settings
from app.routes.mongo_routes import router as mongo_router
from app.routes.aws_routes import router as aws_router
from app.routes.orchestrator_router import router as orchestrator_router
from app.routes.manual_gen_routes import router as manual_gen_router
from app.routes.schema_routes import router as schema_router
from app.routes.spec_gen_routes import router as spec_gen_router
from app.routes.journey_routes import router as journey_router
from app.routes.refiner_routes import router as refiner_router
from app.routes.asset_run_routes import router as asset_run_router
from app.routes.access_routes import router as access_router
from app.routes.usage_routes import router as usage_router
from app.cyclezero.routes import router as cyclezero_router

# Bring in the generator and DB helpers from the biome package
from app.src_biome_gen import database as db_module
from app.src_biome_gen.biome_generator import create_new_biome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Redis client using config
try:
    r = redis.Redis.from_url(settings.CELERY_BROKER_URL)
    r.ping()
    print(f"[FastAPI] Connected to Redis at {settings.CELERY_BROKER_URL}")
except Exception as e:
    print(f"[FastAPI] Failed to connect to Redis: {e}")

# Use lifespan context manager for better startup/shutdown handling
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up FastAPI application...")

    # Test DB connection on startup
    db = get_db()
    if db is not None:
        logger.info("MongoDB connected on startup.")
    else:
        logger.error("MongoDB connection failed on startup.")

    # CycleZero: ensure Postgres tables exist (no-op when already created).
    try:
        from app.cyclezero.db import init_models as _cz_init
        _cz_init()
        logger.info("CycleZero Postgres tables ready.")
    except Exception as exc:  # noqa: BLE001 — never block startup on the optional DB
        logger.warning("CycleZero DB init skipped: %s", exc)

    # CycleZero: seed the metamodel (layers + relation types) and default per-layer
    # schemas into Mongo if absent. Idempotent; best-effort.
    try:
        from worker.lib import manual_gen_schema as _mgs
        from app.cyclezero import metamodel as _cz_mm
        from app.cyclezero.schema_seeds import seed_schemas as _cz_seed_schemas
        _db = _mgs.get_db()
        _cz_mm.ensure_seeded(_db)
        n = _cz_seed_schemas(_db)
        logger.info("CycleZero metamodel ready; seeded %s schema(s).", n)
    except Exception as exc:  # noqa: BLE001 — never block startup on the optional DB
        logger.warning("CycleZero metamodel/schema seed skipped: %s", exc)

    # Start orchestrator + result consumer as background tasks
    orchestrator_task    = asyncio.create_task(orchestrator_main())
    result_consumer_task = asyncio.create_task(result_consumer_main())
    logger.info("Orchestrator + result consumer background tasks started")

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    for task in (orchestrator_task, result_consumer_task):
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("Background tasks cancelled successfully")

app = FastAPI(
    title="Dual Model Generation API",
    lifespan=lifespan
)

# CORS — allow the spark_studio front-end (Netlify + local dev) to call us.
# Extend MANUAL_GEN_CORS_ORIGINS (comma-separated env) for additional origins
# (e.g. a custom domain) without code changes.
_default_cors_origins = [
    "https://sparkaistudio.netlify.app",
    "https://spark-studio.pages.dev",  # Cloudflare Pages (current prod host)
    "http://localhost:5173",   # vite dev
    "http://localhost:5179",   # studio dev (launch.json)
    "http://localhost:4173",   # vite preview
    "http://127.0.0.1:5173",
]
_extra_origins = [
    o.strip() for o in os.getenv("MANUAL_GEN_CORS_ORIGINS", "").split(",") if o.strip()
] if os.getenv("MANUAL_GEN_CORS_ORIGINS") else []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_default_cors_origins + _extra_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers for Mongo and AWS
app.include_router(mongo_router, prefix="/mongo", tags=["MongoDB"])
app.include_router(aws_router, prefix="/aws", tags=["AWS"])

app.include_router(mongo_routes.router, prefix="")
# Orchestrator endpoints (submit_image_tasks, submit_3d_tasks) are exposed under /orchestrate
app.include_router(orchestrator_router, prefix="/orchestrate", tags=["Orchestrator"])
# Also expose the same orchestrator router at root so legacy clients can POST to /submit_image_tasks/
app.include_router(orchestrator_router, prefix="", tags=["Orchestrator"])
# Manual generation pipeline — CRUD + queue (consumed by spark_studio)
app.include_router(manual_gen_router, prefix="/manual-gen", tags=["ManualGen"])
# Versioned spec schemas (CycleZero T00) — consumed by spark_studio /studio/schemas
app.include_router(schema_router, prefix="/schemas", tags=["Schemas"])
# Versioned JSON-artifact runs (CycleZero T03/T04) — paste-mode spec generation
app.include_router(spec_gen_router, prefix="/spec-gen", tags=["SpecGen"])
# Journeys + impact items (CycleZero U01) — hybrid shell persistence
app.include_router(journey_router, prefix="", tags=["Journeys"])
# Refiner chat proxy (CycleZero U02) — keys stay server-side
app.include_router(refiner_router, prefix="/refiner", tags=["Refiner"])
# asset_spec -> GPU pipeline bridge (CycleZero U05)
app.include_router(asset_run_router, prefix="/asset-runs", tags=["AssetRuns"])
# CycleZero game-authoring backend — games/entities/relations/jobs/contract/match
app.include_router(cyclezero_router, prefix="/cyclezero", tags=["CycleZero"])
# Studio access control — email allowlist + waitlist (invite-only sign-in)
app.include_router(access_router, prefix="/access", tags=["Access"])
app.include_router(usage_router, prefix="", tags=["Usage"])

# --- Biome generation endpoints (moved from c_main.py) ---
# In-memory background task store for lightweight single-node async tasks
_BG_TASKS: dict[str, dict | None] = {}


class BiomeGenerationRequest(BaseModel):
    theme_prompt: Any = Field(
        ...,
        example="A sun-scorched desert planet where giant crystalline cacti harvest lightning."
    )
    system_prompt: Any | None = None


class BiomeGenerationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    biome_name: Optional[str] = None
    biome_document: Optional[dict] = None


class BiomeListResponse(BaseModel):
    biome_names: List[str]


@app.post("/biomes/async/", tags=["Generation"])
async def generate_biome_async(request: BiomeGenerationRequest):
    """Schedule generation in a thread and return a task id immediately."""
    prompt_raw = request.theme_prompt
    system_raw = request.system_prompt
    try:
        if not isinstance(prompt_raw, str):
            import json as _json

            prompt_text = _json.dumps(prompt_raw, ensure_ascii=False)
        else:
            prompt_text = prompt_raw
    except Exception:
        prompt_text = str(prompt_raw)

    task_id = str(uuid.uuid4())
    _BG_TASKS[task_id] = None

    async def _run_and_store():
        try:
            res = await run_in_threadpool(create_new_biome, prompt_text, system_raw)
            _BG_TASKS[task_id] = {
                "success": res.success,
                "message": res.message,
                "biome_name": res.biome_name,
                "biome_document": getattr(res, "biome_document", None),
            }
        except Exception as e:
            _BG_TASKS[task_id] = {"success": False, "message": f"Task failed: {e}"}

    asyncio.create_task(_run_and_store())
    return {"task_id": task_id, "status": "queued"}


@app.get("/biomes/result/{task_id}", tags=["Generation"])
async def get_biome_result(task_id: str):
    if task_id not in _BG_TASKS:
        raise HTTPException(status_code=404, detail="Task id not found")
    value = _BG_TASKS[task_id]
    if value is None:
        return {"status": "pending"}
    return {"status": "success", "result": value}


@app.post("/biomes/", response_model=BiomeGenerationResponse, status_code=201, tags=["Generation"])
async def generate_biome_endpoint(request: BiomeGenerationRequest):
    """Generates a new biome based on a creative theme prompt."""
    try:
        start_time = datetime.now().isoformat()
        prompt_raw = request.theme_prompt
        try:
            if not isinstance(prompt_raw, str):
                import json as _json
                prompt_text = _json.dumps(prompt_raw, ensure_ascii=False)
            else:
                prompt_text = prompt_raw
        except Exception:
            prompt_text = str(prompt_raw)

        logging.info(f"Starting biome inference at {start_time} for theme: '{prompt_text}'")
        result = await run_in_threadpool(create_new_biome, prompt_text, request.system_prompt)

        try:
            import re
            match = re.search(r"'_id': '([a-f0-9\-]+)'", result.message)
            inference_uid = match.group(1) if match else "unknown"
        except Exception:
            inference_uid = "unknown"

        logging.info(f"[INFER-{inference_uid}] Started at {start_time}")
        logging.info(f"[INFER-{inference_uid}] Biome generation result: {result.message}")
        logging.info(f"[INFER-{inference_uid}] Finished at {datetime.now().isoformat()}")
        return {
            "success": result.success,
            "message": result.message,
            "biome_name": result.biome_name,
            "biome_document": getattr(result, "biome_document", None)
        }
    except Exception as e:
        logging.error(f"Exception during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")


@app.post("/biomes/raw", response_model=BiomeGenerationResponse, status_code=201, tags=["Generation"])
async def generate_biome_raw(request: Request):
    try:
        body_bytes = await request.body()
        if not body_bytes:
            raise HTTPException(status_code=400, detail="Empty request body")
        prompt_text = body_bytes.decode("utf-8", errors="replace")
        start_time = datetime.now().isoformat()
        logging.info(f"Starting biome inference (raw) at {start_time} for theme: '{prompt_text}'")
        result = await run_in_threadpool(create_new_biome, prompt_text, None)
        return {
            "success": result.success,
            "message": result.message,
            "biome_name": result.biome_name,
            "biome_document": getattr(result, "biome_document", None),
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error("Exception during raw inference: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")


@app.get("/biomes/", response_model=BiomeListResponse, tags=["Retrieval"])
async def list_biomes_endpoint():
    biome_names = await run_in_threadpool(db_module.get_all_biome_names)
    if biome_names is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve biome list from the database.")
    return {"biome_names": biome_names}


@app.get("/biomes/{biome_name}", tags=["Retrieval"])
async def get_biome_endpoint(biome_name: str):
    biome_data = await run_in_threadpool(db_module.get_biome_by_name, biome_name)
    if biome_data is None:
        raise HTTPException(status_code=404, detail=f"Biome '{biome_name}' not found.")
    return biome_data

@app.get("/queue_status/")
async def queue_status():
    """Check status of both queues"""
    image_queue_length = r.llen("image_tasks")
    model_3d_queue_length = r.llen("model_tasks")
    return {
        "image_tasks": image_queue_length,
        "3d_tasks": model_3d_queue_length,
        "total_pending": image_queue_length + model_3d_queue_length
    }




@app.get("/orchestrate/status")
async def get_orchestrator_status():
    """Get orchestrator status (spot instance based)"""
    from app.services.orchestrator_service import orchestrator

    try:
        status = orchestrator.get_status()
        status["timestamp"] = datetime.now().isoformat()
        return status
    except Exception as e:
        logger.error(f"Error getting orchestrator status: {e}")
        return {
            "auto_mode": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.post("/orchestrate/mode")
async def set_orchestrator_mode_endpoint(auto_mode: bool):
    """Enable/disable auto scaling mode"""
    from app.services.orchestrator_service import orchestrator

    try:
        orchestrator.auto_mode = auto_mode
        logger.info(f"Orchestrator auto mode set to: {auto_mode}")
        return {
            "status": "success",
            "auto_mode": auto_mode,
            "message": f"Auto scaling mode {'enabled' if auto_mode else 'disabled'}"
        }
    except Exception as e:
        logger.error(f"Error setting orchestrator mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to set orchestrator mode: {e}")

@app.post("/orchestrate/start")
async def start_orchestrator_endpoint():
    """Start orchestrator monitoring"""
    from app.services.orchestrator_service import orchestrator

    try:
        orchestrator.auto_mode = True
        logger.info("Orchestrator auto mode enabled")
        return {
            "status": "success",
            "auto_mode": True,
            "message": "Orchestrator auto mode started"
        }
    except Exception as e:
        logger.error(f"Error starting orchestrator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start orchestrator: {e}")

@app.post("/orchestrate/stop")
async def stop_orchestrator_endpoint():
    """Stop orchestrator monitoring"""
    from app.services.orchestrator_service import orchestrator

    try:
        orchestrator.auto_mode = False
        logger.info("Orchestrator auto mode disabled")
        return {
            "status": "success",
            "auto_mode": False,
            "message": "Orchestrator auto mode stopped"
        }
    except Exception as e:
        logger.error(f"Error stopping orchestrator: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop orchestrator: {e}")


@app.get("/orchestrate/autoshutdown")
async def get_autoshutdown():
    """Get current autoshutdown state (enabled + idle_minutes)."""
    from app.services.orchestrator_service import orchestrator
    try:
        return orchestrator.get_autoshutdown_state()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/orchestrate/autoshutdown")
async def set_autoshutdown(enabled: bool, idle_minutes: Optional[int] = None):
    """Toggle GPU autoshutdown. When enabling, idle_minutes sets the idle threshold."""
    from app.services.orchestrator_service import orchestrator
    try:
        orchestrator.set_autoshutdown(enabled, idle_minutes)
        state = orchestrator.get_autoshutdown_state()
        return {"status": "success", **state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

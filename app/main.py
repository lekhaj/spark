import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Optional, List

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

import redis, json, uuid
import time
from enum import Enum
import asyncio
from app.routes import mongo_routes 
from app.services.orchestrator_service import orchestrator_main
from app.services.mongo_service import get_db
from app.config import settings
from app.routes.mongo_routes import router as mongo_router
from app.routes.aws_routes import router as aws_router
from app.routes.orchestrator_router import router as orchestrator_router

# Bring in the generator and DB helpers from the biome package
from app.src_biome_gen import database as db_module
from app.src_biome_gen.biome_generator import create_new_biome

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

## main was edited to make api endpoints as a executable function...
## core_logic is unchanged. :)

def create_image_task_dict(prompt, negative_prompt="", width=1024, height=1024, num_inference_steps=30):
    job_id = str(uuid.uuid4())
    return job_id, {
        "job_id": job_id,
        "task_type": TaskType.IMAGE,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "timestamp": time.time(),
        "status": "queued",
        "output_key": f"generated-images/{job_id}/sdxl_output.png"
    }

def create_3d_task_dict(image_s3_url, prompt=""):
    job_id = str(uuid.uuid4())
    return job_id, {
        "job_id": job_id,
        "task_type": TaskType.MODEL_3D,
        "image_s3_url": image_s3_url,
        "prompt": prompt,
        "timestamp": time.time(),
        "status": "queued",
        "output_key": f"3d_assets/{job_id}_mesh.obj"
    }

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

    # Start orchestrator as background task
    orchestrator_task = asyncio.create_task(orchestrator_main())
    logger.info("Orchestrator background task started")

    yield  # App runs here

    # Shutdown
    logger.info("Shutting down FastAPI application...")
    orchestrator_task.cancel()
    try:
        await orchestrator_task
    except asyncio.CancelledError:
        logger.info("Orchestrator task cancelled successfully")

app = FastAPI(
    title="Dual Model Generation API",
    lifespan=lifespan
)

# Mount routers for Mongo and AWS
app.include_router(mongo_router, prefix="/mongo", tags=["MongoDB"])
app.include_router(aws_router, prefix="/aws", tags=["AWS"])

app.include_router(mongo_routes.router, prefix="")
# Orchestrator endpoints (submit_image_tasks, submit_3d_tasks) are exposed under /orchestrate
app.include_router(orchestrator_router, prefix="/orchestrate", tags=["Orchestrator"])
# Also expose the same orchestrator router at root so legacy clients can POST to /submit_image_tasks/
app.include_router(orchestrator_router, prefix="", tags=["Orchestrator"])

class TaskType(str, Enum):
    IMAGE = "image"
    MODEL_3D = "3d_model"

class ImagePromptRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 30

class Model3DRequest(BaseModel):
    image_s3_url: str
    prompt: str = ""  # Optional prompt for guidance

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

# @app.post("/submit_image_task/")
# async def submit_image_task(image_request: ImagePromptRequest):
#     """Submit a prompt for SDXL image generation"""
#     job_id, task_data = create_image_task_dict(
#         prompt=image_request.prompt,
#         negative_prompt=image_request.negative_prompt,
#         width=image_request.width,
#         height=image_request.height,
#         num_inference_steps=image_request.num_inference_steps
#     )
#     print(task_data)
#     r.lpush("image_tasks", json.dumps(task_data))
#     print(task_data)
#     print(f"[FastAPI] Image task pushed to Redis: {job_id}")
#     return {
#         "status": "success",
#         "message": "Image generation task added to queue",
#         "job_id": job_id,
#         "task_type": TaskType.IMAGE
#     }

# @app.post("/submit_3d_task/")
# async def submit_3d_task(model_request: Model3DRequest):
#     """Submit an S3 image URL for 3D model generation"""
#     job_id, task_data = create_3d_task_dict(
#         image_s3_url=model_request.image_s3_url,
#         prompt=model_request.prompt
#     )
#     r.lpush("model_tasks", json.dumps(task_data))
#     print(f"[FastAPI] 3D task pushed to Redis: {job_id}")
#     return {
#         "status": "success",
#         "message": "3D model generation task added to queue",
#         "job_id": job_id,
#         "task_type": TaskType.MODEL_3D
#     }

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

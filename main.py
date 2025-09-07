import logging
from logging.handlers import RotatingFileHandler
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from starlette.concurrency import run_in_threadpool
from src import database as db
from src.biome_generator import create_new_biome
from src.celery_worker import generate_biome_task, celery_app
import uvicorn

LOG_FILE_NAME = "custom_inference.log"

def setup_logger(log_file: str = LOG_FILE_NAME):
    logger = logging.getLogger("inference_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    # Rotating file handler (append mode)
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=2)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger

logger = setup_logger(LOG_FILE_NAME)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application starting up...")
    # we could pre-load the LLM here if desired, but our lazy-loading approach is also fine.
    # For example: await run_in_threadpool(llm_interface.load_llm_pipeline)
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="Procedural Biome Generator API",
    description="An API to generate complex biome documents using an LLM.",
    version="1.0.0",
    lifespan=lifespan
)


# --- Pydantic Models for API Data Contracts ---
class BiomeGenerationRequest(BaseModel):
    theme_prompt: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        example="A sun-scorched desert planet where giant crystalline cacti harvest lightning."
    )

class BiomeGenerationResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    biome_name: Optional[str] = None
    biome_document: Optional[dict] = None

class BiomeListResponse(BaseModel):
    biome_names: List[str]
    

@app.post("/biomes/async/", tags=["Generation"])
async def generate_biome_async(request: BiomeGenerationRequest):
    task = generate_biome_task.delay(request.theme_prompt)
    return {"task_id": task.id, "status": "queued"}

@app.get("/biomes/result/{task_id}", tags=["Generation"])
async def get_biome_result(task_id: str):
    result = celery_app.AsyncResult(task_id)
    if result.state == "PENDING":
        return {"status": "pending"}
    elif result.state == "SUCCESS":
        return {"status": "success", "result": result.result}
    elif result.state == "FAILURE":
        return {"status": "failure", "error": str(result.info)}
    else:
        return {"status": result.state}


@app.post("/biomes/", response_model=BiomeGenerationResponse, status_code=201, tags=["Generation"])
async def generate_biome_endpoint(request: BiomeGenerationRequest):
    """
    Generates a new biome based on a creative theme prompt.
    """
    try:
        start_time = datetime.now().isoformat()
        logger.info(f"Starting biome inference at {start_time} for theme: '{request.theme_prompt}'")
        result = await run_in_threadpool(create_new_biome, request.theme_prompt)
        # Extract UUID from the generated biome document (in result.message)
        try:
            # Try to extract the UUID from the message string
            import re
            match = re.search(r"'_id': '([a-f0-9\-]+)'", result.message)
            inference_uid = match.group(1) if match else "unknown"
        except Exception:
            inference_uid = "unknown"
        # Log all steps and flow
        logger.info(f"[INFER-{inference_uid}] Started at {start_time}")
        logger.info(f"[INFER-{inference_uid}] Biome generation result: {result.message}")
        logger.info(f"[INFER-{inference_uid}] Finished at {datetime.now().isoformat()}")
        return {
            "success": result.success,
            "message": result.message,
            "biome_name": result.biome_name,
            "biome_document": getattr(result, "biome_document", None)
        }
    except Exception as e:
        logger.error(f"Exception during inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An unexpected internal server error occurred.")

@app.get("/biomes/", response_model=BiomeListResponse, tags=["Retrieval"])
async def list_biomes_endpoint():
    """
    Retrieves a list of all unique biome names available in the database.
    """
    biome_names = await run_in_threadpool(db.get_all_biome_names)
    if biome_names is None:
        raise HTTPException(status_code=500, detail="Failed to retrieve biome list from the database.")
    return {"biome_names": biome_names}


@app.get("/biomes/{biome_name}", tags=["Retrieval"])
async def get_biome_endpoint(biome_name: str):
    """
    Fetches the complete data for a specific biome by its name.
    """
    biome_data = await run_in_threadpool(db.get_biome_by_name, biome_name)
    if biome_data is None:
        raise HTTPException(status_code=404, detail=f"Biome '{biome_name}' not found.")
    return biome_data


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
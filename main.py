# main.py - FastAPI Backend

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from celery.result import AsyncResult
from celery import chain
from app import (
    celery_app,
    generate_2d_image_task,
    generate_3d_from_2d_task,
    decimate_3d_task
)
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

app = FastAPI(title="AI-Powered 3D Asset Generation API")

# Define a Pydantic model for the request body
class FullPipelineRequest(BaseModel):
    biome_name: str
    theme_prompt: str
    s3_bucket_name: str
    unique_id: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the 3D Asset Generation API!"}

@app.post("/generate-full-pipeline")
def generate_full_pipeline(request: FullPipelineRequest):
    """
    Kicks off the entire generation and decimation pipeline using a Celery task chain.
    """
    logging.info(f"Received request to start full pipeline for: {request.theme_prompt}")

    # Use a Celery chain to link the tasks together
    # The output of the first task (image bytes) is passed as input to the next task, and so on.
    task_chain = chain(
        generate_2d_image_task.s(
            text_prompt=request.theme_prompt,
            s3_bucket_name=request.s3_bucket_name,
            base_filename=f"2d_{request.unique_id}"
        ),
        generate_3d_from_2d_task.s(
            s3_bucket_name=request.s3_bucket_name,
            base_filename=f"3d_{request.unique_id}"
        ),
        decimate_3d_task.s(
            s3_bucket_name=request.s3_bucket_name,
            base_filename=f"decimated_{request.unique_id}"
        )
    )

    try:
        # Launch the chain
        result = task_chain.delay()
        logging.info(f"Celery task chain launched with task ID: {result.id}")
        return {"task_id": result.id, "status": "PENDING"}
    except Exception as e:
        logging.error(f"Failed to launch Celery task chain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    """
    Polls the status of a Celery task by its ID.
    This endpoint is used by the Gradio frontend to provide real-time updates.
    """
    task_result = AsyncResult(task_id, app=celery_app)

    if task_result is None:
        raise HTTPException(status_code=404, detail="Task not found")

    status = task_result.status
    result = task_result.result if task_result.successful() else None
    
    # Custom status handling for the Gradio frontend
    # This relies on the Celery tasks to correctly update their state with a custom status.
    if status == 'SUCCESS':
        return {
            "status": status,
            "result": result
        }
    elif status == 'FAILURE':
        return {
            "status": status,
            "error": str(task_result.result)
        }
    else:
        # For PENDING, IN_PROGRESS_2D, IN_PROGRESS_3D, IN_PROGRESS_DECIMATION, etc.
        # This relies on the Celery tasks to correctly update their state with a custom status.
        return {
            "status": task_result.info.get('status', 'PENDING'),
            "result": task_result.info.get('result', None)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

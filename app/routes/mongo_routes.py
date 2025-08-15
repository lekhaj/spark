from fastapi import APIRouter, HTTPException
from app.services.mongo_service import ping_db, get_data
from app.services.redis_service import enqueue_task
import uuid
from datetime import datetime
from pydantic import BaseModel

router = APIRouter()

class GenerateRequest(BaseModel):
    prompt: str
#for checkig connection to mongo
@router.get("/check-connection")
def check_connection():
    try:
        return {"status": "success", "mongo_status": ping_db()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#/view-data/biome to view data
@router.get("/view-data/{collection_name}")
def view_data(collection_name: str):
    try:
        data = get_data(collection_name)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
 #main route need to be completed . for creating  task update on mongo and redis and do other things like 
 # grid and biome generation and sending task from redis to gpu
@router.post("/generate")
def generate_asset(req: GenerateRequest):
    try:
        task_id = str(uuid.uuid4())

        task_data = {
            "id": task_id,
            "prompt": req.prompt,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat()
        }

        # Redis queue (worker_hunyuan will process it)
        enqueue_task(task_data)

        return {"task_id": task_id, "status": "queued", "message": "Task sent to GPU worker"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
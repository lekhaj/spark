from fastapi import APIRouter, HTTPException
from app.services.mongo_service import ping_db, get_data, get_biome,get_assets_by_biome,get_db
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
    
@router.get("/biome/{biome_id}")
def read_biome(biome_id: str):
    """
    Fetch a single biome document by its biome_id field.
    """
    try:
        biome = get_biome(biome_id)
        if not biome:
            raise HTTPException(status_code=404, detail="Biome not found")
        # strip _id if needed
        biome.pop("_id", None)
        return {"biome": biome}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
       
# Create GPU generation task
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


from app.services.mongo_service import  extract_all_images_from_biome

@router.get("/biome-images/{biome_id}")
def get_biome_images(biome_id: str):
    """Get all images from a specific biome"""
    try:
        images = extract_all_images_from_biome(biome_id)
        return {
            "biome_id": biome_id,
            "images": images,
            "count": len(images)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/all-images")
def get_all_images():
    """Get all images from all biomes"""
    try:
        db = get_db()
        if db is None:
            raise HTTPException(status_code=500, detail="Database not connected")
        
        all_images = []
        biomes = db["biomes"].find({})
        
        for biome in biomes:
            biome_id = biome.get("_id")
            biome_name = biome.get("biome_name", "Unknown")
            images = extract_all_images_from_biome(biome_id)
            
            for img in images:
                img["biome_id"] = biome_id
                img["biome_name"] = biome_name
                all_images.append(img)
        
        return {"images": all_images, "total": len(all_images)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
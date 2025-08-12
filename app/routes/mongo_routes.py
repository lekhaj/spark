from fastapi import APIRouter, HTTPException
from app.services.mongo_service import ping_db, get_data

router = APIRouter()

@router.get("/check-connection")
def check_connection():
    try:
        return {"status": "success", "mongo_status": ping_db()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/view-data/{collection_name}")
def view_data(collection_name: str):
    try:
        data = get_data(collection_name)
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

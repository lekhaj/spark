from fastapi import APIRouter
from app.services.orchestrator_service import orchestrator

router = APIRouter(prefix="/orchestrator", tags=["Orchestrator"])

@router.get("/status")
async def get_orchestrator_status():
    """Get orchestrator status"""
    return orchestrator.get_status()

@router.post("/mode")
async def set_orchestrator_mode(auto_mode: bool):
    """Enable/disable auto scaling"""
    orchestrator.auto_mode = auto_mode
    return {
        "auto_mode": auto_mode,
        "message": f"Auto mode {'enabled' if auto_mode else 'disabled'}"
    }

@router.post("/start")
async def start_orchestrator():
    """Start orchestrator monitoring"""
    orchestrator.auto_mode = True
    return {"status": "success", "message": "Orchestrator started"}

@router.post("/stop")
async def stop_orchestrator():
    """Stop orchestrator monitoring"""
    orchestrator.auto_mode = False
    return {"status": "success", "message": "Orchestrator stopped"}

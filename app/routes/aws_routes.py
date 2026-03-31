from fastapi import APIRouter, HTTPException
from app.services.aws_service import (
    start_instance, stop_instance, get_instance_state,
    start_gpu_worker, stop_gpu_worker
)
from app.services.orchestrator_service import orchestrator

router = APIRouter(prefix="/aws", tags=["AWS Control"])

@router.post("/start/{instance_id}")
async def start_ec2(instance_id: str):
    """Start EC2 instance"""
    success = start_instance(instance_id)
    if success:
        return {"status": "success", "message": f"Instance {instance_id} started"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to start instance {instance_id}")

@router.post("/stop/{instance_id}")
async def stop_ec2(instance_id: str):
    """Stop EC2 instance"""
    success = stop_instance(instance_id)
    if success:
        return {"status": "success", "message": f"Instance {instance_id} stopped"}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to stop instance {instance_id}")

@router.get("/status/{instance_id}")
async def get_instance_status(instance_id: str):
    """Get instance status"""
    state = get_instance_state(instance_id)
    return {"instance": instance_id, "state": state}

# GPU-specific endpoints
@router.post("/gpu/t4/start")
async def start_gpu_t4():
    """Start GPU T4 with worker"""
    if orchestrator.gpu_states["gpu_t4"] == "running":
        return {"status": "error", "message": "GPU T4 is already running"}

    success = start_instance("gpu_t4") and start_gpu_worker("gpu_t4")
    if success:
        orchestrator.gpu_states["gpu_t4"] = "running"
        return {"status": "success", "message": "GPU T4 started with worker"}
    else:
        orchestrator.gpu_states["gpu_t4"] = "stopped"
        raise HTTPException(status_code=500, detail="Failed to start GPU T4")

@router.post("/gpu/t4/stop")
async def stop_gpu_t4():
    """Stop GPU T4"""
    if orchestrator.gpu_states["gpu_t4"] == "stopped":
        return {"status": "error", "message": "GPU T4 is already stopped"}

    success = stop_instance("gpu_t4")
    if success:
        orchestrator.gpu_states["gpu_t4"] = "stopped"
        return {"status": "success", "message": "GPU T4 stopped"}
    else:
        orchestrator.gpu_states["gpu_t4"] = "running"
        raise HTTPException(status_code=500, detail="Failed to stop GPU T4")

@router.post("/gpu/a10/start")
async def start_gpu_a10():
    """Start GPU A10 with worker"""
    if orchestrator.gpu_states["gpu_a10"] == "running":
        return {"status": "error", "message": "GPU A10 is already running"}

    success = start_instance("gpu_a10") and start_gpu_worker("gpu_a10")
    if success:
        orchestrator.gpu_states["gpu_a10"] = "running"
        return {"status": "success", "message": "GPU A10 started with worker"}
    else:
        orchestrator.gpu_states["gpu_a10"] = "stopped"
        raise HTTPException(status_code=500, detail="Failed to start GPU A10")

@router.post("/gpu/a10/stop")
async def stop_gpu_a10():
    """Stop GPU A10"""
    if orchestrator.gpu_states["gpu_a10"] == "stopped":
        return {"status": "error", "message": "GPU A10 is already stopped"}

    success = stop_instance("gpu_a10")
    if success:
        orchestrator.gpu_states["gpu_a10"] = "stopped"
        return {"status": "success", "message": "GPU A10 stopped"}
    else:
        orchestrator.gpu_states["gpu_a10"] = "running"
        raise HTTPException(status_code=500, detail="Failed to stop GPU A10")

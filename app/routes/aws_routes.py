from fastapi import APIRouter, HTTPException
from app.services.aws_service import (
    start_instance, stop_instance, get_instance_state,
)
from worker.lib import gpu_launcher

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

# ── GPU lifecycle (g7e, spot-first / on-demand-fallback) ──────────────────────
# A single logical GPU. Bring-up is owned by gpu_launcher's ladder (spot first,
# on-demand on capacity loss); shutdown stops the active box. These replace the
# old per-card (/gpu/t4, /gpu/a10) endpoints from the multi-GPU era.

@router.post("/gpu/start")
async def start_gpu():
    """Bring a GPU online (spot-first, on-demand fallback). Force-runs the ladder
    even if GPU_AUTO_LAUNCH is off."""
    import os
    os.environ.setdefault("GPU_AUTO_LAUNCH", "1")
    ok, reason = gpu_launcher.ensure_gpu_ready()
    if ok:
        return {"status": "success", "reason": reason,
                "instance_id": gpu_launcher.get_active_instance_id()}
    raise HTTPException(status_code=500, detail=f"Could not bring GPU online: {reason}")

@router.post("/gpu/stop")
async def stop_gpu():
    """Stop the currently-active GPU box (spot or on-demand)."""
    ok, reason = gpu_launcher.stop_gpu(force=True)
    if ok:
        return {"status": "success", "reason": reason,
                "instance_id": gpu_launcher.get_active_instance_id()}
    raise HTTPException(status_code=500, detail=f"Could not stop GPU: {reason}")

from fastapi import APIRouter
from app.services.aws_service import start_instance, stop_instance

router = APIRouter(prefix="/aws", tags=["AWS Control"])

#/aws/start/gpu or /aws/start/cpu or /aws/start/instance id
#similar for stoping 
#will need to update it to orchastrator which will automatically on/off instance based on need
@router.post("/start/{instance_id}")
def start_ec2(instance_id: str):
    return start_instance(instance_id)

@router.post("/stop/{instance_id}")
def stop_ec2(instance_id: str):
    return stop_instance(instance_id)

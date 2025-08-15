import boto3
import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

INSTANCE_CPU = os.getenv("CPU")
INSTANCE_GPU = os.getenv("GPU")

ec2 = boto3.client(
    "ec2",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

def instance(instance_name_or_id: str) -> str:
    """Return actual instance ID from alias or direct ID."""
    if instance_name_or_id.lower() == "cpu":
        return INSTANCE_CPU
    elif instance_name_or_id.lower() == "gpu":
        return INSTANCE_GPU
    return instance_name_or_id 

def start_instance(instance_name_or_id: str):
    instance_id = instance(instance_name_or_id)
    return ec2.start_instances(InstanceIds=[instance_id])

def stop_instance(instance_name_or_id: str):
    instance_id = instance(instance_name_or_id)
    return ec2.stop_instances(InstanceIds=[instance_id])



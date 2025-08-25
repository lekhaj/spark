import boto3
import os
from dotenv import load_dotenv

load_dotenv()

# AWS and S3 configuration
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")

# Fix: Ensure these variables are defined
INSTANCE_CPU = os.getenv("CPU")
INSTANCE_GPU = os.getenv("GPU")

ec2 = boto3.client(
    "ec2",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

s3_client = boto3.client(
    "s3",
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

def upload_to_s3(file_path: str, object_name: str) -> str:
    """
    Uploads a file to an S3 bucket and returns the public URL.
    
    Args:
        file_path (str): The path to the file to upload.
        object_name (str): The S3 object name (e.g., 'path/to/my-model.obj').
        
    Returns:
        str: The public URL of the uploaded file.
    """
    try:
        s3_client.upload_file(file_path, AWS_S3_BUCKET, object_name)
        # Construct the public URL
        s3_url = f"https://{AWS_S3_BUCKET}.s3.{AWS_REGION}.amazonaws.com/{object_name}"
        return s3_url
    except Exception as e:
        print(f"Error uploading file to S3: {e}")
        return ""
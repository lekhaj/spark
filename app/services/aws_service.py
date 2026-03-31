import boto3
import asyncio
import subprocess
import time
from app.config import settings

# AWS Configuration
ec2 = boto3.client(
    "ec2",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION
)

# Instance Mapping
INSTANCE_MAP = {
    "cpu": getattr(settings, "AWS_CPU_INSTANCE_ID", None),
    "gpu_t4": getattr(settings, "AWS_GPU_T4_INSTANCE_ID", None),
    "gpu_a10": getattr(settings, "AWS_GPU_A10_INSTANCE_ID", None),
}

# GPU SSH Configuration
GPU_CONFIG = {
    "gpu_t4": {
        "ssh_user": getattr(settings, "GPU_T4_SSH_USER", "ubuntu"),
        "public_ip": getattr(settings, "GPU_T4_PUBLIC_IP", None),
        "worker_service": "image-worker"
    },
    "gpu_a10": {
        "ssh_user": getattr(settings, "GPU_A10_SSH_USER", "ubuntu"),
        "public_ip": getattr(settings, "GPU_A10_PUBLIC_IP", None),
        "worker_service": "model-worker"
    }
}

def get_instance_id(instance_name: str) -> str:
    """Get actual instance ID from alias"""
    instance_name = instance_name.lower()
    if instance_name in INSTANCE_MAP:
        return INSTANCE_MAP[instance_name]
    return instance_name  # Assume it's already an instance ID

def start_instance(instance_name: str) -> bool:
    """Start EC2 instance and wait for running state"""
    try:
        instance_id = get_instance_id(instance_name)
        print(f"[AWS] Starting instance: {instance_name} ({instance_id})")

        response = ec2.start_instances(InstanceIds=[instance_id])
        state = response['StartingInstances'][0]['CurrentState']['Name']

        if state == 'pending':
            # Wait for instance to be running
            return wait_for_instance_state(instance_id, 'running')
        return True
    except Exception as e:
        print(f"[AWS ERROR] Failed to start instance {instance_name}: {e}")
        return False

def stop_instance(instance_name: str) -> bool:
    """Stop EC2 instance and wait for stopped state"""
    try:
        instance_id = get_instance_id(instance_name)
        print(f"[AWS] Stopping instance: {instance_name} ({instance_id})")

        response = ec2.stop_instances(InstanceIds=[instance_id])
        state = response['StoppingInstances'][0]['CurrentState']['Name']

        if state == 'stopping':
            # Wait for instance to be stopped
            return wait_for_instance_state(instance_id, 'stopped')
        return True
    except Exception as e:
        print(f"[AWS ERROR] Failed to stop instance {instance_name}: {e}")
        return False

def get_instance_state(instance_name: str) -> str:
    """Get current state of instance"""
    try:
        instance_id = get_instance_id(instance_name)
        response = ec2.describe_instances(InstanceIds=[instance_id])
        return response['Reservations'][0]['Instances'][0]['State']['Name']
    except Exception as e:
        print(f"[AWS ERROR] Failed to get instance state: {e}")
        return "unknown"

def wait_for_instance_state(instance_id: str, target_state: str, timeout: int = 600) -> bool:
    """Wait for instance to reach target state"""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            state = get_instance_state(instance_id)
            print(f"[AWS] Instance {instance_id} state: {state}")

            if state == target_state:
                if target_state == 'running':
                    time.sleep(30)  # Extra wait for SSH readiness
                return True
            elif state in ['pending', 'stopping']:
                time.sleep(10)
            else:
                time.sleep(5)
        except Exception as e:
            print(f"[AWS ERROR] While waiting for instance: {e}")
            time.sleep(10)

    print(f"[AWS TIMEOUT] Instance {instance_id} didn't reach {target_state} in {timeout}s")
    return False

def ssh_to_gpu(gpu_type: str, command: str) -> tuple[bool, str]:
    """Execute SSH command on GPU instance and return success status and output"""
    try:
        if gpu_type not in GPU_CONFIG:
            print(f"[SSH ERROR] Unknown GPU type: {gpu_type}")
            return False, ""

        config = GPU_CONFIG[gpu_type]
        if not config['public_ip']:
            print(f"[SSH ERROR] No public IP configured for {gpu_type}")
            return False, ""

        full_command = [
            "ssh", "-i", "C:/Users/Harsh Thakur/.ssh/s_spu_key.pem",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=30",
            "-o", "BatchMode=yes",
            f"{config['ssh_user']}@{config['public_ip']}",
            command
        ]

        print(f"[SSH {gpu_type.upper()}] Executing: {command}")
        result = subprocess.run(full_command, capture_output=True, text=True, timeout=60)

        if result.returncode == 0:
            print(f"[SSH {gpu_type.upper()}] Command successful")
            return True, result.stdout.strip()
        else:
            print(f"[SSH ERROR {gpu_type.upper()}] Command failed with return code {result.returncode}")
            print(f"[SSH ERROR {gpu_type.upper()}] stderr: {result.stderr}")
            return False, result.stderr

    except subprocess.TimeoutExpired:
        print(f"[SSH ERROR {gpu_type.upper()}] Timeout executing command")
        return False, "timeout"
    except Exception as e:
        print(f"[SSH ERROR {gpu_type.upper()}] {e}")
        return False, str(e)

def start_gpu_worker(gpu_type: str) -> bool:
    """Start worker service on GPU instance"""
    if gpu_type not in GPU_CONFIG:
        return False

    service_name = GPU_CONFIG[gpu_type]['worker_service']
    print(f"[AWS] Starting {service_name} on {gpu_type}")
    success, output = ssh_to_gpu(gpu_type, f"sudo systemctl start {service_name}")
    return success

def stop_gpu_worker(gpu_type: str) -> bool:
    """Stop worker service on GPU instance"""
    if gpu_type not in GPU_CONFIG:
        return False

    service_name = GPU_CONFIG[gpu_type]['worker_service']
    print(f"[AWS] Stopping {service_name} on {gpu_type}")
    success, output = ssh_to_gpu(gpu_type, f"sudo systemctl stop {service_name}")
    return success

def is_gpu_worker_running(gpu_type: str) -> bool:
    """Check if worker service is running on GPU"""
    if gpu_type not in GPU_CONFIG:
        return False

    service_name = GPU_CONFIG[gpu_type]['worker_service']
    print(f"[AWS] Checking if {service_name} is active on {gpu_type}")

    success, output = ssh_to_gpu(gpu_type, f"systemctl is-active {service_name}")

    # systemctl is-active returns:
    # - "active" if service is running
    # - "inactive" if service is stopped
    # - "failed" if service failed
    # - non-zero exit code if service doesn't exist
    if success and output == "active":
        print(f"[AWS] {service_name} is active on {gpu_type}")
        return True
    else:
        print(f"[AWS] {service_name} is not active on {gpu_type}. Output: '{output}'")
        return False

def restart_gpu_worker(gpu_type: str) -> bool:
    """Restart worker service on GPU instance"""
    if gpu_type not in GPU_CONFIG:
        return False

    service_name = GPU_CONFIG[gpu_type]['worker_service']
    print(f"[AWS] Restarting {service_name} on {gpu_type}")
    success, output = ssh_to_gpu(gpu_type, f"sudo systemctl restart {service_name}")
    return success

def ensure_gpu_worker_running(gpu_type: str) -> bool:
    """Ensure worker service is running, start it if not"""
    if gpu_type not in GPU_CONFIG:
        return False

    if is_gpu_worker_running(gpu_type):
        print(f"[AWS] {gpu_type} worker is already running")
        return True
    else:
        print(f"[AWS] {gpu_type} worker is not running, starting it...")
        return start_gpu_worker(gpu_type)

def get_gpu_instance_status(gpu_type: str) -> dict:
    """Get comprehensive status of GPU instance and worker"""
    if gpu_type not in GPU_CONFIG:
        return {"error": f"Unknown GPU type: {gpu_type}"}

    instance_state = get_instance_state(gpu_type)
    worker_running = is_gpu_worker_running(gpu_type) if instance_state == "running" else False

    return {
        "instance_state": instance_state,
        "worker_running": worker_running,
        "worker_service": GPU_CONFIG[gpu_type]['worker_service'],
        "public_ip": GPU_CONFIG[gpu_type]['public_ip']
    }

def debug_service_status(gpu_type: str):
    """Debug service status"""
    if gpu_type not in GPU_CONFIG:
        print(f"[DEBUG] Unknown GPU type: {gpu_type}")
        return

    service_name = GPU_CONFIG[gpu_type]['worker_service']
    print(f"[DEBUG] Checking service: {service_name}")

    # Check if service exists
    success, output = ssh_to_gpu(gpu_type, f"systemctl list-unit-files | grep {service_name}")
    print(f"[DEBUG] Service exists: {success}, Output: {output}")

    # Check service status in detail
    success, output = ssh_to_gpu(gpu_type, f"systemctl status {service_name} --no-pager")
    print(f"[DEBUG] Service status: {success}, Output: {output}")

    # Check if active
    success, output = ssh_to_gpu(gpu_type, f"systemctl is-active {service_name}")
    print(f"[DEBUG] Service active: {success}, Output: '{output}'")

def download_from_s3(bucket, key, download_path):
    """Download file from S3"""
    s3 = boto3.client('s3')
    import os
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3.download_file(bucket, key, download_path)
    print(f"[S3] Downloaded: {bucket}/{key} → {download_path}")

def upload_to_s3(bucket, key, file_path):
    """Upload file to S3"""
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    print(f"[S3] Uploaded: {file_path} → {bucket}/{key}")

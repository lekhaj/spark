import subprocess
import time
import logging
from app.services.mongo_service import biome_assets_for_task, update_or_add_biome_asset, get_biome
from app.services.aws_service import start_instance
from app.services.orchestrator_service import wait_for_gpu
from app.config import settings


from app.main import r, submit_image_task, submit_3d_task, get_result


import argparse
BIOME_ID = ""
REDIS_IMAGE_QUEUE = "image_tasks"
REDIS_3D_QUEUE = "model_tasks"  
IMAGE_OUTPUT_KEY = "generated-images/{job_id}/output.png"
MODEL_OUTPUT_KEY = "generated-models/{job_id}/output.glb"
POLL_INTERVAL = 600 
IMAGE_COMPLETE_STATUS = "image generated"

logger = logging.getLogger("app.orchestrate_biome_image_tasks")
logging.basicConfig(level=logging.INFO)


def orchestrate_image_generation():
    logger.info(f"Starting image generation for BIOME_ID={BIOME_ID}")
    assets = biome_assets_for_task(BIOME_ID, status_filter="not complete")
    if not assets:
        logger.info("No assets with status 'not complete'. Exiting.")
        return []
    for asset_name, asset in assets.items():
        update_or_add_biome_asset(BIOME_ID, asset_name=asset_name, update_dict={"status": "pending"})
        logger.info(f"Asset '{asset_name}' set to 'pending'")
    logger.info("Ensuring GPU instance is running via AWS...")
    try:
        start_instance("gpu")
        wait_for_gpu()
        logger.info("GPU instance is running.")
    except Exception as e:
        logger.error(f"Failed to start GPU instance: {e}")
        return []
    job_ids = []
    for asset_name, asset in assets.items():
        job_id = submit_image_task(
            prompt=asset["description"],
            negative_prompt=asset.get("negative_prompt", ""),
            width=asset.get("width", 1024),
            height=asset.get("height", 1024),
            num_inference_steps=asset.get("num_inference_steps", 30)
        )
        update_or_add_biome_asset(BIOME_ID, asset_name=asset_name, update_dict={"job_id": job_id})
        job_ids.append(job_id)
        logger.info(f"Submitted image task for asset '{asset_name}' with job_id {job_id}")
    logger.info("All image tasks submitted. Waiting for completion...")
    from app.services.redis_service import dequeue_task_by_job_id
    from app.services.mongo_service import update_biome_asset_by_name_or_job_id
    pending_job_ids = set(job_ids)
    while pending_job_ids:
        completed = set()
        for job_id in list(pending_job_ids):
            result = get_result(job_id)
            if result and result.get("status") == "completed" and result.get("image_url"):
                dequeue_task_by_job_id(REDIS_IMAGE_QUEUE, job_id)
                update_biome_asset_by_name_or_job_id(
                    BIOME_ID,
                    job_id=job_id,
                    update_dict={
                        "job_id": job_id,
                        "s3_image_url": result["image_url"],
                        "timestamp": result.get("generated_at", time.time()),
                        "status": "image generated"
                    }
                )
                logger.info(f"Asset with job_id {job_id} completed. MongoDB updated with s3_image_url, timestamp, and status. Job removed from Redis queue.")
                completed.add(job_id)
        pending_job_ids -= completed
        if pending_job_ids:
            logger.info(f"Waiting for {len(pending_job_ids)} jobs to complete...")
            time.sleep(POLL_INTERVAL)
    logger.info("All images generated and MongoDB updated with job_id, s3_image_url, timestamp, and status.")
    return job_ids

def orchestrate_3d_generation():
    logger.info(f"Starting 3D generation for BIOME_ID={BIOME_ID}")
    # Get all assets with status 'image generated'
    assets = biome_assets_for_task(BIOME_ID, status_filter="image generated")
    if not assets:
        logger.info("No assets with status 'image generated'. Exiting 3D gen.")
        return []
    job_ids = []
    for asset_name, asset in assets.items():
        job_id = submit_3d_task(
            image_s3_url=asset.get("s3_image_url", ""),
            prompt=asset["description"]
        )
        update_or_add_biome_asset(BIOME_ID, asset_name=asset_name, update_dict={"status": "3d pending", "job_id_3d": job_id})
        job_ids.append(job_id)
        logger.info(f"Submitted 3D task for asset '{asset_name}' with job_id {job_id}")
    logger.info("All 3D tasks submitted. Waiting for completion...")
    from app.services.redis_service import dequeue_task_by_job_id
    from app.services.mongo_service import update_biome_asset_by_name_or_job_id
    pending_job_ids = set(job_ids)
    while pending_job_ids:
        completed = set()
        for job_id in list(pending_job_ids):
            result = get_result(job_id)
            if result and result.get("status") == "completed" and result.get("model_url"):
                dequeue_task_by_job_id(REDIS_3D_QUEUE, job_id)
                update_biome_asset_by_name_or_job_id(
                    BIOME_ID,
                    job_id=job_id,
                    update_dict={
                        "job_id_3d": job_id,
                        "s3_model_url": result["model_url"],
                        "timestamp_3d": result.get("generated_at", time.time()),
                        "status": "3d generated"
                    }
                )
                logger.info(f"Asset with 3D job_id {job_id} completed. MongoDB updated with s3_model_url, timestamp_3d, and status. Job removed from Redis queue.")
                completed.add(job_id)
        pending_job_ids -= completed
        if pending_job_ids:
            logger.info(f"Waiting for {len(pending_job_ids)} 3D jobs to complete...")
            time.sleep(POLL_INTERVAL)
    logger.info("All 3D models generated and MongoDB updated with job_id_3d, s3_model_url, timestamp_3d, and status.")
    return job_ids



def is_ssh_connection_active(ssh_user, gpu_ip, key_path):
    """Check if SSH connection to GPU instance is active by running a simple command."""
    ssh_cmd = [
        "ssh",
        "-i", key_path,
        "-o", "ConnectTimeout=5",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{gpu_ip}",
        "echo connected"
    ]
    try:
        result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True, timeout=10)
        if "connected" in result.stdout:
            logger.info("SSH connection is active.")
            return True
    except Exception as e:
        logger.warning(f"SSH connection not active: {e}")
    return False

def ensure_ssh_connection(ssh_user, gpu_ip, key_path):
    """Ensure SSH connection is active, try to reconnect if not."""
    if not is_ssh_connection_active(ssh_user, gpu_ip, key_path):
        logger.info("Attempting to re-establish SSH connection...")
        # Optionally, you could add logic to restart SSH agent, re-add key, or wait/retry
        # For now, just log and continue; next SSH command will try to connect

def manage_worker_service(action, service_name=None):
    """
    Manage the worker service on the GPU instance based on logic/config.
    This function uses config or internal logic to determine which service to manage and when.
    Ensures SSH connection is active before running commands.
    """
    ssh_user = settings.GPU_SSH_USER
    gpu_ip = settings.GPU_PUBLIC_IP
    key_path = settings.GPU_SSH_KEY_PATH  # Should be set in config
    if not service_name:
        # Default to image-worker unless config/logic says otherwise
        service_name = settings.DEFAULT_WORKER_SERVICE
    ensure_ssh_connection(ssh_user, gpu_ip, key_path)
    ssh_cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{gpu_ip}",
        "sudo", "systemctl", action, service_name
    ]
    logger.info(f"[SSH] Running: {' '.join(ssh_cmd)}")
    try:
        result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Service '{service_name}' {action}ed on GPU instance. Output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to {action} service '{service_name}' via SSH: {e.stderr}")
        raise
    # Optionally, check which worker is running after the action
    check_active_worker_service(ssh_user, gpu_ip, key_path)

def check_active_worker_service(ssh_user, gpu_ip, key_path):
    """Check which worker service is running on the GPU instance via SSH."""
    ssh_cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{gpu_ip}",
        "systemctl list-units --type=service --state=running"
    ]
    logger.info(f"[SSH] Running: {' '.join(ssh_cmd)}")
    try:
        result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Active services output:\n{result.stdout}")
        if "image-worker.service" in result.stdout:
            print("image-worker is active")
        if "model-worker.service" in result.stdout:
            print("model-worker is active")
        if "image-worker.service" not in result.stdout and "model-worker.service" not in result.stdout:
            print("No worker service running")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check active service via SSH: {e.stderr}")
        raise

def ssh_active_service():
    """Check which worker service is running on the GPU instance via SSH."""
    ssh_user = settings.GPU_SSH_USER
    gpu_ip = settings.GPU_PUBLIC_IP
    key_path = "C:/Users/HP/Downloads/s_spu_key.pem"
    ssh_cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{gpu_ip}",
        "systemctl list-units --type=service --state=running"
    ]
    logger.info(f"[SSH] Running: {' '.join(ssh_cmd)}")
    try:
        result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
        logger.info(f"Active services output:\n{result.stdout}")
        if "image-worker.service" in result.stdout:
            print("image-worker is active")
        if "model-worker.service" in result.stdout:
            print("model-worker is active")
        if "image-worker.service" not in result.stdout and "model-worker.service" not in result.stdout:
            print("No worker service running")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to check active service via SSH: {e.stderr}")
        raise

def ssh_run_command(command):
    """Run an arbitrary shell command on the GPU instance via SSH."""
    ssh_user = settings.GPU_SSH_USER
    gpu_ip = settings.GPU_PUBLIC_IP
    key_path = "C:/Users/Hp/Downloads/s_spu_key.pem"
    ssh_cmd = [
        "ssh",
        "-i", key_path,
        "-o", "StrictHostKeyChecking=accept-new",
        f"{ssh_user}@{gpu_ip}",
        command
    ]
    logger.info(f"[SSH] Running: {' '.join(ssh_cmd)}")
    try:
        result = subprocess.run(ssh_cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run command via SSH: {e.stderr}")
        raise


def main():
    parser = argparse.ArgumentParser(description="Biome Orchestration CLI")
    parser.add_argument("--b_id", type=str, required=False, help="Biome ID to process")
    parser.add_argument("--orch", type=str, choices=["image", "3d", "both"], help="Which orchestration to run")
    args = parser.parse_args()

    global BIOME_ID
    if args.b_id:
        BIOME_ID = args.b_id

    # Example: automatically manage worker service based on orchestration mode
    if args.orch == "image":
        manage_worker_service(action="start", service_name=settings.DEFAULT_WORKER_SERVICE)
        orchestrate_image_generation()
        manage_worker_service(action="stop", service_name=settings.DEFAULT_WORKER_SERVICE)
    elif args.orch == "3d":
        manage_worker_service(action="start", service_name="model-worker")
        orchestrate_3d_generation()
        manage_worker_service(action="stop", service_name="model-worker")
    elif args.orch == "both":
        manage_worker_service(action="start", service_name=settings.DEFAULT_WORKER_SERVICE)
        orchestrate_image_generation()
        manage_worker_service(action="stop", service_name=settings.DEFAULT_WORKER_SERVICE)
        manage_worker_service(action="start", service_name="model-worker")
        orchestrate_3d_generation()
        manage_worker_service(action="stop", service_name="model-worker")
        
if __name__ == "__main__":
    main()
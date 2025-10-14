from fastapi import APIRouter
import logging
import json
import time
import uuid
import redis
from app.config import settings
from app.services.mongo_service import (
    biome_assets_for_task,
    update_or_add_biome_asset,
    get_biome_asset_update_key,
)

router = APIRouter()

REDIS_IMAGE_QUEUE = "image_tasks"
REDIS_3D_QUEUE = "model_tasks"
logger = logging.getLogger("app.orchestrator")

# Create a Redis client safely (may be None if not configured)
try:
    _r = redis.Redis.from_url(settings.CELERY_BROKER_URL)
    _r.ping()
    r = _r
    logger.info(f"Connected to Redis at {settings.CELERY_BROKER_URL}")
except Exception as e:
    r = None
    logger.warning(f"Redis not available for orchestrator router: {e}")


def create_image_task_dict(prompt, negative_prompt="", width=1024, height=1024, num_inference_steps=30):
    job_id = str(uuid.uuid4())
    return job_id, {
        "job_id": job_id,
        "task_type": "image",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "width": width,
        "height": height,
        "num_inference_steps": num_inference_steps,
        "timestamp": time.time(),
        "status": "queued",
        "output_key": f"generated-images/{job_id}/sdxl_output.png",
    }


def create_3d_task_dict(image_s3_url, prompt=""):
    job_id = str(uuid.uuid4())
    return job_id, {
        "job_id": job_id,
        "task_type": "3d_model",
        "image_s3_url": image_s3_url,
        "prompt": prompt,
        "timestamp": time.time(),
        "status": "queued",
        "output_key": f"3d_assets/{job_id}_mesh.obj",
    }


@router.post("/submit_image_tasks/")
def submit_image_tasks(biome_id: str):
    assets = biome_assets_for_task(biome_id, status_filter="not complete")
    if not assets:
        return {"message": "No assets with status 'not complete' for this biome."}
    job_ids = []
    for asset_name, asset in assets.items():
        job_id, task = create_image_task_dict(
            prompt=asset["description"],
            negative_prompt=asset.get("negative_prompt", ""),
            width=asset.get("width", 1024),
            height=asset.get("height", 1024),
            num_inference_steps=asset.get("num_inference_steps", 30),
        )
        update_key = get_biome_asset_update_key(biome_id, asset_name)
        if update_key:
            update_or_add_biome_asset(
                biome_id,
                update_key,
                {"status": "pending", "job_id": job_id},
            )
            task["update_key"] = update_key
        else:
            task["update_key"] = None
        task["biome_id"] = biome_id
        if r:
            r.rpush(REDIS_IMAGE_QUEUE, json.dumps(task))
            count = r.llen(REDIS_IMAGE_QUEUE)
        else:
            logger.warning("Redis client not available; task not pushed")
            count = 0
        job_ids.append(job_id)
    return {"biome_id": biome_id, "submitted_job_ids": job_ids, "count": count}


@router.post("/submit_3d_tasks/")
def submit_3d_tasks(biome_id: str):
    assets = biome_assets_for_task(biome_id, status_filter="image generated")
    if not assets:
        return {"message": "No assets with status 'image generated' for this biome."}
    job_ids = []
    image_url_key = ["image_s3_url", "image_url", "s3_image_url", "s3_image_uri"]
    for asset_name, asset in assets.items():
        # prefer top-level keys, else look inside attributes dict
        chosen_key = next((x for x in image_url_key if x in asset and asset.get(x)), None)
        image_url_val = asset.get(chosen_key) if chosen_key else None
        if not image_url_val and isinstance(asset.get("attributes"), dict):
            image_attrs = asset.get("attributes")
            chosen_key = next((x for x in image_url_key if x in image_attrs and image_attrs.get(x)), None)
            image_url_val = image_attrs.get(chosen_key) if chosen_key else None

        job_id, task = create_3d_task_dict(
            image_s3_url=image_url_val or "",
            prompt=asset.get("description", ""),
        )
        update_key = get_biome_asset_update_key(biome_id, asset_name)
        if update_key:
            update_or_add_biome_asset(
                biome_id,
                update_key,
                {"status": "3d pending", "job_id_3d": job_id},
            )
            task["update_key"] = update_key
        else:
            task["update_key"] = None
        task["biome_id"] = biome_id
        if r:
            r.rpush(REDIS_3D_QUEUE, json.dumps(task))
            count = r.llen(REDIS_3D_QUEUE)
        else:
            logger.warning("Redis client not available; task not pushed")
            count = 0
        job_ids.append(job_id)
    return {"biome_id": biome_id, "submitted_job_ids": job_ids, "count": count}

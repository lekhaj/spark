import logging
from fastapi import FastAPI

import json
from app.services.mongo_service import biome_assets_for_task, update_or_add_biome_asset, get_biome_asset_update_key
from app.main import r, create_image_task_dict, create_3d_task_dict

app = FastAPI()

REDIS_IMAGE_QUEUE = "image_tasks"
REDIS_3D_QUEUE = "model_tasks"
logger = logging.getLogger("app.orchestrate_biome_image_tasks")
logging.basicConfig(level=logging.INFO)

@app.post("/submit_image_tasks/")
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
            num_inference_steps=asset.get("num_inference_steps", 30)
        )
        update_key = get_biome_asset_update_key(biome_id, asset_name)
        if update_key:
            update_or_add_biome_asset(
                biome_id,
                update_key,
                {"status": "pending", "job_id": job_id}
            )
            task["update_key"] = update_key # example: "possible_structures.buildings.House1"
        else:
            task["update_key"] = None
        task["biome_id"] = biome_id
        r.rpush(REDIS_IMAGE_QUEUE, json.dumps(task))
        job_ids.append(job_id)
    return {"biome_id": biome_id, "submitted_job_ids": job_ids, "count": r.llen(REDIS_IMAGE_QUEUE)}

@app.post("/submit_3d_tasks/")
def submit_3d_tasks(biome_id: str):
    assets = biome_assets_for_task(biome_id, status_filter="image generated")
    if not assets:
        return {"message": "No assets with status 'image generated' for this biome."}
    job_ids = []
    for asset_name, asset in assets.items():
        job_id, task = create_3d_task_dict(
            image_s3_url=asset.get("s3_image_url", ""),
            prompt=asset["description"]
        )
        
        update_key = get_biome_asset_update_key(biome_id, asset_name)
        if update_key:
            update_or_add_biome_asset(
                biome_id,
                update_key,
                {"status": "3d pending", "job_id_3d": job_id}
            )
            task["update_key"] = update_key
        else:
            task["update_key"] = None
        task["biome_id"] = biome_id
        r.rpush(REDIS_3D_QUEUE, json.dumps(task))
        job_ids.append(job_id)
    return {"biome_id": biome_id, "submitted_job_ids": job_ids, "count": r.llen(REDIS_3D_QUEUE)}
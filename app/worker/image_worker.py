"""
Configurable Image Generation Worker
=====================================
Reads task from Redis image_tasks queue, generates image using a configurable
pipeline, uploads to S3, updates MongoDB, then forwards task to model_tasks.

Model is fully configurable via .env — switching models requires only env var changes.

Supported pipeline classes (diffusers):
  - ZImagePipeline          → Tongyi-MAI/Z-Image-Turbo  (default)
  - StableDiffusionXLPipeline → SDXL base
  - FluxPipeline             → FLUX.1-dev / FLUX.1-schnell

.env keys:
  IMAGE_MODEL_ID          = Tongyi-MAI/Z-Image-Turbo
  IMAGE_PIPELINE_CLASS    = ZImagePipeline
  IMAGE_STEPS             = 9
  IMAGE_GUIDANCE_SCALE    = 0.0
  IMAGE_TORCH_DTYPE       = bfloat16       # or float16
  IMAGE_HEIGHT            = 1024
  IMAGE_WIDTH             = 1024
  IMAGE_T_POSE_SUFFIX     = , T-pose, arms extended horizontally, full body, character sheet
  IMAGE_IDLE_TIMEOUT      = 30             # seconds idle → worker exits (model_runner restarts as needed)
"""

import os, json, time, importlib, logging, boto3, redis, torch, gc
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [IMAGE-WORKER] %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ── Config (all overridable via .env) ────────────────────────────────────────
REDIS_HOST       = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT       = int(os.getenv("REDIS_PORT", 6380))
REDIS_DB         = int(os.getenv("REDIS_DB", 0))

MONGO_URI        = os.getenv("MONGODB_URL")
DB_NAME          = os.getenv("MONGODB_DB_NAME")

BUCKET_NAME      = os.getenv("BUCKET_NAME", "sparkassets")
S3_REGION        = os.getenv("S3_REGION", "ap-south-1")

IMAGE_QUEUE      = os.getenv("IMAGE_QUEUE", "image_tasks")
MODEL_QUEUE      = os.getenv("MODEL_QUEUE", "model_tasks")  # forward to 3D worker

# ── Model config ─────────────────────────────────────────────────────────────
MODEL_ID         = os.getenv("IMAGE_MODEL_ID",       "Tongyi-MAI/Z-Image-Turbo")
PIPELINE_CLASS   = os.getenv("IMAGE_PIPELINE_CLASS", "ZImagePipeline")
STEPS            = int(os.getenv("IMAGE_STEPS",      "9"))
GUIDANCE_SCALE   = float(os.getenv("IMAGE_GUIDANCE_SCALE", "0.0"))
TORCH_DTYPE_STR  = os.getenv("IMAGE_TORCH_DTYPE",   "bfloat16")
HEIGHT           = int(os.getenv("IMAGE_HEIGHT",     "1024"))
WIDTH            = int(os.getenv("IMAGE_WIDTH",      "1024"))
# Appended to every prompt to enforce T-pose for clean rigging
T_POSE_SUFFIX    = os.getenv(
    "IMAGE_T_POSE_SUFFIX",
    ", T-pose, arms extended horizontally, legs straight, full body, front view, character sheet, white background"
)
IDLE_TIMEOUT     = int(os.getenv("IMAGE_IDLE_TIMEOUT", "30"))  # seconds

TORCH_DTYPE = getattr(torch, TORCH_DTYPE_STR, torch.bfloat16)


def load_pipeline():
    """Dynamically load the configured pipeline class from diffusers."""
    logger.info(f"Loading pipeline: {PIPELINE_CLASS} | model: {MODEL_ID} | dtype: {TORCH_DTYPE_STR}")
    try:
        diffusers = importlib.import_module("diffusers")
        PipelineClass = getattr(diffusers, PIPELINE_CLASS)
    except AttributeError:
        raise ImportError(
            f"Pipeline class '{PIPELINE_CLASS}' not found in diffusers. "
            f"Check IMAGE_PIPELINE_CLASS in .env. "
            f"Available example values: ZImagePipeline, StableDiffusionXLPipeline, FluxPipeline"
        )

    pipe = PipelineClass.from_pretrained(
        MODEL_ID,
        torch_dtype=TORCH_DTYPE,
        low_cpu_mem_usage=True,
    )
    pipe.enable_model_cpu_offload()
    logger.info(f"✅ Pipeline loaded: {PIPELINE_CLASS} with CPU offload")
    return pipe


def build_prompt(base_prompt: str) -> str:
    """Append T-pose suffix to prompt for consistent character generation."""
    prompt = base_prompt.strip().rstrip(",")
    return f"{prompt}{T_POSE_SUFFIX}"


def generate_image(pipe, task: dict) -> str:
    """Run inference and return local file path."""
    job_id     = task["job_id"]
    raw_prompt = task.get("prompt", "")
    prompt     = build_prompt(raw_prompt)

    logger.info(f"[{job_id}] Generating | steps={STEPS} guidance={GUIDANCE_SCALE}")
    logger.info(f"[{job_id}] Prompt: {prompt[:120]}...")

    result = pipe(
        prompt=prompt,
        height=task.get("height", HEIGHT),
        width=task.get("width", WIDTH),
        num_inference_steps=task.get("steps", STEPS),
        guidance_scale=task.get("guidance_scale", GUIDANCE_SCALE),
        generator=torch.Generator("cuda").manual_seed(task.get("seed", int(time.time()))),
    )
    image = result.images[0]

    out_path = f"/tmp/{job_id}.png"
    image.save(out_path)
    logger.info(f"[{job_id}] ✅ Image saved to {out_path}")
    return out_path


def upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload file to S3 and return public URL."""
    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.upload_file(local_path, BUCKET_NAME, s3_key)
    url = f"https://{BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    logger.info(f"✅ Uploaded to S3: {url}")
    return url


def update_mongo(biome_id: str, update_key: str, image_url: str, status: str = "image_generated"):
    """Write image URL and status back to the biome asset in MongoDB."""
    if not biome_id or not update_key:
        logger.warning("biome_id or update_key missing — skipping MongoDB update")
        return False
    try:
        client = MongoClient(MONGO_URI)
        db     = client[DB_NAME]
        result = db.biomes.update_one(
            {"_id": biome_id},
            {"$set": {
                f"{update_key}.image_url": image_url,
                f"{update_key}.status":    status,
            }}
        )
        client.close()
        ok = result.modified_count > 0
        logger.info(f"✅ MongoDB updated biome={biome_id} key={update_key} status={status} modified={ok}")
        return ok
    except Exception as e:
        logger.error(f"❌ MongoDB update failed: {e}")
        return False


def forward_to_model_queue(redis_client, task: dict, image_url: str):
    """Push a model_tasks entry so the 3D worker picks up the generated image."""
    model_task = {
        "job_id":       task["job_id"],
        "image_s3_url": image_url,
        "biome_id":     task.get("biome_id"),
        "update_key":   task.get("update_key"),
        "prompt":       task.get("prompt", ""),
        "output_key":   task.get("output_key"),
    }
    redis_client.rpush(MODEL_QUEUE, json.dumps(model_task))
    logger.info(f"[{task['job_id']}] ✅ Forwarded to {MODEL_QUEUE}")


def process_task(task: dict, pipe, redis_client) -> bool:
    """Full pipeline: generate → S3 → MongoDB → forward to 3D queue."""
    job_id    = task.get("job_id", "unknown")
    biome_id  = task.get("biome_id")
    update_key = task.get("update_key")

    # Mark as processing
    update_mongo(biome_id, update_key, "", "image_processing")

    local_path = None
    try:
        local_path = generate_image(pipe, task)

        s3_key    = task.get("output_key") or f"generated-images/{job_id}/output.png"
        image_url = upload_to_s3(local_path, s3_key)

        update_mongo(biome_id, update_key, image_url, "image_generated")
        forward_to_model_queue(redis_client, task, image_url)
        return True

    except Exception as e:
        logger.error(f"[{job_id}] ❌ Task failed: {e}", exc_info=True)
        update_mongo(biome_id, update_key, "", "image_error")
        return False

    finally:
        if local_path and os.path.exists(local_path):
            os.remove(local_path)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def main():
    logger.info("=" * 60)
    logger.info(f"IMAGE WORKER STARTING")
    logger.info(f"  Model     : {MODEL_ID}")
    logger.info(f"  Pipeline  : {PIPELINE_CLASS}")
    logger.info(f"  Steps     : {STEPS}  |  Guidance: {GUIDANCE_SCALE}")
    logger.info(f"  Resolution: {WIDTH}x{HEIGHT}  |  dtype: {TORCH_DTYPE_STR}")
    logger.info(f"  Queue     : {IMAGE_QUEUE} → {MODEL_QUEUE}")
    logger.info(f"  Idle exit : {IDLE_TIMEOUT}s")
    logger.info("=" * 60)

    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)
    redis_client.ping()
    logger.info("✅ Redis connected")

    pipe = load_pipeline()

    last_task_time = time.time()

    while True:
        # Exit when idle — model_runner.py will restart when next task arrives
        if time.time() - last_task_time > IDLE_TIMEOUT:
            logger.info(f"⏰ Idle for {IDLE_TIMEOUT}s — shutting down to free GPU memory")
            break

        raw = redis_client.blpop(IMAGE_QUEUE, timeout=5)
        if not raw:
            continue

        last_task_time = time.time()
        _, payload = raw

        try:
            task = json.loads(payload)
        except json.JSONDecodeError:
            logger.error("❌ Invalid JSON in queue — skipping")
            continue

        logger.info(f"📥 Task received: {task.get('job_id')} biome={task.get('biome_id')}")
        process_task(task, pipe, redis_client)

    logger.info("Image worker stopped.")


if __name__ == "__main__":
    main()

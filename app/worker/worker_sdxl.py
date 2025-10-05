import os, json, time, boto3, torch, redis
from dotenv import load_dotenv
from diffusers import DiffusionPipeline
from pymongo import MongoClient
load_dotenv()
# ------------ Config ------------
REDIS_HOST = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
S3_REGION = os.getenv("S3_REGION", "ap-south-1")
BUCKET_NAME = os.getenv("BUCKET_NAME", "sparkassets")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")

IMAGE_QUEUE = "image_tasks"   # FastAPI → image worker
MODEL_QUEUE = "model_tasks"   # image worker → GPU worker
# --------------------------------

def init_pipeline():
    print("Loading SDXL...")
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to("cuda")
    try: pipe.enable_xformers_memory_efficient_attention()
    except: pass
    return pipe

def upload_s3(path, key):
    s3 = boto3.client("s3", region_name=S3_REGION)
    s3.upload_file(path, BUCKET_NAME, key)
    return f"https://{BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{key}"

def update_mongo(biome_id, update_key, image_url, status="image_generated"):
    """Update MongoDB with image URL and status"""
    try:
        mongo = MongoClient(MONGO_URI)
        db = mongo[DB_NAME]
        
        # Update nested field using dot notation
        result = db.biomes.update_one(
            {"_id": biome_id},
            {"$set": {
                f"{update_key}.image_url": image_url,
                f"{update_key}.status": status
            }}
        )
        mongo.close()
        return result.modified_count > 0
    except Exception as e:
        print(f"❌ MongoDB update failed: {e}")
        return False

def process_task(task, pipe):
    job_id = task.get("job_id", str(int(time.time())))
    prompt = task.get("prompt", "")
    biome_id = task.get("biome_id")
    update_key = task.get("update_key")
    print(f"Biome ID: [{biome_id}]")
    print(f"Job ID of asset:[{job_id}] Generating: {prompt[:50]}...")

    image = pipe(
        prompt=prompt,
        negative_prompt=task.get("negative_prompt", ""),
        width=task.get("width", 1024),
        height=task.get("height", 1024),
        num_inference_steps=task.get("steps", 30),
        guidance_scale=task.get("guidance_scale", 7.5),
        generator=torch.Generator("cuda").manual_seed(int(time.time()))
    ).images[0]

    tmp = f"/tmp/{job_id}.png"
    image.save(tmp)

    # Use provided output_key or default path
    s3_key = task.get("output_key", f"generated/{job_id}.png")
    image_url = upload_s3(tmp, s3_key)
    os.remove(tmp)

    # Update MongoDB if we have the required fields
    if biome_id and update_key:
        if update_mongo(biome_id, update_key, image_url, "completed"):
            print(f"[{job_id}] ✅ MongoDB updated")
        else:
            print(f"[{job_id}] ❌ MongoDB update failed")

    # Prepare task for model worker
    model_task = {
        "job_id": job_id,
        "image_s3_url": image_url,
        "biome_id": biome_id,
        "update_key": update_key,
        "output_key": task.get("output_key"),  # Pass along for model worker
        "prompt": prompt
    }

    return model_task

def image_worker():
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    r.ping()
    print("✅ Connected to Redis")
    
    pipe = init_pipeline()
    print("✅ Pipeline ready. Listening...")

    idle_timeout = 30  # seconds
    last_task_time = time.time()

    while True:
        # Check if we've been idle too long
        if time.time() - last_task_time > idle_timeout:
            print("⏰ 30 seconds without tasks. Shutting down.")
            break

        task_data = r.blpop(IMAGE_QUEUE, timeout=5)
        if not task_data: 
            continue

        # Reset idle timer when we get a task
        last_task_time = time.time()
        _, raw = task_data

        try: 
            task = json.loads(raw)
        except:
            print("❌ Bad JSON")
            continue

        result = process_task(task, pipe)
        r.rpush(MODEL_QUEUE, json.dumps(result))
        print(f"[{result['job_id']}] ✅ Forwarded to {MODEL_QUEUE}")

if __name__ == "__main__":
    image_worker()
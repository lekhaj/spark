from fastapi import FastAPI, UploadFile, File
from app.routes import aws_routes
from pydantic import BaseModel
import redis, json, uuid, boto3

REDIS_HOST = "15.206.99.66"
REDIS_PORT = 6380
BUCKET_NAME = "sparkassets"

# Redis client
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    r.ping()
    print(f"[FastAPI] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f"[FastAPI] Failed to connect to Redis: {e}")

# S3 client
s3 = boto3.client("s3")

app = FastAPI(title="FastAPI MongoDB AWS Control")

#app.include_router(mongo_routes.router)
app.include_router(aws_routes.router)


@app.get("/")
def home():
    return {"message": "text to 3d pipeline"}

class Task(BaseModel):
    image_url: str

@app.post("/submit_task/")
async def submit_task(task: Task):
    job_id = str(uuid.uuid4())
    
    task_data = {
        "job_id": job_id,
        "image_s3_url": task.image_url,
        "output_key": f"spackassets/3d_assets/{job_id}_mesh.obj"
    }

    r.lpush("tasks", json.dumps(task_data))
    print(f"[FastAPI] Task pushed into Redis queue: {task_data}")
    print(f"[MongoDB] Inserted job {job_id} with status='queued'")

    return {"status": "queued", "task": task_data}

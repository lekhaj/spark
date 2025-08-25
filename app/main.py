from fastapi import FastAPI
from app.routes import  aws_routes#, mongo_routes
from pydantic import BaseModel
import redis, json
 
REDIS_HOST = "15.206.99.66"
REDIS_PORT = "6380"
try:
    r = redis.Redis(host=REDIS_HOST, port=6380, db=0)
    r.ping()
    print(f"[FastAPI] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    print(f"[FastAPI]  Failed to connect to Redis: {e}")

app = FastAPI(title="FastAPI MongoDB AWS Control")

#app.include_router(mongo_routes.router)
app.include_router(aws_routes.router)
class Task(BaseModel):
    image_path: str
    output_key: str
@app.get("/")
def home():
    return {"message": "text to 3d pipeline"}


@app.post("/submit_task/")
def submit_task(task: Task):
    print(f"[FastAPI] Received task: {task.dict()}")
    r.lpush("tasks", json.dumps(task.dict()))
    print("[FastAPI] Task pushed into Redis queue.")
    return {"status": "queued", "task": task.dict()}
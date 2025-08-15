import redis
import json
import os

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True
)

def enqueue_task(task_data: dict):
    redis_client.lpush("task_queue", json.dumps(task_data))

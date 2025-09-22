
# Recommended pattern:
# - Use Redis queue (list) only for task delivery and ordering.
# - Store all task state (status, metadata, results, etc.) in MongoDB (biome asset).
# - Workers pop tasks from the queue, process, and update MongoDB only.
# - To check status, always query MongoDB.
# - When a task is completed, remove it from the Redis queue (if still present).

# Remove (dequeue) a task from a Redis queue by job_id
import redis
import json
from app.config import settings

redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

def enqueue_task(task_data: dict):
    redis_client.lpush("task_queue", json.dumps(task_data))
    
def dequeue_task_by_job_id(queue_name: str, job_id: str):
    tasks = redis_client.lrange(queue_name, 0, -1)
    removed = False
    for task_json in tasks:
        try:
            task = json.loads(task_json)
        except Exception:
            continue
        if str(task.get("job_id")) == str(job_id):
            redis_client.lrem(queue_name, 1, task_json)
            removed = True
            break
    return removed



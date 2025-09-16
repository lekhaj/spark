
import redis
import json
from app.config import settings

redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

def enqueue_task(task_data: dict):
    redis_client.lpush("task_queue", json.dumps(task_data))

from app.config import settings
import redis
import boto3

# Redis
redis_client = redis.Redis.from_url(settings.CELERY_BROKER_URL, decode_responses=True)

# S3
s3 = boto3.client("s3", region_name=settings.AWS_REGION)
BUCKET_NAME = settings.AWS_S3_BUCKET

# For workers: get redis host/port if needed
REDIS_HOST = settings.CELERY_BROKER_URL.split('@')[-1].split(':')[0] if '@' in settings.CELERY_BROKER_URL else 'localhost'
REDIS_PORT = int(settings.CELERY_BROKER_URL.split(':')[-1]) if ':' in settings.CELERY_BROKER_URL else 6379

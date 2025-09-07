from celery import Celery
import os

redis_url = os.getenv("REDIS_URL")
celery_app = Celery("biome_tasks", broker=redis_url, backend=redis_url)

@celery_app.task
def generate_biome_task(theme_prompt):
    from src.biome_generator import create_new_biome
    result = create_new_biome(theme_prompt)
    return {
        "success": result.success,
        "message": result.message,
        "biome_name": result.biome_name,
        "biome_document": result.biome_document,
    }

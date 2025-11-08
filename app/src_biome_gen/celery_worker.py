from celery import Celery
import os

# Support legacy CELERY_BROKER_URL / CELERY_RESULT_BACKEND or a single REDIS_URL
redis_url = os.getenv("REDIS_URL") or os.getenv("CELERY_BROKER_URL") or os.getenv("CELERY_RESULT_BACKEND")
if not redis_url:
    raise ValueError("FATAL ERROR: REDIS_URL or CELERY_BROKER_URL / CELERY_RESULT_BACKEND must be set for Celery.")

celery_app = Celery("biome_tasks", broker=redis_url, backend=redis_url)

@celery_app.task
def generate_biome_task(theme_prompt):
    from src_biome_gen.biome_generator import create_new_biome
    # Backward compatible: accept either (theme_prompt) or (theme_prompt, system_prompt)
    try:
        # If Celery forwarded args as a tuple in the payload, it may reach here as a
        # single composite object; handle the common case where a second arg was passed
        # by expecting `theme_prompt` to be either a string or a tuple/list.
        if isinstance(theme_prompt, (list, tuple)) and len(theme_prompt) >= 1:
            tp = theme_prompt[0]
            sp = theme_prompt[1] if len(theme_prompt) > 1 else None
            result = create_new_biome(tp, sp)
        else:
            result = create_new_biome(theme_prompt, None)
    except Exception:
        # Default safe call
        result = create_new_biome(theme_prompt, None)
    return {
        "success": result.success,
        "message": result.message,
        "biome_name": result.biome_name,
        "biome_document": result.biome_document,
    }

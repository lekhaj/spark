
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Application settings are defined in this class, loaded from environment
    variables or a .env file using pydantic-settings.

    This provides a single, type-safe source of truth for all configuration.
    """
    
    # --- Project Metadata ---
    PROJECT_NAME: str = "Biome Generation Orchestrator"
    API_V1_STR: str = "/api/v1"
    

    MONGODB_URL: str
    
    
    MONGODB_DB_NAME: str
    
    CELERY_BROKER_URL: str
    
    CELERY_RESULT_BACKEND: str
    LOGS_DIR: str = "logs"
    
    LOG_LEVEL: str = "INFO"


    AWS_REGION: str = "ap-south-1"
    
    
    AWS_S3_BUCKET: str
    
    
    AWS_GPU_INSTANCE_ID: str
    
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    
    GPU_SSH_USER: str
    GPU_PUBLIC_IP: str

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')

settings = Settings()
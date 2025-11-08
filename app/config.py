from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    redis_host: str = "localhost"
    redis_port: int = 6379
    api_base_url: str = "http://localhost:8000"
    """
    Application settings are defined in this class, loaded from environment
    variables or a .env file using pydantic-settings.

    This provides a single, type-safe source of truth for all configuration.
    """
    
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
    GPU_PUBLIC_IP:str
    # Optional explicit blender executable path (set in .env on the host that will run Blender)
    BLENDER_PATH: str | None = None
    # Resource tuning for Blender worker (controls threaded libraries)
    # These defaults are conservative for small EC2 instances. Override in .env per host.
    BLENDER_OMP_THREADS: int = 2
    BLENDER_MKL_THREADS: int = 2
    BLENDER_OPENBLAS_THREADS: int = 2
    BLENDER_BLIS_THREADS: int = 2
    # Niceness to apply when launching Blender on POSIX systems (higher is lower priority)
    BLENDER_NICE: int = 10

  
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8',extra="ignore" )

settings = Settings()
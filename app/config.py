from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
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

    
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    # ── EC2 Instance IDs ───────────────────────────────────────────────────
    # See app/infra.py for all infra constants — these override defaults from .env
    AWS_CPU_INSTANCE_ID: Optional[str] = None          # CPU — FastAPI / orchestrator
    AWS_GPU_INSTANCE_ID: Optional[str] = None          # GPU — g5.2xlarge A10G (active)

    # ── GPU SSH Configuration ──────────────────────────────────────────────
    GPU_A10_PUBLIC_IP: Optional[str] = None            # Public IP of GPU instance
    # SSH key path on the orchestrator (CPU instance) used to reach the GPU instance
    GPU_SSH_KEY_PATH: str = "/home/ubuntu/.ssh/s_spu_key.pem"

    # ── Spot GPU Instance Config (FUTURE — disabled until custom AMI ready) ─
    # SPOT_AMI_ID: Optional[str] = None
    # SPOT_KEY_NAME: Optional[str] = None
    # SPOT_SECURITY_GROUP_IDS: str = ""
    # SPOT_SUBNET_ID: Optional[str] = None
    # SPOT_IAM_INSTANCE_PROFILE: Optional[str] = None
    # SPOT_SSH_USER: str = "ubuntu"
    # SPOT_IDLE_TIMEOUT: int = 300

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

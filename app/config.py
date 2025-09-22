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
    
    # --- MongoDB Configuration ---
    # The full connection string for your MongoDB server.
    # Example for local Docker: "mongodb://localhost:27017/"
    MONGODB_URL: str
    
    # The name of the database to use within your MongoDB instance.
    MONGODB_DB_NAME: str
    
    # --- Celery and Redis Configuration ---
    # The URL for your Redis server, used as the message broker for Celery.
    # Example for local Docker: "redis://localhost:6379/0"
    CELERY_BROKER_URL: str
    
    # The URL for your Redis server, used as the result backend for Celery.
    # It's common to use a different database index (e.g., /1) to keep data separate.
    # Example for local Docker: "redis://localhost:6379/1"
    CELERY_RESULT_BACKEND: str

    # --- Logging Configuration ---
    # The directory where log files will be stored, relative to the project root.
    LOGS_DIR: str = "logs"
    # The minimum level of logs to capture. Can be DEBUG, INFO, WARNING, ERROR, CRITICAL.
    LOG_LEVEL: str = "INFO"

    # --- AWS Configuration ---
    # The default AWS region for services like S3 and EC2.
    AWS_REGION: str = "ap-south-1"
    
    # The S3 bucket where generated assets (images, models) will be stored.
    AWS_S3_BUCKET: str
    
    # The specific EC2 instance ID of the GPU machine to be started and stopped.
    AWS_GPU_INSTANCE_ID: str
    
    # Boto3 will automatically look for these standard environment variables for authentication.
    # You should set them in your .env file.
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str
    
    
    GPU_SSH_USER:str
    GPU_PUBLIC_IP:str

    # This configuration tells Pydantic to look for a .env file and use UTF-8 encoding.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding='utf-8')


# Create a single, importable instance of the settings class.
# Throughout your application, you will import `settings` from `app.config`
# to access configuration variables like `settings.AWS_GPU_INSTANCE_ID`.
settings = Settings()
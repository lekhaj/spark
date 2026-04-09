import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("src_biome_gen.config")

# --- LLM Provider Configuration ---
# Validation is intentionally LAZY — errors are logged at startup but only
# raised when LLM inference is actually attempted.  This prevents a missing
# LLM_PROVIDER / LOCAL_MODEL_PATH env var from crashing the entire Gradio /
# FastAPI app on boot (all other tabs keep working normally).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
VALID_PROVIDERS = {"local", "api", "gemini", "aws"}

# Initialize variables for all providers
OPENAI_API_KEY    = None
OPENAI_MODEL_NAME = None
LOCAL_MODEL_PATH  = None
GEMINI_API_KEY    = None
GEMINI_MODEL      = None
AWS_API           = None
AWS_REGION        = None
AWS_BEDROCK_MODEL = None

# Config error message (set if misconfigured; raised lazily inside llm_inference)
LLM_CONFIG_ERROR: str | None = None

if LLM_PROVIDER not in VALID_PROVIDERS:
    LLM_CONFIG_ERROR = (
        f"LLM_PROVIDER='{LLM_PROVIDER}' is invalid. "
        f"Must be one of: {sorted(VALID_PROVIDERS)}."
    )
    logger.warning(f"[config] {LLM_CONFIG_ERROR}")
elif LLM_PROVIDER == "api":
    OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    if not OPENAI_API_KEY:
        LLM_CONFIG_ERROR = "LLM_PROVIDER is 'api' but OPENAI_API_KEY is not set."
        logger.warning(f"[config] {LLM_CONFIG_ERROR}")
elif LLM_PROVIDER == "local":
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
    if not LOCAL_MODEL_PATH:
        LLM_CONFIG_ERROR = "LLM_PROVIDER is 'local' but LOCAL_MODEL_PATH is not set."
        logger.warning(f"[config] {LLM_CONFIG_ERROR}")
elif LLM_PROVIDER == "gemini":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "models/gemini-pro")
    if not GEMINI_API_KEY:
        LLM_CONFIG_ERROR = "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set."
        logger.warning(f"[config] {LLM_CONFIG_ERROR}")
elif LLM_PROVIDER == "aws":
    AWS_API           = os.getenv("aws_bedrock_api")
    AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
    AWS_BEDROCK_MODEL = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    if not AWS_API:
        LLM_CONFIG_ERROR = "LLM_PROVIDER is 'aws' but AWS_BEDROCK_API / aws_bedrock_api is not set."
        logger.warning(f"[config] {LLM_CONFIG_ERROR}")


# --- MongoDBConfiguration ---
# Support legacy env naming used in the project (.env provides MONGODB_URL / MONGODB_DB_NAME)
# Keep backwards compatibility: prefer MONGODB_* and fall back to MONGO_*
MONGO_URL = os.getenv("MONGODB_URL") or os.getenv("MONGO_URL") or os.getenv("MONGO_URI")
if not MONGO_URL:
    raise ValueError("FATAL ERROR: MONGODB_URL (or MONGO_URL/MONGO_URI) environment variable is not set.")

MONGO_DB_NAME = os.getenv("MONGODB_DB_NAME") or os.getenv("MONGO_DB_NAME") or os.getenv("MONGO_DB") or "biome_db"
MONGO_BIOME_COLLECTION = os.getenv("MONGO_BIOME_COLLECTION", "biomes")

# --- AWS / Bedrock configuration ---
# Support both legacy lowercase 'aws_bedrock_api' and uppercase 'AWS_BEDROCK_API' env names
AWS_API = os.getenv("AWS_BEDROCK_API") or os.getenv("aws_bedrock_api")
AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
AWS_BEDROCK_MODEL = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-5-haiku-20241022-v1:0")

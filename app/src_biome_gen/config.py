import os
from dotenv import load_dotenv

# Load the .env file from the project root.
# This path works assuming you run the app from the `biome-generator/` directory.
load_dotenv()

# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
if LLM_PROVIDER not in ["local", "api", "gemini", "aws"]:
    raise ValueError(f"Invalid LLM_PROVIDER: '{LLM_PROVIDER}'. Must be 'local', 'api', 'gemini', or 'aws'.")


# Initialize variables for all providers
OPENAI_API_KEY = None
OPENAI_MODEL_NAME = None
LOCAL_MODEL_PATH = None
GEMINI_API_KEY = None
GEMINI_MODEL = None
AWS_API = None
AWS_REGION = None
AWS_BEDROCK_MODEL = None


# _LLM_ provider prefixes
if LLM_PROVIDER == "api":
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
    if not OPENAI_API_KEY:
        raise ValueError("FATAL ERROR: LLM_PROVIDER is 'api' but OPENAI_API_KEY is not set.")
elif LLM_PROVIDER == "local":
    LOCAL_MODEL_PATH = os.getenv("LOCAL_MODEL_PATH")
    if not LOCAL_MODEL_PATH:    
        raise ValueError("FATAL ERROR: LLM_PROVIDER is 'local' but LOCAL_MODEL_PATH is not set.")
elif LLM_PROVIDER == "gemini":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-pro")
    if not GEMINI_API_KEY:
        raise ValueError("FATAL ERROR: LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set.")
elif LLM_PROVIDER == "aws":
    AWS_API = os.getenv("aws_bedrock_api")
    AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
    AWS_BEDROCK_MODEL = os.getenv("AWS_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0")
    if not AWS_API:
        raise ValueError("FATAL ERROR: LLM_PROVIDER is 'aws' but AWS_API is not set.")
    if not AWS_REGION:
        raise ValueError("FATAL ERROR: LLM_PROVIDER is 'aws' but AWS_REGION is not set.")
    if not AWS_BEDROCK_MODEL:
        raise ValueError("FATAL ERROR: LLM_PROVIDER is 'aws' but AWS_BEDROCK_MODEL is not set.")


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

import os
from dotenv import load_dotenv

# Load the .env file from the project root.
# This path works assuming you run the app from the `biome-generator/` directory.
load_dotenv()

# --- LLM Provider Configuration ---
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local").lower()
if LLM_PROVIDER not in ["local", "api", "gemini"]:
    raise ValueError(f"Invalid LLM_PROVIDER: '{LLM_PROVIDER}'. Must be 'local', 'api', or 'gemini'.")


# Initialize variables for all providers
OPENAI_API_KEY = None
OPENAI_MODEL_NAME = None
LOCAL_MODEL_PATH = None
GEMINI_API_KEY = None
GEMINI_MODEL = None


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


# --- MongoDBConfiguration ---
MONGO_URL = os.getenv("MONGO_URL")
if not MONGO_URL:
    raise ValueError("FATAL ERROR: MONGO_URL environment variable is not set.")

MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "biome_db")
MONGO_BIOME_COLLECTION = os.getenv("MONGO_BIOME_COLLECTION", "biomes")
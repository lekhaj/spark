import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("rpg_fast_api.src_biome_gen.config")

# ---------------------------------------------------------------------------
# LLM Provider Configuration
#
# RCA note (2026-04-11): This file previously raised ValueError at module
# scope when the provider was misconfigured. That caused the entire Gradio
# app to crash at startup even when the biome-generation feature was never
# used. Fix: validate lazily inside get_provider_config(), called only when
# generate_structured_output() is actually invoked.
# ---------------------------------------------------------------------------

# Valid providers
_VALID_PROVIDERS = {"local", "api", "gemini", "aws"}

# Default changed from "local" (requires a local GPU model) to "aws"
# (Bedrock creds are present on this infra and need no extra setup).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "aws").lower()

if LLM_PROVIDER not in _VALID_PROVIDERS:
    # Log a warning at import time but do NOT raise — still let the app start.
    logger.warning(
        f"LLM_PROVIDER='{LLM_PROVIDER}' is not valid. "
        f"Must be one of {sorted(_VALID_PROVIDERS)}. "
        f"Biome generation will fail when called."
    )

# Pre-read all possible values now (cheap, no side-effects).
# Actual validation happens in get_provider_config() below.
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")

LOCAL_MODEL_PATH  = os.getenv("LOCAL_MODEL_PATH")

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL      = os.getenv("GEMINI_MODEL", "models/gemini-pro")
GEMINI_MODE       = os.getenv("GEMINI_MODE", "vertex")

# Support both casing conventions for Bedrock
AWS_API           = os.getenv("AWS_BEDROCK_API") or os.getenv("aws_bedrock_api")
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
AWS_BEDROCK_MODEL = os.getenv(
    "AWS_BEDROCK_MODEL",
    "anthropic.claude-3-5-haiku-20241022-v1:0"
)


def get_provider_config() -> dict:
    """
    Validate and return the active provider's config.
    Called lazily from generate_structured_output() — NOT at import time.
    Raises ValueError with a clear message if the config is incomplete.
    """
    provider = LLM_PROVIDER

    if provider not in _VALID_PROVIDERS:
        raise ValueError(
            f"LLM_PROVIDER='{provider}' is invalid. "
            f"Set LLM_PROVIDER to one of: {sorted(_VALID_PROVIDERS)}"
        )

    if provider == "api":
        if not OPENAI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is 'api' but OPENAI_API_KEY is not set. "
                "Add OPENAI_API_KEY to your .env file."
            )
        return {"provider": "api", "api_key": OPENAI_API_KEY, "model": OPENAI_MODEL_NAME}

    if provider == "local":
        if not LOCAL_MODEL_PATH:
            raise ValueError(
                "LLM_PROVIDER is 'local' but LOCAL_MODEL_PATH is not set. "
                "Set LOCAL_MODEL_PATH to the model directory, or switch to "
                "LLM_PROVIDER=aws to use Bedrock instead."
            )
        return {"provider": "local", "model_path": LOCAL_MODEL_PATH}

    if provider == "gemini":
        if not GEMINI_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is 'gemini' but GEMINI_API_KEY is not set. "
                "Add GEMINI_API_KEY to your .env file."
            )
        return {"provider": "gemini", "api_key": GEMINI_API_KEY,
                "model": GEMINI_MODEL, "mode": GEMINI_MODE}

    if provider == "aws":
        # Bedrock uses the instance IAM role OR explicit env creds.
        # AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY are already in .env —
        # boto3 picks them up automatically. No explicit check needed here;
        # boto3 will raise a clear NoCredentialsError if they're missing.
        return {
            "provider": "aws",
            "region":   AWS_REGION,
            "model":    AWS_BEDROCK_MODEL,
        }

    # Unreachable but satisfies linters
    raise ValueError(f"Unhandled provider: {provider}")


# ---------------------------------------------------------------------------
# MongoDB Configuration
# ---------------------------------------------------------------------------
# Support legacy env naming used in the project:
# MONGODB_URL > MONGO_URL > MONGO_URI (in priority order)
MONGO_URL = (
    os.getenv("MONGODB_URL")
    or os.getenv("MONGO_URL")
    or os.getenv("MONGO_URI")
    or "mongodb://kartik:Kartikg421@localhost:27017"
)

MONGO_DB_NAME = (
    os.getenv("MONGODB_DB_NAME")
    or os.getenv("MONGO_DB_NAME")
    or os.getenv("MONGO_DB")
    or "World_builder"
)

MONGO_BIOME_COLLECTION = os.getenv("MONGO_BIOME_COLLECTION", "biomes")

# src/text_grid/llm.py

import os
import torch 
import logging
import json
import httpx # For making raw HTTP requests

# Imports for local model (kept for TextProcessor's 'local' model option)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# OpenRouter configuration from config.py
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL 

# Logging setup
logger = logging.getLogger("LLM")
logger.setLevel(logging.DEBUG) # Set to DEBUG for detailed tracing

# Online model client (OpenRouter)
openrouter_client = None
if OPENROUTER_API_KEY and OPENROUTER_BASE_URL:
    openrouter_client = httpx.AsyncClient(base_url=OPENROUTER_BASE_URL)
    logger.info("OpenRouter LLM client initialized.")
else:
    logger.warning("OPENROUTER_API_KEY or OPENROUTER_BASE_URL not set. OpenRouter LLM client not initialized.")

# Local model variables (kept for TextProcessor's 'local' model option)
_local_model = None
_local_tokenizer = None
_local_pipeline = None

def load_local_pipeline():
    """
    Loads the local Qwen 7B model and tokenizer for text generation.
    This is a lazy loading function, so it only loads once.
    """
    global _local_model, _local_tokenizer, _local_pipeline
    if _local_pipeline is None:
        logger.info("🔄 Loading local Qwen 7B model...")
        try:
            _local_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
            _local_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True,
                                                                 torch_dtype=torch.float16, device_map="auto")
            _local_pipeline = pipeline("text-generation", model=_local_model, tokenizer=_local_tokenizer)
            logger.info("Local Qwen 7B model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading local Qwen 7B model: {e}", exc_info=True)
            _local_model = None
            _local_tokenizer = None
            _local_pipeline = None
            raise # Re-raise to indicate failure to caller
    return _local_pipeline

# --- JSON Schema Definition for get_biome_generation_hints ---
# This schema defines the structure that the LLM *must* adhere to.
# It matches the expected output in get_grid_placement_hints_prompt from utils.py
GRID_HINTS_SCHEMA = {
    "type": "object",
    "properties": {
        "biome_name_suggestion": {"type": "string", "description": "A concise, max 4-word biome name."},
        "grid_dimensions": {
            "type": "object",
            "properties": {
                "width": {"type": "integer"},
                "height": {"type": "integer"}
            },
            "required": ["width", "height"]
        },
        "placement_rules": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "structure_id": {"type": "integer", "description": "Unique integer ID of the structure."},
                    "type": {"type": "string", "description": "The name of the structure type."},
                    "min_count": {"type": "integer", "description": "Minimum instances to place."},
                    "max_count": {"type": "integer", "description": "Maximum instances to place."},
                    "size": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                        "description": "[width, height] in grid cells."
                    },
                    "priority_zones": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Preferred locations: 'corner', 'edge', 'center', 'any'."
                    },
                    "adjacent_to_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of structure IDs this should be near."
                    },
                    "avoid_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of structure IDs this should avoid."
                    }
                },
                "required": ["structure_id", "type", "min_count", "max_count", "size", "priority_zones", "adjacent_to_ids", "avoid_ids"]
            }
        },
        "general_density": {"type": "number", "format": "float", "description": "Overall structure density (0.0-1.0)."}
    },
    "required": ["biome_name_suggestion", "grid_dimensions", "placement_rules", "general_density"]
}


# General LLM call function, now with structured output capability
async def call_online_llm(prompt: str, model_name: str = "google/gemini-2.0-flash-exp:free", output_schema: dict = None) -> str:
    """
    Makes a general call to the online LLM (OpenRouter) with a given prompt.
    Supports requesting structured JSON output via `output_schema`.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str): The specific model name to use on OpenRouter.
                         Defaults to "google/gemini-1.5-flash" (a valid OpenRouter ID).
        output_schema (dict): An optional JSON schema to include in the request
                              to guide the LLM's output format.

    Returns:
        str: The raw text or JSON string response from the LLM, or None if an error occurs.
    """
    global openrouter_client

    if openrouter_client is None:
        logger.error("OpenRouter LLM client is not initialized. Cannot make API call.")
        return None

    try:
        chat_messages = []
        chat_messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name, # Now uses the correct OpenRouter ID for Gemini Flash
            "messages": chat_messages,
            "temperature": 0.7, # Adjust as needed for creativity vs. adherence
            "max_tokens": 2048, # Ensure enough tokens for the response
            # OpenRouter's way to request JSON output
            "response_format": {"type": "json_object"}, 
        }

        api_url_path = "/api/v1/chat/completions" # OpenRouter's chat completions endpoint

        logger.debug(f"Calling OpenRouter API: {OPENROUTER_BASE_URL}{api_url_path}")
        logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

        response = await openrouter_client.post(
            api_url_path,
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {OPENROUTER_API_KEY}',
                'HTTP-Referer': 'http://localhost:8080', # Replace with your actual app URL if deployed
                'X-Title': 'TerrainGenApp', # Your app's title
            },
            json=payload,
            timeout=120.0 # Increased timeout for LLM responses
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()

        logger.debug(f"LLM.PY DEBUG: Raw API response object: {result}")

        if result.get('choices') and result['choices'][0].get('message'):
            message_content = result['choices'][0]['message'].get('content')
            
            if message_content and message_content.strip():
                return message_content.strip()
            else:
                # If content is empty, log and return None for robustness
                error_message = result.get('error', {}).get('message', 'Unknown error from OpenRouter LLM (empty content).')
                logger.error(f"[LLM ERROR] OpenRouter response did not contain expected content. Error: {error_message}. Full result: {result}")
                return None
        else:
            error_message = result.get('error', {}).get('message', 'Unknown error from OpenRouter LLM (no choices/message).')
            logger.error(f"[LLM ERROR] OpenRouter response did not contain expected 'choices' or 'message' structure. Error: {error_message}. Full result: {result}")
            return None

    except httpx.HTTPStatusError as e:
        logger.error(f"[LLM ERROR] OpenRouter API HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True) 
        return None
    except httpx.RequestError as e:
        logger.error(f"[LLM ERROR] OpenRouter network error: {e}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[LLM ERROR] An unexpected error occurred during OpenRouter API call: {e}", exc_info=True)
        return None

# The call_llm_for_structure_definitions is specifically for defining structures
# and can remain as is, as its prompt already asks for JSON.
# It now calls the updated call_online_llm with the default model.
async def call_llm_for_structure_definitions(theme_prompt: str, structure_types: list) -> str:
    """
    Calls the LLM to generate fictional structure definitions based on a theme and types.
    The prompt is specifically crafted for this task, now asking for a JSON object.
    """
    # This prompt is designed to elicit JSON output.
    structure_prompt = (
        f"Given the theme '{theme_prompt}', generate 5 fictional structure definitions. "
        f"Each definition must include a 'type' (from: {', '.join(structure_types)}), "
        f"a 'description', and 'attributes' (with at least 'hp'). "
        "Return a single valid JSON object where keys are arbitrary structure names and values are their definitions. "
        "The response MUST contain ONLY the JSON object, no other text or markdown fences." # CRITICAL: No markdown fences
        "\n\nExample Output Format:\n"
        "{\n"
        "  \"Structure A\": {\"type\": \"TypeA\", \"description\": \"DescA\", \"attributes\": {\"hp\": 100}},\n"
        "  \"Structure B\": {\"type\": \"TypeB\", \"description\": \"DescB\", \"attributes\": {\"hp\": 150}}\n"
        "}"
    )
    # This function now calls the OpenRouter LLM with the default model,
    # which is configured for google/gemini-1.5-flash and requests JSON output.
    return await call_online_llm(structure_prompt)


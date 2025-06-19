# src/text_grid/llm.py

import os
import torch 
import logging
import json
import httpx # For making raw HTTP requests

# Imports for local model (kept for TextProcessor's 'local' model option)
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from transformers.utils import is_flash_attn_2_available

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
_local_model = "Qwen/Qwen3-30B-A3B"  # Local model identifier
_local_tokenizer = AutoTokenizer.from_pretrained(_local_model, trust_remote_code=True) if openrouter_client else None
_local_pipeline = None
_local_model_loaded = False # Flag to track successful local model load attempt

def load_local_pipeline():
    """
    Loads the local Qwen 7B model and tokenizer for text generation.
    This is a lazy loading function, so it only loads once.
    Sets _local_model_loaded to True on success, False on failure.
    """
    global _local_model, _local_tokenizer, _local_pipeline, _local_model_loaded
    
    if _local_model_loaded: # If already successfully loaded, just return
        return _local_pipeline
    
    if _local_pipeline is not None: # If already tried loading and failed, don't try again immediately
        return _local_pipeline # Will be None if previous load failed

    logger.info("🔄 Attempting to load local Qwen 7B model...")
    try:
        # Check for Flash Attention 2 for better performance if available
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "eager"
        
        _local_tokenizer = AutoTokenizer.from_pretrained(_local_model, trust_remote_code=True)
        _local_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True,
                                                             torch_dtype=torch.float16, device_map="auto",
                                                             attn_implementation=attn_implementation)
        _local_pipeline = pipeline("text-generation", model=_local_model, tokenizer=_local_tokenizer)
        logger.info("Local Qwen 7B model loaded successfully.")
        _local_model_loaded = True
    except Exception as e:
        logger.error(f"Error loading local Qwen 30B model: {e}", exc_info=True)
        _local_model = None
        _local_tokenizer = None
        _local_pipeline = None
        _local_model_loaded = False # Explicitly set to False on failure
        # Do NOT re-raise here, as we want to gracefully fall back in call_online_llm
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
    Attempts to make a call to the local LLM first.
    If local model is not available or fails, falls back to the online LLM (OpenRouter).
    
    Supports requesting structured JSON output via `output_schema` for the online model.
    For the local model, JSON formatting must be explicitly requested in the prompt.

    Args:
        prompt (str): The prompt to send to the LLM.
        model_name (str): The specific model name to use on OpenRouter (if falling back).
        output_schema (dict): An optional JSON schema for the online model to adhere to.

    Returns:
        str: The raw text or JSON string response from the LLM, or None if an error occurs.
    """
    global openrouter_client, _local_pipeline

    # --- Step 1: Try Local Model First ---
    if not USE_CELERY: # Only attempt local model if in DEV mode (not using Celery workers)
        try:
            local_pipe = load_local_pipeline() # Attempt to load or retrieve local pipeline
            if local_pipe:
                logger.info("Attempting to generate response using local LLM.")
                # Local model expects chat-like messages, even for single prompts
                messages_for_local_model = [
                    {"role": "user", "content": prompt}
                ]
                
                # Generate response from local model
                # Ensure max_new_tokens is sufficient for expected output size
                # The output_schema is ignored here; JSON must be requested in the prompt.
                local_response_raw = local_pipe(messages_for_local_model, max_new_tokens=2048) 
                
                if local_response_raw and isinstance(local_response_raw, list) and local_response_raw[0].get('generated_text'):
                    # Extract only the LLM's response part, not the original prompt and system message
                    generated_text = local_response_raw[0]['generated_text']
                    # Qwen's pipeline often includes the prompt in the generated_text,
                    # so we need to extract only the actual generated part.
                    # This is a common challenge with local transformers pipelines.
                    # A robust way is to find where the assistant's reply starts.
                    
                    # For Qwen, it often looks like: user_prompt ... <|im_start|>assistant\nASSISTANT_RESPONSE<|im_end|>
                    # A simpler heuristic for now: just return the full generated text and rely on
                    # the calling function (e.g., extract_first_json_block) to parse.
                    logger.debug(f"Local LLM raw response: {generated_text}")
                    return generated_text.strip()
                else:
                    logger.warning("Local LLM generated empty or unexpected response structure.")
                    _local_model_loaded = False # Reset if it produced bad output
            else:
                logger.info("Local LLM pipeline not available after loading attempt. Falling back to online.")
        except Exception as e:
            logger.error(f"Error during local LLM inference: {e}", exc_info=True)
            _local_model_loaded = False # Mark as failed so it tries to reload or skips
            logger.info("Local LLM failed. Falling back to online LLM.")
    else:
        logger.info("Skipping local LLM attempt as USE_CELERY is True.")

    # --- Step 2: Fallback to Online Model if Local Fails or Not Used ---
    if openrouter_client is None:
        logger.error("OpenRouter LLM client is not initialized. Cannot make API call for online model.")
        return None

    try:
        chat_messages = []
        chat_messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model_name,
            "messages": chat_messages,
            "temperature": 0.7,
            "max_tokens": 2048,
            # OpenRouter's way to request JSON output via response_format
            "response_format": {"type": "json_object"}, 
        }
        
        # If an explicit output_schema is provided, OpenRouter supports it via 'schema' field
        if output_schema:
            payload["schema"] = output_schema # Some OpenRouter models might directly use this

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
        "   \"Structure A\": {\"type\": \"TypeA\", \"description\": \"DescA\", \"attributes\": {\"hp\": 100}},\n"
        "   \"Structure B\": {\"type\": \"TypeB\", \"description\": \"DescB\", \"attributes\": {\"hp\": 150}}\n"
        "}"
    )
    # This function now calls the OpenRouter LLM (which will try local first)
    # with the default model and expects JSON output.
    return await call_online_llm(structure_prompt) # No change here, it will now use the smart fallback

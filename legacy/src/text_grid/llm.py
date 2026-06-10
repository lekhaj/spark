# src/text_grid/llm.py

import os
import torch
import logging
import json
# httpx is no longer used in this file as there is no call_online_llm
# LlamaTokenizer is imported but not explicitly used if AutoTokenizer handles it
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from transformers.utils import is_flash_attn_2_available
import nltk
nltk.download('punkt', quiet=True, download_dir='/home/ubuntu/nltk_data')  # Download punkt tokenizer models if not already present

# Logging setup
logger = logging.getLogger("LLM")
logger.setLevel(logging.DEBUG) # Keep at DEBUG for detailed tracing

# Local model variables
_local_model_id_str = "/home/ubuntu/sagar/spark/Meta-Llama-3.1-8B-Instruct/original" # Local model identifier (string)
_local_tokenizer_instance = None # Will store the loaded tokenizer object
_local_model_instance = None # Will store the loaded model object
_local_pipeline = None # Will store the loaded transformers pipeline
_local_model_loaded = False # Flag to track successful local model load attempt

def load_local_pipeline():
    """
    Loads the local Llama 3.1 8B model and tokenizer for text generation.
    This is a lazy loading function, so it only loads once per worker process.
    Sets _local_model_loaded to True on success, False on failure.
    This function MUST be synchronous (def), not async def, as it's called in Celery worker init.
    """
    global _local_model_id_str, _local_tokenizer_instance, _local_model_instance, _local_pipeline, _local_model_loaded

    # Check for both pipeline and tokenizer to ensure full readiness
    if _local_model_loaded and _local_pipeline is not None and _local_tokenizer_instance is not None:
        logger.info("Local model pipeline and tokenizer already loaded. Returning existing instances.")
        return _local_pipeline

    logger.info(f"🔄 Attempting to load local model: {_local_model_id_str}...")
    try:
        # Check for Flash Attention 2 for better performance if available
        attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "eager"

        # Load tokenizer and model using the _local_model_id_str (string)
        _local_tokenizer_instance = AutoTokenizer.from_pretrained(_local_model_id_str, trust_remote_code=True)

        # Define the quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_quant_type="nf4", # nf4 is generally preferred over nf16 for better precision
            bnb_8bit_compute_dtype=torch.float16,
            bnb_8bit_use_double_quant=True,
        )

        _local_model_instance = AutoModelForCausalLM.from_pretrained(
            _local_model_id_str,
            trust_remote_code=True,
            quantization_config=bnb_config, # Pass the BitsAndBytesConfig object here
            device_map="auto", # Force use of GPU only
            attn_implementation=attn_implementation
        )
        
        # Explicitly set pad_token_id for the tokenizer if not already set, common for text-generation pipelines
        if _local_tokenizer_instance.pad_token_id is None:
            _local_tokenizer_instance.pad_token_id = _local_tokenizer_instance.eos_token_id

        _local_pipeline = pipeline("text-generation", model=_local_model_instance, tokenizer=_local_tokenizer_instance)
        logger.info(f"Local model '{_local_model_id_str}' loaded successfully.")
        _local_model_loaded = True # Set flag to True on success
    except Exception as e:
        logger.error(f"Error loading local model '{_local_model_id_str}': {e}", exc_info=True)
        # Reset relevant globals on failure
        _local_tokenizer_instance = None
        _local_model_instance = None
        _local_pipeline = None
        _local_model_loaded = False # Explicitly set to False on failure
    return _local_pipeline

# --- JSON Schema Definition for get_biome_generation_hints (remains the same) ---
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

# Only function for LLM calls - exclusively local
def call_local_llm(prompt: str) -> str:
    """
    Calls the local LLM to generate a response.
    This function assumes the local pipeline and tokenizer are already loaded or attempts to load them.
    It applies the chat template with enable_thinking=False.
    Returns the raw generated text (which should be JSON) for the caller to parse.
    """
    global _local_pipeline, _local_tokenizer_instance

    local_pipe = load_local_pipeline()
    if not local_pipe:
        logger.error("Local LLM pipeline is not available. Cannot process request locally.")
        return None
    
    if _local_tokenizer_instance is None:
        logger.error("Local tokenizer is not available. Cannot apply chat template.")
        return None

    try:
        logger.info("Attempting to generate response using local LLM.")
        
        messages_for_local_model = [
            {"role": "user", "content": prompt}
        ]

        templated_prompt = _local_tokenizer_instance.apply_chat_template(
            messages_for_local_model,
            tokenize=False,
            add_generation_prompt=True,
            # This should prevent the model's internal monologue/thinking process
            # for Llama models, this is typically part of their default chat template behavior.
            # Explicit `enable_thinking=False` might not be a direct parameter on all models/tokenizers.
            # The prompt itself should guide the model to output *only* JSON.
        )
        
        logger.debug(f"Templated prompt sent to pipeline:\n---\n{templated_prompt}\n---")

        local_response_raw = local_pipe(
            templated_prompt,
            max_new_tokens=2048, # Keep high enough for full JSON
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_return_sequences=1,
            # Prevent the pipeline from adding its own `eos_token` if the model does not generate it itself
            # This helps in getting the raw generated JSON without truncation or extra tokens.
            # Adjust `return_full_text` to False if you only want the new tokens, not the prompt echo.
            return_full_text=False # This ensures only the *newly generated* text is returned
        )
        
        logger.debug(f"Raw local_pipe output: {local_response_raw}")

        generated_text = None
        if local_response_raw and isinstance(local_response_raw, list) and \
           len(local_response_raw) > 0 and 'generated_text' in local_response_raw[0] and \
           isinstance(local_response_raw[0]['generated_text'], str):
            
            generated_text = local_response_raw[0]['generated_text'].strip()
            logger.debug(f"Full generated text received from pipeline:\n{generated_text}")

            # Now, return this raw generated text. The `utils.extract_first_json_block`
            # function will handle the robust extraction of the JSON from this text.
            return generated_text
        else:
            logger.warning("Local LLM generated empty or unexpected response structure from pipeline.")
            return None
    except Exception as e:
        logger.error(f"Error during local LLM inference: {e}", exc_info=True)
        return None

# This function now exclusively calls the local LLM.
def call_llm_for_structure_definitions(theme_prompt: str, structure_types: list) -> str:
    """
    Calls the LLM to generate fictional structure definitions based on a theme and types.
    This function now exclusively uses the local LLM.
    """
    structure_prompt = (
        f"Given the theme '{theme_prompt}', generate 5 fictional structure definitions. "
        f"Each definition must include a 'type' (from: {', '.join(structure_types)}), "
        f"a 'description', and 'attributes' (with at least 'hp'). "
        f"For each structure, also include an 'adjacent_environmental_objects' field: a list of possible environmental objects (e.g., trees, fences, props, structure-specific items) that can be placed adjacent to this structure. Each object should have:"
        f"- 'type': string (e.g., 'Tree', 'Fence', 'Herb Drying Platform')"
        f"- 'description': string"
        f"- 'attributes': dictionary (e.g., material, size, special properties)"
        "Return a single valid JSON object where keys are arbitrary structure names and values are their definitions. "
        "The response MUST contain ONLY the JSON object, no other text or markdown fences."
        # REMOVED: "\n\nExample Output Format:\n" + "{\n...}"
        # The example was causing the LLM to echo it, leading to JSON parsing errors.
        # The prompt is explicit enough without it.
    )
    # Ensure this calls the local LLM
    return call_local_llm(structure_prompt)

def call_llm_for_placement_hints(placement_prompt: str) -> str:
    """
    Calls the local LLM to generate placement hints based on the placement prompt.
    Returns the raw generated text (which should be JSON) for the caller to parse.
    """
    return call_local_llm(placement_prompt)
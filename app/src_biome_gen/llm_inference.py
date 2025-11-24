import logging
import os
from . import config
import requests
import boto3
import json

def get_logger():
    return logging.getLogger("rpg_fast_api.llm_inference")
logger = get_logger()

# -- conditional import for LLM -- 
if config.LLM_PROVIDER == "api":
    import openai
    # Initialize the OpenAI client once
    try:
        openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        openai_client = None

elif config.LLM_PROVIDER == "local":
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
    from transformers.utils import is_flash_attn_2_available
    
    _local_pipeline = None
    _local_tokenizer = None


# --- Private Helper Functions ---

def _call_openai_api(prompt: str) -> str | None:
    """Sends a prompt to the OpenAI API and returns the response."""
    if not openai_client:
        logger.error("OpenAI client not initialized. Cannot make API call.")
        return None
    try:
        logger.info("Sending request to OpenAI API...")
        response = openai_client.chat.completions.create(
            model=config.OPENAI_MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are an expert assistant that only responds in valid JSON format."},
                {"role": "user", "content": prompt},
            ],
            # Use JSON mode for guaranteed valid JSON output
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        content = response.choices[0].message.content
        logger.info("Successfully received response from OpenAI API.")
        return content
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return None


def _call_local_model(prompt: str) -> str | None:
    """
    Sends a prompt to the local transformer model and returns the response.
    Uses a singleton pattern to lazily load the model on the first call.
    """
    global _local_pipeline, _local_tokenizer

    if _local_pipeline is None:
        try:
            # local pipeline loading (only for in house models)
            logger.info("First call detected. Loading local LLM pipeline...")
            attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(config.LOCAL_MODEL_PATH)
            model = AutoModelForCausalLM.from_pretrained(
                config.LOCAL_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation=attn_implementation
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            _local_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
            _local_tokenizer = tokenizer
            logger.info("Local LLM pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local LLM pipeline: {e}", exc_info=True)
            return None # Prevent further execution if loading fails

    # --- Generate text with the loaded pipeline ---
    try:
        logger.info("Sending request to local LLM...")
        messages = [{"role": "user", "content": prompt}]
        templated_prompt = _local_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        output = _local_pipeline(
            templated_prompt,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )
        content = output[0]['generated_text'].strip()
        logger.info("Successfully received response from local LLM.")
        return content
    except Exception as e:
        logger.error(f"Error during local LLM inference: {e}", exc_info=True)
        return None

def _call_claude_bedrock(prompt: str, max_tokens: int = 8000) -> str | None:
    """
    Sends a single-turn message to Anthropic Claude via AWS Bedrock and returns the response text.
    """
    try:
        bedrock_runtime = boto3.client(service_name="bedrock-runtime", region_name=config.AWS_REGION)
        model_id = config.AWS_BEDROCK_MODEL
        user_theme = prompt.strip()
        messages = [
            {
                "role": "user",
                "content": [
                    {"text": user_theme}
                ]
            }
        ]
        response = bedrock_runtime.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": 0.7
            }
        )
        # Parse response as in test_bedrock_inference.py
        response_body = response.get('output').get('message').get('content')[0].get('text')
        return response_body
    except Exception as e:
        logger.error(f"Claude Bedrock API call failed: {e}")
        return None

# --- Public Function ---
# this is the function that biome_generation calls :) ---

def generate_structured_output(prompt: str) -> str | None:
    """
    The main, public function to generate a response from the configured LLM.
    """
    logger.info(f"Generating structured output using '{config.LLM_PROVIDER}' provider.")
    provider = config.LLM_PROVIDER.lower()
    if provider == "api":
        return _call_openai_api(prompt)
    elif provider == "local":
        return _call_local_model(prompt)
    elif provider == "gemini":
        gemini_mode = getattr(config, "GEMINI_MODE", "vertex").lower()
        if gemini_mode == "vertex":
            try:
                import google.generativeai as genai
                api_key = config.GEMINI_API_KEY
                if not api_key:
                    logger.error("GEMINI_API_KEY not set for Vertex Gemini.")
                    return None
                genai.configure(api_key=api_key)
                model_name = getattr(config, "GEMINI_MODEL", "gemini-1.5-flash-latest")
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                return response.text
            except Exception as e:
                logger.error(f"Vertex Gemini API call failed: {e}")
                return None
        else:  # fallback to REST
            url = f"https://generativelanguage.googleapis.com/v1beta/{config.GEMINI_MODEL}:generateContent?key={config.GEMINI_API_KEY}"
            headers = {"Content-Type": "application/json"}
            data = {
                "contents": [
                    {"parts": [{"text": prompt}]}
                ]
            }
            try:
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                result = response.json()
                return result["candidates"][0]["content"]["parts"][0]["text"]
            except Exception as e:
                logger.error(f"Gemini REST API call failed: {e}")
                return None
    elif provider == "aws":
        return _call_claude_bedrock(prompt)
    else:
        logger.error("Invalid LLM_PROVIDER configured.")
        return None

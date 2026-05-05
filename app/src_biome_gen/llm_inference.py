import logging
import os
from . import config
import requests
import boto3
import json

def get_logger():
    return logging.getLogger("rpg_fast_api.llm_inference")
logger = get_logger()

# ---------------------------------------------------------------------------
# RCA note (2026-04-11): Previously, provider-specific imports (torch,
# transformers, openai) ran at module scope, so importing this file would
# crash if those packages weren't installed. Moved all provider imports
# inside their respective call functions (lazy import). The app starts
# regardless of which packages are installed; missing packages only error
# when the specific provider is actually called.
# ---------------------------------------------------------------------------


# --- Private Helper Functions ---

def _call_openai_api(prompt: str) -> str | None:
    """Sends a prompt to the OpenAI API and returns the response."""
    try:
        import openai
        cfg = config.get_provider_config()
        client = openai.OpenAI(api_key=cfg["api_key"])
        response = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": "You are an expert assistant that only responds in valid JSON format."},
                {"role": "user", "content": prompt},
            ],
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
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
        from transformers.utils import is_flash_attn_2_available
    except ImportError as e:
        logger.error(f"Local model dependencies not installed: {e}. "
                     f"Install torch + transformers, or switch LLM_PROVIDER=aws.")
        return None

    cfg = config.get_provider_config()
    model_path = cfg["model_path"]

    # Singleton pipeline — loaded once per process
    if not hasattr(_call_local_model, "_pipeline"):
        _call_local_model._pipeline = None
        _call_local_model._tokenizer = None

    if _call_local_model._pipeline is None:
        try:
            logger.info("First call: loading local LLM pipeline...")
            attn_implementation = "flash_attention_2" if is_flash_attn_2_available() else "eager"
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_compute_dtype=torch.float16,
                bnb_8bit_use_double_quant=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                attn_implementation=attn_implementation,
            )
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = tokenizer.eos_token_id
            _call_local_model._pipeline  = pipeline("text-generation", model=model, tokenizer=tokenizer)
            _call_local_model._tokenizer = tokenizer
            logger.info("Local LLM pipeline loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load local LLM pipeline: {e}", exc_info=True)
            return None

    try:
        logger.info("Sending request to local LLM...")
        messages = [{"role": "user", "content": prompt}]
        templated_prompt = _call_local_model._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = _call_local_model._pipeline(
            templated_prompt,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False,
        )
        content = output[0]["generated_text"].strip()
        logger.info("Successfully received response from local LLM.")
        return content
    except Exception as e:
        logger.error(f"Error during local LLM inference: {e}", exc_info=True)
        return None


def _call_claude_bedrock(prompt: str, max_tokens: int = 8000) -> str | None:
    """Sends a message to Anthropic Claude via AWS Bedrock."""
    try:
        cfg = config.get_provider_config()
        bedrock_runtime = boto3.client(
            service_name="bedrock-runtime",
            region_name=cfg["region"],
        )
        response = bedrock_runtime.converse(
            modelId=cfg["model"],
            messages=[{"role": "user", "content": [{"text": prompt.strip()}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": 0.7},
        )
        return response["output"]["message"]["content"][0]["text"]
    except Exception as e:
        logger.error(f"Claude Bedrock API call failed: {e}", exc_info=True)
        return None


def _call_gemini(prompt: str) -> str | None:
    """Sends a prompt to Google Gemini."""
    try:
        cfg = config.get_provider_config()
        if cfg.get("mode", "vertex") == "vertex":
            import google.generativeai as genai
            genai.configure(api_key=cfg["api_key"])
            model = genai.GenerativeModel(cfg["model"])
            return model.generate_content(prompt).text
        else:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/"
                f"{cfg['model']}:generateContent?key={cfg['api_key']}"
            )
            resp = requests.post(
                url,
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}", exc_info=True)
        return None


# --- Public Function ---

def generate_structured_output(prompt: str) -> str | None:
    """
    Main public function — routes to the configured LLM provider.
    Validation happens here (lazily), not at import time.
    """
    try:
        config.get_provider_config()   # validate early, clear error if misconfigured
    except ValueError as e:
        logger.error(f"LLM provider misconfigured: {e}")
        return None

    provider = config.LLM_PROVIDER
    logger.info(f"Generating structured output using '{provider}' provider.")

    if provider == "api":
        return _call_openai_api(prompt)
    elif provider == "local":
        return _call_local_model(prompt)
    elif provider == "gemini":
        return _call_gemini(prompt)
    elif provider == "aws":
        return _call_claude_bedrock(prompt)
    else:
        logger.error(f"Unhandled LLM_PROVIDER: '{provider}'")
        return None

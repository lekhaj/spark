# llm.py
import os
import torch
import logging
import json
from openai import OpenAI, OpenAIError
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL

# Logging setup
logger = logging.getLogger("LLM")
logger.setLevel(logging.INFO)

# Online model client
online_client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url=OPENROUTER_BASE_URL
)

# Load local model (lazy)
_local_model = None
_local_tokenizer = None
_local_pipeline = None

def load_local_pipeline():
    global _local_model, _local_tokenizer, _local_pipeline
    if _local_pipeline is None:
        logger.info("🔄 Loading local Qwen 7B model...")
        _local_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
        _local_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat", trust_remote_code=True)
        _local_pipeline = pipeline("text-generation", model=_local_model, tokenizer=_local_tokenizer)
    return _local_pipeline

def call_structure_generator(prompt: str) -> list:
    structure_prompt = (
        f"Given the theme '{prompt}', generate 5 fictional structure definitions. "
        "Return them in JSON array format, each with 'type', 'description', and 'attributes' (with at least 'hp')."
    )
    return call_online_llm(structure_prompt)


def call_online_llm(prompt: str):
    try:
        response = online_client.chat.completions.create(
            model="qwen/qwen3-30b-a3b:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=1500,
            top_p=0.95
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        print(f"[LLM ERROR] {e}")
        return None

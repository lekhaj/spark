import random
from llm import call_structure_generator
from utils import get_suggestion_prompt, extract_first_json_block
import json

from structure_registry import get_matching_structures
# placement_suggestor.py


def suggest_structure_placement(structure_definitions, theme_prompt):
    prompt = get_suggestion_prompt(list(structure_definitions.keys()), theme_prompt)

    suggestion = call_structure_generator(prompt)  # your LLM call

    if not suggestion.strip():
        raise ValueError("[ERROR] LLM returned an empty response")

    try:
        cleaned = extract_first_json_block(suggestion)
        if not cleaned:
            raise ValueError("No JSON block found in LLM response")
        parsed = json.loads(cleaned)
    except Exception as e:
        raise ValueError(f"[ERROR] Failed to parse LLM response: {e}\nRaw: {suggestion}")

    layout = parsed.get("layout", [])
    biome_name = parsed.get("biome_name", "Unnamed Biome")
    return layout, biome_name

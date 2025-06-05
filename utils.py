import os
import json
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Logging setup
LOG_FILE = "logs/biome_generator.log"
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO)


def extract_first_json_block(text):
    """
    Extract the first valid JSON object from a string using bracket matching.
    """
    brace_stack = []
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                start_idx = i
            brace_stack.append('{')
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack:
                    return text[start_idx:i+1]
    return None


def get_suggestion_prompt(structure_types, theme_prompt):
    return (
        f"You are generating a 10x10 layout for a biome themed around: '{theme_prompt}'.\n\n"
        f"The available structure types are: {', '.join(structure_types)}.\n\n"
        "Your task:\n"
        "1. Generate a creative and concise biome name (max 4 words).\n"
        "2. Generate a 10x10 2D layout using structure IDs (integers), using 0 for empty tiles.\n\n"
        "Return your output strictly as JSON in the format:\n"
        "{\n"
        '  "biome_name": "Your generated biome name",\n'
        '  "layout": [[1, 0, 2, ..., 0], [...], ...]\n'
        "}"
    )


def build_structure_definitions(structure_defs, start_id):
    """
    Converts structure definitions into an ID-keyed dictionary.
    Accepts either a stringified JSON or a list/dict of structure definitions.
    """
    if isinstance(structure_defs, str):
        try:
            cleaned_json = extract_first_json_block(structure_defs)
            if not cleaned_json:
                raise ValueError("No valid JSON block found.")
            structure_defs = json.loads(cleaned_json)
        except Exception as e:
            print(f"[ERROR] Invalid JSON from LLM: {e}")
            print(f"[DEBUG] Raw input: {structure_defs}")
            raise ValueError("Failed to decode structure_defs JSON string")

    if isinstance(structure_defs, dict):
        structure_defs = list(structure_defs.values())
    elif not isinstance(structure_defs, list):
        raise ValueError("Expected structure_defs to be a list or dict of structures.")

    structured = {}
    current_id = start_id
    for struct in structure_defs:
        if not isinstance(struct, dict):
            print(f"[DEBUG] Invalid struct in list: {struct}")
            raise ValueError(f"[ERROR] Each structure must be a dict. Got: {type(struct)}")
        structured[str(current_id)] = {
            "type": struct["type"],
            "description": struct["description"],
            "attributes": struct["attributes"]
        }
        current_id += 1
    return structured
def get_structure_definition_prompt(theme_prompt, structure_types):
    return (
        f"You are an intelligent generator for biome structure metadata.\n"
        f"The biome theme is: '{theme_prompt}'.\n"
        f"The structure types required are: {', '.join(structure_types)}.\n\n"
        "For each structure type, return a JSON dictionary containing:\n"
        "- 'type': name of the structure (string)\n"
        "- 'description': short description of its role in the biome (string)\n"
        "- 'attributes': a dictionary of relevant properties (e.g. material, usage, magical affinity)\n\n"
        "Return a single valid JSON object where keys are structure type names, and values are their definitions.\n\n"
        "Example output:\n"
        "{\n"
        "  \"Herbalist's Hut\": {\n"
        "    \"type\": \"Herbalist's Hut\",\n"
        "    \"description\": \"A small wooden hut used to prepare herbal remedies.\",\n"
        "    \"attributes\": {\"HP\"}\n"
        "  },\n"
        "  \"Forest Shrine\": {\n"
        "    \"type\": \"Forest Shrine\",\n"
        "    \"description\": \"A mystical shrine to forest spirits.\",\n"
        "    \"attributes\": {\"HP\"}\n"
        "  }\n"
        "}"
    )


def generate_biome_name_from_prompt(prompt: str) -> str:
    """
    Generates a concise biome name using the first few keywords from the prompt.
    """
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(prompt.lower())
    keywords = [w.capitalize() for w in words if w.isalpha() and w not in stop_words]
    name = " ".join(keywords[:4])  # Allow up to 4 words
    return name or "Generated Biome"

# src/text_grid/utils.py
import os
import json
import logging
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from . import llm # Relative import

# Download necessary NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Logging setup
# Adjust log file path to be relative to the project root, not src/text_grid
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
# FIX: Set logging level to DEBUG for more verbose output during debugging
logging.basicConfig(filename=os.path.join(LOG_DIR, "biome_generator.log"), level=logging.DEBUG) 

# Define the logger at the module level
logger = logging.getLogger(__name__)


def extract_first_json_block(text):
    """
    Extract the first valid JSON object from a string using bracket matching.
    Improved to handle markdown code blocks.
    """
    # First, try to strip markdown code block fences if they exist
    if text.strip().startswith('```json') and text.strip().endswith('```'):
        text = text.strip()[len('```json'):-len('```')].strip()
    elif text.strip().startswith('```') and text.strip().endswith('```'):
        text = text.strip()[len('```'):-len('```')].strip()

    brace_stack = []
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if not brace_stack:
                start_idx = i
            brace_stack.append('{')
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_idx != -1:
                    return text[start_idx:i+1]
    return None


# --- NEW or MODIFIED LLM PROMPT FUNCTIONS ---

def get_grid_placement_hints_prompt(structure_definitions, theme_prompt, grid_dimensions=(10, 10)):
    """
    Generates a prompt for the LLM to get high-level placement hints and biome name.
    It expects a JSON output with biome name, grid dimensions, placement rules, and general density.
    """
    # Create a string representation of structure IDs to types for LLM to use
    structure_info = "\n".join([
        f"{idx}: {details['type']} (Description: {details['description']})"
        for idx, details in structure_definitions.items()
    ])

    return (
        f"You are an expert game environment designer, specializing in procedural generation.\n"
        f"The biome theme is: '{theme_prompt}'.\n"
        f"The grid size is {grid_dimensions[0]}x{grid_dimensions[1]}.\n"
        f"Available structures and their IDs:\n{structure_info}\n\n"
        "Your task is to suggest strategic placement rules and parameters for these structures "
        "to create a realistic and thematic layout, without generating the actual grid.\n"
        "Consider density, clustering, relation to edges/corners, and proximity between structure types.\n"
        "Use 0 for empty tiles if you need to refer to empty spaces in your rules.\n\n"
        "Return your output strictly as a JSON object with the following keys, "
        "enclosed within a markdown code block for JSON:\n"
        "```json\n"
        "{\n"
        '   "biome_name_suggestion": "Your generated biome name (max 4 words)",\n'
        '   "grid_dimensions": {"width": 10, "height": 10},\n'
        '   "placement_rules": [\n'
        '     { "structure_id": <int>, "type": "<str>", "min_count": <int>, "max_count": <int>, "size": [<int>, <int>], "priority_zones": ["corner", "edge", "center", "any"], "adjacent_to_ids": [<int>], "avoid_ids": [<int>] }\n'
        '   ],\n'
        '   "general_density": <float> // (0.0 to 1.0, overall structure density)\n'
        '}\n'
        "```\n"
        "Briefly describe each parameter within the JSON, e.g., 'priority_zones': list of preferred locations like 'corner', 'edge', 'center', 'any'. "
        "The `size` is [width, height], `adjacent_to_ids` means structures should be near these IDs, `avoid_ids` means structures should avoid these IDs. "
        "Keep `max_count` realistic for a 10x10 grid (e.g., total count should not exceed grid_size * density). "
        "Ensure `structure_id` and `type` match the provided structure_info exactly."
    )

def parse_llm_hints(llm_response: str):
    """
    Parses the LLM's JSON response containing grid placement hints.
    Includes robust error handling and sets default values if parsing fails or data is missing.
    """
    current_logger = logging.getLogger(__name__) # Ensure logger is accessible
    current_logger.info(f"Raw LLM response for hints:\n{llm_response}") # Added print for debugging
    try:
        cleaned = extract_first_json_block(llm_response)
        if not cleaned:
            raise ValueError("No JSON block found in LLM response for hints")
        parsed = json.loads(cleaned)

        # Basic validation and default values for grid dimensions
        grid_dims = parsed.get("grid_dimensions", {"width": 10, "height": 10})
        if not isinstance(grid_dims, dict) or "width" not in grid_dims or "height" not in grid_dims:
             grid_dims = {"width": 10, "height": 10} # Default if invalid

        rules = []
        for r in parsed.get("placement_rules", []):
            try: # Try parsing each rule robustly
                rule_id = int(r.get("structure_id")) if isinstance(r.get("structure_id"), (str, int)) else 0
                rule_type = str(r.get("type", "unknown"))
                min_c = int(r.get("min_count", 0))
                max_c = int(r.get("max_count", 1))
                size_w, size_h = (1,1)
                if isinstance(r.get("size"), list) and len(r["size"]) == 2:
                    size_w, size_h = int(r["size"][0]), int(r["size"][1])
                p_zones = [str(z) for z in r.get("priority_zones", ["any"])]
                adj_to = [int(i) for i in r.get("adjacent_to_ids", []) if isinstance(i, (str, int))]
                avoid = [int(i) for i in r.get("avoid_ids", []) if isinstance(i, (str, int))]

                rules.append({
                    "structure_id": rule_id,
                    "type": rule_type,
                    "min_count": min_c,
                    "max_count": max_c,
                    "size": [size_w, size_h],
                    "priority_zones": p_zones,
                    "adjacent_to_ids": adj_to,
                    "avoid_ids": avoid
                })
            except Exception as rule_e:
                current_logger.warning(f"Skipping malformed placement rule: {r} - Error: {rule_e}")
        
        general_density = float(parsed.get("general_density", 0.3))

        return {
            "biome_name_suggestion": parsed.get("biome_name_suggestion", "Unnamed Biome"),
            "grid_dimensions": grid_dims,
            "placement_rules": rules,
            "general_density": general_density
        }

    except Exception as e:
        current_logger.error(f"[ERROR] Failed to parse LLM hints: {e}\nRaw: {llm_response}")
        # Return a default, basic set of hints if parsing fails
        return {
            "biome_name_suggestion": "Default Biome",
            "grid_dimensions": {"width": 10, "height": 10},
            "placement_rules": [], # No specific rules
            "general_density": 0.3
        }


# --- Existing Utility Functions (no change from previous update) ---
def build_structure_definitions(structure_defs_raw_str, start_id): # Renamed for clarity
    """
    Converts structure definitions into an ID-keyed dictionary.
    Accepts a stringified JSON (expected to be a single object where keys are structure names).
    """
    current_logger = logging.getLogger(__name__) # Ensure logger is accessible
    current_logger.info(f"Raw LLM response for structure definitions:\n{structure_defs_raw_str}") # Added print for debugging
    
    # --- ADDED: Check for None or empty string before processing ---
    if structure_defs_raw_str is None or not structure_defs_raw_str.strip():
        raise ValueError("LLM response for structure definitions is empty or None. Cannot process.")

    parsed_json = None
    try:
        cleaned_json = extract_first_json_block(structure_defs_raw_str)
        if not cleaned_json:
            raise ValueError("No valid JSON block found in LLM response for structure definitions.")
        parsed_json = json.loads(cleaned_json)
    except json.JSONDecodeError as e:
        current_logger.error(f"[ERROR] JSON decoding failed for structure definitions: {e}")
        current_logger.debug(f"[DEBUG] Raw input: {structure_defs_raw_str}")
        raise ValueError("Failed to decode structure_defs JSON string. Invalid JSON format.")
    except Exception as e:
        current_logger.error(f"[ERROR] An unexpected error occurred during JSON extraction/parsing for structure definitions: {e}")
        current_logger.debug(f"[DEBUG] Raw input: {structure_defs_raw_str}")
        raise ValueError(f"Failed to process raw structure definitions: {e}")

    # Now, explicitly check if the parsed JSON is a dictionary, as per the prompt's example
    if not isinstance(parsed_json, dict):
        current_logger.error(f"[ERROR] LLM returned unexpected type for structure definitions. Expected dict, got {type(parsed_json)}")
        current_logger.debug(f"[DEBUG] Parsed JSON: {parsed_json}")
        raise ValueError("Expected LLM response for structure definitions to be a dictionary.")

    # Convert the dictionary of structures into a list of structure dictionaries (their values)
    structure_list = list(parsed_json.values())
    
    structured = {}
    current_id = start_id
    for struct in structure_list:
        if not isinstance(struct, dict):
            current_logger.debug(f"[DEBUG] Invalid struct in list: {struct}")
            raise ValueError(f"[ERROR] Each structure in the LLM-provided list must be a dict. Got: {type(struct)}")
        
        # Ensure 'attributes' is a dictionary if it comes as something else
        attributes = struct.get("attributes", {})
        if not isinstance(attributes, dict):
            current_logger.warning(f"Attributes for structure type '{struct.get('type')}' is not a dict. Defaulting to empty dict.")
            attributes = {}

        structured[str(current_id)] = {
            "type": struct.get("type", "Unnamed Structure"), # Add .get with default for robustness
            "description": struct.get("description", ""),
            "attributes": attributes
        }
        current_id += 1
    return structured

def get_structure_definition_prompt(theme_prompt, structure_types):
    """
    Generates a prompt for the LLM to get detailed structure definitions.
    """
    return (
        f"You are an intelligent generator for biome structure metadata.\n"
        f"The biome theme is: '{theme_prompt}'.\n"
        f"The structure types required are: {', '.join(structure_types)}.\n\n"
        "For each structure type, return a JSON dictionary containing:\n"
        "- 'type': name of the structure (string)\n"
        "- 'description': short description of its role in the biome (string)\n"
        "- 'attributes': a dictionary of relevant properties (e.g. material, usage, magical affinity, HP)\n\n"
        "Return a single valid JSON object where keys are structure type names, and values are their definitions.\n\n"
        "Example output:\n"
        "{\n"
        "   \"Herbalist's Hut\": {\n"
        "     \"type\": \"Herbalist's Hut\",\n"
        "     \"description\": \"A small wooden hut used to prepare herbal remedies.\",\n"
        "     \"attributes\": {\"hp\": 50}\n"
        "   },\n"
        "   \"Forest Shrine\": {\n"
        "     \"type\": \"Forest Shrine\",\n"
        "     \"description\": \"A mystical shrine to forest spirits.\",\n"
        "     \"attributes\": {\"hp\": 100}\n"
        "   }\n"
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
# src/text_grid/utils.py
import os
import json
import logging
import re # Added for more robust JSON extraction
from datetime import datetime
import nltk
os.environ["NLTK_DATA"] = "/home/ubuntu/nltk_data"
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import demjson3
# Note: Removed relative import of .llm as it's not directly used in utils for LLM calls
# and was causing potential circular dependency issues.

# Download necessary NLTK resources
# Ensure download_dir is accessible and persistent on your worker.
nltk.download("punkt", quiet=True, download_dir='/home/ubuntu/nltk_data')
nltk.download("stopwords", quiet=True, download_dir='/home/ubuntu/nltk_data')

# Logging setup
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "biome_generator.log"), level=logging.DEBUG)

logger = logging.getLogger(__name__)

def clean_llm_output(text):
    """
    Removes common chat template artifacts and special tokens from LLM output.
    """
    # Remove lines starting with <| or similar artifacts
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not line.strip().startswith('<|')]
    # Remove empty lines
    cleaned = "\n".join(line for line in cleaned_lines if line.strip())
    return cleaned

def extract_first_json_block(text):
    """
    Extract the first valid JSON object from a string.
    Returns only the substring from the first '{' to its matching '}' (balanced).
    Ignores any trailing text after the closing brace.
    """
    # Try to extract from markdown code blocks first
    json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1).strip()

    # Fallback: Find the first balanced {...} block
    brace_stack = []
    start_idx = None
    for i, char in enumerate(text):
        if char == '{':
            if start_idx is None:
                start_idx = i
            brace_stack.append('{')
        elif char == '}':
            if brace_stack:
                brace_stack.pop()
                if not brace_stack and start_idx is not None:
                    # Return only the first complete JSON object
                    return text[start_idx:i+1]
    # Fallback: Try to extract a placement_rules array fragment
    match = re.search(r'"placement_rules"\s*:\s*\[.*?\]', text, re.DOTALL)
    if match:
        return '{' + match.group(0) + '}'
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

    # IMPORTANT: The prompt is now even more aggressive in demanding ONLY JSON output.
    # Removed explicit `--- JSON_OUTPUT_START ---` and `--- JSON_OUTPUT_END ---`
    # as the `extract_first_json_block` handles ```json` fences primarily.
    return (
        f"```json\n"
        f"{{\n"
        f'   "biome_name_suggestion": "Your generated biome name (max 4 words)",\n'
        f'   "grid_dimensions": {{"width": {grid_dimensions[0]}, "height": {grid_dimensions[1]}}},\n'
        f'   "placement_rules": [\n'
        f'     {{"structure_id": <int>, "type": "<str>", "min_count": <int>, "max_count": <int>, "size": [<int>, <int>], "priority_zones": ["corner", "edge", "center", "any"], "adjacent_to_ids": [<int>], "avoid_ids": [<int>] }}\n'
        f'   ],\n'
        f'   "general_density": <float> // (0.0 to 1.0, overall structure density)\n'
        f'}}\n'
        f"```\n"
        f"As an expert game environment designer specializing in procedural generation, "
        f"your task is to suggest strategic placement rules and parameters for a "
        f"{grid_dimensions[0]}x{grid_dimensions[1]} grid based on the theme: '{theme_prompt}'.\n"
        f"Available structures and their IDs:\n{structure_info}\n\n"
        "Crucial Guidelines:\n"
        "- Do NOT generate the actual grid. Provide only the rules.\n"
        "- Consider density, clustering, relation to edges/corners, and proximity between structure types.\n"
        "- Use 0 for empty tiles when referring to space.\n"
        "- `structure_id`: Must match IDs from the provided list.\n"
        "- `type`: Must match types from the provided list.\n"
        "- `min_count`, `max_count`: Number of instances of this structure.\n"
        "- `size`: [width, height] for the structure's footprint (e.g., [1,1] for a single tile, [2,2] for a 2x2 building).\n"
        "- `priority_zones`: List of preferred locations: 'corner', 'edge', 'center', 'any'.\n"
        "- `adjacent_to_ids`: List of structure IDs this one should be near (Manhattan distance <= 3).\n"
        "- `avoid_ids`: List of structure IDs this one should avoid (Manhattan distance <= 2).\n"
        "**VERY IMPORTANT**: Ensure `max_count` and `size` values are realistic so that the sum of `max_count * size[0] * size[1]` for all structures does not significantly exceed `grid_width * grid_height * general_density`."
    )

def parse_llm_hints(llm_response: str):
    """
    Parses the LLM's JSON response or fragment containing grid placement hints.
    Handles both full JSON objects and fragments like "placement_rules": [...]
    Includes robust error handling and sets default values if parsing fails or data is missing.
    """
    import re
    current_logger = logging.getLogger(__name__)
    current_logger.info(f"Raw LLM response for hints:\n{llm_response}")

    try:
        cleaned_response = clean_llm_output(llm_response)
        cleaned = extract_first_json_block(cleaned_response)
        
        # If no full JSON block, try to extract a placement_rules array fragment
        if not cleaned:
            match = re.search(r'"placement_rules"\s*:\s*\[.*?\]', llm_response, re.DOTALL)
            if match:
                # Wrap the fragment in a JSON object
                cleaned = '{' + match.group(0) + '}'
            else:
                raise ValueError("No JSON block or placement_rules fragment found in LLM response for hints")

        parsed = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as e:
            current_logger.warning(f"Initial JSON parse failed for hints. Attempting repair: {e}")
            repaired_json = _repair_json_string(cleaned)
            try:
                parsed = json.loads(repaired_json)
                current_logger.info("JSON hints successfully parsed after repair.")
            except json.JSONDecodeError as repair_e:
                current_logger.error(f"[ERROR] JSON repair failed for hints: {repair_e}\nRepaired string: {repaired_json}")
                raise ValueError(f"Failed to parse LLM hints even after repair: {repair_e}")

        if parsed is None:
            raise ValueError("Parsed JSON is None after all attempts.")

        # If only placement_rules are present, fill in defaults for other fields
        grid_dims = parsed.get("grid_dimensions", {"width": 10, "height": 10})
        if not isinstance(grid_dims, dict) or "width" not in grid_dims or "height" not in grid_dims:
            grid_dims = {"width": 10, "height": 10}

        rules = []
        for r in parsed.get("placement_rules", []):
            try:
                rule_id = int(r.get("structure_id")) if isinstance(r.get("structure_id"), (str, int)) else 0
                rule_type = str(r.get("type", "unknown"))
                min_c = int(r.get("min_count", 0))
                max_c = int(r.get("max_count", 1))
                size_w, size_h = (1, 1)
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
        return {
            "biome_name_suggestion": "Default Biome",
            "grid_dimensions": {"width": 10, "height": 10},
            "placement_rules": [],
            "general_density": 0.3
        }
# --- NEW: JSON Repair Helper Function ---
def _repair_json_string(malformed_json_str: str) -> str:
    """
    Attempts to repair common JSON errors, including:
    - unquoted string values (including those with spaces/units)
    - missing commas between object fields or array items
    - removes comments (// ...)
    """
    logger.info("Attempting to repair malformed JSON string...")

    # 0. Remove comments (// ...)
    repaired_str = re.sub(r'//.*', '', malformed_json_str)

    # 1. Fix unquoted string values (single words)
    pattern = re.compile(r'("[a-zA-Z_][a-zA-Z0-9_]*"\s*:\s*)([a-zA-Z_][a-zA-Z0-9_]*)([\s,\]\}])')
    def replacer(match):
        before = match.group(1)
        value = match.group(2)
        after = match.group(3)
        # Only quote if not a JSON literal
        if value not in ['true', 'false', 'null'] and not value.isdigit():
            value = f'"{value}"'
        return f"{before}{value}{after}"
    repaired_str = pattern.sub(replacer, repaired_str)

    # 1b. Fix unquoted string values with spaces/units (e.g., 100TB, 1000 credits/month, 500 meters)
    # This pattern matches: "key": value with spaces or units (not quoted)
    pattern_units = re.compile(r'("([a-zA-Z_][a-zA-Z0-9_]*)"\s*:\s*)([^\[\{"][^,\}\n]*)')
    def replacer_units(match):
        before = match.group(1)
        value = match.group(3).strip()
        # If already quoted or a number, skip
        if value.startswith('"') or value in ['true', 'false', 'null']:
            return match.group(0)
        # Try to convert to float/int, if fails, quote it
        try:
            float(value)
            return match.group(0)
        except Exception:
            # Remove trailing commas/brackets/braces from value
            value_clean = re.sub(r'[\s,}\]]*$', '', value)
            return f'{before}"{value_clean}"'
    repaired_str = pattern_units.sub(replacer_units, repaired_str)

    # 1c. Fix unquoted values with units (e.g., 100TB, 100mph, 5m, 10kg, 3.5GHz)
    # This pattern matches: "key": value_with_units (not quoted)
    pattern_units = re.compile(r'("[a-zA-Z_][a-zA-Z0-9_]*"\s*:\s*)([+-]?\d+(?:\.\d+)?[a-zA-Z%/]+)([\s,}\]])')
    def replacer_units(match):
        before = match.group(1)
        value = match.group(2)
        after = match.group(3)
        # Quote the value if not already quoted
        return f'{before}"{value}"{after}'
    repaired_str = pattern_units.sub(replacer_units, repaired_str)

    # 2. Insert missing commas at the end of lines before a new field/object/array
    repaired_str = re.sub(r'([}\]"])\s*\n\s*(")', r'\1,\n\2', repaired_str)
    repaired_str = re.sub(r'([}\]"])\s*\n\s*([}\]])', r'\1,\n\2', repaired_str)
    repaired_str = re.sub(r'([}\]0-9"])\s*([\[{""])', r'\1,\2', repaired_str)
    repaired_str = re.sub(r'("|\d|\]|\})\s*("|\{)', r'\1,\2', repaired_str)
    repaired_str = re.sub(r'(\}|\])\s*("|\{)', r'\1,\2', repaired_str)

    # Remove trailing commas before closing braces/brackets (which can be introduced by the above)
    repaired_str = re.sub(r',(\s*[}\]])', r'\1', repaired_str)

    logger.debug(f"Original JSON string (first 200 chars): {malformed_json_str[:200]}")
    logger.debug(f"Repaired JSON string (first 200 chars): {repaired_str[:200]}")

    return repaired_str
    """
    Attempts to repair common JSON errors, including:
    - unquoted string values
    - missing commas between object fields or array items
    """
    logger.info("Attempting to repair malformed JSON string...")

    # 1. Fix unquoted string values (existing logic)
    pattern = re.compile(r'("[a-zA-Z_][a-zA-Z0-9_]*"\s*:\s*)([a-zA-Z_][a-zA-Z0-9_]*)([\s,\]\}])')
    def replacer(match):
        before = match.group(1)
        value = match.group(2)
        after = match.group(3)
        # Only quote if not a JSON literal
        if value not in ['true', 'false', 'null'] and not value.isdigit():
            value = f'"{value}"'
        return f"{before}{value}{after}"
    repaired_str = pattern.sub(replacer, malformed_json_str)

    # 2. Insert missing commas between object fields or array items
    # Add a comma between }" or }{ or ]" or ]{ or }[ or ][ etc.
    repaired_str = re.sub(r'([}\]0-9"])\s*([\[{""])', r'\1,\2', repaired_str)
    # Add a comma between closing quote/number/bracket and a quote or opening brace/bracket, but not inside strings
    repaired_str = re.sub(r'("|\d|\]|\})\s*("|\{)', r'\1,\2', repaired_str)
    # Add a comma between closing brace/bracket and a quote or opening brace/bracket, but not inside strings
    repaired_str = re.sub(r'(\}|\])\s*("|\{)', r'\1,\2', repaired_str)

    # Remove trailing commas before closing braces/brackets (which can be introduced by the above)
    repaired_str = re.sub(r',(\s*[}\]])', r'\1', repaired_str)

    logger.debug(f"Original JSON string (first 200 chars): {malformed_json_str[:200]}")
    logger.debug(f"Repaired JSON string (first 200 chars): {repaired_str[:200]}")

    return repaired_str
# --- Existing Utility Functions (modified for robustness) ---

def build_structure_definitions(structure_defs_raw_str, start_id):
    """
    Converts structure definitions into an ID-keyed dictionary.
    Accepts a stringified JSON (expected to be a single object where keys are structure names).
    Also returns a mapping from structure type to structure_id for use in placement rules.
    """
    current_logger = logging.getLogger(__name__)
    current_logger.info(f"Raw LLM response for structure definitions:\n{structure_defs_raw_str}")
    current_logger.info(f"Type of structure_defs_raw_str: {type(structure_defs_raw_str)}")
    current_logger.info(f"Value of start_id: {start_id} (type: {type(start_id)})")

    if structure_defs_raw_str is None or not isinstance(structure_defs_raw_str, str) or not structure_defs_raw_str.strip():
        current_logger.error("LLM response for structure definitions is empty, None, or not a string. Cannot process.")
        raise ValueError("LLM response for structure definitions is empty, None, or not a string. Cannot process.")


    parsed_json = None
    cleaned_json = None

    try:
        cleaned_json = extract_first_json_block(structure_defs_raw_str)
        current_logger.debug(f"[DEBUG] cleaned_json: {cleaned_json}")
        if not cleaned_json:
            # Try to extract all top-level "name": { ... } pairs (handles LLM output with multiple objects)
            matches = re.findall(r'"[^"]+"\s*:\s*\{[^}]*\}', structure_defs_raw_str)
            current_logger.debug(f"[DEBUG] matches: {matches}")
            if matches:
                cleaned_json = '{' + ','.join(matches) + '}'
            else:
                # Fallback: try to wrap the whole output in braces if it looks like multiple objects
                lines = [line for line in structure_defs_raw_str.strip().splitlines() if line.strip()]
                if lines and not structure_defs_raw_str.strip().startswith("{"):
                    # Remove trailing commas and wrap in braces
                    joined = "\n".join(lines)
                    # Remove trailing commas before closing braces
                    joined = re.sub(r',\s*}', '}', joined)
                    joined = re.sub(r',\s*$', '', joined)
                    cleaned_json = '{' + joined + '}'
                else:
                    current_logger.error("No valid JSON block or structure fragments found in LLM response for structure definitions.")
                    raise ValueError("No valid JSON block or structure fragments found in LLM response for structure definitions.")
        cleaned_json = re.sub(r',([\s]*[}\]])', r'\1', cleaned_json)

        # Try parsing as-is first (demjson3, then ast, then json)
        try:
            parsed_json = demjson3.decode(cleaned_json)
            current_logger.debug(f"[DEBUG] parsed_json (demjson3, no repair): {parsed_json}")
        except Exception as dem_e:
            current_logger.warning(f"demjson3 failed to parse structure definitions (no repair): {dem_e}. Trying ast.literal_eval as next step.")
            import ast
            try:
                no_indent = "\n".join(line.lstrip() for line in cleaned_json.splitlines())
                parsed_json = ast.literal_eval(no_indent)
                current_logger.debug(f"[DEBUG] parsed_json (ast.literal_eval, no repair): {parsed_json}")
            except Exception as ast_e:
                current_logger.warning(f"ast.literal_eval failed (no repair): {ast_e}. Trying json.loads as next step.")
                try:
                    parsed_json = json.loads(cleaned_json)
                    current_logger.debug(f"[DEBUG] parsed_json (json.loads, no repair): {parsed_json}")
                except Exception as json_e:
                    current_logger.warning(f"json.loads failed (no repair): {json_e}. Will attempt repair.")
                    # Only attempt repair if all direct parses fail
                    repaired_json_str = _repair_json_string(cleaned_json)
                    current_logger.debug(f"[DEBUG] repaired_json_str: {repaired_json_str}")
                    try:
                        parsed_json = demjson3.decode(repaired_json_str)
                        current_logger.debug(f"[DEBUG] parsed_json (demjson3, after repair): {parsed_json}")
                    except Exception as dem_e2:
                        current_logger.warning(f"demjson3 failed to parse structure definitions after repair: {dem_e2}. Trying ast.literal_eval as last resort.")
                        try:
                            no_indent = "\n".join(line.lstrip() for line in repaired_json_str.splitlines())
                            parsed_json = ast.literal_eval(no_indent)
                            current_logger.debug(f"[DEBUG] parsed_json (ast.literal_eval, after repair): {parsed_json}")
                        except Exception as ast_e2:
                            current_logger.warning(f"ast.literal_eval failed after repair: {ast_e2}. Trying json.loads as last resort.")
                            try:
                                parsed_json = json.loads(repaired_json_str)
                                current_logger.debug(f"[DEBUG] parsed_json (json.loads, after repair): {parsed_json}")
                            except Exception as json_e2:
                                current_logger.error(f"json.loads failed to parse structure definitions after repair: {json_e2}")
                                current_logger.error(f"Raw input for debugging:\n{structure_defs_raw_str}")
                                current_logger.error(f"Failed to parse structure_defs even with demjson3, ast.literal_eval, and json.loads after repair: {json_e2}")
                                parsed_json = {}

    except Exception as e:
        current_logger.error(f"[ERROR] An unexpected error occurred during JSON extraction/parsing for structure definitions: {e}")
        current_logger.error(f"[DEBUG] Raw input: {structure_defs_raw_str}")
        parsed_json = {}

    if not isinstance(parsed_json, dict):
        current_logger.error(f"[ERROR] LLM returned unexpected type for structure definitions. Expected dict, got {type(parsed_json)}")
        current_logger.debug(f"[DEBUG] Parsed JSON: {parsed_json}")
        raise ValueError("Expected LLM response for structure definitions to be a dictionary.")

    structure_list = list(parsed_json.values())

    structured = {}
    type_to_id = {}
    current_id = start_id
    for idx, struct in enumerate(structure_list):
        current_logger.debug(f"[DEBUG] Processing struct at index {idx}: {struct}")
        if not isinstance(struct, dict):
            current_logger.warning(f"[SKIP] Structure at index {idx} is not a dict: {struct}")
            continue
        attributes = struct.get("attributes", {})
        if not isinstance(attributes, dict):
            current_logger.warning(f"Attributes for structure type '{struct.get('type')}' is not a dict. Defaulting to empty dict.")
            attributes = {}
        # --- Parse adjacent_environmental_objects robustly ---
        adj_env_objs = struct.get("adjacent_environmental_objects", [])
        normalized_adj_env_objs = []
        if isinstance(adj_env_objs, dict):
            for k, v in adj_env_objs.items():
                if isinstance(v, dict):
                    obj_type = v.get("type", k)
                    desc = v.get("description", "")
                    attrs = v.get("attributes", {})
                    if not isinstance(attrs, dict):
                        attrs = {}
                    normalized_adj_env_objs.append({
                        "type": obj_type,
                        "description": desc,
                        "attributes": attrs
                    })
        elif isinstance(adj_env_objs, list):
            for obj in adj_env_objs:
                if isinstance(obj, dict):
                    obj_type = obj.get("type", "Unknown Object")
                    desc = obj.get("description", "")
                    attrs = obj.get("attributes", {})
                    if not isinstance(attrs, dict):
                        attrs = {}
                    normalized_adj_env_objs.append({
                        "type": obj_type,
                        "description": desc,
                        "attributes": attrs
                    })
                elif isinstance(obj, str):
                    normalized_adj_env_objs.append({
                        "type": obj,
                        "description": "",
                        "attributes": {}
                    })
        elif isinstance(adj_env_objs, str):
            normalized_adj_env_objs.append({
                "type": adj_env_objs,
                "description": "",
                "attributes": {}
            })
        # --- End robust parsing ---
        # Accept even if some fields are missing
        structured[str(current_id)] = {
            "type": struct.get("type", f"Unnamed Structure {current_id}"),
            "description": struct.get("description", ""),
            "attributes": attributes,
            "adjacent_environmental_objects": normalized_adj_env_objs
        }
        type_to_id[struct.get("type", f"Unnamed Structure {current_id}")] = str(current_id)
        current_logger.debug(f"[DEBUG] Added structure with id {current_id}: {structured[str(current_id)]}")
        current_id += 1
    current_logger.info(f"[INFO] Final structured dict: {structured}")
    return structured, type_to_id

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
        "- 'description': a detailed, multi-sentence description (at least 3 sentences) focusing on appearance, materials, function, and unique features. Avoid generic or vague statements. Do not invent features not plausible for the theme.\n"
        "- 'attributes': a dictionary of relevant properties (e.g. material, usage, magical affinity, HP). Ensure all attribute values are JSON-compatible (strings, numbers, booleans, arrays, or objects).\n\n"
        "- 'adjacent_environmental_objects': a list of objects, each with:\n"
           " - 'type': string\n"
           " - 'description': a detailed, multi-sentence description (at least 3 sentences), focusing mainly on appearance and materials, and. Do not invent features not plausible for the theme and do not add living beings like animals and humans, plants are fine.\n"
           " - 'attributes': dictionary\n"
        "If a detail is not clear, state 'unknown' or omit it.\n" # Added instruction
        "Return a single valid JSON object where keys are structure type names, and values are their definitions.\n\n"
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


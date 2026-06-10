import json
import logging

# Import the correct generic local structured output function
from text_grid.llm import call_local_llm as call_llm_for_structured_output


logger = logging.getLogger(__name__)

# Define the JSON schema for the expected LLM output for biome generation hints
# This schema DIRECTLY corresponds to the MongoDB $jsonSchema you provided.
BIOME_HINTS_SCHEMA = {
    "type": "object",
    "required": [
        "biome_name",
        "possible_structures",
        "possible_grids",
        "theme_prompt"
    ],
    "properties": {
        "biome_name": {
            "type": "string",
            "description": "A concise, max 4-word suggested name for the biome."
        },
        "theme_prompt": {
            "type": "string",
            "description": "The user-provided biome theme used for generation (should be the input prompt itself)."
        },
        "possible_structures": {
            "type": "object",
            "description": "Can include any category of structures (e.g. buildings, trees, etc.)",
            "additionalProperties": {
                "type": "object",
                "description": "Structure category (e.g. buildings, trees, etc.)",
                "additionalProperties": {
                    "type": "object",
                    "required": [
                        "type",
                        "attributes",
                        "description"
                    ],
                    "properties": {
                        "type": {
                            "type": "string",
                            "description": "structure type"
                        },
                        "attributes": {
                            "type": "object",
                            "description": "structure-specific attributes (e.g., {'hp': 100})",
                            "additionalProperties": {
                                "oneOf": [ # Allows int, float, or string
                                    {"type": "integer"},
                                    {"type": "number"},
                                    {"type": "string"}
                                ]
                            }
                        },
                        "description": {
                            "type": "string",
                            "description": "textual description"
                        },
                        "imageUrl": {
                            "type": "string",
                            "description": "URL for the 2D image representation of the structure (initialize as empty string).",
                            "default": "" # Explicit default for the LLM to follow
                        },
                        "model3dUrl": {
                            "type": "string",
                            "description": "URL for the 3D model asset of the structure (initialize as empty string).",
                            "default": "" # Explicit default for the LLM to follow
                        },
                        "Status": {
                            "type": "string",
                            "description": "Current processing status of the structure (initialize as 'Yet to start').",
                            "enum": ["Yet to start", "Image Generated", "3D Model Generated", "Decimated", "Decimating", "Decimation Failed", "Generation Failed", "Dispatch Failed"], # Include all potential statuses
                            "default": "Yet to start" # Explicit default for the LLM to follow
                        }
                    }
                }
            }
        },
        "possible_grids": {
            "type": "array",
            "description": "Array of grid definitions for the biome layout.",
            "items": {
                "type": "object",
                "required": [
                    "grid_id",
                    "layout"
                ],
                "properties": {
                    "grid_id": {
                        "type": "string",
                        "description": "Unique ID for this grid (e.g., 'main_layout', 'alternate_layout')."
                    },
                    "layout": {
                        "type": "array",
                        "description": "2D array of integers",
                        "items": {
                            "type": "array",
                            "items": {
                                "type": "integer",
                                "description": "Integer representing a terrain type (e.g., 0 for plain, 1 for forest)."
                            }
                        },
                        "minItems": 5, # Example: Ensure at least a 5x5 grid (rows)
                        "maxItems": 15 # Example: Max 15x15 grid (rows)
                    }
                }
            },
            "minItems": 1 # Ensure at least one grid is generated
        }
    }
}


def get_biome_generation_hints(structured_buildings: list, theme_prompt: str) -> dict:
    """
    Generates hints for biome generation using an LLM (synchronous).

    Args:
        structured_buildings (list): A list of existing or pre-defined structured building types.
                                     This context helps the LLM suggest relevant structures.
        theme_prompt (str): The overall theme prompt for the biome.

    Returns:
        dict: A dictionary derived from the LLM's JSON output,
              containing 'biome_name', 'theme_prompt', 'possible_structures', and 'possible_grids'.
    Raises:
        ValueError: If the LLM output is not valid JSON or doesn't match the schema.
    """
    # Embed the schema directly into the prompt for the LLM to follow
    schema_str = json.dumps(BIOME_HINTS_SCHEMA, indent=2)

    system_instruction = f"""
    You are an AI assistant specialized in generating creative suggestions for 3D environment biomes.
    Your task is to generate a comprehensive JSON object for a biome, strictly adhering to the provided JSON schema.
    The output MUST contain ONLY the JSON object, no other text or markdown fences.

    **JSON Schema to Follow:**
    {schema_str}

    Here are the key requirements for the JSON output:
    1.  **`biome_name`**: A concise, creative name for the biome (max 4 words).
    2.  **`theme_prompt`**: Directly copy the user's original theme prompt provided.
    3.  **`possible_structures`**:
        * Include diverse categories (e.g., 'buildings', 'terrain_features', 'vegetation').
        * For each specific structure, provide:
            * `type`: A general type (e.g., 'building', 'tower', 'tree').
            * `description`: A detailed textual description for 3D model generation.
            * `attributes`: An object, MUST include at least `hp` (e.g., `{{ "hp": 100 }}`). You can add other relevant attributes (e.g., `size`, `material`, `function`).
            * `imageUrl`: MUST be an empty string `""`.
            * `model3dUrl`: MUST be an empty string `""`.
            * `Status`: MUST be `"Yet to start"`.
        * Suggest 5-7 distinct structures in total across categories.
    4.  **`possible_grids`**:
        * Provide at least one 2D grid layout (e.g., 10x10 or 15x15), representing a top-down view of the biome.
        * Each grid object needs a `grid_id` (e.g., "main_layout") and `layout` (a 2D array of integers).
        * Use integers to represent different terrain/structure types (e.g., 0: empty, 1: building, 2: tree, 3: water, 4: road, etc.). Define your own consistent mapping.
        * Aim for a grid that is at least 5x5 and at most 15x15.

    Consider the following existing building types for context, although you are free to suggest new, relevant ones:
    {json.dumps(structured_buildings, indent=2)}

    Example Output Structure (DO NOT copy directly, generate based on prompt. ENSURE all fields as per schema):
    {{
        "biome_name": "Mystic Grove",
        "theme_prompt": "A magical forest with glowing flora and ancient ruins.",
        "possible_structures": {{
            "vegetation": {{
                "glowing_tree": {{
                    "type": "tree",
                    "description": "A tall, ancient tree with bioluminescent leaves and roots.",
                    "attributes": {{"hp": 200, "light_intensity": 0.8}},
                    "imageUrl": "",
                    "model3dUrl": "",
                    "Status": "Yet to start"
                }},
                "shimmering_bush": {{
                    "type": "bush",
                    "description": "A small, dense bush emitting a soft, ethereal shimmer.",
                    "attributes": {{"hp": 50}},
                    "imageUrl": "",
                    "model3dUrl": "",
                    "Status": "Yet to start"
                }}
            }},
            "ruins": {{
                "ancient_archway": {{
                    "type": "ruin",
                    "description": "A crumbling stone archway overgrown with moss and glowing vines.",
                    "attributes": {{"hp": 500}},
                    "imageUrl": "",
                    "model3dUrl": "",
                    "Status": "Yet to start"
                }}
            }}
        }},
        "possible_grids": [
            {{
                "grid_id": "main_layout",
                "layout": [
                    [0,0,1,0,0],
                    [0,1,1,1,0],
                    [1,1,0,1,1],
                    [0,1,1,1,0],
                    [0,0,1,0,0]
                ]
            }}
        ]
    }}
    """

    user_prompt_for_llm = f"{system_instruction}\n\nGenerate a detailed biome JSON based on the theme: '{theme_prompt}'. Strictly follow the provided JSON schema."

    logger.info(f"Sending request to LLM for biome generation hints for theme: '{theme_prompt}'")
    
    try:
        # CORRECTED CALL: Call the generic local structured output function
        raw_llm_output_json_string = call_llm_for_structured_output(
            prompt=user_prompt_for_llm # Pass the combined system_instruction and user_prompt
        )

        if not raw_llm_output_json_string:
            logger.error("LLM returned empty or None response for biome generation hints.")
            raise ValueError("LLM response for biome generation hints is empty or None. Cannot process.")

        llm_output_dict = json.loads(raw_llm_output_json_string)
        
        # Basic validation against the schema top-level keys
        required_keys = ["biome_name", "possible_structures", "possible_grids", "theme_prompt"]
        if not all(key in llm_output_dict for key in required_keys):
            logger.error(f"LLM output is missing required top-level keys. Expected: {required_keys}. Got: {llm_output_dict.keys()}")
            raise ValueError(f"LLM output is missing required top-level keys. Expected: {required_keys}")

        # Also ensure theme_prompt in output matches input theme_prompt
        if llm_output_dict.get("theme_prompt") != theme_prompt:
             logger.warning(f"LLM's 'theme_prompt' in output '{llm_output_dict.get('theme_prompt')}' does not exactly match input '{theme_prompt}'. Adjusting.")
             llm_output_dict["theme_prompt"] = theme_prompt # Force consistency

        logger.info(f"Received LLM hints for biome '{llm_output_dict.get('biome_name')}'.")
        return llm_output_dict # Return the parsed dictionary

    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from LLM response in placement_suggestor: {e}. Raw response: {raw_llm_output_json_string}", exc_info=True)
        raise ValueError(f"LLM did not return valid JSON for biome hints: {e}") from e
    except ValueError as e:
        logger.error(f"Validation error for LLM biome hints: {e}", exc_info=True)
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred in get_biome_generation_hints: {e}", exc_info=True)
        raise
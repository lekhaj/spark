
import json

from matplotlib.pylab import integer

# BIOME_DOCUMENT_SCHEMA = {
#     "type": "object",
#     "required": ["biome_name", "theme_prompt", "possible_structures", "possible_grids"],
#     "properties": {
#         "biome_name": {
#             "type": "string",
#             "description": "A concise, evocative, and unique name for the biome (e.g., 'The Ashen Caldera', 'Whispering Sunken City')."
#         },
#         "theme_prompt": {
#             "type": "string",
#             "description": "The original user-provided theme prompt, copied exactly."
#         },
#         "possible_structures": {
#             "type": "object",
#             "description": "A dictionary of structure categories (e.g., 'buildings', 'vegetation', 'terrain_features'). Each key is a category name.",
#             "minProperties": 1,
#             "patternProperties": {
#                 "^[a-zA-Z_]+$": {
#                     "type": "object",
#                     "description": "A dictionary of specific structures within this category. Each key is a unique structure name (e.g., 'obsidian_tower').",
#                     "minProperties": 1,
#                     "patternProperties": {
#                         "^[a-zA-Z_]+$": {
#                             "type": "object",
#                             "required": ["type", "description", "attributes"],
#                             "properties": {
#                                 "type": {"type": "string", "description": "A general classification for this structure (e.g., 'tower', 'tree', 'geyser')."},
#                                 "description": {"type": "string", "description": "A detailed, multi-sentence textual description suitable for inspiring a 3D model. Focus on materials, shape, scale, and unique features."},
#                                 "attributes": {"type": "object", "description": "A dictionary of game-related properties. Must include 'hp'. Can include others like 'size', 'material', 'function'."}
#                             }
#                         }
#                     }
#                 }
#             }
#         },
#         "possible_grids": {
#             "type": "array",
#             "description": "An array containing at least one grid layout definition.",
#             "minItems": 1,
#             "items": {
#                 "type": "object",
#                 "required": ["grid_id", "layout"],
#                 "properties": {
#                     "grid_id": {"type": "string", "description": "A unique identifier for this grid layout (e.g., 'main_layout')."},
#                     "layout": {
#                         "type": "array",
#                         "description": "A 2D array of integers representing the biome map (e.g., 10x10). Use 0 for empty space. Assign a unique positive integer to each structure type defined in 'possible_structures'.",
#                         "minItems": 5, # Grid must be at least 5 rows
#                         "items": {
#                             "type": "array",
#                             "minItems": 5, # Grid must be at least 5 columns
#                             "items": {"type": "integer"}
#                         }
#                     }
#                 }
#             }
#         }
#     }
# }

BIOME_DOCUMENT_SCHEMA =    """
  "biome_name": "string",
  "theme_prompt": "string",
  "possible_structures": {
    "category_name (e.g., 'buildings', 'characters', 'flora', 'terrain_features', 'props')": {
      "structure_name": {
        "id": integer,
        "type": "string",
        "description": "string (Physical details of the asset ONLY, no surroundings)",
        "background_prompt": "string (Simple, 1-sentence context for the asset's surroundings)",
        "attributes": { 
          "hp": integer, 
          "scale": "string (e.g., 'human-scale', 'massive', 'colossal')",
          "material_primary": "string (e.g., 'sandstone', 'glowing fungus', 'obsidian')",
          ... 
        }
      }
    }
  },
  "possible_grids": [{
    "grid_id": "main_layout",
    "dimensions": { "width": 10, "height": 10 },
    "layout": "A full 10x10 2D array of integers."
  }]
"""

def get_biome_generation_prompt(theme_prompt: str) -> str:
    """
    Constructs the final, detailed prompt to be sent to the LLM.

    It embeds the JSON schema and provides clear, explicit instructions to generate
    a complete biome document in a single call.

    Args:
        theme_prompt (str): The user's creative theme for the biome.

    Returns:
        str: The fully-formed prompt string.
    """
    # We serialize the schema to include it directly in the prompt.
    schema_str = json.dumps(BIOME_DOCUMENT_SCHEMA, indent=2)

    prompt = f"""AI task: Generate a single, raw JSON object from a user theme. Output ONLY the JSON object, nothing else. All generated structures must be directly relevant to the user's theme; avoid generic or redundant objects.

Strictly follow this structure and rules:

{BIOME_DOCUMENT_SCHEMA}


- Define unique and thematically relevant structure types (assets), each with a unique positive `id` (1, 2, 3...).
- **Description Guidelines:** This field must ONLY describe the physical asset itself, as if it were on a blank background.
    - Focus on its shape, materials, textures, specific details, and state of decay.
    - **DO NOT** describe the ground it sits on, the sky above it, nearby objects, or any other environmental details. That context belongs ONLY in the `background_prompt`.
- **Attribute Guidelines:** For the `attributes` object, provide ONLY properties useful for image/3D generation and core gameplay. Focus on visual and physical characteristics that guide an artist's "feel and view" of the object.
    - **Must include:** `hp` (integer), `scale` (string), and `material_primary` (string).
    - **Can also include:** `state_of_decay` (e.g., 'weathered', 'crumbled'), `color_palette` (e.g., 'earth_tones', 'monochromatic'), or `texture` (e.g., 'rough-hewn', 'pitted').
    - **AVOID** abstract or non-visual concepts like 'security', 'importance', or 'magic_level'.
- **Grid Layout Guidelines:** The `layout` grid should represent a plausible and interesting game space.
    - **Create Clusters:** Group related assets together logically (e.g., rubble near a ruin, flora in a grove).
    - **Establish Points of Interest (POIs):** Use one or two major assets as focal points.
    - **Control Density:** Based on the biome's theme, decide on the density. A “visual” should have few empty spaces (`0`s). A "make creative decision" or “open”should make more deliberate use of negative space to create paths and wide-open areas. The amount of empty space is a design choice, not a default.
- Place asset `id`s on the grid, reusing them for multiple instances.
- Use 0 for empty space. Every non-zero integer in the `layout` MUST correspond to a defined `id`.
- Respond ONLY with a valid JSON object. All property names and string values must use double quotes.

**User_theme:** "{theme_prompt}"
"""

    # prompt = f"""
    # You are an expert procedural world-building AI. Your task is to generate a complete biome document as a single, valid JSON object that strictly adheres to the provided schema. Do not include any explanatory text, markdown fences, or anything outside of the JSON object.

    # **JSON Schema to Follow:**
    # ```json
    # {schema_str}
    # ```

    # **Key Generation Guidelines:**
    # 1.  **`biome_name`**: Invent a creative and fitting name for the biome based on the theme.
    # 2.  **`theme_prompt`**: This MUST be an exact copy of the user's theme: "{theme_prompt}".
    # 3.  **`possible_structures`**:
    #     *   Define 3-5 unique and interesting structures that fit the theme.
    #     *   Group them into logical categories (e.g., 'buildings', 'flora', 'geology').
    #     *   The keys for categories and structures must be snake_case strings.
    #     *   Write compelling descriptions for each structure.
    # 4.  **`possible_grids`**:
    #     *   Generate one 10x10 grid layout.
    #     *   Map the structures you defined to unique positive integer IDs. For example, if you defined 3 structures, map them to 1, 2, and 3.
    #     *   Place these IDs on the grid to create a plausible and interesting layout. Use 0 for empty cells. Do not use integer IDs for which you have not defined a corresponding structure.

    # **User's Theme:** "{theme_prompt}"

    # # Generate the JSON object 
    # """
    
    return prompt.strip()
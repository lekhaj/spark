
import json

# BIOME_DOCUMENT_SCHEMA - Updated for Image-to-3D Asset Generation
BIOME_DOCUMENT_SCHEMA = {
    "type": "object",
    "required": ["biome_name", "status", "theme_prompt", "possible_structures", "possible_grids"],
    "properties": {
        "biome_name": {"type": "string"},
        "status": {"type": "string", "enum": ["pending"]},
        "theme_prompt": {"type": "string"},
        "possible_structures": {
            "type": "object",
            "description": "Categories containing up to 20 unique modelable assets",
            "patternProperties": {
                "^[a-z_]+$": {
                    "type": "object",
                    "patternProperties": {
                        "^[a-z_]+$": {
                            "type": "object",
                            "required": ["id", "type", "description", "background_prompt", "status", "attributes"],
                            "properties": {
                                "id": {"type": "integer"},
                                "type": {"type": "string"},
                                "description": {
                                    "type": "string",
                                    "description": "Detailed physical description of the isolated asset + correct posing instruction + rendering keywords."
                                },
                                "background_prompt": {"type": "string"},
                                "status": {"type": "string", "enum": ["not complete"]},
                                "attributes": {
                                    "type": "object",
                                    "required": ["hp", "scale", "material_primary"],
                                    "properties": {
                                        "hp": {"type": "integer"},
                                        "scale": {"type": "string"},
                                        "material_primary": {"type": "string"},
                                        "state_of_decay": {"type": "string"},
                                        "color_palette": {"type": "string"},
                                        "texture": {"type": "string"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        },
        "possible_grids": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["grid_id", "dimensions", "layout"],
                "properties": {
                    "grid_id": {"type": "string"},
                    "dimensions": {
                        "type": "object",
                        "properties": {
                            "width": {"type": "integer", "maximum": 20},
                            "height": {"type": "integer", "maximum": 20}
                        }
                    },
                    "layout": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "integer"}
                        }
                    }
                }
            }
        }
    }
}


def get_biome_generation_prompt(theme_prompt: str) -> str:
    """
    Returns the full system prompt for generating a biome JSON document.
    Updated: No token limit on descriptions, max 20 assets.
    """
    schema_str = json.dumps(BIOME_DOCUMENT_SCHEMA, indent=2)

    prompt = f"""AI task: Generate a single, raw JSON object based on the user theme. Output ONLY the raw JSON object, without any surrounding text or markdown fences.

You are generating the initial state for image-to-3D asset pipeline.

**CRITICAL RULES:**
- Generate up to 20 unique, high-quality, modelable assets. Aim for rich variety while staying relevant to the theme.
- Maximize grid area utilization — use as few zeroes as possible while maintaining a logical and interesting layout.
- All assets must be discrete, modelable objects (never vast environments).
- Deconstruct large concepts into single representative assets.

**description field rules:**
- Provide a detailed, clean physical description of the asset only (form, silhouette, materials, PBR textures, specific details, state of wear).
- Do NOT describe environment, ground, or surroundings in the description field.
- Append the correct posing instruction based on asset category:
  - Humanoid characters: "A single, accurate, T-pose image of the asset."
  - Creatures/beasts/monsters: "A single, accurate, neutral pose image of the asset."
  - Buildings/props/flora/terrain: "A single, centered image of the asset."
- Always end the description with: "full shot, concept art, game asset, isolated on a white background, soft studio lighting, masterpiece, 4k."

**JSON Structure to follow exactly:**
{schema_str}

Additional instructions:
- Top level "status" must be "pending"
- Every individual asset's "status" must be "not complete"
- Use unique positive integer ids starting from 1
- Create logical clusters in the grid with major assets as Points of Interest (POIs)
- Use 0 for empty space. Every non-zero number must match a defined asset id.
- Make the grid as filled and visually interesting as possible.

**User Theme:** "{theme_prompt}"

Respond with ONLY the valid JSON object.
"""

    return prompt.strip()

# src/text_grid/grid_generator.py

import uuid
import json
import logging
from . import llm # Relative import
from . import utils # Relative import
from . import structure_registry # Relative import
from . import placement_suggestor # Relative import
from .grid_placement_logic import generate_grid_from_hints # Corrected: now a relative import

logger = logging.getLogger(__name__) # Ensure logger is defined at module level

def generate_biome(theme_prompt, structure_type_list):
    """
    Generates a biome based on a theme prompt and a list of structure type names.
    """

    next_id = structure_registry.get_next_structure_id() 

    structure_defs_raw = llm.call_llm_for_structure_definitions(theme_prompt, structure_type_list) 
    logger.info(f"DEBUG: Raw LLM structure response from llm.py: {structure_defs_raw}") 

    try:
        structured_buildings = utils.build_structure_definitions(structure_defs_raw, next_id) 
        if not structured_buildings:
            logger.error("[ERROR] No structures defined by LLM. Aborting biome generation.")
            return "❌ Error: LLM failed to define structures or returned empty definitions."
    except ValueError as e:
        logger.error(f"[ERROR] Failed to build structure definitions: {e}")
        return f"❌ Error: Failed to process structure definitions: {e}"

    try:
        # DEBUG: Log the input to placement_suggestor
        logger.debug(f"Calling get_biome_generation_hints with structured_buildings: {json.dumps(structured_buildings, indent=2)} and theme_prompt: {theme_prompt}")
        llm_hints = placement_suggestor.get_biome_generation_hints(structured_buildings, theme_prompt)
        logger.debug(f"Raw LLM hints output: {llm_hints}")
        # Extra: Log placement_rules if present
        if isinstance(llm_hints, dict):
            logger.debug(f"Parsed placement_rules: {json.dumps(llm_hints.get('placement_rules', []), indent=2)}")
        biome_name = llm_hints.get("biome_name", utils.generate_biome_name_from_prompt(theme_prompt))
    except ValueError as e:
        logger.error(f"[ERROR] Failed to get LLM placement hints: {e}")
        biome_name = utils.generate_biome_name_from_prompt(theme_prompt) 
        llm_hints = { 
            "grid_dimensions": {"width": 10, "height": 10},
            "placement_rules": [], 
            "general_density": 0.3
        }
        logger.warning(f"Using fallback biome name: '{biome_name}' and default grid hints due to LLM error.")

    # --- NEW DEBUG LOGS HERE ---
    logger.info(f"DEBUG: LLM Hints received by grid_generator (before passing to placement_logic): {json.dumps(llm_hints, indent=2)}")
    logger.info(f"DEBUG: Structured Buildings received by grid_generator (before passing to placement_logic): {json.dumps(structured_buildings, indent=2)}")
    # --- END NEW DEBUG LOGS ---

    try:
        layout = generate_grid_from_hints(llm_hints, structured_buildings)
        
        # Original check for empty layout - now more robust due to `all(cell == 0 ...)`
        if not layout or all(cell == 0 for row in layout for cell in row): 
            logger.warning("Generated layout is empty or contains no structures. Overwriting with an all-zero grid for safety.")
            grid_width = llm_hints["grid_dimensions"]["width"] if "grid_dimensions" in llm_hints else 10
            grid_height = llm_hints["grid_dimensions"]["height"] if "grid_dimensions" in llm_hints else 10
            layout = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
            # --- NEW DEBUG ---
            logger.debug("DEBUG: Layout was empty, reset to all zeros.")
            # --- END NEW DEBUG ---
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate grid from hints: {e}", exc_info=True)
        grid_width = llm_hints["grid_dimensions"]["width"] if "grid_dimensions" in llm_hints else 10
        grid_height = llm_hints["grid_dimensions"]["height"] if "grid_dimensions" in llm_hints else 10
        layout = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        return f"❌ Error generating grid: {e}"

    logger.info(f"Type of 'layout' before document assembly: {type(layout)}")
    if not isinstance(layout, list) or not (len(layout) > 0 and isinstance(layout[0], list)):
        logger.error(f"Layout is not a list of lists at document assembly! Actual type: {type(layout)}")
        grid_width = llm_hints["grid_dimensions"]["width"] if "grid_dimensions" in llm_hints else 10
        grid_height = llm_hints["grid_dimensions"]["height"] if "grid_dimensions" in llm_hints else 10
        layout = [[0 for _ in range(grid_width)] for _ in range(grid_height)]

    grid_id = f"{biome_name.replace(' ', '_')}_grid_{uuid.uuid4().hex[:8]}" 
    doc = {
        "_id": str(uuid.uuid4()), 
        "biome_name": biome_name,
        "theme_prompt": theme_prompt,
        "possible_structures": {
            "buildings": structured_buildings 
        },
        "possible_grids": [
            {
                "grid_id": grid_id,
                "layout": layout 
            }
        ]
    }

    try:
        # Ensure structures are correctly registered along with the layout
        structure_registry.register_new_structures(doc["possible_structures"], theme_prompt, biome_name, doc["possible_grids"][0]["layout"]) 
        structure_registry.insert_biome(doc) 
        return f"✅ Biome '{biome_name}' generated and saved."
    except Exception as e:
        logger.error(f"[DB] ❌ Failed to insert biome: {e}", exc_info=True) 
        return f"❌ Error saving biome: {e}"

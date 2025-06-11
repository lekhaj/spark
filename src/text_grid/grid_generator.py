import uuid
import logging
from . import llm # Relative import
from . import utils # Relative import
from . import structure_registry # Relative import
from . import placement_suggestor # Relative import
from .grid_placement_logic import generate_grid_from_hints # Corrected: now a relative import

logger = logging.getLogger(__name__) # Ensure logger is defined at module level

# NOTE: Made generate_biome async because it calls an async function (llm.call_llm_for_structure_definitions)
async def generate_biome(theme_prompt, structure_type_list):
    """
    Generates a biome based on a theme prompt and a list of structure type names.

    Args:
        theme_prompt (str): The user's prompt describing the biome's theme.
        structure_type_list (list): List of structure type names (e.g., ['Herbalist Hut', 'Forest Shrine'])

    Returns:
        str: Success or error message
    """

    # 🔢 Step 1: Get the next structure ID to begin numbering
    next_id = structure_registry.get_next_structure_id() 

    # 🤖 Step 2: Ask LLM to generate structure metadata for each structure type
    # Using the new, specific LLM call for structure definitions
    # CRITICAL FIX: AWAIT the async call
    structure_defs_raw = await llm.call_llm_for_structure_definitions(theme_prompt, structure_type_list) 
    logger.info(f"DEBUG: Raw LLM structure response from llm.py: {structure_defs_raw}") # Using logger.info

    # 🧱 Step 3: Parse and map definitions into ID-based structure dictionary
    try:
        # Pass the raw string to utils.build_structure_definitions as it expects a string
        structured_buildings = utils.build_structure_definitions(structure_defs_raw, next_id) 
        if not structured_buildings:
            logging.error("[ERROR] No structures defined by LLM. Aborting biome generation.")
            return "❌ Error: LLM failed to define structures or returned empty definitions."
    except ValueError as e:
        logging.error(f"[ERROR] Failed to build structure definitions: {e}")
        return f"❌ Error: Failed to process structure definitions: {e}"


    # 💡 Step 4: Ask LLM for high-level placement hints and biome name suggestion
    try:
        # Pass the newly generated 'structured_buildings' to the hint generator
        # CRITICAL FIX: AWAIT the async call if placement_suggestor.get_biome_generation_hints becomes async
        # For now, assuming it's synchronous. If it calls an LLM, it must be async and awaited.
        llm_hints, biome_name_suggestion = placement_suggestor.get_biome_generation_hints(structured_buildings, theme_prompt) 
        # Use the biome name suggested by the LLM from the hints
        biome_name = biome_name_suggestion
    except ValueError as e:
        logging.error(f"[ERROR] Failed to get LLM placement hints: {e}")
        # Fallback to programmatic name generation and default grid hints if LLM hints fail
        biome_name = utils.generate_biome_name_from_prompt(theme_prompt) 
        llm_hints = { # Provide default hints if LLM fails
            "grid_dimensions": {"width": 10, "height": 10},
            "placement_rules": [], # No specific rules, will lead to sparse grid
            "general_density": 0.3
        }
        logging.warning(f"Using fallback biome name: '{biome_name}' and default grid hints due to LLM error.")


    # 🗺️ Step 5: Programmatically generate the grid layout using LLM hints
    try:
        # Pass the LLM hints and the *ID-mapped* structured_buildings
        layout = generate_grid_from_hints(llm_hints, structured_buildings)
        if not layout or not any(0 not in row for row in layout): # Check if layout is empty or all zeros
            logging.warning("Generated layout is empty or contains no structures. Check grid_placement_logic or LLM hints.")
            # Fallback to an empty grid if generation fails completely
            # Use dimensions from llm_hints or default 10x10 if not available
            grid_width = llm_hints["grid_dimensions"]["width"] if "grid_dimensions" in llm_hints else 10
            grid_height = llm_hints["grid_dimensions"]["height"] if "grid_dimensions" in llm_hints else 10
            layout = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    except Exception as e:
        logging.error(f"[ERROR] Failed to generate grid from hints: {e}")
        # Fallback to an empty grid in case of critical error
        grid_width = llm_hints["grid_dimensions"]["width"] if "grid_dimensions" in llm_hints else 10
        grid_height = llm_hints["grid_dimensions"]["height"] if "grid_dimensions" in llm_hints else 10
        layout = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
        return f"❌ Error generating grid: {e}"

    # --- ADDED: Explicit Type Check for layout before assembling document ---
    logging.info(f"Type of 'layout' before document assembly: {type(layout)}")
    if not isinstance(layout, list) or not (len(layout) > 0 and isinstance(layout[0], list)):
        logging.error(f"Layout is not a list of lists at document assembly! Actual type: {type(layout)}")
        # Fallback to a default empty layout if it's not the expected type
        grid_width = llm_hints["grid_dimensions"]["width"] if "grid_dimensions" in llm_hints else 10
        grid_height = llm_hints["grid_dimensions"]["height"] if "grid_dimensions" in llm_hints else 10
        layout = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    # --- END ADDED ---

    # 📦 Step 6: Assemble final biome document
    grid_id = f"{biome_name.replace(' ', '_')}_grid_{uuid.uuid4().hex[:8]}" # Add UUID to grid_id for uniqueness
    doc = {
        "_id": str(uuid.uuid4()), 
        "biome_name": biome_name,
        "theme_prompt": theme_prompt,
        "possible_structures": {
            "buildings": list(structured_buildings.values()) # Convert the dict of structures to a list of their values as required by structure_registry
        },
        "possible_grids": [
            {
                "grid_id": grid_id,
                "layout": layout # Should be the actual list of lists here
            }
        ]
    }

    # 🗃️ Step 7: Save structures and biome to database
    try:
        # Ensure register_new_structures and insert_biome are called with correct args
        # And ensure structured_buildings is passed as a list of dicts to register_new_structures if it expects that
        structure_registry.register_new_structures(doc["possible_structures"], theme_prompt, biome_name, doc["possible_grids"][0]["layout"]) 
        structure_registry.insert_biome(doc) 
        return f"✅ Biome '{biome_name}' generated and saved."
    except Exception as e:
        logging.error(f"[DB] ❌ Failed to insert biome: {e}")
        return f"❌ Error saving biome: {e}"


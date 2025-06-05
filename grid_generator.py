import uuid
import logging
from llm import call_online_llm
from utils import (
    build_structure_definitions,
    get_structure_definition_prompt,
)
from structure_registry import (
    get_next_structure_id,
    register_new_structures,
    insert_biome,
)
from placement_suggestor import suggest_structure_placement


def generate_biome(theme_prompt, structure_type_list):
    """
    Generates a biome based on a theme prompt and a list of structure type names.

    Args:
        theme_prompt (str): The user's prompt describing the biome's theme.
        structure_type_list (list): List of structure type names (e.g., ['Herbalist Hut', 'Forest Shrine'])

    Returns:
        str: Success or error message
    """

    # 🔢 Step 1: Get the next structure ID to begin numbering
    next_id = get_next_structure_id()

    # 🤖 Step 2: Ask LLM to generate structure metadata for each structure type
    structure_prompt = get_structure_definition_prompt(theme_prompt, structure_type_list)
    structure_defs_raw = call_online_llm(structure_prompt)

    # 🧱 Step 3: Parse and map definitions into ID-based structure dictionary
    structured = build_structure_definitions(structure_defs_raw, next_id)

    # 🌐 Step 4: Generate biome layout + name from placement_suggestor
    layout, biome_name = suggest_structure_placement(structured, theme_prompt)

    # 📦 Step 5: Assemble final biome document
    grid_id = f"{biome_name.replace(' ', '_')}_grid_1"
    doc = {
        "biome_name": biome_name,
        "theme_prompt": theme_prompt,
        "possible_structures": {
            "buildings": structured
        },
        "possible_grids": [
            {
                "grid_id": grid_id,
                "layout": [[int(val) for val in row] for row in layout]
            }
        ]
    }

    # 🗃️ Step 6: Save structures and biome to database
    try:
        register_new_structures(doc["possible_structures"], theme_prompt, biome_name, doc["possible_grids"][0]["layout"])
        insert_biome(doc)
        return f"✅ Biome '{biome_name}' generated and saved."
    except Exception as e:
        logging.error(f"[DB] ❌ Failed to insert biome: {e}")
        return f"❌ Error generating biome: {e}"

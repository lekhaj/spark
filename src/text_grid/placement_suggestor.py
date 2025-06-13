# src/text_grid/placement_suggestor.py
import random
import json
from . import llm # Relative import
from . import utils # Relative import

# This file is now responsible for getting *hints* from the LLM, not the final layout.

# CRITICAL FIX: Make this function async and await the LLM call
async def get_biome_generation_hints(structure_definitions, theme_prompt):
    """
    Asks the LLM for high-level grid placement hints and a biome name suggestion.
    
    Args:
        structure_definitions (dict): A dictionary of structure IDs mapped to their definitions.
                                      This is passed to the prompt so LLM knows what structures exist.
        theme_prompt (str): The theme of the biome.

    Returns:
        tuple: (dict: parsed LLM hints, str: biome name suggestion)
    """
    prompt = utils.get_grid_placement_hints_prompt(structure_definitions, theme_prompt) 
    
    # CRITICAL FIX: AWAIT the async call to the LLM
    llm_response = await llm.call_online_llm(prompt) 

    if not llm_response: # Check for None explicitly or empty string
        # Re-raise as ValueError to be caught by generate_biome for fallback
        raise ValueError("[ERROR] LLM returned an empty or error response for hints.")

    llm_hints = utils.parse_llm_hints(llm_response) 

    # The biome name suggestion will now come from the LLM hints
    biome_name_suggestion = llm_hints.get("biome_name_suggestion", "Unnamed Biome")

    return llm_hints, biome_name_suggestion


# src/text_grid/placement_suggestor.py

import random
import json
import logging # Import logging
from . import llm # Relative import
from . import utils # Relative import

logger = logging.getLogger(__name__) # Get logger for this module

# This file is now responsible for getting *hints* from the LLM, not the final layout.

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
    
    # Pass the output_schema to guide the LLM to produce JSON
    llm_response = await llm.call_online_llm(prompt, output_schema=llm.GRID_HINTS_SCHEMA) 

    if not llm_response: # Check for None explicitly or empty string
        # Re-raise as ValueError to be caught by generate_biome for fallback
        raise ValueError("[ERROR] LLM returned an empty or error response for hints.")

    # CRITICAL FIX: Use extract_first_json_block to clean the response BEFORE parsing
    cleaned_llm_response = utils.extract_first_json_block(llm_response)

    try:
        parsed_response = json.loads(cleaned_llm_response)
        llm_hints = parsed_response
        logger.info("Successfully parsed LLM response as direct JSON (after cleaning).")
    except json.JSONDecodeError as e:
        logger.error(f"[ERROR] JSONDecodeError parsing cleaned LLM response: {e}\nRaw Cleaned: {cleaned_llm_response}\nOriginal Raw: {llm_response}")
        # If even after cleaning it fails, something is fundamentally wrong, no fallback here.
        raise ValueError(f"Failed to parse LLM response into JSON after cleaning: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during JSON parsing of cleaned LLM response: {e}\nRaw Cleaned: {cleaned_llm_response}\nOriginal Raw: {llm_response}")
        raise ValueError(f"Unexpected error during JSON parsing: {e}")


    # The biome name suggestion will now come from the LLM hints
    biome_name_suggestion = llm_hints.get("biome_name_suggestion", "Unnamed Biome")

    return llm_hints, biome_name_suggestion


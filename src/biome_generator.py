# src/biome_generator.py

import logging
import uuid
from src import llm_inference as llm_interface
from src import database
from src import utils
from prompt_repo import get_biome_generation_prompt


def get_logger():
    return logging.getLogger("rpg_fast_api.biome_generator")
logger = get_logger()

class GenerationResult:
    def __init__(self, success: bool, message: str, biome_name: str | None = None, biome_document: dict | None = None):
        self.success = success
        self.message = message
        self.biome_name = biome_name
        self.biome_document = biome_document


def create_new_biome(theme_prompt: str) -> GenerationResult:
    """
    Orchestrates the entire process of generating and saving a new biome.

    This function is the high-level workflow controller. It delegates tasks
    to the specialized modules (llm_interface, utils, database).

    Args:
        theme_prompt (str): The user's creative theme for the biome.

    Returns:
        GenerationResult: An object containing the status and result message.
    """
    logger.info(f"Starting new biome generation for theme: '{theme_prompt}'")

    try:
        prompt_for_llm = get_biome_generation_prompt(theme_prompt)
    except Exception as e:
        error_msg = f"Failed to construct prompt: {e}"
        logger.error(error_msg, exc_info=True)
        return GenerationResult(success=False, message=error_msg)

    # 2. Call the LLM interface to get the raw response
    raw_llm_response = llm_interface.generate_structured_output(prompt_for_llm)
    if not raw_llm_response:
        error_msg = "LLM failed to generate a response or an error occurred."
        logger.error(error_msg)
        return GenerationResult(success=False, message=error_msg)

    # 3. Parse the raw response into a dictionary using our utility
    biome_document = utils.parse_llm_json_output(raw_llm_response)
    if not biome_document:
        error_msg = "Failed to parse a valid JSON object from the LLM response."
        logger.error(error_msg)
        return GenerationResult(success=False, message=error_msg)

    biome_document['theme_prompt'] = theme_prompt
    biome_document['_id'] = str(uuid.uuid4())
    
    biome_name = biome_document.get("biome_name", "Unnamed Biome")

    save_result = database.save_biome_document(biome_document)
    if not save_result:
        error_msg = f"Failed to save the generated biome '{biome_name}' to the database."
        logger.error(error_msg)
        return GenerationResult(success=False, message=error_msg)

    # --- Success ---
    success_msg = f"Successfully generated and saved biome: '{biome_name}'"
    logger.info(success_msg)
    return GenerationResult(success=True, message=success_msg, biome_name=biome_name, biome_document=biome_document)
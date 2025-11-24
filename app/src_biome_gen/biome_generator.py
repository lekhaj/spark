# src_biome_gen/biome_generator.py

import logging
import uuid
from . import llm_inference as llm_interface
from . import database
from . import utils
from app.prompt_repo import get_biome_generation_prompt


def get_logger():
    return logging.getLogger("rpg_fast_api.biome_generator")
logger = get_logger()

class GenerationResult:
    def __init__(self, success: bool, message: str, biome_name: str | None = None, biome_document: dict | None = None):
        self.success = success
        self.message = message
        self.biome_name = biome_name
        self.biome_document = biome_document


def create_new_biome(theme_prompt: str, system_prompt: str | None = None, save_to_db: bool = True) -> GenerationResult:
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
        # Compose the full prompt sent to the LLM by combining an optional system
        # instruction with the user's theme prompt. We only persist the user's
        # theme prompt to the database per your request.
        user_prompt = theme_prompt
        if system_prompt and system_prompt.strip():
            prompt_for_llm = get_biome_generation_prompt(f"System Instruction:\n{system_prompt}\n\nUser Prompt:\n{user_prompt}")
        else:
            prompt_for_llm = get_biome_generation_prompt(user_prompt)
    except Exception as e:
        error_msg = f"Failed to construct prompt: {e}"
        logger.error(error_msg, exc_info=True)
        return GenerationResult(success=False, message=error_msg)

    # 2. Call the LLM interface to get the raw response
    raw_llm_response = llm_interface.generate_structured_output(prompt_for_llm)
    if isinstance(raw_llm_response, list):
        raw_llm_response = " ".join(str(x) for x in raw_llm_response)
    if not raw_llm_response:
        error_msg = "LLM failed to generate a response or an error occurred."
        logger.error(error_msg)
        return GenerationResult(success=False, message=error_msg)

    # 3. Parse the raw response into a dictionary using our utility
    biome_document = utils.parse_llm_json_output(raw_llm_response)
    if not biome_document:
        error_msg = "Failed to parse a valid JSON object from the LLM response. _in biome_generator.py_"
        logger.error(error_msg)
        return GenerationResult(success=False, message=error_msg)

    # Persist only the user's theme prompt in the stored document.
    biome_document['theme_prompt'] = theme_prompt

    biome_name = biome_document.get("biome_name") or "Unnamed Biome"

    save_result = None
    if save_to_db:
        # Assign an _id only when performing the DB save so generated-but-unsaved
        # documents don't carry a stale or conflicting '_id'.
        biome_document['_id'] = str(uuid.uuid4())
        save_result = database.save_biome_document(biome_document)
        if not save_result:
            error_msg = f"Failed to save the generated biome '{biome_name}' to the database."
            logger.error(error_msg)
            return GenerationResult(success=False, message=error_msg)

    # --- Success ---
    if save_to_db:
        success_msg = f"Successfully generated and saved biome: '{biome_name}'"
    else:
        success_msg = f"Successfully generated biome (not saved): '{biome_name}'"

    logger.info(success_msg)
    return GenerationResult(success=True, message=success_msg, biome_name=biome_name, biome_document=biome_document)

# src_biome_gen/utils.py

import json
import logging
import re

# --- Module-level logger ---
logger = logging.getLogger(__name__)

def parse_llm_json_output(llm_response_text: str) -> dict | None:
    """
    Parses a string containing a JSON object, designed to be robust against
    common LLM output issues like markdown fences and leading/trailing text.

    Args:
        llm_response_text (str): The raw text response from the LLM.

    Returns:
        dict | None: A parsed dictionary if successful, otherwise None.
    """

    temp_file = "temp_llm_response.txt"
    with open(temp_file, "w") as f:
        f.write(llm_response_text)
    logger.info(f"LLM response written to {temp_file} for debugging.")
    if not llm_response_text:
        logger.warning("LLM response was empty or None. Cannot parse JSON.")
        return None

    # --- Step 1: Find the JSON block ---
    # The most common pattern is a markdown-style JSON block.
    # The `re.DOTALL` flag allows `.` to match newlines.
    match = re.search(r'```json\s*(\{.*?\})\s*```', llm_response_text, re.DOTALL)
    
    json_string = ""
    if match:
        json_string = match.group(1)
    else:
        # If no markdown block is found, assume the entire string might be a
        # JSON object, or a JSON object surrounded by other text. We find the
        # first opening brace '{' to the last closing brace '}'.
        first_brace = llm_response_text.find('{')
        last_brace = llm_response_text.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            json_string = llm_response_text[first_brace : last_brace + 1]
        else:
            logger.error("Could not find a valid JSON block in the LLM response.")
            logger.debug(f"LLM Response Text for Debugging:\n{llm_response_text}")
            return None

    # --- Step 2: Parse the extracted JSON string ---
    try:
        parsed_json = json.loads(json_string)
        logger.info("Successfully parsed JSON from LLM response.")
        return parsed_json
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from the extracted block: {e}")
        logger.debug(f"Problematic JSON String for Debugging:\n{json_string}")
        # Here, you could add more advanced repair logic if needed,
        # but for now, we will fail gracefully.
        return None

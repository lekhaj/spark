# test_runner.py

import sys
import os

# --- The Magic Part ---
# 1. Import your new local config module first
from scripts import config_local

# 2. Trick Python: Tell it that whenever any script asks for the module named 'config',
#    it should be given our 'config_local' module instead.
#    This must be done BEFORE we import the main_processor.
sys.modules['config'] = config_local

# --- The Rest is the Same ---
# 3. Now, import the function you want to test.
#    When main_processor.py runs its "import config" line, Python will give it
#    our fake one (config_local) because of the trick above.
from scripts.main_processor import run_blender_script


# Define paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(PROJECT_ROOT, "input_models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_models")
INPUT_FILENAME = "109_car.glb" # The file you want to test


# Run the test
if __name__ == "__main__":
    input_file_path = os.path.join(INPUT_DIR, INPUT_FILENAME)

    print("--- Starting Isolated Local Test ---")
    print(f"Using temporary config from: {config_local.__file__}")

    if not os.path.exists(input_file_path):
        print(f"!!! ERROR: Input file not found at {input_file_path}")
    else:
        # Call the function. It will now use the settings from config_local.py!
        success = run_blender_script(input_file_path, OUTPUT_DIR)
        
        if success:
            print("--- Local Test Finished Successfully ---")
        else:
            print("--- Local Test Failed ---")
import json
import re

INPUT_FILE = "biome_raw_single.txt"
OUTPUT_FILE = "biome_final.json"
DEBUG_FILE = "debug_cleaned.json"

# Load raw LLM output
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    raw = f.read()

# Step 1: Extract content between <think>```json and final ```
match = re.search(r"<think>\\s*```json\\s*(\\{.*?\\})\\s*```", raw, re.DOTALL)
if not match:
    print("❌ Could not find valid JSON block between <think>```json and ```.")
    exit(1)

json_block = match.group(1)

# Step 2: Clean non-printable characters
json_block = ''.join(c for c in json_block if c.isprintable())

# Step 3: Save cleaned JSON to debug file
with open(DEBUG_FILE, "w", encoding="utf-8") as debug:
    debug.write(json_block)

# Step 4: Attempt to parse
try:
    biome = json.loads(json_block)
except json.JSONDecodeError as e:
    print("❌ JSON parsing failed:", e)
    print(f"💡 Check '{DEBUG_FILE}' for the cleaned JSON content.")
    exit(1)

# Step 5: Filter invalid grid IDs
defined_ids = set(biome["possible_structures"]["buildings"].keys())
for grid in biome.get("possible_grids", []):
    if "layout" in grid:
        grid["layout"] = [
            [cell if str(cell) in defined_ids or cell == 0 else 0 for cell in row]
            for row in grid["layout"]
        ]

# Step 6: Save clean result
with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    json.dump(biome, out, indent=4)

print(f"✅ Final cleaned biome JSON saved to {OUTPUT_FILE}")
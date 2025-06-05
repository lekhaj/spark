import json
from openai import OpenAI, OpenAIError
from config import OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from db import insert_biome, get_next_structure_id
from utils import build_prompt, extract_json_block, remap_structure_ids

client = OpenAI(api_key="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", base_url=OPENROUTER_BASE_URL)

def generate_biome(theme):
    prompt = build_prompt(theme)
    try:
        response = client.chat.completions.create(
            model="qwen/qwen2.5-vl-32b-instruct:free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=1024,
            top_p=0.95
        )
        content = response.choices[0].message.content.strip()
        doc = extract_json_block(content)
        doc["theme_prompt"] = theme

        # Reuse previously generated structures if available
        existing_structures = get_structures_by_theme(theme)
        if existing_structures:
            doc["possible_structures"] = existing_structures

            # remap grid to use only existing IDs
            all_ids = []
            for cat in existing_structures.values():
                all_ids.extend(map(int, cat.keys()))
            min_id = min(all_ids)
            mapping = {int(k): int(k) for k in range(min_id, min_id + len(all_ids))}
            for grid in doc["possible_grids"]:
                grid["layout"] = [
                    [mapping.get(val, val) for val in row] for row in grid["layout"]
                ]
        else:
            doc = remap_structure_ids(doc, get_next_structure_id())

        insert_biome(doc)
        return f"✅ Biome '{doc['biome_name']}' generated and saved."
    except (OpenAIError, json.JSONDecodeError) as e:
        return f"❌ Error: {str(e)}"



import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

output_file = "biome_raw_single.txt"

example = """Here is an example JSON biome:
{
  "biome_name": "Medieval City",
  "possible_structures": {
    "buildings": {
      "1": {
        "type": "mill",
        "attributes": { "hp": 600 },
        "description": "A grand brick mill with arched windows and a tall smokestack, functioning as a key hub for grain processing."
      },
      "2": {
        "type": "blacksmith",
        "attributes": { "hp": 700 },
        "description": "A massive blacksmith forge with thick stone walls, a wide hearth, and an abundance of ironworking equipment."
      },
      "3": {
        "type": "shop",
        "attributes": { "hp": 400 },
        "description": "A well-furnished wooden shop with intricate carvings and a tiled roof, serving as a local merchant's quarters."
      },
      "4": {
        "type": "school",
        "attributes": { "hp": 500 },
        "description": "An ancient wooden schoolhouse with high ceilings and large windows, housing scholars and students alike."
      },
      "5": {
        "type": "inn",
        "attributes": { "hp": 450 },
        "description": "A cozy wooden inn with a thatched roof, frequented by travelers and merchants during the day."
      }
    }
  },
  "possible_grids": [
    {
      "grid_id": "grid_1",
      "layout": [[1,1,0,2,2,0,3,3,4,5], [1,1,0,2,2,0,3,3,4,5], [0,0,0,0,0,0,0,0,0,0],
                 [2,2,0,1,1,0,3,3,4,5], [2,2,0,1,1,0,3,3,4,5], [0,0,0,0,0,0,0,0,0,0],
                 [3,3,0,1,1,0,2,2,4,5], [3,3,0,1,1,0,2,2,4,5], [0,0,0,0,0,0,0,0,0,0],
                 [4,4,0,5,5,0,1,1,0,4]]
    }
  ]
}
"""

# Single prompt asking for the full JSON
prompt = f"""
You are a world-building assistant. Generate a complete JSON document for a medieval biome, structured exactly like the example below.

{example}

Now generate a different biome with a new name, 5 original buildings (with type, attributes, and descriptions), and a 10x10 grid using the IDs. Respond ONLY with valid JSON output.
"""

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1000,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

# Run and save result
response = generate(prompt)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(response)

print(f"✓ Single-prompt biome JSON saved to {output_file}")
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os

# Initialize the model and tokenizer
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Change this to your preferred model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)

# Prompt for user input
biome_description = input("Enter a description of the biome or location: ")

# Build the LLM prompt
prompt = f"""
Generate a list of typical objects you would find in the following environment:
Description: {biome_description}

Respond in a strict JSON array format, with each object as a string.
Example:
[
    "tree",
    "river",
    "cottage",
    "path"
]
"""

# Tokenize and generate
input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model.generate(
        **input_ids,
        max_new_tokens=300,
        temperature=0.5,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# Decode and parse
# Initialize an empty dictionary to store the valid items
cell_mapping = {}

# Step 1: Process the string and extract lines
lines = output.splitlines()

# Step 2: Loop through each line and match the pattern
for line in lines:
    match = re.match(r'^\s*"(\d+)":\s*\{"type":\s*"([^"]+)"\}\s*,?$', line.strip())
    if match:
        # Extract the ID and type
        id_ = match.group(1)
        type_ = match.group(2)
        
        # Add to the dictionary
        cell_mapping[id_] = {"type": type_}

# Step 3: Save the dictionary to a JSON file
with open('cell_mapping.json', 'a') as f:
    json.dump(cell_mapping, f, indent=4)

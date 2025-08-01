import torch
import os
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# Step 1: Setup paths
output_dir = "outputs/prompt_to_3d"
os.makedirs(output_dir, exist_ok=True)

# Step 2: Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# Step 3: Define your prompt
prompt = "A futuristic red sports car with glowing lights in a showroom"
image_path = os.path.join(output_dir, "generated_image.png")
mesh_path = os.path.join(output_dir, "generated_model.ply")

# Step 4: Generate image from prompt using SDXL Turbo
print("🎨 Generating image from prompt...")
sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to(device)

image = sdxl_pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0).images[0]
image.save(image_path)
print(f"✅ Image saved at: {image_path}")

# Step 5: Load the image for 3D generation
image = load_image(image_path).convert("RGB")

# Step 6: Generate 3D model using Hunyuan3D
print("🧱 Generating 3D mesh from image...")
hunyuan_pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1",
    subfolder="hunyuan3d-dit-v2-mini",
    torch_dtype=torch.float16
).to(device)

output = hunyuan_pipe(image, output_type="mesh")
output["mesh"].export(mesh_path)

print(f"✅ 3D model saved at: {mesh_path}")

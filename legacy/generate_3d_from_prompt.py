import torch
from PIL import Image, ImageDraw
from diffusers import StableDiffusionPipeline
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import numpy as np
import os
import boto3
import logging
import datetime
from botocore.exceptions import ClientError
import trimesh
from uuid import uuid4
import json
import gradio as gr
import tempfile
# Removed google.generativeai as it's no longer used
from celery import Celery
from celery.result import AsyncResult

# Configure logging
logging.basicConfig(level=logging.INFO)

# --- Celery Configuration ---
# IMPORTANT: Replace 'your_app_name_without_py_extension' with the actual filename
# For example, if your file is 'my_generator.py', use 'my_generator'
app = Celery('asset_generator', 
             broker='redis://localhost:6379/0', 
             backend='redis://localhost:6379/0')

# --- S3 Upload Function ---
def upload_file_to_s3(file_name, bucket, object_name=None):
    if object_name is None:
        object_name = os.path.basename(file_name)

    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_name, bucket, object_name)
        logging.info(f"Successfully uploaded {file_name} to s3://{bucket}/{object_name}")
        return f"https://{bucket}.s3.amazonaws.com/{object_name}"
    except ClientError as e:
        logging.error(f"Failed to upload {file_name}: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred during S3 upload: {e}")
        return None

# --- Decimation Function ---
def decimate_mesh(mesh_filename, target_faces=1000):
    print(f"\n--- Starting Decimation Step ---")
    print(f"Decimating mesh to approximately {target_faces} faces...")
    
    mesh = trimesh.load(mesh_filename)
    if isinstance(mesh, trimesh.Scene):
        if not mesh.geometry:
            print("Warning: Scene contains no geometry. Decimation skipped.")
            return None
        if mesh.dump():
            mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
        else:
            print("Warning: Scene has no dumpable geometry. Decimation skipped.")
            return None

    decimated_mesh = mesh.simplify_quadric_decimation(face_count=target_faces)
    print(f"Original faces: {len(mesh.faces)}, Decimated faces: {len(decimated_mesh.faces)}")
    return decimated_mesh

# --- Celery Tasks ---

@app.task
def generate_2d_image_task(text_prompt: str, width: int, height: int, num_images: int, s3_bucket_name: str, base_filename: str):
    optimized_prompt = f"{text_prompt}, 3D render, octane render, unreal engine, cinematic, highly detailed"
    
    stable_diffusion_model_id = "runwayml/stable-diffusion-v1-5"
    try:
        sd_pipe = StableDiffusionPipeline.from_pretrained(
            stable_diffusion_model_id, torch_dtype=torch.float16
        ).to("cuda")
    except Exception:
        sd_pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_model_id).to("cpu")

    negative_prompt = "blurry, low quality, bad anatomy, ugly, artifacts, text, watermark"
    
    generated_image_urls = []
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(num_images):
            image = sd_pipe(
                optimized_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=25,
                width=width,
                height=height
            ).images[0]
            
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{base_filename}_{timestamp}_{i}.png"
            local_path = os.path.join(tmpdir, filename)
            image.save(local_path)
            
            s3_url = upload_file_to_s3(local_path, s3_bucket_name, f"images/{filename}")
            if s3_url:
                generated_image_urls.append(s3_url)
                print(f"Uploaded 2D image to: {s3_url}")
            else:
                print(f"Failed to upload 2D image {filename} to S3.")
    
    return {"status": "completed", "image_urls": generated_image_urls}

@app.task
def generate_grid_image_task(grid_data_str: str, width: int, height: int, num_images: int, s3_bucket_name: str, base_filename: str):
    try:
        grid_data = json.loads(grid_data_str)
        if not isinstance(grid_data, list) or not all(isinstance(row, list) and all(isinstance(cell, int) for cell in row) for row in grid_data):
            raise ValueError("Grid data must be a valid JSON array of arrays of integers.")

        if not grid_data or not grid_data[0]:
            raise ValueError("Grid data is empty.")

        cell_size = 50 
        img_width = len(grid_data[0]) * cell_size
        img_height = len(grid_data) * cell_size

        terrain_colors = {
            0: (144, 238, 144),
            1: (34, 139, 34),
            2: (139, 69, 19),
            3: (65, 105, 225),
            4: (244, 164, 96),
            5: (240, 248, 255),
            6: (102, 51, 153),
            7: (160, 82, 45),
            8: (105, 105, 105),
            9: (128, 128, 128)
        }

        generated_visualization_urls = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(num_images):
                img = Image.new('RGB', (img_width, img_height), color = 'white')
                d = ImageDraw.Draw(img)

                for r_idx, row in enumerate(grid_data):
                    for c_idx, cell_value in enumerate(row):
                        color = terrain_colors.get(cell_value, (0, 0, 0))
                        x1 = c_idx * cell_size
                        y1 = r_idx * cell_size
                        x2 = x1 + cell_size
                        y2 = y1 + cell_size
                        d.rectangle([x1, y1, x2, y2], fill=color, outline=(0,0,0))
                
                timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"{base_filename}_{timestamp}_{i}.png"
                local_path = os.path.join(tmpdir, filename)
                img.save(local_path)

                s3_url = upload_file_to_s3(local_path, s3_bucket_name, f"images/grid_visualizations/{filename}")
                if s3_url:
                    generated_visualization_urls.append(s3_url)
                    print(f"Uploaded grid visualization to: {s3_url}")
                else:
                    print(f"Failed to upload grid visualization {filename} to S3.")

        return {"status": "completed", "visualization_urls": generated_visualization_urls}

    except json.JSONDecodeError as e:
        return {"status": "failed", "error": f"Invalid JSON format for Grid Data: {e}"}
    except ValueError as ve:
        return {"status": "failed", "error": f"Grid data error: {ve}"}
    except Exception as e:
        return {"status": "failed", "error": f"An unexpected error occurred during grid visualization: {e}"}

@app.task
def generate_3d_from_2d_task(image_input_base64: str, s3_bucket_name: str, base_filename: str):
    try:
        # Decode base64 image back to PIL Image
        from io import BytesIO
        import base64
        image_bytes = base64.b64decode(image_input_base64)
        image_2d_input = Image.open(BytesIO(image_bytes))

        with tempfile.TemporaryDirectory() as tmpdir:
            initial_model_local_path = os.path.join(tmpdir, f"3d_{base_filename}.glb")

            hunyuan_model_id = 'tencent/Hunyuan3D-2mini'
            hunyuan_subfolder = 'hunyuan3d-dit-v2-mini'
            
            hunyuan_pipeline = None
            try:
                hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                    hunyuan_model_id,
                    subfolder=hunyuan_subfolder,
                    use_safetensors=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            except Exception as e:
                print(f"Failed to load Hunyuan3D-2mini with GPU. Error: {e}")
                try:
                    hunyuan_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
                        hunyuan_model_id,
                        subfolder=hunyuan_subfolder,
                        use_safetensors=True
                    ).to("cpu")
                    print("Successfully loaded model on CPU.")
                except Exception as cpu_e:
                    print(f"Failed to load Hunyuan3D-2mini on CPU. Error: {cpu_e}")
                    return {"status": "failed", "error": "Failed to load Hunyuan3D model."}

            with torch.no_grad():
                mesh = hunyuan_pipeline(
                    image=image_2d_input,
                    num_inference_steps=30,
                    octree_resolution=256,
                    generator=torch.Generator(device=hunyuan_pipeline.device).manual_seed(42)
                )[0]
            
            mesh.export(initial_model_local_path)
            
            s3_url = upload_file_to_s3(initial_model_local_path, s3_bucket_name, f"3d_assets/{base_filename}.glb")
            
            if s3_url:
                return {"status": "completed", "model_url": s3_url}
            else:
                return {"status": "failed", "error": "Failed to upload 3D model to S3."}
    except Exception as e:
        return {"status": "failed", "error": f"Error during 3D generation: {e}"}

@app.task
def decimate_3d_task(input_3d_file_base64: str, s3_bucket_name: str, base_filename: str, original_file_extension: str):
    try:
        # Decode base64 file back to a temporary file
        from io import BytesIO
        import base64
        file_bytes = base64.b64decode(input_3d_file_base64)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_local_path = os.path.join(tmpdir, f"input_model.{original_file_extension}")
            with open(input_local_path, "wb") as f:
                f.write(file_bytes)

            output_decimated_path = os.path.join(tmpdir, f"decimated_{base_filename}.glb")

            decimated_mesh_obj = decimate_mesh(input_local_path, target_faces=1000)
            if decimated_mesh_obj is not None:
                decimated_mesh_obj.export(output_decimated_path)
                s3_url = upload_file_to_s3(output_decimated_path, s3_bucket_name, f"processed/{base_filename}_decimated.glb")
                if s3_url:
                    return {"status": "completed", "decimated_model_url": s3_url}
                else:
                    return {"status": "failed", "error": "Failed to upload decimated 3D model to S3."}
            else:
                return {"status": "failed", "error": "Decimation failed: No valid mesh produced."}
    except Exception as e:
        return {"status": "failed", "error": f"Error during decimation: {e}"}

# --- Gradio UI Functions (to interact with Celery) ---

def submit_2d_image_task(text_prompt: str, width: int, height: int, num_images: int, s3_bucket_name: str, base_filename: str):
    task = generate_2d_image_task.delay(text_prompt, width, height, num_images, s3_bucket_name, base_filename)
    return f"Task submitted! ID: {task.id}", None

def submit_grid_image_task(grid_data_str: str, width: int, height: int, num_images: int, s3_bucket_name: str, base_filename: str):
    task = generate_grid_image_task.delay(grid_data_str, width, height, num_images, s3_bucket_name, base_filename)
    return f"Task submitted! ID: {task.id}", None, None

def submit_3d_from_2d_task(image_2d_input: Image.Image, s3_bucket_name: str, base_filename: str):
    if image_2d_input is None:
        return "Please upload a 2D image first.", None
    
    # Convert PIL Image to base64 string for Celery task
    from io import BytesIO
    import base64
    buffered = BytesIO()
    image_2d_input.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    task = generate_3d_from_2d_task.delay(image_base64, s3_bucket_name, base_filename)
    return f"Task submitted! ID: {task.id}", None

def submit_decimate_3d_task(input_3d_file: gr.File, s3_bucket_name: str, base_filename: str):
    if input_3d_file is None:
        return "Please upload a 3D model (GLB/OBJ/STL) first.", None
    
    # Read file content and encode to base64
    with open(input_3d_file.name, "rb") as f:
        file_bytes = f.read()
    import base64
    file_base64 = base64.b64encode(file_bytes).decode('utf-8')
    original_file_extension = input_3d_file.name.split('.')[-1]

    task = decimate_3d_task.delay(file_base64, s3_bucket_name, base_filename, original_file_extension)
    return f"Task submitted! ID: {task.id}", None

def check_task_status(task_id: str):
    if not task_id:
        return "Please enter a Task ID.", "", "", ""
    
    task_result = AsyncResult(task_id, app=app)
    
    if task_result.ready():
        result = task_result.get()
        if result["status"] == "completed":
            if "image_urls" in result: # For 2D image generation
                image_urls = result["image_urls"]
                return "Task completed successfully!", image_urls, None, json.dumps(result, indent=2)
            elif "visualization_urls" in result: # For Grid to Image
                vis_urls = result["visualization_urls"]
                return "Task completed successfully!", vis_urls, None, json.dumps(result, indent=2)
            elif "model_url" in result: # For 3D generation
                model_url = result["model_url"]
                return "Task completed successfully!", None, gr.HTML(f"<a href='{model_url}' target='_blank'>Download 3D Model</a>"), json.dumps(result, indent=2)
            elif "decimated_model_url" in result: # For Decimated 3D
                decimated_url = result["decimated_model_url"]
                return "Task completed successfully!", None, gr.HTML(f"<a href='{decimated_url}' target='_blank'>Download Decimated 3D Model</a>"), json.dumps(result, indent=2)
            else:
                return "Task completed, but result format is unexpected.", None, None, json.dumps(result, indent=2)
        else:
            return f"Task failed: {result.get('error', 'Unknown error')}", None, None, json.dumps(result, indent=2)
    elif task_result.pending:
        return "Task is pending...", None, None, ""
    elif task_result.started:
        return "Task is running...", None, None, ""
    else:
        return "Task ID not found or invalid.", None, None, ""

# --- Gradio Interface ---
if __name__ == "__main__":
    with gr.Blocks(title="AI-Powered 3D Asset Generator (Celery Offloading)") as demo:
        gr.Markdown("# AI-Powered 3D Asset Generator")
        gr.Markdown("This application offloads heavy AI generation tasks to Celery workers (on GPU instances), allowing the Gradio UI to remain responsive on a CPU instance.")

        s3_bucket_input_global = gr.Textbox(label="S3 Bucket Name", value="sparkassets-us", interactive=True)

        with gr.Tabs():
            with gr.TabItem("Text to Image"):
                gr.Markdown("## Text-to-Image Generation")
                gr.Markdown("Generate images from text descriptions. **All prompts are automatically optimized for 3D asset generation** with added keywords for better 3D model quality.")
                
                with gr.Row():
                    gr.Markdown("### � 3D Generation Optimization")
                    gr.Checkbox(label="Enabled", value=True, interactive=False) 

                text_to_image_prompt = gr.Textbox(
                    label="Text Prompt", 
                    placeholder="💡 Tip: Describe objects clearly for best 3D generation results. Background and environment terms will be automatically optimized.",
                    lines=3
                )
                
                base_filename_txt2img = gr.Textbox(label="Base Filename for Image(s)", placeholder="e.g., my_2d_image")

                with gr.Row():
                    width_slider_txt2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                    height_slider_txt2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                
                with gr.Row():
                    num_images_slider_txt2img = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                    model_dropdown_txt2img = gr.Dropdown(
                        label="Model", 
                        choices=["SDXL Turbo: High-quality local GPU image generation optimized for 3D"], 
                        value="SDXL Turbo: High-quality local GPU image generation optimized for 3D",
                        interactive=False
                    )
                
                generate_image_button = gr.Button("🚀 Generate Image from Text (3D-Optimized)")
                image_generation_status = gr.Textbox(label="Task Status", lines=1)
                image_generation_output = gr.Gallery(label="Generated Images", columns=2, height='auto')

                generate_image_button.click(
                    fn=submit_2d_image_task,
                    inputs=[text_to_image_prompt, width_slider_txt2img, height_slider_txt2img, num_images_slider_txt2img, s3_bucket_input_global, base_filename_txt2img],
                    outputs=[image_generation_status, image_generation_output]
                )
            
            with gr.TabItem("Grid to Image"):
                gr.Markdown("## Grid to Image Visualization")
                gr.Markdown("""
                **Grid Format**
                Use numbers to represent different terrain types:
                * **0**: Plain
                * **1**: Forest
                * **2**: Mountain
                * **3**: Water
                * **4**: Desert
                * **5**: Snow
                * **6**: Swamp
                * **7**: Hills
                * **8**: Urban
                * **9**: Ruins
                """)
                
                grid_data_input = gr.Textbox(label="Grid Data (JSON array of arrays)", lines=10, 
                                             placeholder="Example: [[0,0,1,1],[0,1,1,0]]")
                load_sample_grid_button = gr.Button("Load Sample Grid")
                
                base_filename_grid2img = gr.Textbox(label="Base Filename for Visualization", placeholder="e.g., my_grid_map")

                with gr.Row():
                    width_slider_grid2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                    height_slider_grid2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                
                with gr.Row():
                    num_images_slider_grid2img = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                    model_dropdown_grid2img = gr.Dropdown(
                        label="Model", 
                        choices=["SDXL Turbo: High-quality local GPU image generation optimized for 3D"], 
                        value="SDXL Turbo: High-quality local GPU image generation optimized for 3D",
                        interactive=False
                    )
                
                generate_grid_image_button = gr.Button("Generate Image from Grid")
                grid_generation_status = gr.Textbox(label="Task Status", lines=1)
                grid_visualization_output = gr.Gallery(label="Grid Visualization", columns=2, height='auto')
                generated_terrain_output = gr.Gallery(label="Generated Terrain (Future Feature)", columns=2, height='auto') 

                load_sample_grid_button.click(
                    fn=load_sample_grid,
                    inputs=[],
                    outputs=[grid_data_input]
                )

                generate_grid_image_button.click(
                    fn=submit_grid_image_task,
                    inputs=[grid_data_input, width_slider_grid2img, height_slider_grid2img, num_images_slider_grid2img, s3_bucket_input_global, base_filename_grid2img],
                    outputs=[grid_generation_status, grid_visualization_output, generated_terrain_output]
                )

            with gr.TabItem("3D Generation"):
                gr.Markdown("## 3D Model Generation from 2D Image")
                gr.Markdown("Upload a 2D image to generate a 3D GLB model. The generated 3D model will be stored in your S3 bucket.")
                
                input_2d_image_for_3d = gr.Image(label="Upload 2D Image", type="pil")
                base_filename_3d_gen = gr.Textbox(label="Base Filename for 3D Model (e.g., my_3d_asset)")
                
                generate_3d_button = gr.Button("Generate 3D Model")
                status_3d_gen = gr.Textbox(label="Task Status", lines=1)
                output_3d_model_link = gr.HTML(label="Generated 3D Model Link")

                generate_3d_button.click(
                    fn=submit_3d_from_2d_task,
                    inputs=[input_2d_image_for_3d, s3_bucket_input_global, base_filename_3d_gen],
                    outputs=[status_3d_gen, output_3d_model_link]
                )

            with gr.TabItem("Decimated 3D"):
                gr.Markdown("## Decimate 3D Model")
                gr.Markdown("Upload an existing 3D GLB/OBJ/STL model to reduce its polygon count. The decimated model will be stored in your S3 bucket.")
                
                input_3d_file_decimate = gr.File(label="Upload 3D Model (GLB, OBJ, STL)", type="filepath")
                base_filename_decimate = gr.Textbox(label="Base Filename for Decimated Model (e.g., my_decimated_asset)")
                
                decimate_button = gr.Button("Decimate 3D Model")
                status_decimate = gr.Textbox(label="Task Status", lines=1)
                output_decimated_model_link = gr.HTML(label="Decimated 3D Model Link")

                decimate_button.click(
                    fn=submit_decimate_3d_task,
                    inputs=[input_3d_file_decimate, s3_bucket_input_global, base_filename_decimate],
                    outputs=[status_decimate, output_decimated_model_link]
                )
            
            with gr.TabItem("Check Task Results"):
                gr.Markdown("## Check Status of Submitted Tasks")
                task_id_input = gr.Textbox(label="Enter Task ID")
                check_status_button = gr.Button("Check Status")
                result_status_output = gr.Textbox(label="Status", lines=1)
                result_image_output = gr.Gallery(label="Result Images", columns=2, height='auto')
                result_link_output = gr.HTML(label="Result Link")
                full_result_json = gr.JSON(label="Full Result JSON")

                check_status_button.click(
                    fn=check_task_status,
                    inputs=[task_id_input],
                    outputs=[result_status_output, result_image_output, result_link_output, full_result_json]
                )

    demo.launch(server_name="0.0.0.0", server_port=7000)
�

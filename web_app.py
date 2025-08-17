# web_app.py
import gradio as gr
from app import generate_2d_image_task, generate_image_from_grid_task
import json
import time
import io
from PIL import Image
from datetime import datetime

# --- Simulated Database (In-Memory) ---
# This dictionary simulates a MongoDB collection of prompts.
mock_prompts_db = {
    "prompts": [
        {"_id": "prompt1", "name": "Vibrant medieval fantasy scene, high detail", "content": "A high-quality 3D render of a vibrant medieval fantasy scene with a knight, dragon, and a castle. Optimized for 3D asset generation."},
        {"_id": "prompt2", "name": "Minimalist sci-fi spaceship blueprint", "content": "A clean, high-resolution 2D blueprint of a minimalist sci-fi spaceship. Optimized for 3D asset generation."},
        {"_id": "prompt3", "name": "Stylized robot character concept", "content": "A detailed concept art of a stylized robot character with a glossy, metallic finish. Optimized for 3D asset generation."},
    ],
    "grids": [
        {"_id": "grid1", "name": "Sample Grid Map 1", "content": "[[0,0,1,1,0,0,2,2,0,0],[0,1,1,1,1,0,2,2,2,0],[1,1,1,1,1,1,0,2,2,2],[1,1,1,1,1,1,0,0,2,2],[0,1,1,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[3,3,3,3,3,3,3,3,3,3],[3,3,3,3,3,3,3,3,3,3],[4,4,4,4,0,0,0,0,0,0],[4,4,4,4,0,0,0,0,0,0]]"},
        {"_id": "grid2", "name": "Sample Grid Map 2", "content": "[[1,1,1,1,1],[1,2,2,2,1],[1,2,9,2,1],[1,2,2,2,1],[1,1,1,1,1]]"},
        {"_id": "grid3", "name": "Sample Grid Map 3", "content": "[[3,3,3,3,3],[3,0,0,0,3],[3,0,4,0,3],[3,0,0,0,3],[3,3,3,3,3]]"},
    ]
}

# In-memory database for completed tasks (retained from previous version)
mock_database = []

def fetch_prompts(db_name, col_name):
    """Simulates fetching prompts from a database collection."""
    # In a real app, this would query the DB with db_name and col_name
    # Here, we just return the mock data for all "collections"
    prompts = mock_prompts_db.get("prompts", [])
    prompt_names = [p["name"] for p in prompts]
    return gr.Dropdown.update(choices=prompt_names, value=None), gr.Textbox.update(value=f"Fetched {len(prompts)} prompts from simulated database.")

def fetch_grids(db_name, col_name):
    """Simulates fetching grids from a database collection."""
    grids = mock_prompts_db.get("grids", [])
    grid_names = [g["name"] for g in grids]
    return gr.Dropdown.update(choices=grid_names, value=None), gr.Textbox.update(value=f"Fetched {len(grids)} grids from simulated database.")

def get_prompt_content(prompt_name):
    """Retrieves the full prompt text based on the name selected from the dropdown."""
    for p in mock_prompts_db.get("prompts", []):
        if p["name"] == prompt_name:
            return p["content"]
    return ""

def launch_2d_generation(text_prompt, width, height, num_images, s3_bucket_name, base_filename):
    """Starts the 2D image generation task and returns the task ID."""
    task = generate_2d_image_task.delay(text_prompt, width, height, num_images, s3_bucket_name, base_filename)
    return task.id

def track_2d_generation_progress(task_id):
    """Tracks the status of the 2D image generation task and yields updates."""
    if not task_id:
        yield "Waiting for task to start...", [], None
        return
    
    while True:
        task = generate_2d_image_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            yield f"Task is pending... (ID: {task_id})", [], None
        elif task.state == 'PROGRESS':
            status = task.info.get('status', 'Processing...')
            results = task.info.get('result', [])
            yield f"{status} (ID: {task_id})", results, None
        elif task.state == 'SUCCESS':
            status = task.info.get('status', "Task complete!")
            results = task.info.get('result', [])
            
            # Add to mock database with s3 bucket info
            global mock_database
            mock_database.append({
                "task_id": task_id,
                "type": "2D Image",
                "status": "SUCCESS",
                "description": f"Text-to-Image for prompt: '{task.info.get('prompt', 'N/A')}'",
                "files": results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "s3_bucket": task.info.get('s3_bucket', 'N/A')
            })

            html_output = f"<h3>Generated Images:</h3>"
            for url in results:
                html_output += f"<a href='{url}' target='_blank'>Download Image</a><br>"
            yield f"{status} (ID: {task_id})", results, html_output
            return
        elif task.state == 'FAILURE':
            error_msg = task.info.get('error', "An error occurred.")
            yield f"Error: {error_msg} (ID: {task_id})", [], None
            return
        else:
            yield f"Task state: {task.state} (ID: {task_id})", [], None
            
        time.sleep(2)

def launch_grid_generation(grid_data_str, width, height, num_images, s3_bucket_name, base_filename):
    """Starts the grid visualization task and returns the task ID."""
    task = generate_image_from_grid_task.delay(grid_data_str, width, height, num_images, s3_bucket_name, base_filename)
    return task.id

def track_grid_generation_progress(task_id):
    """Tracks the status of the grid generation task and yields updates."""
    if not task_id:
        yield "Waiting for task to start...", None
        return
        
    while True:
        task = generate_image_from_grid_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            yield f"Task is pending... (ID: {task_id})", None
        elif task.state == 'PROGRESS':
            status = task.info.get('status', 'Processing...')
            results = task.info.get('result', [])
            images = [url for url in results]
            yield f"{status} (ID: {task_id})", images
        elif task.state == 'SUCCESS':
            status = task.info.get('status', "Task complete!")
            results = task.info.get('result', [])
            images = [url for url in results]
            
            # Add to mock database with s3 bucket info
            global mock_database
            mock_database.append({
                "task_id": task_id,
                "type": "Grid Visualization",
                "status": "SUCCESS",
                "description": "Image generated from grid data",
                "files": results,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "s3_bucket": task.info.get('s3_bucket', 'N/A')
            })

            yield f"{status} (ID: {task_id})", images
            return
        elif task.state == 'FAILURE':
            error_msg = task.info.get('error', "An error occurred.")
            yield f"Error: {error_msg} (ID: {task_id})", None
            return
        else:
            yield f"Task state: {task.state} (ID: {task_id})", None

        time.sleep(2)

def get_stored_files():
    """Retrieves and formats the stored file data for display."""
    headers = ["Task ID", "Type", "Description", "Status", "Timestamp", "S3 Bucket", "Download Link"]
    data = []
    
    for item in mock_database:
        link_html = ""
        if item.get("files"):
            for url in item["files"]:
                link_html += f"<a href='{url}' target='_blank'>Download</a> "
        
        row = [
            item.get("task_id", "N/A"),
            item.get("type", "N/A"),
            item.get("description", "N/A"),
            item.get("status", "N/A"),
            item.get("timestamp", "N/A"),
            item.get("s3_bucket", "N/A"),
            link_html
        ]
        data.append(row)
    
    return gr.Dataframe(headers=headers, data=data)

def get_grid_content(grid_name):
    """Retrieves the full grid content based on the name selected from the dropdown."""
    for g in mock_prompts_db.get("grids", []):
        if g["name"] == grid_name:
            return g["content"]
    return ""

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# AI-Powered 3D Asset Generator")
    
    # State variable to store the task ID
    task_id_state = gr.State(None)
    
    with gr.Tabs():
        with gr.TabItem("Text to 2D Image"):
            gr.Markdown("## Generate Images from a Text Prompt with SDXL Turbo")
            gr.Markdown("Enter a text prompt below and use SDXL Turbo to generate 2D images.")
            
            with gr.Row():
                text_prompt_input = gr.Textbox(label="Text Prompt", lines=3)
            
            with gr.Row():
                width_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                height_slider = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                num_images_slider = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
            
            generate_2d_btn = gr.Button("🚀 Generate 2D Image")
            
            status_text_2d = gr.Textbox(label="Status", lines=1)
            generated_image_gallery_2d = gr.Gallery(label="Generated Image", columns=2)
            
            generate_2d_btn.click(
                fn=launch_2d_generation,
                inputs=[text_prompt_input, width_slider, height_slider, num_images_slider, gr.Textbox(value="sparkassets", visible=False), gr.Textbox(value="manual_2d_image", visible=False)],
                outputs=[task_id_state]
            ).then(
                fn=track_2d_generation_progress,
                inputs=[task_id_state],
                outputs=[status_text_2d, generated_image_gallery_2d, gr.HTML()]
            )
            
        with gr.TabItem("Generate Images from Prompts"):
            gr.Markdown("## Generate Images from MongoDB Prompts with SDXL Turbo")
            gr.Markdown("""
            🚀 **SDXL Turbo Integration:** Ultra-fast, high-quality local image generation optimized for 3D assets!
            🎯 All prompts are automatically enhanced for 3D asset generation - perfect for creating images that will work well with the 3D Generation tab!
            """)
            
            with gr.Row():
                db_name_input = gr.Textbox(label="Database Name", value="mock_db", interactive=True)
                col_name_input = gr.Textbox(label="Collection Name", value="prompts", interactive=True)
                fetch_prompts_btn = gr.Button("Fetch Prompts")
                
            gr.Markdown("### 🎯 SDXL Turbo Features:")
            gr.Markdown("""
            - ⚡ **Ultra-fast:** 2-4 inference steps vs 20-50 for other models
            - 🏠 **Local processing:** No API costs, completely private
            - 🎨 **3D-optimized:** Clean backgrounds, perfect lighting for 3D generation
            """)
            
            with gr.Row():
                width_slider_db = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                height_slider_db = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                num_images_slider_db = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
            
            model_dropdown = gr.Dropdown(
                label="Model", 
                choices=["SDXL Turbo (Local GPU-accelerated generation)"], 
                value="SDXL Turbo (Local GPU-accelerated generation)", 
                interactive=False
            )
            
            with gr.Row():
                select_prompt_dropdown = gr.Dropdown(label="Select a Prompt")
                generate_prompt_btn = gr.Button("🚀 Generate with SDXL Turbo")

            status_text_db = gr.Textbox(label="Status", lines=1)
            generated_image_gallery_db = gr.Gallery(label="Generated Image (SDXL Turbo Output)", columns=2)
            
            # Linking components
            fetch_prompts_btn.click(
                fn=fetch_prompts,
                inputs=[db_name_input, col_name_input],
                outputs=[select_prompt_dropdown, status_text_db]
            )

            generate_prompt_btn.click(
                fn=get_prompt_content,
                inputs=[select_prompt_dropdown],
                outputs=[gr.State()]
            ).then(
                fn=launch_2d_generation,
                inputs=[gr.State(), width_slider_db, height_slider_db, num_images_slider_db, gr.Textbox(value="sparkassets", visible=False), gr.Textbox(value="db_image", visible=False)],
                outputs=[task_id_state]
            ).then(
                fn=track_2d_generation_progress,
                inputs=[task_id_state],
                outputs=[status_text_db, generated_image_gallery_db, gr.HTML()]
            )

        with gr.TabItem("3D Generation"):
            gr.Markdown("## Generate 3D Assets")
            gr.Markdown("This functionality is not yet available. Please use the other tabs to generate 2D images that can be used for 3D generation later.")

        with gr.TabItem("Decimated Generation"):
            gr.Markdown("## Generate Decimated Assets")
            gr.Markdown("This functionality is not yet available. Please use the other tabs to generate images that can be used for decimated asset generation later.")

        with gr.TabItem("Grid to Image"):
            gr.Markdown("## Batch Processing with SDXL Turbo")
            gr.Markdown("""
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
            
            with gr.Row():
                grid_db_name_input = gr.Textbox(label="Database Name", value="mock_db", interactive=True)
                grid_col_name_input = gr.Textbox(label="Collection Name", value="grids", interactive=True)
            
            fetch_grids_btn = gr.Button("Fetch Grids")
            
            with gr.Row():
                width_slider_grid = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                height_slider_grid = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
                num_images_slider_grid = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
            
            model_dropdown_grid = gr.Dropdown(
                label="Model", 
                choices=["SDXL Turbo (Optimized for grid-based generation)"], 
                value="SDXL Turbo (Optimized for grid-based generation)", 
                interactive=False
            )
            
            with gr.Row():
                select_grid_dropdown = gr.Dropdown(label="Select a Grid")
                generate_grid_btn = gr.Button("🚀 Generate with SDXL Turbo")

            status_text_grid = gr.Textbox(label="Status", lines=1)
            generated_image_gallery_grid = gr.Gallery(label="Generated Image (SDXL Turbo Output)", columns=2)
            
            # Linking components
            fetch_grids_btn.click(
                fn=fetch_grids,
                inputs=[grid_db_name_input, grid_col_name_input],
                outputs=[select_grid_dropdown, status_text_grid]
            )

            generate_grid_btn.click(
                fn=get_grid_content,
                inputs=[select_grid_dropdown],
                outputs=[gr.State()]
            ).then(
                fn=launch_grid_generation,
                inputs=[gr.State(), width_slider_grid, height_slider_grid, num_images_slider_grid, gr.Textbox(value="sparkassets", visible=False), gr.Textbox(value="db_grid", visible=False)],
                outputs=[task_id_state]
            ).then(
                fn=track_grid_generation_progress,
                inputs=[task_id_state],
                outputs=[status_text_grid, generated_image_gallery_grid]
            )
            
        with gr.TabItem("Task Management & Storage"):
            gr.Markdown("## Stored Files (Simulated Database)")
            gr.Markdown("This table shows a history of all successfully completed tasks.")
            
            refresh_button = gr.Button("Refresh Stored Files")
            
            stored_files_output = gr.Dataframe(
                headers=["Task ID", "Type", "Description", "Status", "Timestamp", "S3 Bucket", "Download Link"],
                row_count=10,
                col_count=(7, "fixed"),
                interactive=False
            )
            
            refresh_button.click(fn=get_stored_files, outputs=stored_files_output)

demo.launch(server_name="0.0.0.0", server_port=7860)

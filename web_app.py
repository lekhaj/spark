# web_app.py
import gradio as gr
from app import generate_2d_image_task, generate_image_from_grid_task, generate_3d_from_2d_task, decimate_3d_task, generate_2d_from_db_task, generate_grid_from_db_task
import json
import time
import io
from PIL import Image

def load_sample_grid():
    """Loads a predefined sample grid as a string."""
    sample_grid = """
[[0,0,1,1,0,0,2,2,0,0],
[0,1,1,1,1,0,2,2,2,0],
[1,1,1,1,1,1,0,2,2,2],
[1,1,1,1,1,1,0,0,2,2],
[0,1,1,1,0,0,0,0,0,0],
[0,0,0,0,0,0,0,0,0,0],
[3,3,3,3,3,3,3,3,3,3],
[3,3,3,3,3,3,3,3,3,3],
[4,4,4,4,0,0,0,0,0,0],
[4,4,4,4,0,0,0,0,0,0]]
    """
    return sample_grid.strip()

def launch_2d_generation(text_prompt, width, height, num_images, s3_bucket_name, base_filename):
    """Starts the 2D image generation task and returns the task ID."""
    task = generate_2d_image_task.delay(text_prompt, width, height, num_images, s3_bucket_name, base_filename)
    return task.id

def track_2d_generation_progress(task_id):
    """Tracks the status of the 2D image generation task and yields updates."""
    if not task_id:
        yield "Waiting for task to start...", [], None, "No Task ID"
        return
    
    while True:
        task = generate_2d_image_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            yield "Task is pending...", [], None, task_id
        elif task.state == 'PROGRESS':
            status = task.info.get('status', 'Processing...')
            results = task.info.get('result', [])
            yield status, results, None, task_id
        elif task.state == 'SUCCESS':
            status = task.info.get('status', "Task complete!")
            results = task.info.get('result', [])
            html_output = f"<h3>Generated Images:</h3>"
            for url in results:
                html_output += f"<a href='{url}' target='_blank'>Download Image</a><br>"
            yield status, results, html_output, task_id
            return
        elif task.state == 'FAILURE':
            error_msg = task.info.get('error', "An error occurred.")
            yield f"Error: {error_msg}", [], None, task_id
            return
        else:
            yield f"Task state: {task.state}", [], None, task_id
        
        time.sleep(2)

def launch_grid_generation(grid_data_str, width, height, num_images, s3_bucket_name, base_filename):
    """Starts the grid visualization task and returns the task ID."""
    task = generate_image_from_grid_task.delay(grid_data_str, width, height, num_images, s3_bucket_name, base_filename)
    return task.id

def track_grid_generation_progress(task_id):
    """Tracks the status of the grid generation task and yields updates."""
    if not task_id:
        yield "Waiting for task to start...", None, "No Task ID"
        return
        
    while True:
        task = generate_image_from_grid_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            yield "Task is pending...", None, task_id
        elif task.state == 'PROGRESS':
            status = task.info.get('status', 'Processing...')
            results = task.info.get('result', [])
            images = [url for url in results]
            yield status, images, task_id
        elif task.state == 'SUCCESS':
            status = task.info.get('status', "Task complete!")
            results = task.info.get('result', [])
            images = [url for url in results]
            yield status, images, task_id
            return
        elif task.state == 'FAILURE':
            error_msg = task.info.get('error', "An error occurred.")
            yield f"Error: {error_msg}", None, task_id
            return
        else:
            yield f"Task state: {task.state}", None, task_id

        time.sleep(2)

def launch_3d_generation(image_2d_input, s3_bucket_name, base_filename):
    """Starts the 3D model generation task."""
    if image_2d_input is None:
        return None, "No image uploaded.", None
    
    image_bytes = io.BytesIO()
    image_2d_input.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    
    task = generate_3d_from_2d_task.delay(image_bytes.getvalue(), s3_bucket_name, base_filename)
    return task.id, "Task started...", None

def track_3d_generation_progress(task_id):
    """Tracks the status of the 3D model generation task and yields updates."""
    if not task_id:
        yield "Waiting for task to start...", None, "No Task ID"
        return
        
    while True:
        task = generate_3d_from_2d_task.AsyncResult(task_id)
        
        if task.state == 'PENDING':
            yield "Task is pending...", None, task_id
        elif task.state == 'PROGRESS':
            status = task.info.get('status', 'Processing...')
            yield status, None, task_id
        elif task.state == 'SUCCESS':
            status = task.info.get('status', "Task complete!")
            url = task.info.get('result', None)
            yield status, gr.HTML(f"<a href='{url}' target='_blank'>Download 3D Model</a>"), task_id
            return
        elif task.state == 'FAILURE':
            error_msg = task.info.get('error', "An error occurred.")
            yield f"Error: {error_msg}", None, task_id
            return
        else:
            yield f"Task state: {task.state}", None, task_id

        time.sleep(2)
        
def launch_decimation_task(input_3d_file, s3_bucket_name, base_filename):
    """Starts the 3D decimation task."""
    if input_3d_file is None:
        return None, "No file uploaded.", None
    
    with open(input_3d_file.name, 'rb') as f:
        file_bytes = f.read()

    task = decimate_3d_task.delay(file_bytes, s3_bucket_name, base_filename)
    return task.id, "Task started...", None

def track_decimation_progress(task_id):
    """Tracks the status of the 3D decimation task and yields updates."""
    if not task_id:
        yield "Waiting for task to start...", None, "No Task ID"
        return

    while True:
        task = decimate_3d_task.AsyncResult(task_id)

        if task.state == 'PENDING':
            yield "Task is pending...", None, task_id
        elif task.state == 'PROGRESS':
            status = task.info.get('status', 'Processing...')
            yield status, None, task_id
        elif task.state == 'SUCCESS':
            status = task.info.get('status', "Task complete!")
            url = task.info.get('result', None)
            yield status, gr.HTML(f"<a href='{url}' target='_blank'>Download Decimated 3D Model</a>"), task_id
            return
        elif task.state == 'FAILURE':
            error_msg = task.info.get('error', "An error occurred.")
            yield f"Error: {error_msg}", None, task_id
            return
        else:
            yield f"Task state: {task.state}", None, task_id
            
        time.sleep(2)

def fetch_prompts_from_db(db_name, collection_name):
    """Fetches text prompts from MongoDB."""
    # This is a placeholder. You would implement the actual MongoDB connection and fetch logic here.
    # For now, it will return a sample list.
    print(f"Fetching prompts from database: {db_name}, collection: {collection_name}")
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        collection = db[collection_name]
        
        # Fetch all documents and get the 'prompt' field
        prompts = [doc.get('prompt') for doc in collection.find({}) if doc.get('prompt')]
        return prompts
    except Exception as e:
        print(f"Error fetching prompts: {e}")
        return gr.Dropdown.update(choices=["Error fetching prompts"], value="Error fetching prompts")

def fetch_grids_from_db(db_name, collection_name):
    """Fetches grids from MongoDB."""
    # This is a placeholder. You would implement the actual MongoDB connection and fetch logic here.
    # For now, it will return a sample list.
    print(f"Fetching grids from database: {db_name}, collection: {collection_name}")
    try:
        from pymongo import MongoClient
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        collection = db[collection_name]
        
        # Fetch all documents and get the 'grid' field
        grids = [json.dumps(doc.get('grid')) for doc in collection.find({}) if doc.get('grid')]
        return grids
    except Exception as e:
        print(f"Error fetching grids: {e}")
        return gr.Dropdown.update(choices=["Error fetching grids"], value="Error fetching grids")

def launch_2d_db_generation(prompt, width, height, num_images, s3_bucket_name, base_filename):
    """Starts 2D image generation from a database prompt."""
    if not prompt:
        return None, "Please select a prompt.", None
    task = generate_2d_from_db_task.delay(prompt, width, height, num_images, s3_bucket_name, base_filename)
    return task.id, "Task started...", None

def launch_grid_db_generation(grid_data_str, width, height, num_images, s3_bucket_name, base_filename):
    """Starts grid image generation from a database grid."""
    if not grid_data_str:
        return None, "Please select a grid.", None
    task = generate_grid_from_db_task.delay(grid_data_str, width, height, num_images, s3_bucket_name, base_filename)
    return task.id, "Task started...", None

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# AI-Powered 3D Asset Generator")
    gr.Markdown("This application uses Celery to run generation tasks in the background, keeping the Gradio app responsive. The generated assets are uploaded to S3.")

    s3_bucket_input_global = gr.Textbox(label="S3 Bucket Name", value="sparkassets", interactive=True)
    
    # State variables to store task IDs
    task_id_state = gr.State(None)
    task_id_states = {
        "txt2img": gr.State(None),
        "grid2img": gr.State(None),
        "3dgen": gr.State(None),
        "decimate": gr.State(None),
        "db_txt2img": gr.State(None),
        "db_grid2img": gr.State(None)
    }

    with gr.Tabs():
        with gr.TabItem("Text to Image"):
            with gr.Tabs():
                with gr.TabItem("Generator"):
                    gr.Markdown("## Text-to-Image Generation")
                    gr.Markdown("Generate images from text descriptions. **All prompts are automatically optimized for 3D asset generation**.")
                    
                    with gr.Row():
                        gr.Markdown("### 🎯 3D Generation Optimization")
                        gr.Checkbox(label="Enabled", value=True, interactive=False) 

                    text_to_image_prompt = gr.Textbox(
                        label="Text Prompt", 
                        placeholder="💡 Tip: Describe objects clearly for best 3D generation results.",
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
                    
                    generate_image_button = gr.Button("🚀 Generate Image from Text")
                    image_generation_status = gr.Textbox(label="Image Generation Status", lines=1)
                    image_generation_output = gr.Gallery(label="Generated Images", columns=2, height='auto')
                    image_generation_link = gr.HTML(label="Download Links")
                    
                    generate_image_button.click(
                        fn=launch_2d_generation,
                        inputs=[text_to_image_prompt, width_slider_txt2img, height_slider_txt2img, num_images_slider_txt2img, s3_bucket_input_global, base_filename_txt2img],
                        outputs=[task_id_states["txt2img"]]
                    ).then(
                        fn=track_2d_generation_progress,
                        inputs=[task_id_states["txt2img"]],
                        outputs=[image_generation_status, image_generation_output, image_generation_link, gr.State(None)]
                    )
                with gr.TabItem("Task Status"):
                    task_id_display_txt2img = gr.Textbox(label="Submitted Task ID", interactive=False)
                    task_id_states["txt2img"].change(fn=lambda x: x, inputs=task_id_states["txt2img"], outputs=task_id_display_txt2img)

        with gr.TabItem("Grid to Image"):
            with gr.Tabs():
                with gr.TabItem("Generator"):
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
                    grid_generation_status = gr.Textbox(label="Status", lines=1)
                    grid_visualization_output = gr.Gallery(label="Grid Visualization", columns=2, height='auto')
                    
                    load_sample_grid_button.click(
                        fn=load_sample_grid,
                        inputs=[],
                        outputs=[grid_data_input]
                    )

                    generate_grid_image_button.click(
                        fn=launch_grid_generation,
                        inputs=[grid_data_input, width_slider_grid2img, height_slider_grid2img, num_images_slider_grid2img, s3_bucket_input_global, base_filename_grid2img],
                        outputs=[task_id_states["grid2img"]]
                    ).then(
                        fn=track_grid_generation_progress,
                        inputs=[task_id_states["grid2img"]],
                        outputs=[grid_generation_status, grid_visualization_output, gr.State(None)]
                    )
                with gr.TabItem("Task Status"):
                    task_id_display_grid2img = gr.Textbox(label="Submitted Task ID", interactive=False)
                    task_id_states["grid2img"].change(fn=lambda x: x, inputs=task_id_states["grid2img"], outputs=task_id_display_grid2img)

        with gr.TabItem("3D Generation"):
            with gr.Tabs():
                with gr.TabItem("Generator"):
                    gr.Markdown("## 3D Model Generation from 2D Image")
                    gr.Markdown("Upload a 2D image to generate a 3D GLB model.")
                    
                    input_2d_image_for_3d = gr.Image(label="Upload 2D Image", type="pil")
                    base_filename_3d_gen = gr.Textbox(label="Base Filename for 3D Model (e.g., my_3d_asset)")
                    
                    generate_3d_button = gr.Button("Generate 3D Model")
                    status_3d_gen = gr.Textbox(label="3D Generation Status", lines=1)
                    output_3d_model_link = gr.HTML(label="Generated 3D Model Link")

                    generate_3d_button.click(
                        fn=launch_3d_generation,
                        inputs=[input_2d_image_for_3d, s3_bucket_input_global, base_filename_3d_gen],
                        outputs=[task_id_states["3dgen"], status_3d_gen, output_3d_model_link]
                    ).then(
                        fn=track_3d_generation_progress,
                        inputs=[task_id_states["3dgen"]],
                        outputs=[status_3d_gen, output_3d_model_link, gr.State(None)]
                    )
                with gr.TabItem("Task Status"):
                    task_id_display_3dgen = gr.Textbox(label="Submitted Task ID", interactive=False)
                    task_id_states["3dgen"].change(fn=lambda x: x, inputs=task_id_states["3dgen"], outputs=task_id_display_3dgen)

        with gr.TabItem("Decimated 3D"):
            with gr.Tabs():
                with gr.TabItem("Decimator"):
                    gr.Markdown("## Decimate 3D Model")
                    gr.Markdown("Upload an existing 3D GLB/OBJ/STL model to reduce its polygon count.")
                    
                    input_3d_file_decimate = gr.File(label="Upload 3D Model (GLB, OBJ, STL)", type="filepath")
                    base_filename_decimate = gr.Textbox(label="Base Filename for Decimated Model (e.g., my_decimated_asset)")
                    
                    decimate_button = gr.Button("Decimate 3D Model")
                    status_decimate = gr.Textbox(label="Decimation Status", lines=1)
                    output_decimated_model_link = gr.HTML(label="Decimated 3D Model Link")

                    decimate_button.click(
                        fn=launch_decimation_task,
                        inputs=[input_3d_file_decimate, s3_bucket_input_global, base_filename_decimate],
                        outputs=[task_id_states["decimate"], status_decimate, output_decimated_model_link]
                    ).then(
                        fn=track_decimation_progress,
                        inputs=[task_id_states["decimate"]],
                        outputs=[status_decimate, output_decimated_model_link, gr.State(None)]
                    )
                with gr.TabItem("Task Status"):
                    task_id_display_decimate = gr.Textbox(label="Submitted Task ID", interactive=False)
                    task_id_states["decimate"].change(fn=lambda x: x, inputs=task_id_states["decimate"], outputs=task_id_display_decimate)
        
        with gr.TabItem("MongoDB"):
            with gr.Tabs():
                with gr.TabItem("Text Prompts"):
                    gr.Markdown("## Generate Images from MongoDB Prompts with SDXL Turbo")
                    gr.Markdown("""
                    🚀 **SDXL Turbo Integration**: Ultra-fast, high-quality local image generation optimized for 3D assets!
                    🎯 **All prompts are automatically enhanced for 3D asset generation** - perfect for creating images that will work well with the 3D Generation tab!
                    """)
                    
                    with gr.Row():
                        db_name_prompts = gr.Textbox(label="Database Name", placeholder="e.g., my_prompts_db")
                        collection_name_prompts = gr.Textbox(label="Collection Name", placeholder="e.g., prompts")
                        fetch_prompts_button = gr.Button("Fetch Prompts")

                    with gr.Accordion("🎯 SDXL Turbo Features:", open=False):
                        gr.Markdown("""
                        - ⚡ **Ultra-fast**: 2-4 inference steps vs 20-50 for other models
                        - 🏠 **Local processing**: No API costs, completely private
                        - 🎨 **3D-optimized**: Clean backgrounds, perfect lighting for 3D generation
                        """)

                    with gr.Row():
                        width_slider_db_txt2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                        height_slider_db_txt2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")

                    with gr.Row():
                        num_images_slider_db_txt2img = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        model_dropdown_db_txt2img = gr.Dropdown(
                            label="Model",
                            choices=["SDXL Turbo (Local GPU-accelerated generation)"],
                            value="SDXL Turbo (Local GPU-accelerated generation)",
                            interactive=False
                        )
                    
                    prompt_dropdown = gr.Dropdown(label="Select a Prompt", choices=[], interactive=True)
                    generate_db_image_button = gr.Button("🚀 Generate with SDXL Turbo")
                    
                    db_image_status = gr.Textbox(label="Status")
                    db_image_output = gr.Gallery(label="Generated Image (SDXL Turbo Output)")
                    db_image_gen_status = gr.Textbox(label="Generation Status")

                    fetch_prompts_button.click(
                        fn=fetch_prompts_from_db,
                        inputs=[db_name_prompts, collection_name_prompts],
                        outputs=[prompt_dropdown]
                    )

                    generate_db_image_button.click(
                        fn=launch_2d_db_generation,
                        inputs=[prompt_dropdown, width_slider_db_txt2img, height_slider_db_txt2img, num_images_slider_db_txt2img, s3_bucket_input_global, gr.Textbox(value="db_2d_image", visible=False)],
                        outputs=[task_id_states["db_txt2img"], db_image_status, db_image_output]
                    ).then(
                        fn=track_2d_generation_progress,
                        inputs=[task_id_states["db_txt2img"]],
                        outputs=[db_image_gen_status, db_image_output, gr.State(None)]
                    )

                with gr.TabItem("Grid Data"):
                    gr.Markdown("## Grid Data")
                    
                    with gr.Row():
                        db_name_grids = gr.Textbox(label="Database Name", placeholder="e.g., my_grids_db")
                        collection_name_grids = gr.Textbox(label="Collection Name", placeholder="e.g., grids")
                        fetch_grids_button = gr.Button("Fetch Grids")

                    with gr.Row():
                        width_slider_db_grid2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                        height_slider_db_grid2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")

                    with gr.Row():
                        num_images_slider_db_grid2img = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")
                        model_dropdown_db_grid2img = gr.Dropdown(
                            label="Model",
                            choices=["SDXL Turbo (Optimized for grid-based generation)"],
                            value="SDXL Turbo (Optimized for grid-based generation)",
                            interactive=False
                        )
                    
                    grid_dropdown = gr.Dropdown(label="Select a Grid", choices=[], interactive=True)
                    generate_db_grid_button = gr.Button("🚀 Generate with SDXL Turbo")

                    db_grid_status = gr.Textbox(label="Status")
                    db_grid_output = gr.Gallery(label="Generated Image (SDXL Turbo Output)")
                    grid_visualization_output_db = gr.Gallery(label="Grid Visualization")
                    db_grid_gen_status = gr.Textbox(label="Generation Status")

                    fetch_grids_button.click(
                        fn=fetch_grids_from_db,
                        inputs=[db_name_grids, collection_name_grids],
                        outputs=[grid_dropdown]
                    )

                    generate_db_grid_button.click(
                        fn=launch_grid_db_generation,
                        inputs=[grid_dropdown, width_slider_db_grid2img, height_slider_db_grid2img, num_images_slider_db_grid2img, s3_bucket_input_global, gr.Textbox(value="db_grid_image", visible=False)],
                        outputs=[task_id_states["db_grid2img"], db_grid_status, db_grid_output]
                    ).then(
                        fn=track_grid_generation_progress,
                        inputs=[task_id_states["db_grid2img"]],
                        outputs=[db_grid_gen_status, db_grid_output, gr.State(None)]
                    )
                
                with gr.TabItem("Task Status"):
                    task_id_display_db_txt2img = gr.Textbox(label="Text Prompts Task ID", interactive=False)
                    task_id_display_db_grid2img = gr.Textbox(label="Grid Data Task ID", interactive=False)
                    task_id_states["db_txt2img"].change(fn=lambda x: x, inputs=task_id_states["db_txt2img"], outputs=task_id_display_db_txt2img)
                    task_id_states["db_grid2img"].change(fn=lambda x: x, inputs=task_id_states["db_grid2img"], outputs=task_id_display_db_grid2img)

    demo.launch(server_name="0.0.0.0", server_port=7860)


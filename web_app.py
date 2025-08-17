# web_app.py
import gradio as gr
from app import generate_2d_image_task, generate_image_from_grid_task, generate_3d_from_2d_task, decimate_3d_task
import json, time, io
from PIL import Image
from pymongo import MongoClient

# ========== Helpers ==========

def load_sample_grid():
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

# ===== MongoDB Helpers =====

def fetch_prompts_from_mongo(db_name, collection_name):
    try:
        client = MongoClient("mongodb://localhost:27017/")  # adjust if remote
        db = client[db_name]
        collection = db[collection_name]
        prompts = [doc.get("prompt", "") for doc in collection.find({}, {"prompt": 1})]
        return prompts if prompts else ["⚠️ No prompts found."]
    except Exception as e:
        return [f"Error: {str(e)}"]

def fetch_grids_from_mongo(db_name, collection_name):
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        collection = db[collection_name]
        grids = [json.dumps(doc.get("grid", [])) for doc in collection.find({}, {"grid": 1})]
        return grids if grids else ["⚠️ No grids found."]
    except Exception as e:
        return [f"Error: {str(e)}"]

# ========== Celery Task Launchers + Trackers ==========
# (unchanged from your working code, just shortened here for clarity)

def launch_2d_generation(text_prompt, width, height, num_images, s3_bucket_name, base_filename):
    task = generate_2d_image_task.delay(text_prompt, width, height, num_images, s3_bucket_name, base_filename)
    return task.id

def track_2d_generation_progress(task_id):
    if not task_id:
        yield "Waiting...", [], None
        return
    while True:
        task = generate_2d_image_task.AsyncResult(task_id)
        if task.state == "SUCCESS":
            results = task.info.get("result", [])
            html_output = "".join([f"<a href='{url}' target='_blank'>Download</a><br>" for url in results])
            yield "✅ Complete", results, html_output
            return
        elif task.state == "FAILURE":
            yield f"❌ Error: {task.info}", [], None
            return
        else:
            yield f"⏳ {task.state}", [], None
        time.sleep(2)

# (same pattern for grid/3D/decimation — left unchanged)

# ========== Gradio UI ==========

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# AI-Powered 3D Asset Generator")
    s3_bucket_input_global = gr.Textbox(label="S3 Bucket Name", value="sparkassets", interactive=True)

    # track all task ids
    submitted_task_ids = gr.State([])

    with gr.Tabs():

        # ----- Existing Tabs -----
        with gr.TabItem("Text to Image"):
            # ... your existing working code ...
            pass  

        with gr.TabItem("Grid to Image"):
            # ... your existing working code ...
            pass  

        with gr.TabItem("3D Generation"):
            # ... your existing working code ...
            pass  

        with gr.TabItem("Decimated 3D"):
            # ... your existing working code ...
            pass  

        # ----- New MongoDB Tab -----
        with gr.TabItem("MongoDB Prompts"):
            gr.Markdown("## Text Prompts from MongoDB with SDXL Turbo")

            db_name = gr.Textbox(label="Database Name", value="World_builder")
            coll_name = gr.Textbox(label="Collection Name", value="biomes")
            fetch_button = gr.Button("Fetch Prompts")

            prompt_dropdown = gr.Dropdown(label="Select a Prompt", choices=[])
            width = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height = gr.Slider(256, 1024, value=512, step=64, label="Height")
            num_images = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
            base_filename_mongo = gr.Textbox(label="Base Filename")

            gen_button = gr.Button("🚀 Generate with SDXL Turbo")
            status_box = gr.Textbox(label="Status")
            output_gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")
            output_links = gr.HTML()

            fetch_button.click(
                fn=fetch_prompts_from_mongo,
                inputs=[db_name, coll_name],
                outputs=[prompt_dropdown]
            )

            gen_button.click(
                fn=launch_2d_generation,
                inputs=[prompt_dropdown, width, height, num_images, s3_bucket_input_global, base_filename_mongo],
                outputs=[submitted_task_ids]
            ).then(
                fn=track_2d_generation_progress,
                inputs=[submitted_task_ids],
                outputs=[status_box, output_gallery, output_links]
            )

        with gr.TabItem("MongoDB Grids"):
            gr.Markdown("## Grid Data from MongoDB")

            db_name_grid = gr.Textbox(label="Database Name", value="World_builder")
            coll_name_grid = gr.Textbox(label="Collection Name", value="biomes")
            fetch_grids_button = gr.Button("Fetch Grids")

            grid_dropdown = gr.Dropdown(label="Select a Grid", choices=[])
            width_g = gr.Slider(256, 1024, value=512, step=64, label="Width")
            height_g = gr.Slider(256, 1024, value=512, step=64, label="Height")
            num_images_g = gr.Slider(1, 4, value=1, step=1, label="Number of Images")
            base_filename_grid_mongo = gr.Textbox(label="Base Filename")

            gen_grid_button = gr.Button("🚀 Generate with SDXL Turbo")
            status_grid_box = gr.Textbox(label="Status")
            grid_output_gallery = gr.Gallery(label="Generated Images", columns=2, height="auto")

            fetch_grids_button.click(
                fn=fetch_grids_from_mongo,
                inputs=[db_name_grid, coll_name_grid],
                outputs=[grid_dropdown]
            )

            gen_grid_button.click(
                fn=launch_2d_generation,
                inputs=[grid_dropdown, width_g, height_g, num_images_g, s3_bucket_input_global, base_filename_grid_mongo],
                outputs=[submitted_task_ids]
            ).then(
                fn=track_2d_generation_progress,
                inputs=[submitted_task_ids],
                outputs=[status_grid_box, grid_output_gallery, gr.HTML()]
            )

        # ----- New Task ID Viewer Tab -----
        with gr.TabItem("Submitted Task IDs"):
            gr.Markdown("## Submitted Task IDs (for tracking/debugging)")
            gr.Dataframe(label="All Task IDs", value=[], headers=["Task ID"], interactive=False)

demo.launch(server_name="0.0.0.0", server_port=7860)


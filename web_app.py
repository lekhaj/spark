# web_app.py
import gradio as gr
from app import generate_2d_image_task, generate_image_from_grid_task, generate_3d_from_2d_task, decimate_3d_task
import json
import time
import io
from PIL import Image
from datetime import datetime
from pymongo import MongoClient
import pandas as pd

# ================== MongoDB Setup ==================
MONGO_URI = "mongodb://localhost:27017"   # change if needed
DB_NAME = "spark_ai"
COLLECTION_NAME = "tasks"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def save_to_mongo(task_id, task_type, prompt, base_filename):
    """Save task details to MongoDB."""
    doc = {
        "task_id": task_id,
        "task_type": task_type,
        "prompt": prompt,
        "base_filename": base_filename,
        "timestamp": datetime.utcnow()
    }
    collection.insert_one(doc)

def fetch_from_mongo(collection_name):
    """Fetch all prompts from MongoDB for a given collection."""
    coll = db[collection_name]
    records = list(coll.find({}, {"_id": 0}))
    if not records:
        return pd.DataFrame([{"message": "No records found"}])
    return pd.DataFrame(records)

# ================== Utility ==================
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

# ================== Task Launchers ==================
def launch_2d_generation(text_prompt, width, height, num_images, s3_bucket_name, base_filename):
    task = generate_2d_image_task.delay(text_prompt, width, height, num_images, s3_bucket_name, base_filename)
    save_to_mongo(task.id, "text_to_image", text_prompt, base_filename)
    return task.id

def launch_grid_generation(grid_data_str, width, height, num_images, s3_bucket_name, base_filename):
    task = generate_image_from_grid_task.delay(grid_data_str, width, height, num_images, s3_bucket_name, base_filename)
    save_to_mongo(task.id, "grid_to_image", grid_data_str, base_filename)
    return task.id

def launch_3d_generation(image_2d_input, s3_bucket_name, base_filename):
    if image_2d_input is None:
        return None
    image_bytes = io.BytesIO()
    image_2d_input.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    task = generate_3d_from_2d_task.delay(image_bytes.getvalue(), s3_bucket_name, base_filename)
    save_to_mongo(task.id, "image_to_3d", "uploaded_image", base_filename)
    return task.id

def launch_decimation_task(input_3d_file, s3_bucket_name, base_filename):
    if input_3d_file is None:
        return None
    with open(input_3d_file.name, 'rb') as f:
        file_bytes = f.read()
    task = decimate_3d_task.delay(file_bytes, s3_bucket_name, base_filename)
    save_to_mongo(task.id, "decimate_3d", "uploaded_3d_file", base_filename)
    return task.id

# ================== Trackers (unchanged except outputs) ==================
def track_2d_generation_progress(task_id):
    if not task_id:
        yield "Waiting for task to start...", [], None
        return
    while True:
        task = generate_2d_image_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            yield "Task is pending...", [], None
        elif task.state == 'PROGRESS':
            yield task.info.get('status', 'Processing...'), task.info.get('result', []), None
        elif task.state == 'SUCCESS':
            results = task.info.get('result', [])
            html_output = "<h3>Generated Images:</h3>" + "".join([f"<a href='{url}' target='_blank'>Download</a><br>" for url in results])
            yield "Task complete!", results, html_output
            return
        elif task.state == 'FAILURE':
            yield f"Error: {task.info.get('error', 'Unknown error')}", [], None
            return
        time.sleep(2)

def track_grid_generation_progress(task_id):
    if not task_id:
        yield "Waiting for task to start...", None
        return
    while True:
        task = generate_image_from_grid_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            yield "Task is pending...", None
        elif task.state == 'PROGRESS':
            yield task.info.get('status', 'Processing...'), task.info.get('result', [])
        elif task.state == 'SUCCESS':
            yield "Task complete!", task.info.get('result', [])
            return
        elif task.state == 'FAILURE':
            yield f"Error: {task.info.get('error', 'Unknown error')}", None
            return
        time.sleep(2)

def track_3d_generation_progress(task_id):
    if not task_id:
        yield "Waiting for task to start...", None
        return
    while True:
        task = generate_3d_from_2d_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            yield "Task is pending...", None
        elif task.state == 'PROGRESS':
            yield task.info.get('status', 'Processing...'), None
        elif task.state == 'SUCCESS':
            url = task.info.get('result', None)
            yield "Task complete!", gr.HTML(f"<a href='{url}' target='_blank'>Download 3D Model</a>")
            return
        elif task.state == 'FAILURE':
            yield f"Error: {task.info.get('error', 'Unknown error')}", None
            return
        time.sleep(2)

def track_decimation_progress(task_id):
    if not task_id:
        yield "Waiting for task to start...", None
        return
    while True:
        task = decimate_3d_task.AsyncResult(task_id)
        if task.state == 'PENDING':
            yield "Task is pending...", None
        elif task.state == 'PROGRESS':
            yield task.info.get('status', 'Processing...'), None
        elif task.state == 'SUCCESS':
            url = task.info.get('result', None)
            yield "Task complete!", gr.HTML(f"<a href='{url}' target='_blank'>Download Decimated 3D</a>")
            return
        elif task.state == 'FAILURE':
            yield f"Error: {task.info.get('error', 'Unknown error')}", None
            return
        time.sleep(2)

# ================== Gradio UI ==================
with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# AI-Powered 3D Asset Generator")
    s3_bucket_input_global = gr.Textbox(label="S3 Bucket Name", value="sparkassets", interactive=True)
    task_id_state = gr.State(None)

    with gr.Tabs():
        # -------- Text-to-Image --------
        with gr.TabItem("Text to Image"):
            text_to_image_prompt = gr.Textbox(label="Text Prompt", lines=3)
            base_filename_txt2img = gr.Textbox(label="Base Filename")
            width_slider = gr.Slider(256, 1024, 512, 64, label="Width")
            height_slider = gr.Slider(256, 1024, 512, 64, label="Height")
            num_images_slider = gr.Slider(1, 4, 1, 1, label="Num Images")
            generate_button = gr.Button("🚀 Generate Image from Text")
            task_id_box_txt2img = gr.Textbox(label="Task ID", interactive=False)
            status_box = gr.Textbox(label="Status", lines=1)
            gallery = gr.Gallery(columns=2)
            link_box = gr.HTML()

            generate_button.click(
                fn=launch_2d_generation,
                inputs=[text_to_image_prompt, width_slider, height_slider, num_images_slider, s3_bucket_input_global, base_filename_txt2img],
                outputs=[task_id_box_txt2img]
            ).then(
                fn=track_2d_generation_progress,
                inputs=[task_id_box_txt2img],
                outputs=[status_box, gallery, link_box]
            )

        # -------- Grid-to-Image --------
        with gr.TabItem("Grid to Image"):
            grid_data_input = gr.Textbox(label="Grid JSON", lines=10)
            load_sample = gr.Button("Load Sample Grid")
            base_filename_grid = gr.Textbox(label="Base Filename")
            width_slider_g = gr.Slider(256, 1024, 512, 64, label="Width")
            height_slider_g = gr.Slider(256, 1024, 512, 64, label="Height")
            num_images_slider_g = gr.Slider(1, 4, 1, 1, label="Num Images")
            generate_grid_button = gr.Button("Generate Image from Grid")
            task_id_box_grid = gr.Textbox(label="Task ID", interactive=False)
            grid_status = gr.Textbox(label="Status", lines=1)
            grid_gallery = gr.Gallery(columns=2)

            load_sample.click(load_sample_grid, [], [grid_data_input])
            generate_grid_button.click(
                fn=launch_grid_generation,
                inputs=[grid_data_input, width_slider_g, height_slider_g, num_images_slider_g, s3_bucket_input_global, base_filename_grid],
                outputs=[task_id_box_grid]
            ).then(
                fn=track_grid_generation_progress,
                inputs=[task_id_box_grid],
                outputs=[grid_status, grid_gallery]
            )

        # -------- 3D Generation --------
        with gr.TabItem("3D Generation"):
            input_2d_image = gr.Image(type="pil")
            base_filename_3d = gr.Textbox(label="Base Filename")
            gen3d_button = gr.Button("Generate 3D")
            task_id_box_3d = gr.Textbox(label="Task ID", interactive=False)
            status_3d = gr.Textbox(label="Status")
            model_link = gr.HTML()

            gen3d_button.click(
                fn=launch_3d_generation,
                inputs=[input_2d_image, s3_bucket_input_global, base_filename_3d],
                outputs=[task_id_box_3d]
            ).then(
                fn=track_3d_generation_progress,
                inputs=[task_id_box_3d],
                outputs=[status_3d, model_link]
            )

        # -------- Decimation --------
        with gr.TabItem("Decimated 3D"):
            input_3d_file = gr.File(type="filepath")
            base_filename_dec = gr.Textbox(label="Base Filename")
            dec_button = gr.Button("Decimate 3D")
            task_id_box_dec = gr.Textbox(label="Task ID", interactive=False)
            status_dec = gr.Textbox(label="Status")
            dec_link = gr.HTML()

            dec_button.click(
                fn=launch_decimation_task,
                inputs=[input_3d_file, s3_bucket_input_global, base_filename_dec],
                outputs=[task_id_box_dec]
            ).then(
                fn=track_decimation_progress,
                inputs=[task_id_box_dec],
                outputs=[status_dec, dec_link]
            )

        # -------- MongoDB Viewer --------
        with gr.TabItem("MongoDB Viewer"):
            collection_input = gr.Textbox(label="Collection Name", value=COLLECTION_NAME)
            fetch_button = gr.Button("Fetch Records")
            records_output = gr.Dataframe(headers=["task_id", "task_type", "prompt", "base_filename", "timestamp"])

            fetch_button.click(
                fn=fetch_from_mongo,
                inputs=[collection_input],
                outputs=[records_output]
            )

demo.launch(server_name="0.0.0.0", server_port=7860)


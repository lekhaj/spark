# web_app.py
import gradio as gr
from app import generate_2d_image_task, generate_image_from_grid_task, generate_3d_from_2d_task, decimate_3d_task
import json
import time
import io
from PIL import Image
from pymongo import MongoClient

# ------------------ MongoDB Setup ------------------
def fetch_prompts_from_mongo(db_name, collection_name):
    try:
        client = MongoClient("mongodb://localhost:27017/")  # Change if needed
        db = client[db_name]
        collection = db[collection_name]
        prompts = [doc.get("prompt", "") for doc in collection.find({}, {"prompt": 1})]
        return prompts if prompts else ["⚠️ No prompts found in collection"]
    except Exception as e:
        return [f"Error: {str(e)}"]

def fetch_grids_from_mongo(db_name, collection_name):
    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client[db_name]
        collection = db[collection_name]
        grids = [json.dumps(doc.get("grid", [])) for doc in collection.find({}, {"grid": 1})]
        return grids if grids else ["⚠️ No grids found in collection"]
    except Exception as e:
        return [f"Error: {str(e)}"]

# ------------------ Helper Functions ------------------
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

# ------------------ Task Launchers & Trackers ------------------
# (same as your original code, kept unchanged for Celery background tasks)

# ------------------ Gradio App ------------------
with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# AI-Powered 3D Asset Generator")
    gr.Markdown("This application uses Celery to run generation tasks in the background, keeping the Gradio app responsive. The generated assets are uploaded to S3.")

    s3_bucket_input_global = gr.Textbox(label="S3 Bucket Name", value="sparkassets", interactive=True)
    task_id_state = gr.State(None)

    with gr.Tabs():
        # ---------------- Text-to-Image ----------------
        with gr.TabItem("Text to Image"):
            gr.Markdown("## Text-to-Image Generation")
            
            # MongoDB Inputs
            mongo_db_name = gr.Textbox(label="Database Name", value="World_builder")
            mongo_collection_name = gr.Textbox(label="Collection Name", value="biomes")
            fetch_prompts_button = gr.Button("Fetch Prompts")
            mongo_prompt_dropdown = gr.Dropdown(label="Select a Prompt", choices=[], interactive=True)

            fetch_prompts_button.click(
                fn=fetch_prompts_from_mongo,
                inputs=[mongo_db_name, mongo_collection_name],
                outputs=[mongo_prompt_dropdown]
            )

            text_to_image_prompt = gr.Textbox(label="Or Enter Custom Prompt", lines=3)
            base_filename_txt2img = gr.Textbox(label="Base Filename for Image(s)")

            with gr.Row():
                width_slider_txt2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
                height_slider_txt2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")

            num_images_slider_txt2img = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")

            generate_image_button = gr.Button("🚀 Generate Image from Text")
            image_generation_status = gr.Textbox(label="Status")
            image_generation_output = gr.Gallery(label="Generated Images", columns=2)
            image_generation_link = gr.HTML(label="Download Links")

            # Use dropdown OR textbox for prompt
            def choose_prompt(selected, custom):
                return custom if custom else selected

            generate_image_button.click(
                fn=choose_prompt,
                inputs=[mongo_prompt_dropdown, text_to_image_prompt],
                outputs=[text_to_image_prompt]
            ).then(
                fn=launch_2d_generation,
                inputs=[text_to_image_prompt, width_slider_txt2img, height_slider_txt2img, num_images_slider_txt2img, s3_bucket_input_global, base_filename_txt2img],
                outputs=[task_id_state]
            ).then(
                fn=track_2d_generation_progress,
                inputs=[task_id_state],
                outputs=[image_generation_status, image_generation_output, image_generation_link]
            )

        # ---------------- Grid-to-Image ----------------
        with gr.TabItem("Grid to Image"):
            gr.Markdown("## Grid to Image Visualization")
            
            mongo_db_grid = gr.Textbox(label="Database Name", value="World_builder")
            mongo_collection_grid = gr.Textbox(label="Collection Name", value="biomes")
            fetch_grids_button = gr.Button("Fetch Grids")
            mongo_grid_dropdown = gr.Dropdown(label="Select a Grid", choices=[], interactive=True)

            fetch_grids_button.click(
                fn=fetch_grids_from_mongo,
                inputs=[mongo_db_grid, mongo_collection_grid],
                outputs=[mongo_grid_dropdown]
            )

            grid_data_input = gr.Textbox(label="Or Enter Grid Data (JSON)", lines=10)
            load_sample_grid_button = gr.Button("Load Sample Grid")
            load_sample_grid_button.click(fn=load_sample_grid, inputs=[], outputs=[grid_data_input])

            base_filename_grid2img = gr.Textbox(label="Base Filename")
            width_slider_grid2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Width")
            height_slider_grid2img = gr.Slider(minimum=256, maximum=1024, value=512, step=64, label="Height")
            num_images_slider_grid2img = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Number of Images")

            generate_grid_image_button = gr.Button("🚀 Generate Image from Grid")
            grid_generation_status = gr.Textbox(label="Status")
            grid_visualization_output = gr.Gallery(label="Grid Visualization", columns=2)

            def choose_grid(selected, custom):
                return custom if custom else selected

            generate_grid_image_button.click(
                fn=choose_grid,
                inputs=[mongo_grid_dropdown, grid_data_input],
                outputs=[grid_data_input]
            ).then(
                fn=launch_grid_generation,
                inputs=[grid_data_input, width_slider_grid2img, height_slider_grid2img, num_images_slider_grid2img, s3_bucket_input_global, base_filename_grid2img],
                outputs=[task_id_state]
            ).then(
                fn=track_grid_generation_progress,
                inputs=[task_id_state],
                outputs=[grid_generation_status, grid_visualization_output]
            )

        # ---------------- 3D Generation ----------------
        with gr.TabItem("3D Generation"):
            input_2d_image_for_3d = gr.Image(label="Upload 2D Image", type="pil")
            base_filename_3d_gen = gr.Textbox(label="Base Filename for 3D Model")
            generate_3d_button = gr.Button("Generate 3D Model")
            status_3d_gen = gr.Textbox(label="3D Generation Status")
            output_3d_model_link = gr.HTML(label="3D Model Link")

            generate_3d_button.click(
                fn=launch_3d_generation,
                inputs=[input_2d_image_for_3d, s3_bucket_input_global, base_filename_3d_gen],
                outputs=[task_id_state]
            ).then(
                fn=track_3d_generation_progress,
                inputs=[task_id_state],
                outputs=[status_3d_gen, output_3d_model_link]
            )

        # ---------------- Decimate 3D ----------------
        with gr.TabItem("Decimated 3D"):
            input_3d_file_decimate = gr.File(label="Upload 3D Model", type="filepath")
            base_filename_decimate = gr.Textbox(label="Base Filename for Decimated Model")
            decimate_button = gr.Button("Decimate 3D Model")
            status_decimate = gr.Textbox(label="Decimation Status")
            output_decimated_model_link = gr.HTML(label="Decimated Model Link")

            decimate_button.click(
                fn=launch_decimation_task,
                inputs=[input_3d_file_decimate, s3_bucket_input_global, base_filename_decimate],
                outputs=[task_id_state]
            ).then(
                fn=track_decimation_progress,
                inputs=[task_id_state],
                outputs=[status_decimate, output_decimated_model_link]
            )

demo.launch(server_name="0.0.0.0", server_port=7860)

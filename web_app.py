# web_app.py - Gradio Frontend

import gradio as gr
import requests
import time
from uuid import uuid4

# --- FastAPI Backend URL ---
# IMPORTANT: Replace this with the PRIVATE IP address of your FastAPI server.
# This allows for secure internal communication within your network.
FASTAPI_URL = "http://172.31.25.10:8000" # Placeholder for a private IP address

def launch_full_pipeline(text_prompt, s3_bucket_name):
    """
    Makes a POST request to the FastAPI backend to start the full generation pipeline.
    Returns the task ID to the Gradio frontend.
    """
    # Using a single text prompt for both biome and theme for simplicity.
    unique_id = str(uuid4())
    data = {
        "biome_name": text_prompt,
        "theme_prompt": text_prompt,
        "s3_bucket_name": s3_bucket_name,
        "unique_id": unique_id
    }
    
    try:
        response = requests.post(f"{FASTAPI_URL}/generate-full-pipeline", json=data)
        response.raise_for_status() # Raise an exception for bad status codes
        task_info = response.json()
        return task_info.get("task_id")
    except requests.exceptions.RequestException as e:
        print(f"Error calling FastAPI backend: {e}")
        return None

def track_pipeline_progress(task_id):
    """
    Polls the FastAPI /tasks/{id} endpoint to track the status of the entire pipeline.
    Yields status updates back to the Gradio UI.
    """
    if not task_id:
        yield "Waiting for task to start...", [], None
        return
    
    # Store previous state to avoid redundant updates
    previous_status = ""
    
    while True:
        try:
            response = requests.get(f"{FASTAPI_URL}/tasks/{task_id}")
            response.raise_for_status()
            task_info = response.json()
            status = task_info.get("status", "UNKNOWN")
            
            # Only update if the status has changed
            if status == previous_status:
                time.sleep(2)
                continue
            
            previous_status = status
            
            if status == 'PENDING':
                yield "Task is queued...", [], None
            elif status == 'IN_PROGRESS_2D':
                yield "Generating 2D image from text prompt...", [], None
            elif status == 'IN_PROGRESS_3D':
                yield "2D image generation complete. Starting 3D model generation...", [], None
            elif status == 'IN_PROGRESS_DECIMATION':
                yield "3D model generation complete. Starting decimation...", [], None
            elif status == 'SUCCESS':
                result = task_info.get("result", {})
                # Note: The result from the task chain is the final result of the last task.
                # In this case, it's the URL of the decimated model.
                decimated_url = result.get("decimated_url")
                
                # To get the image URL from the first task, we would need to store it
                # in a different way or retrieve it from S3.
                # For this example, we will just display the decimated URL.
                
                # Create HTML for download links
                html_output = "<h3>Generated Assets:</h3>"
                if decimated_url:
                    html_output += f"<h4>Final 3D Model:</h4>"
                    html_output += f"<a href='{decimated_url}' target='_blank'>Download Decimated 3D Model</a><br>"

                yield "All tasks complete!", [], html_output
                return
            elif status == 'FAILURE':
                error_msg = task_info.get('error', "An unknown error occurred.")
                yield f"Error: {error_msg}", [], None
                return
            else:
                yield f"Task state: {status}", [], None
        
        except requests.exceptions.RequestException as e:
            yield f"Error polling FastAPI backend: {e}", [], None
            return
            
        time.sleep(2) # Poll every 2 seconds

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# AI-Powered 3D Asset Generator")
    gr.Markdown("This application uses an external FastAPI backend and Celery queue to run generation tasks on a GPU, keeping the Gradio app responsive. The generated assets are uploaded to S3.")

    s3_bucket_input_global = gr.Textbox(label="S3 Bucket Name", value="sparkassets", interactive=True)
    task_id_state = gr.State(None)

    with gr.TabItem("Text to 3D Pipeline"):
        gr.Markdown("## Text-to-3D Generation Pipeline")
        gr.Markdown("Enter a text prompt to generate a 2D image, which is then used to generate and decimate a 3D model.")
        
        text_to_3d_prompt = gr.Textbox(label="Text Prompt", placeholder="e.g., 'a detailed model of a stone lantern'")

        generate_button = gr.Button("🚀 Start Full Pipeline")
        generation_status = gr.Textbox(label="Generation Status", lines=1)
        final_output = gr.HTML(label="Final Download Links")

        # The image_output gallery has been removed as the final output is now a single 3D model.
        # The track_pipeline_progress function will need to be updated to handle this change
        # and provide more granular updates if needed.
        generate_button.click(
            fn=launch_full_pipeline,
            inputs=[text_to_3d_prompt, s3_bucket_input_global],
            outputs=[task_id_state]
        ).then(
            fn=track_pipeline_progress,
            inputs=[task_id_state],
            outputs=[generation_status, final_output]
        )
            
demo.launch(server_name="0.0.0.0", server_port=7860)

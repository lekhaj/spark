import gradio as gr
import json
import time
import redis
import os
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# Load environment variables
load_dotenv()

# --- Configuration from environment variables ---
MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "biomes")
REDIS_HOST = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_QUEUE_2D = "image_tasks"
REDIS_QUEUE_3D = "model_tasks"
REDIS_QUEUE_DECIMATE = "decimation_tasks"

# Initialize Redis client
redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# --- DATABASE INTERACTION FUNCTIONS ---

def get_db_client():
    """Establishes and returns a MongoDB client connection."""
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except ConnectionFailure:
        print("Failed to connect to MongoDB.")
        return None

def find_documents(query: dict):
    """Finds documents based on a query."""
    client = get_db_client()
    if not client:
        return []
    collection = client[MONGO_DB][MONGO_COLLECTION]
    return list(collection.find(query))

def start_generation_pipeline(doc_id):
    """
    Enqueues the generation tasks to Redis.
    """
    client = get_db_client()
    if not client:
        return "Failed to connect to MongoDB."

    try:
        doc = client[MONGO_DB][MONGO_COLLECTION].find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found."

        prompt = doc.get('text_prompt')
        if not prompt:
            return "Document has no text prompt."

        # Enqueue 2D image generation task
        image_task = {
            "job_id": str(doc['_id']),
            "prompt": prompt
        }
        redis_client.lpush(REDIS_QUEUE_2D, json.dumps(image_task))
        
        # Enqueue 3D model generation task
        model_task = {
            "job_id": str(doc['_id']),
            "prompt": prompt,
            "image_url": doc.get('image_generation_details', {}).get('s3_links', [None])[0]
        }
        redis_client.lpush(REDIS_QUEUE_3D, json.dumps(model_task))

        # Enqueue decimation task
        decimate_task = {
            "job_id": str(doc['_id']),
            "model_url": doc.get('3d_generation_details', {}).get('s3_link', '')
        }
        redis_client.lpush(REDIS_QUEUE_DECIMATE, json.dumps(decimate_task))
        
        return f"Pipeline started for document ID `{doc_id}`. Check the status tabs for progress."

    except Exception as e:
        return f"Failed to start pipeline: {e}"

def check_image_status(doc_id):
    """Checks the status of the image generation task."""
    client = get_db_client()
    if not client:
        return "Failed to connect to MongoDB."

    try:
        doc = client[MONGO_DB][MONGO_COLLECTION].find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found."
            
        details = doc.get("image_generation_details", {})
        status = details.get("status", "pending")
        s3_links = details.get("s3_links", [])
        
        if status == "COMPLETED" and s3_links:
            return f"Status: {status}\nLink: {s3_links[0]}"
        else:
            return f"Status: {status}"
    except Exception as e:
        return f"Error checking image status: {e}"

def check_3d_status(doc_id):
    """Checks the status of the 3D generation task."""
    client = get_db_client()
    if not client:
        return "Failed to connect to MongoDB."

    try:
        doc = client[MONGO_DB][MONGO_COLLECTION].find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found."
            
        details = doc.get("3d_generation_details", {})
        status = details.get("status", "pending")
        s3_link = details.get("s3_link")
        
        if status == "COMPLETED" and s3_link:
            return f"Status: {status}\nLink: {s3_link}"
        else:
            return f"Status: {status}"
    except Exception as e:
        return f"Error checking 3D status: {e}"

def check_decimation_status(doc_id):
    """Checks the status of the decimation task."""
    client = get_db_client()
    if not client:
        return "Failed to connect to MongoDB."

    try:
        doc = client[MONGO_DB][MONGO_COLLECTION].find_one({"_id": ObjectId(doc_id)})
        if not doc:
            return "Document not found."
            
        details = doc.get("decimation_details", {})
        status = details.get("status", "pending")
        s3_link = details.get("model_url")
        
        if status == "COMPLETED" and s3_link:
            return f"Status: {status}\nLink: {s3_link}"
        else:
            return f"Status: {status}"
    except Exception as e:
        return f"Error checking decimation status: {e}"

# --- GRADIO INTERFACE LAYOUT ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 Generative Biome Pipeline")
    
    with gr.Tab("Dashboard"):
        gr.Markdown("## 📋 Pending Biomes")
        refresh_pending_button = gr.Button("Refresh List")
        pending_output = gr.Textbox(label="Pending Biomes (ID, Prompt, Status)", interactive=False)
        
        refresh_pending_button.click(
            fn=lambda: json.dumps(find_documents({"status": "pending"}), indent=2, default=str), 
            outputs=pending_output
        )
        
    with gr.Tab("Start Pipeline"):
        gr.Markdown("## ▶️ Start a New Pipeline")
        doc_id_input = gr.Textbox(label="MongoDB Document ID to process", placeholder="Enter the ID of a pending document...")
        start_button = gr.Button("Start Pipeline")
        start_output = gr.Textbox(label="Status", interactive=False)
        
        start_button.click(
            fn=start_generation_pipeline,
            inputs=doc_id_input,
            outputs=start_output
        )

    with gr.Tab("Check Status"):
        gr.Markdown("## 🔄 Check a Document's Status")
        status_doc_id_input = gr.Textbox(label="Document ID", placeholder="Enter the document ID to check...")
        
        with gr.Row():
            check_image_button = gr.Button("Check 2D Image Status")
            check_3d_button = gr.Button("Check 3D Model Status")
            check_decimation_button = gr.Button("Check Decimated Assets Status")
            
        image_status_output = gr.Textbox(label="2D Image Status", interactive=False)
        model_status_output = gr.Textbox(label="3D Model Status", interactive=False)
        decimation_status_output = gr.Textbox(label="Decimated Assets Status", interactive=False)

        check_image_button.click(check_image_status, inputs=status_doc_id_input, outputs=image_status_output)
        check_3d_button.click(check_3d_status, inputs=status_doc_id_input, outputs=model_status_output)
        check_decimation_button.click(check_decimation_status, inputs=status_doc_id_input, outputs=decimation_status_output)

# Launch the Gradio application
demo.launch(server_name="0.0.0.0", server_port=7860)

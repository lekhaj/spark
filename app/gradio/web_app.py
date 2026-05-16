import gradio as gr
import json
import time
import random
import os
import redis
import requests
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from app.gradio.pages.s3_asset_viewer_page import s3_asset_viewer_ui
from app.gradio.pages.decimation_page import decimation_page_ui
from app.gradio.pages.biome_generation_page import mount_into as mount_generator_ui
from app.gradio.pages.orchestrator_control_page import create_orchestrator_ui
from app.gradio.pages.biome_editor import biome_editor_ui
from app.gradio.pages.pipeline_dashboard_page import pipeline_dashboard_ui
from app.gradio.pages.generation_studio_page import generation_studio_ui
from app.services.mongo_service import get_db, get_biome_choices_live

# Load environment variables from .env file
load_dotenv()


# --- Configuration from environment variables ---

MONGO_URI = os.getenv("MONGODB_URL")
MONGO_COLLECTION = os.getenv("MONGODB_DB_NAME", "biomes")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
REDIS_QUEUE_2D = "image_tasks"
REDIS_QUEUE_3D = "model_tasks"
REDIS_QUEUE_DECIMATE = "__tasks"
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Initialize Redis client
try:
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    print("Successfully connected to Redis.")
except redis.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    r = None

# Global variables to hold the active database and collection.
db_client = None
active_db = None
active_collection = None

# --- DATABASE INTERACTION FUNCTIONS ---    

def get_database_names():
    """Lists all database names from the connected client."""
    db = get_db()
    if db is None:
        return []
    try:
        return db.client.list_database_names()
    except Exception as e:
        print(f"Failed to list databases: {e}")
        return []

def get_collection_names(database_name):
    """
    Lists all collection names in a given database.
    Added a try-except block to handle potential authentication errors.
    """
    db = get_db()
    if db is None:
        return []
    try:
        return db.client[database_name].list_collection_names()
    except Exception as e:
        print(f"Failed to list collections in database '{database_name}': {e}.")
        return []


def fetch_live_biome_details(database_name, collection_name, doc_id):
    """
    Fetches a single document from the live database by its ID.
    """
    db = get_db()
    if db is None:
        return None
    try:
        collection = db[collection_name]
        try:
            query_id = ObjectId(doc_id)
        except Exception:
            query_id = doc_id
        return collection.find_one({"_id": query_id})
    except Exception as e:
        print(f"Failed to fetch biome details for ID {doc_id}: {e}")
        return None

def update_live_biome_details(database_name, collection_name, doc_id, section, new_data):
    """
    Updates a specific section of a document in the live database.
    """
    db = get_db()
    if db is None:
        return False
    try:
        collection = db[collection_name]
        try:
            query_id = ObjectId(doc_id)
        except Exception:
            query_id = doc_id
        result = collection.update_one({"_id": query_id}, {"$set": {section: new_data}})
        return result.modified_count > 0
    except Exception as e:
        print(f"Failed to update document for ID {doc_id}: {e}")
        return False

def create_new_biome(database_name, collection_name, biome_name):
    """
    Creates a new biome document in the specified database and collection.
    """
    db = get_db()
    if db is None or not database_name or not collection_name or not biome_name:
        return (None, "Failed to create biome. Please check inputs.")
    try:
        collection = db[collection_name]
        if collection.find_one({"biome_name": biome_name}):
            return (None, f"Biome '{biome_name}' already exists.")
        new_doc = {
            "biome_name": biome_name,
            "status": "created",
            "image_generation_details": {},
            "3d_generation_details": {},
            "decimation_details": {},
            "timestamp": time.time()
        }
        result = collection.insert_one(new_doc)
        return (str(result.inserted_id), "Biome created successfully!")
    except Exception as e:
        print(f"Failed to create new biome: {e}")
        return (None, f"Error creating biome: {e}")

# --- UI LOGIC FUNCTIONS ---

def update_collections_dropdown(database_name):
    """Updates the collections dropdown based on the selected database."""
    collections = get_collection_names(database_name)
    return gr.Dropdown(choices=collections, value=collections[0] if collections else None)

def update_biomes_dropdown(database_name, collection_name):
    """
    Updates the biomes dropdown and the biome choices state.
    """
    biome_choices = get_biome_choices_live(database_name, collection_name)
    biome_names = [name for name, _ in biome_choices]
    return (
        gr.Dropdown(choices=biome_names, value=biome_names[0] if biome_names else None),
        biome_choices
    )

def refresh_orchestrate_biomes():
    from app.config import settings
    biome_choices = get_biome_choices_live(settings.MONGODB_DB_NAME, "biomes")
    return gr.update(choices=[(name, _id) for name, _id in biome_choices], value=biome_choices[0][1] if biome_choices else None)

# --- ORCHESTRATOR ENDPOINT CALLERS ---
def submit_image_tasks_api(biome_id):
    """Call orchestrator /submit_image_task/ endpoint for the given biome_id."""
    try:
        resp = requests.post(f"{API_BASE_URL}/submit_image_task/", params={"biome_id": biome_id}, timeout=30)
        if resp.ok:
            return json.dumps(resp.json(), indent=2)
        else:
            return f"Error: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"Exception: {e}"

def submit_3d_tasks_api(biome_id):
    """Call orchestrator /submit_3d_tasks/ endpoint for the given biome_id."""
    try:
        resp = requests.post(f"{API_BASE_URL}/submit_3d_tasks/", params={"biome_id": biome_id}, timeout=30)
        if resp.ok:
            return json.dumps(resp.json(), indent=2)
        else:
            return f"Error: {resp.status_code} {resp.text}"
    except Exception as e:
        return f"Exception: {e}"

def load_biome_pipeline_live(biome_name, biome_choices, database_name, collection_name):
    """
    Loads the pipeline status for the selected biome from the live database.
    This version includes more robust checks for JSON and data format to prevent errors.
    """
    doc_id = next((_id for name, _id in biome_choices if name == biome_name), None)

    if not doc_id:
        return (gr.State(None), "Not Started", "{}", [], "Not Started", "{}", "")

    biome_doc = fetch_live_biome_details(database_name, collection_name, doc_id)

    if not biome_doc:
        return (gr.State(doc_id), "Not Started", "{}", [], "Not Started", "{}", "")

    status_2d_text = "Not Started"
    images_2d_list = []

    # Return the full biome_doc as the 2D JSON output
    json_2d_dict = biome_doc.copy() if isinstance(biome_doc, dict) else {}

    details_2d = biome_doc.get("image_generation_details", {}) if isinstance(biome_doc, dict) else {}
    if isinstance(details_2d, dict):
        status_2d_text = details_2d.get("status", "Not Started")
        generated_images = details_2d.get("generated_images", [])
        if isinstance(generated_images, list):
            images_2d_list = generated_images
        else:
            images_2d_list = []

    status_3d_text = "Not Started"
    json_3d_dict = {}
    model_link = ""

    details_3d = biome_doc.get("3d_generation_details", {})
    if isinstance(details_3d, dict):
        status_3d_text = details_3d.get("status", "Not Started")
        json_3d_dict = details_3d
        if status_3d_text == "COMPLETED" and "model_url" in details_3d:
            model_url = details_3d.get("model_url")
            if model_url:
                model_link = f"<a href='{model_url}' target='_blank'>Download 3D Model</a>"

    return (
        gr.State(doc_id),
        status_2d_text,
        json_2d_dict,
        images_2d_list,
        status_3d_text,
        json_3d_dict,
        model_link
    )

def get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name):
    """
    Helper function to get the document ID for a new or existing biome.
    Returns (doc_id, error_message)
    """
    if biome_action_type == "Create New Biome":
        doc_id, msg = create_new_biome(database_name, collection_name, new_biome_name)
        if not doc_id:
            return None, msg
        print(msg)
    else:
        doc_id = next((_id for name, _id in biome_choices if name == selected_biome_name), None)
        if not doc_id:
            return None, "Selected biome not found."
    return doc_id, None
def _start_2d_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, prompt):
    """Submits a 2D generation task to the Redis queue."""
    # ... (existing code to get doc_id)
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return (None, "Failed to submit task.", "{}")

    # This is the task payload for the worker
    task_payload = {
        'job_id': str(doc_id),
        'prompt': prompt,
    }
    
    # Send task to Redis queue
    if r:
        r.lpush("image_tasks", json.dumps(task_payload))
        status_message = f"Task {doc_id} for 2D generation submitted. Status: PENDING"
    else:
        status_message = "Failed to connect to Redis. Task not submitted."
        
    initial_data = {
        "status": "PENDING",
        "prompt": prompt,
        "model_used": "Stable Diffusion XL",
        "generated_images": [],
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "image_generation_details", initial_data)
    
    print(status_message)
    
    return (gr.State(doc_id), status_message, initial_data)

def _start_3d_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, input_2d_image):
    """Submits a 3D generation task to the Redis queue."""
    # ... (existing code)
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return (gr.State(None), "Failed to submit task.", "")

    if input_2d_image is None:
        return (gr.State(None), "Please upload a 2D image.", "")

    # ── Upload image to S3 so GPU worker can fetch it ────────────────────────
    import boto3 as _boto3, uuid as _uuid
    S3_BUCKET = os.getenv("BUCKET_NAME", "sparkassets")
    S3_REGION = os.getenv("S3_REGION", "ap-south-1")
    s3_key    = f"manual_gen_inputs/{doc_id}_{_uuid.uuid4().hex[:8]}.png"
    image_s3_url = ""
    try:
        s3 = _boto3.client("s3", region_name=S3_REGION)
        s3.upload_file(input_2d_image.name, S3_BUCKET, s3_key)
        image_s3_url = f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        print(f"Uploaded input image → {image_s3_url}")
    except Exception as _e:
        print(f"S3 upload failed: {_e}")
        image_s3_url = input_2d_image.name

    task_payload = {
        "job_id":       str(doc_id),
        "biome_id":     str(doc_id),
        "image_s3_url": image_s3_url,
        "update_key":   "3d_generation_details.model_url",
        "output_key":   f"models/{doc_id}/output.glb",
        "timestamp":    time.time()
    }
    
    # Send task to Redis queue
    if r:
        r.lpush("model_tasks", json.dumps(task_payload))
        status_message = f"Task {doc_id} for 3D generation submitted. Status: PENDING"
    else:
        status_message = "Failed to connect to Redis. Task not submitted."

    initial_data = {
        "status": "PENDING",
        "input_images_count": 1,
        "model_url": "",
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "3d_generation_details", initial_data)

    print(status_message)

    return (gr.State(doc_id), status_message, "")

def _start_decimation_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, input_3d_file):
    """Simulates starting a 3D decimation task."""
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return (gr.State(None), "Failed to submit task.", "")

    if input_3d_file is None:
        return (gr.State(None), "Please upload a 3D model.", "")
    
    task_id = f"task-decimate-{doc_id}-{int(time.time())}"
    
    initial_data = {
        "status": "PENDING",
        "input_file": input_3d_file.name,
        "model_url": "",
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "decimation_details", initial_data)

    print(f"Task {task_id} for decimation submitted. Status: PENDING")
    
    return (gr.State(task_id), "Task submitted: PENDING", "")


# --- GRADIO INTERFACE LAYOUT (modified to include AWS control) ---

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# 🌎 AI World Builder")
    

    current_biome_id = gr.State(None)
    biome_choices_list = gr.State([])
    current_task_id_2d = gr.State(None)
    current_task_id_grid = gr.State(None)
    current_task_id_3d = gr.State(None)
    current_task_id_decimate = gr.State(None)

    # Pre-populate dropdowns
    dbs = get_database_names()
    default_db = dbs[0] if dbs else None
    colls = get_collection_names(default_db) if default_db else []
    default_coll = colls[2] if colls else None
    initial_biome_names = []
    initial_biome_value = None
    initial_biome_choices_val = []
    # Prefer explicitly loading from the 'biomes' collection for the Orchestrate tab
    if default_db:
        # Try the explicit 'biomes' collection first
        initial_biome_choices_val = get_biome_choices_live(default_db, "biomes")
        # Fallback to the default collection if 'biomes' is empty or not present
        if not initial_biome_choices_val and default_coll:
            initial_biome_choices_val = get_biome_choices_live(default_db, default_coll)
        initial_biome_names = [name for name, _ in initial_biome_choices_val]
        initial_biome_value = initial_biome_names[0] if initial_biome_names else None

    with gr.Tabs() as tabs:
        # S3 Asset Viewer Tab
        with gr.TabItem("S3 Asset Viewer"):
            s3_asset_viewer_ui()

        # Generator Tab
        with gr.TabItem("Generate Biome"):
            mount_generator_ui()

        # Biome editor tab
        with gr.TabItem("Biome Editor"):
            biome_editor_ui()

        # Orchestrator Control Tab
        with gr.TabItem("Orchestrator Control"):
            create_orchestrator_ui()


        # Orchestrate Biome Tab
        with gr.TabItem("Orchestrate Biome"):
            gr.Markdown("## Orchestrate Biome Tasks")
            gr.Markdown("Submit image and 3D generation tasks for a biome using backend orchestration API.")

            # Biome selection dropdown: show name but value is document id (choices are (label, value))
            if initial_biome_choices_val:
                orchestrate_biome_dropdown = gr.Dropdown(
                    label="Select Biome",
                    choices=[(name, _id) for name, _id in initial_biome_choices_val],
                    value=initial_biome_choices_val[0][1],
                    interactive=True
                )
            else:
                orchestrate_biome_dropdown = gr.Dropdown(label="Select Biome", choices=[], value=None, interactive=False)

            submit_image_btn = gr.Button("Submit Image Tasks")
            submit_3d_btn = gr.Button("Submit 3D Tasks")
            orchestration_result = gr.Json(label="Orchestration Result")
            refresh_biomes_btn = gr.Button("Refresh Biomes")

            def submit_image_tasks_gradio(biome_id):
                try:
                    resp = requests.post(f"http://localhost:8000/submit_image_tasks/", params={"biome_id": biome_id})
                    return resp.json()
                except Exception as e:
                    return {"error": str(e)}

            def submit_3d_tasks_gradio(biome_id):
                try:
                    resp = requests.post(f"http://localhost:8000/submit_3d_tasks/", params={"biome_id": biome_id})
                    return resp.json()
                except Exception as e:
                    return {"error": str(e)}

            # Bind clicks: pass the dropdown value (document id) directly to the handlers
            submit_image_btn.click(fn=submit_image_tasks_gradio, inputs=[orchestrate_biome_dropdown], outputs=[orchestration_result])
            submit_3d_btn.click(fn=submit_3d_tasks_gradio, inputs=[orchestrate_biome_dropdown], outputs=[orchestration_result])
            refresh_biomes_btn.click(fn=refresh_orchestrate_biomes, inputs=[], outputs=[orchestrate_biome_dropdown])

        # Asset Decimation Tab
        with gr.TabItem("Asset Decimation"):
            decimation_page_ui()

        # Pipeline Dashboard Tab
        with gr.TabItem("Pipeline Dashboard"):
            pipeline_dashboard_ui()

        # Generation Studio Tab — manual stage-by-stage generation
        with gr.TabItem("🎨 Generation Studio"):
            generation_studio_ui()


# ─── FastAPI mount so we can serve /static/babylon_viewer.html alongside Gradio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

_static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "static")
_static_dir = os.path.normpath(_static_dir)

# queue() must be called before gr.mount_gradio_app — without it Gradio's
# websocket queue never starts and the UI hangs on the loading screen.
demo.queue()

fastapi_app = FastAPI()
fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["GET", "HEAD"], allow_headers=["*"],
)
if os.path.isdir(_static_dir):
    fastapi_app.mount("/static", StaticFiles(directory=_static_dir, html=True), name="static")

fastapi_app = gr.mount_gradio_app(fastapi_app, demo, path="/")

uvicorn.run(fastapi_app, host="0.0.0.0", port=7860)

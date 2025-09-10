# =========================
# File: web_app.py
# =========================
import gradio as gr
import json
import time
import random
import os
import redis
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import requests

# Load environment variables from .env file
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
API_BASE_URL = os.getenv("API_BASE_URL", "http://15.206.99.66:8000")

# Initialize Redis client
try:
    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    redis_client.ping()
    print("Successfully connected to Redis.")
except redis.exceptions.ConnectionError as e:
    print(f"Could not connect to Redis: {e}")
    redis_client = None

# Global variables to hold the active database and collection.
db_client = None
active_db = None
active_collection = None

# --- DATABASE INTERACTION FUNCTIONS ---

def get_db_client():
    """Establishes and returns a MongoDB client connection."""
    global db_client
    if db_client is None:
        try:
            db_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            db_client.admin.command('ping')
            print("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            return None
    return db_client

def get_database_names():
    """Lists all database names from the connected client."""
    client = get_db_client()
    if not client:
        return []
    try:
        return client.list_database_names()
    except OperationFailure as e:
        print(f"Failed to list databases: {e}")
        return []

def get_collection_names(database_name):
    """
    Lists all collection names in a given database.
    Added a try-except block to handle potential authentication errors.
    """
    client = get_db_client()
    if not client:
        return []
    try:
        db = client[database_name]
        return db.list_collection_names()
    except OperationFailure as e:
        print(f"Failed to list collections in database '{database_name}': {e}. Check user permissions.")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

def get_biome_choices_live(database_name, collection_name):
    """
    Fetches all biome names and their IDs from a live MongoDB collection.
    Returns a list of tuples: [(biome_name, doc_id), ...].
    """
    client = get_db_client()
    if not client or not database_name or not collection_name:
        return []
    try:
        db = client[database_name]
        collection = db[collection_name]
        documents = list(collection.find({}, {"biome_name": 1}))
        choices = [(doc.get("biome_name", "Unknown Biome"), str(doc["_id"])) for doc in documents]
        return choices
    except OperationFailure as e:
        print(f"Failed to fetch biomes from collection '{collection_name}': {e}")
        return []

def fetch_live_biome_details(database_name, collection_name, doc_id):
    """
    Fetches a single document from the live database by its ID.
    """
    client = get_db_client()
    if not client:
        return None
    try:
        db = client[database_name]
        collection = db[collection_name]
        return collection.find_one({"_id": ObjectId(doc_id)})
    except Exception as e:
        print(f"Failed to fetch biome details for ID {doc_id}: {e}")
        return None

def update_live_biome_details(database_name, collection_name, doc_id, section, new_data):
    """
    Updates a specific section of a document in the live database.
    """
    client = get_db_client()
    if not client:
        return False
    try:
        db = client[database_name]
        collection = db[collection_name]
        result = collection.update_one({"_id": ObjectId(doc_id)}, {"$set": {section: new_data}})
        return result.modified_count > 0
    except Exception as e:
        print(f"Failed to update document for ID {doc_id}: {e}")
        return False

def create_new_biome(database_name, collection_name, biome_name):
    """
    Creates a new biome document in the specified database and collection.
    """
    client = get_db_client()
    if not client or not database_name or not collection_name or not biome_name:
        return (None, "Failed to create biome. Please check inputs.")
    try:
        db = client[database_name]
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
    json_2d_text = "{}"
    images_2d_list = []
    
    details_2d = biome_doc.get("image_generation_details", {})
    if isinstance(details_2d, dict):
        status_2d_text = details_2d.get("status", "Not Started")
        try:
            json_2d_text = json.dumps(details_2d, indent=2)
        except TypeError:
            json_2d_text = "Error: Invalid JSON data"
            
        generated_images = details_2d.get("generated_images", [])
        if isinstance(generated_images, list):
            images_2d_list = generated_images
        else:
            images_2d_list = []

    status_3d_text = "Not Started"
    json_3d_text = "{}"
    model_link = ""

    details_3d = biome_doc.get("3d_generation_details", {})
    if isinstance(details_3d, dict):
        status_3d_text = details_3d.get("status", "Not Started")
        try:
            json_3d_text = json.dumps(details_3d, indent=2)
        except TypeError:
            json_3d_text = "Error: Invalid JSON data"
            
        if status_3d_text == "COMPLETED" and "model_url" in details_3d:
            model_url = details_3d.get("model_url")
            if model_url:
                model_link = f"<a href='{model_url}' target='_blank'>Download 3D Model</a>"
    
    return (
        gr.State(doc_id),
        status_2d_text,
        json_2d_text,
        images_2d_list,
        status_3d_text,
        json_3d_text,
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
    """Simulates starting a 2D generation task."""
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return (None, "Failed to submit task.", "{}")

    task_id = f"task-2d-{doc_id}-{int(time.time())}"
    
    initial_data = {
        "status": "PENDING",
        "prompt": prompt,
        "model_used": "Simulated AI Model",
        "generated_images": [],
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "image_generation_details", initial_data)
    
    print(f"Task {task_id} for 2D generation submitted. Status: PENDING")
    
    return (gr.State(task_id), "Task submitted: PENDING", initial_data)

def run_2d_generation(task_id_input, database_name, collection_name):
    """Simulates the completion of a 2D task and updates the main pipeline view."""
    if not task_id_input:
        return (gr.Json({}), gr.Gallery([]), "No task to run.")

    doc_id = task_id_input.split('-')[2]
    
    time.sleep(2)
    
    new_images = [
        f"https://placehold.co/600x400/1e88e5/ffffff?text=Biome+Image+{random.randint(1,9)}",
        f"https://placehold.co/600x400/43a047/ffffff?text=Biome+Image+{random.randint(10,19)}",
    ]
    new_data = {
        "status": "COMPLETED",
        "generated_images": new_images,
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "image_generation_details", new_data)

    print(f"Task {task_id_input} completed.")

    return (gr.Json(new_data), gr.Gallery(new_images), "Task Completed.")

def _start_grid_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, grid_data_str):
    """Simulates starting a Grid to Image generation task."""
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return (None, "Failed to submit task.", "{}")
        
    task_id = f"task-grid-{doc_id}-{int(time.time())}"

    initial_data = {
        "status": "PENDING",
        "grid_data": json.loads(grid_data_str),
        "model_used": "Simulated Grid Model",
        "generated_images": [],
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "grid_generation_details", initial_data)
    
    print(f"Task {task_id} for Grid generation submitted. Status: PENDING")
    
    return (gr.State(task_id), "Task submitted: PENDING", initial_data)

def run_grid_generation(task_id_input, database_name, collection_name):
    """Simulates the completion of a Grid task and updates the main pipeline view."""
    if not task_id_input:
        return (gr.Json({}), gr.Gallery([]), "No task to run.")

    doc_id = task_id_input.split('-')[2]

    time.sleep(2)

    new_images = [
        f"https://placehold.co/600x400/ff9800/ffffff?text=Grid+Image+{random.randint(20,29)}",
        f"https://placehold.co/600x400/9e9e9e/ffffff?text=Grid+Image+{random.randint(30,39)}",
    ]
    new_data = {
        "status": "COMPLETED",
        "generated_images": new_images,
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "grid_generation_details", new_data)
    
    print(f"Task {task_id_input} completed.")

    return (gr.Json(new_data), gr.Gallery(new_images), "Task Completed.")

def _start_3d_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, input_2d_image):
    """Simulates starting a 3D generation task."""
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return (gr.State(None), "Failed to submit task.", "")
        
    if input_2d_image is None:
        return (gr.State(None), "Please upload a 2D image.", "")
    
    task_id = f"task-3d-{doc_id}-{int(time.time())}"

    initial_data = {
        "status": "PENDING",
        "input_images_count": 1,
        "model_url": "",
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "3d_generation_details", initial_data)

    print(f"Task {task_id} for 3D generation submitted. Status: PENDING")

    return (gr.State(task_id), "Task submitted: PENDING", "")

def run_3d_generation(task_id_input, database_name, collection_name):
    """Simulates the completion of a 3D task and updates the main pipeline view."""
    if not task_id_input:
        return (gr.HTML(""), "No task to run.")

    doc_id = task_id_input.split('-')[2]

    time.sleep(2)

    new_model_url = f"s3://sparkassets/3d_assets/model-{random.randint(100, 999)}.glb"
    new_data = {
        "status": "COMPLETED",
        "model_url": new_model_url,
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "3d_generation_details", new_data)

    model_link = f"<a href='{new_model_url}' target='_blank'>Download 3D Model</a>"
    
    print(f"Task {task_id_input} completed.")

    return (gr.HTML(model_link), "Task Completed.")

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

def run_decimation(task_id_input, database_name, collection_name):
    """Simulates the completion of a decimation task and updates the main pipeline view."""
    if not task_id_input:
        return (gr.HTML(""), "No task to run.")

    doc_id = task_id_input.split('-')[2]

    time.sleep(2)

    new_model_url = f"s3://sparkassets/processed/decimated-model-{random.randint(100, 999)}.glb"
    new_data = {
        "status": "COMPLETED",
        "model_url": new_model_url,
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "decimation_details", new_data)
    
    model_link = f"<a href='{new_model_url}' target='_blank'>Download Decimated 3D Model</a>"
    
    print(f"Task {task_id_input} completed.")

    return (gr.HTML(model_link), "Task Completed.")

# --- AWS Control Functions ---

def control_aws_instance(instance_type: str, action: str):
    """
    Sends a request to the FastAPI backend to start or stop a specific EC2 instance.
    """
    endpoint = f"{API_BASE_URL}/aws/{action}/{instance_type}"
    try:
        response = requests.post(endpoint)
        response.raise_for_status()  # This will raise an exception for HTTP errors
        return f"Successfully sent command to {action} {instance_type} instance."
    except requests.exceptions.RequestException as e:
        return f"Failed to connect to API: {e}"
    except Exception as e:
        return f"An error occurred: {e}"

# --- GRADIO INTERFACE LAYOUT (modified to include AWS control) ---

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# 🌎 AI World Builder")
    
    current_biome_id = gr.State(None)
    biome_choices_list = gr.State([]) 
    current_task_id_2d = gr.State(None)
    current_task_id_grid = gr.State(None)
    current_task_id_3d = gr.State(None)
    current_task_id_decimate = gr.State(None)

    with gr.Tabs() as tabs:
        # Asset Pipeline Tab
        with gr.TabItem("Asset Pipeline"):
            gr.Markdown("## ⚙️ MongoDB Connection")
            gr.Markdown("First, select your database and collection. Then, click 'Refresh Biomes' to populate the dropdown below.")
            
            with gr.Row():
                database_dropdown = gr.Dropdown(
                    label="Select Database", 
                    choices=get_database_names(), 
                    interactive=True
                )
                collection_dropdown = gr.Dropdown(
                    label="Select Collection", 
                    choices=[], 
                    interactive=True
                )
                refresh_button = gr.Button("Refresh Biomes")
            
            gr.Markdown("## 📦 Asset Generation Pipeline")
            gr.Markdown("Select a biome to view its asset generation status.")
            
            biome_dropdown = gr.Dropdown(
                label="Select Biome", 
                choices=[],
                interactive=True
            )
            
            with gr.Accordion("2D Image Generation", open=True):
                gr.Markdown("### Status")
                status_2d_output = gr.Textbox(label="2D Status", show_label=False)
                gr.Markdown("### Details (JSON)")
                json_2d_output = gr.Json(label="2D Generation Details")
                gr.Markdown("### Generated Images")
                images_2d_gallery = gr.Gallery(label="Generated 2D Images", columns=2)
            
            with gr.Accordion("3D Model Generation", open=False):
                gr.Markdown("### Status")
                status_3d_output = gr.Textbox(label="3D Status", show_label=False)
                gr.Markdown("### Details (JSON)")
                json_3d_output = gr.Json(label="3D Generation Details")
                gr.Markdown("### 3D Model Link")
                model_link_output = gr.HTML(label="3D Model Link")
            
            database_dropdown.change(
                fn=update_collections_dropdown, 
                inputs=[database_dropdown], 
                outputs=[collection_dropdown]
            )
            collection_dropdown.change(
                fn=update_biomes_dropdown, 
                inputs=[database_dropdown, collection_dropdown], 
                outputs=[biome_dropdown, biome_choices_list]
            )
            refresh_button.click(
                fn=update_biomes_dropdown, 
                inputs=[database_dropdown, collection_dropdown], 
                outputs=[biome_dropdown, biome_choices_list]
            )
            
            biome_dropdown.change(
                fn=load_biome_pipeline_live,
                inputs=[biome_dropdown, biome_choices_list, database_dropdown, collection_dropdown],
                outputs=[current_biome_id, status_2d_output, json_2d_output, images_2d_gallery, status_3d_output, json_3d_output, model_link_output]
            )

        # Text to Image Tab (unchanged)
        with gr.TabItem("Text to Image"):
            gr.Markdown("## Text-to-Image Generation")
            gr.Markdown("Generate 2D images based on a text prompt for a new or existing biome.")
            
            with gr.Row():
                text_to_image_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                text_to_image_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)
            
            with gr.Row():
                biome_action_type_txt2img = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")
            
            with gr.Column(visible=True) as new_biome_col_txt2img:
                new_biome_name_txt2img = gr.Textbox(label="New Biome Name")
            
            with gr.Column(visible=False) as existing_biome_col_txt2img:
                existing_biome_dropdown_txt2img = gr.Dropdown(label="Select Biome", choices=[], interactive=True)
            
            task_status_2d = gr.Textbox(label="Task Status", interactive=False)
            task_id_2d = gr.State(None)
            
            text_to_image_prompt = gr.Textbox(label="Text Prompt", placeholder="Describe the biome, e.g., 'a serene crystal cave with luminous flora'")
            generate_image_button = gr.Button("🚀 Generate Image from Text")
            
            with gr.Row():
                check_2d_status_button = gr.Button("Run Task & Refresh")
                
            json_2d_results = gr.Json(label="Generation Results")
            images_2d_results = gr.Gallery(label="Generated Images")
            
            text_to_image_db.change(fn=update_collections_dropdown, inputs=[text_to_image_db], outputs=[text_to_image_collection])
            text_to_image_collection.change(fn=update_biomes_dropdown, inputs=[text_to_image_db, text_to_image_collection], outputs=[existing_biome_dropdown_txt2img, biome_choices_list])
            biome_action_type_txt2img.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_txt2img,
                outputs=[new_biome_col_txt2img, existing_biome_col_txt2img]
            )
            generate_image_button.click(
                fn=_start_2d_task,
                inputs=[text_to_image_db, text_to_image_collection, biome_action_type_txt2img, new_biome_name_txt2img, existing_biome_dropdown_txt2img, biome_choices_list, text_to_image_prompt],
                outputs=[task_id_2d, task_status_2d, json_2d_results]
            )
            check_2d_status_button.click(
                fn=run_2d_generation,
                inputs=[task_id_2d, text_to_image_db, text_to_image_collection],
                outputs=[json_2d_results, images_2d_results, task_status_2d]
            )

        # Grid to Image Tab (unchanged)
        with gr.TabItem("Grid to Image"):
            gr.Markdown("## Grid-to-Image Generation")
            gr.Markdown("Generate 2D images from a grid for a new or existing biome.")
            
            with gr.Row():
                grid_to_image_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                grid_to_image_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            with gr.Row():
                biome_action_type_grid2img = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")

            with gr.Column(visible=True) as new_biome_col_grid2img:
                new_biome_name_grid2img = gr.Textbox(label="New Biome Name")

            with gr.Column(visible=False) as existing_biome_col_grid2img:
                existing_biome_dropdown_grid2img = gr.Dropdown(label="Select Biome", choices=[], interactive=True)
                
            task_status_grid = gr.Textbox(label="Task Status", interactive=False)
            task_id_grid = gr.State(None)

            grid_data_input = gr.Textbox(label="Grid Data (JSON array of arrays)", lines=10, placeholder="Example: [[0,0,1,1],[0,1,1,0]]")
            generate_grid_image_button = gr.Button("Generate Image from Grid")
            
            with gr.Row():
                check_grid_status_button = gr.Button("Run Task & Refresh")

            json_grid_results = gr.Json(label="Generation Results")
            images_grid_results = gr.Gallery(label="Generated Images")

            grid_to_image_db.change(fn=update_collections_dropdown, inputs=[grid_to_image_db], outputs=[grid_to_image_collection])
            grid_to_image_collection.change(fn=update_biomes_dropdown, inputs=[grid_to_image_db, grid_to_image_collection], outputs=[existing_biome_dropdown_grid2img, biome_choices_list])
            biome_action_type_grid2img.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_grid2img,
                outputs=[new_biome_col_grid2img, existing_biome_col_grid2img]
            )
            generate_grid_image_button.click(
                fn=_start_grid_task,
                inputs=[grid_to_image_db, grid_to_image_collection, biome_action_type_grid2img, new_biome_name_grid2img, existing_biome_dropdown_grid2img, biome_choices_list, grid_data_input],
                outputs=[task_id_grid, task_status_grid, json_grid_results]
            )
            check_grid_status_button.click(
                fn=run_grid_generation,
                inputs=[task_id_grid, grid_to_image_db, grid_to_image_collection],
                outputs=[json_grid_results, images_grid_results, task_status_grid]
            )

        # 3D Generation Tab (unchanged)
        with gr.TabItem("3D Generation"):
            gr.Markdown("## 3D Model Generation from 2D Image")
            gr.Markdown("Upload or select a 2D image to generate a 3D GLB model for a new or existing biome.")
            
            with gr.Row():
                _3d_gen_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                _3d_gen_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            with gr.Row():
                biome_action_type_3d = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")

            with gr.Column(visible=True) as new_biome_col_3d:
                new_biome_name_3d = gr.Textbox(label="New Biome Name")

            with gr.Column(visible=False) as existing_biome_col_3d:
                existing_biome_dropdown_3d = gr.Dropdown(label="Select Biome", choices=[], interactive=True)
                
            task_status_3d = gr.Textbox(label="Task Status", interactive=False)
            task_id_3d = gr.State(None)

            input_2d_image_for_3d = gr.Image(label="Upload 2D Image", type="pil")
            generate_3d_button = gr.Button("Generate 3D Model")

            with gr.Row():
                check_3d_status_button = gr.Button("Run Task & Refresh")

            model_link_3d_results = gr.HTML(label="3D Model Link")
            
            _3d_gen_db.change(fn=update_collections_dropdown, inputs=[_3d_gen_db], outputs=[_3d_gen_collection])
            _3d_gen_collection.change(fn=update_biomes_dropdown, inputs=[_3d_gen_db, _3d_gen_collection], outputs=[existing_biome_dropdown_3d, biome_choices_list])
            biome_action_type_3d.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_3d,
                outputs=[new_biome_col_3d, existing_biome_col_3d]
            )
            generate_3d_button.click(
                fn=_start_3d_task,
                inputs=[_3d_gen_db, _3d_gen_collection, biome_action_type_3d, new_biome_name_3d, existing_biome_dropdown_3d, biome_choices_list, input_2d_image_for_3d],
                outputs=[task_id_3d, task_status_3d, model_link_3d_results]
            )
            check_3d_status_button.click(
                fn=run_3d_generation,
                inputs=[task_id_3d, _3d_gen_db, _3d_gen_collection],
                outputs=[model_link_3d_results, task_status_3d]
            )

        # Decimated 3D Tab (unchanged)
        with gr.TabItem("Decimated 3D"):
            gr.Markdown("## Decimate 3D Model")
            gr.Markdown("Upload an existing 3D GLB/OBJ/STL model to reduce its polygon count for a new or existing biome.")

            with gr.Row():
                decimate_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                decimate_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            with gr.Row():
                biome_action_type_decimate = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")

            with gr.Column(visible=True) as new_biome_col_decimate:
                new_biome_name_decimate = gr.Textbox(label="New Biome Name")

            with gr.Column(visible=False) as existing_biome_col_decimate:
                existing_biome_dropdown_decimate = gr.Dropdown(label="Select Biome", choices=[], interactive=True)

            task_status_decimate = gr.Textbox(label="Task Status", interactive=False)
            task_id_decimate = gr.State(None)
            
            input_3d_file_decimate = gr.File(label="Upload 3D Model (GLB, OBJ, STL)", type="filepath")
            decimate_button = gr.Button("Decimate 3D Model")
            
            with gr.Row():
                check_decimate_status_button = gr.Button("Run Task & Refresh")

            model_link_decimate_results = gr.HTML(label="Decimated 3D Model Link")
            
            decimate_db.change(fn=update_collections_dropdown, inputs=[decimate_db], outputs=[decimate_collection])
            decimate_collection.change(fn=update_biomes_dropdown, inputs=[decimate_db, decimate_collection], outputs=[existing_biome_dropdown_decimate, biome_choices_list])
            biome_action_type_decimate.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_decimate,
                outputs=[new_biome_col_decimate, existing_biome_col_decimate]
            )
            decimate_button.click(
                fn=_start_decimation_task,
                inputs=[decimate_db, decimate_collection, biome_action_type_decimate, new_biome_name_decimate, existing_biome_dropdown_decimate, biome_choices_list, input_3d_file_decimate],
                outputs=[task_id_decimate, task_status_decimate, model_link_decimate_results]
            )
            check_decimate_status_button.click(
                fn=run_decimation,
                inputs=[task_id_decimate, decimate_db, decimate_collection],
                outputs=[model_link_decimate_results, task_status_decimate]
            )
            
        # New Tab for AWS Control
        with gr.TabItem("AWS Control"):
            gr.Markdown("## 🚀 AWS EC2 Instance Control")
            gr.Markdown("Control the GPU and CPU instances directly from this interface.")
            
            status_output = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                start_gpu_button = gr.Button("Start GPU Instance")
                stop_gpu_button = gr.Button("Stop GPU Instance")
            
            with gr.Row():
                start_cpu_button = gr.Button("Start CPU Instance")
                stop_cpu_button = gr.Button("Stop CPU Instance")
            
            # Button actions
            start_gpu_button.click(
                fn=lambda: control_aws_instance(instance_type="gpu", action="start"),
                inputs=[],
                outputs=[status_output]
            )
            stop_gpu_button.click(
                fn=lambda: control_aws_instance(instance_type="gpu", action="stop"),
                inputs=[],
                outputs=[status_output]
            )
            start_cpu_button.click(
                fn=lambda: control_aws_instance(instance_type="cpu", action="start"),
                inputs=[],
                outputs=[status_output]
            )
            stop_cpu_button.click(
                fn=lambda: control_aws_instance(instance_type="cpu", action="stop"),
                inputs=[],
                outputs=[status_output]
            )
            
# Launch the Gradio application
demo.launch(server_name="0.0.0.0", server_port=7860)

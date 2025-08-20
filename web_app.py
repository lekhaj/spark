# The Asset Pipeline Tracker is a Gradio application that allows users to
# monitor and manage the status of AI-generated 3D assets. It connects
# to a mock database to display a list of biomes and their progress
# through a 2D image generation and 3D model generation pipeline.

# The app features:
# - A main "Asset Pipeline" tab.
# - Dynamic retrieval and display of biomes from a mock MongoDB collection.
# - A detailed view of each biome's pipeline status, including JSON details.
# - The ability to re-trigger incomplete pipeline steps.
# - A real-time update of the UI upon task completion.

import gradio as gr
import json
import time
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure

# MongoDB Connection String and details provided by the user.
CONNECTION_STRING = "mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin"

# Global variables to hold the active database and collection.
# These will be set by the UI.
db_client = None
active_db = None
active_collection = None

# --- DATABASE INTERACTION FUNCTIONS ---

def get_db_client():
    """Establishes and returns a MongoDB client connection."""
    global db_client
    if db_client is None:
        try:
            db_client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
            db_client.admin.command('ping')  # Test the connection
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
        # Use ObjectId to query by the document's unique ID
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
        # Use update_one to modify the specific field
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
        # Check if a biome with this name already exists
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
    """Updates the biomes dropdown based on the selected collection."""
    biome_choices = get_biome_choices_live(database_name, collection_name)
    biome_names = [name for name, _ in biome_choices]
    return (
        gr.Dropdown(choices=biome_names, value=biome_names[0] if biome_names else None),
        gr.State(biome_choices)
    )

def load_biome_pipeline_live(biome_name, biome_choices, database_name, collection_name):
    """
    Loads the pipeline status for the selected biome from the live database.
    """
    doc_id = None
    for name, _id in biome_choices:
        if name == biome_name:
            doc_id = _id
            break

    if not doc_id:
        return (gr.State(None), "Biome not found.", {}, [], "Biome not found.", {}, "")

    biome_doc = fetch_live_biome_details(database_name, collection_name, doc_id)
    if not biome_doc:
        return (gr.State(None), "Failed to load details.", {}, [], "Failed to load details.", {}, "")

    # Initialize outputs for the 2D section
    status_2d_text = "Not Started"
    json_2d_text = "{}"
    images_2d_list = []
    
    # Check if 2D details exist and update outputs
    if "image_generation_details" in biome_doc:
        details_2d = biome_doc["image_generation_details"]
        status_2d_text = details_2d.get("status", "NOT_STARTED")
        json_2d_text = json.dumps(details_2d, indent=2)
        images_2d_list = details_2d.get("generated_images", [])

    # Initialize outputs for the 3D section
    status_3d_text = "Not Started"
    json_3d_text = "{}"
    model_link = ""

    # Check if 3D details exist and update outputs
    if "3d_generation_details" in biome_doc:
        details_3d = biome_doc["3d_generation_details"]
        status_3d_text = details_3d.get("status", "NOT_STARTED")
        json_3d_text = json.dumps(details_3d, indent=2)
        if "model_url" in details_3d:
            model_link = f"<a href='{details_3d['model_url']}' target='_blank'>Download 3D Model</a>"

    return (
        gr.State(doc_id),
        status_2d_text,
        json_2d_text,
        images_2d_list,
        status_3d_text,
        json_3d_text,
        model_link
    )

# --- NEW GENERATION FUNCTIONS ---
def run_2d_generation(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, prompt):
    """Handles 2D generation for a new or existing biome."""
    doc_id = None
    if biome_action_type == "Create New Biome":
        doc_id, msg = create_new_biome(database_name, collection_name, new_biome_name)
        if not doc_id:
            return (None, msg, {}, [], "")
        print(msg)
    elif biome_action_type == "Select Existing Biome":
        biome_choices = get_biome_choices_live(database_name, collection_name)
        for name, _id in biome_choices:
            if name == selected_biome_name:
                doc_id = _id
                break
        if not doc_id:
            return (None, "Selected biome not found.", {}, [], "")

    print(f"Starting 2D generation for document ID: {doc_id}")
    task_id = f"task-2d-{doc_id}-{int(time.time())}"
    
    # Simulate Celery task start
    new_images = [
        f"https://placehold.co/400x400/222222/EEEEEE?text={selected_biome_name}+Image+1",
        f"https://placehold.co/400x400/222222/EEEEEE?text={selected_biome_name}+Image+2",
    ]
    new_data = {
        "status": "COMPLETED",
        "prompt": prompt,
        "model_used": "Simulated AI Model",
        "generated_images": new_images,
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "image_generation_details", new_data)

    return (task_id, "COMPLETED", json.dumps(new_data, indent=2), new_images, new_images)

def run_grid_generation(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, grid_data_str):
    """Handles Grid to Image generation for a new or existing biome."""
    doc_id = None
    if biome_action_type == "Create New Biome":
        doc_id, msg = create_new_biome(database_name, collection_name, new_biome_name)
        if not doc_id:
            return (None, msg, {}, [], "")
        print(msg)
    elif biome_action_type == "Select Existing Biome":
        biome_choices = get_biome_choices_live(database_name, collection_name)
        for name, _id in biome_choices:
            if name == selected_biome_name:
                doc_id = _id
                break
        if not doc_id:
            return (None, "Selected biome not found.", {}, [], "")
            
    print(f"Starting Grid to Image generation for document ID: {doc_id}")
    task_id = f"task-grid-{doc_id}-{int(time.time())}"
    
    # Simulate Celery task start
    new_images = [
        f"https://placehold.co/400x400/222222/EEEEEE?text={selected_biome_name}+Grid+1",
        f"https://placehold.co/400x400/222222/EEEEEE?text={selected_biome_name}+Grid+2",
    ]
    new_data = {
        "status": "COMPLETED",
        "grid_data": json.loads(grid_data_str),
        "model_used": "Simulated Grid Model",
        "generated_images": new_images,
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "grid_generation_details", new_data)
    
    return (task_id, "COMPLETED", json.dumps(new_data, indent=2), new_images, new_images)

def run_3d_generation(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, image_2d_input):
    """Handles 3D generation for a new or existing biome."""
    doc_id = None
    if biome_action_type == "Create New Biome":
        doc_id, msg = create_new_biome(database_name, collection_name, new_biome_name)
        if not doc_id:
            return (None, msg, {}, "")
        print(msg)
    elif biome_action_type == "Select Existing Biome":
        biome_choices = get_biome_choices_live(database_name, collection_name)
        for name, _id in biome_choices:
            if name == selected_biome_name:
                doc_id = _id
                break
        if not doc_id:
            return (None, "Selected biome not found.", {}, "")
            
    if image_2d_input is None:
        return (None, "Please upload a 2D image.", {}, "")
    
    print(f"Starting 3D generation for document ID: {doc_id}")
    task_id = f"task-3d-{doc_id}-{int(time.time())}"
    
    # Simulate Celery task start
    new_data = {
        "status": "COMPLETED",
        "input_images_count": 1,
        "model_url": f"https://placehold.co/400x200/50C878/000000?text={selected_biome_name}+3D+Model",
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "3d_generation_details", new_data)

    model_link = f"<a href='{new_data['model_url']}' target='_blank'>Download 3D Model</a>"
    return (task_id, "COMPLETED", json.dumps(new_data, indent=2), model_link)

def run_decimation(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, input_3d_file):
    """Handles 3D decimation for a new or existing biome."""
    doc_id = None
    if biome_action_type == "Create New Biome":
        doc_id, msg = create_new_biome(database_name, collection_name, new_biome_name)
        if not doc_id:
            return (None, msg, {}, "")
        print(msg)
    elif biome_action_type == "Select Existing Biome":
        biome_choices = get_biome_choices_live(database_name, collection_name)
        for name, _id in biome_choices:
            if name == selected_biome_name:
                doc_id = _id
                break
        if not doc_id:
            return (None, "Selected biome not found.", {}, "")

    if input_3d_file is None:
        return (None, "Please upload a 3D model.", {}, "")
    
    print(f"Starting 3D decimation for document ID: {doc_id}")
    task_id = f"task-decimate-{doc_id}-{int(time.time())}"
    
    # Simulate Celery task start
    new_data = {
        "status": "COMPLETED",
        "input_file": input_3d_file.name,
        "model_url": f"https://placehold.co/400x200/FF0000/FFFFFF?text={selected_biome_name}+Decimated+Model",
        "timestamp": time.time()
    }
    update_live_biome_details(database_name, collection_name, doc_id, "decimation_details", new_data)
    
    model_link = f"<a href='{new_data['model_url']}' target='_blank'>Download Decimated 3D Model</a>"
    return (task_id, "COMPLETED", json.dumps(new_data, indent=2), model_link)

# --- GRADIO INTERFACE LAYOUT ---

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# 🌎 AI World Builder")
    
    # Hidden states to manage the live connection
    current_biome_id = gr.State(None)
    biome_choices_list = gr.State([])

    with gr.Tabs():
        # Asset Pipeline Tab (Unchanged)
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
                choices=[],  # Initially empty
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

        # Text to Image Tab
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
                
            task_id_txt2img = gr.Textbox(label="Submitted Task ID", interactive=False)
            
            text_to_image_prompt = gr.Textbox(label="Text Prompt", placeholder="Describe the biome, e.g., 'a serene crystal cave with luminous flora'")
            generate_image_button = gr.Button("🚀 Generate Image from Text")
            image_generation_status = gr.Textbox(label="Image Generation Status", lines=1)
            image_generation_output = gr.Gallery(label="Generated Images", columns=2, height='auto')
            
            # Event listeners
            text_to_image_db.change(fn=update_collections_dropdown, inputs=[text_to_image_db], outputs=[text_to_image_collection])
            text_to_image_collection.change(fn=update_biomes_dropdown, inputs=[text_to_image_db, text_to_image_collection], outputs=[existing_biome_dropdown_txt2img, biome_choices_list])
            biome_action_type_txt2img.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_txt2img,
                outputs=[new_biome_col_txt2img, existing_biome_col_txt2img]
            )
            generate_image_button.click(
                fn=run_2d_generation,
                inputs=[text_to_image_db, text_to_image_collection, biome_action_type_txt2img, new_biome_name_txt2img, existing_biome_dropdown_txt2img, text_to_image_prompt],
                outputs=[task_id_txt2img, image_generation_status, gr.Json(), image_generation_output, images_2d_gallery]
            )

        # Grid to Image Tab
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
            
            task_id_grid2img = gr.Textbox(label="Submitted Task ID", interactive=False)

            grid_data_input = gr.Textbox(label="Grid Data (JSON array of arrays)", lines=10, placeholder="Example: [[0,0,1,1],[0,1,1,0]]")
            generate_grid_image_button = gr.Button("Generate Image from Grid")
            grid_generation_status = gr.Textbox(label="Status", lines=1)
            grid_visualization_output = gr.Gallery(label="Grid Visualization", columns=2, height='auto')

            # Event listeners
            grid_to_image_db.change(fn=update_collections_dropdown, inputs=[grid_to_image_db], outputs=[grid_to_image_collection])
            grid_to_image_collection.change(fn=update_biomes_dropdown, inputs=[grid_to_image_db, grid_to_image_collection], outputs=[existing_biome_dropdown_grid2img, biome_choices_list])
            biome_action_type_grid2img.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_grid2img,
                outputs=[new_biome_col_grid2img, existing_biome_col_grid2img]
            )
            generate_grid_image_button.click(
                fn=run_grid_generation,
                inputs=[grid_to_image_db, grid_to_image_collection, biome_action_type_grid2img, new_biome_name_grid2img, existing_biome_dropdown_grid2img, grid_data_input],
                outputs=[task_id_grid2img, grid_generation_status, gr.Json(), grid_visualization_output, images_2d_gallery]
            )

        # 3D Generation Tab
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
            
            task_id_3d_gen = gr.Textbox(label="Submitted Task ID", interactive=False)

            input_2d_image_for_3d = gr.Image(label="Upload 2D Image", type="pil")
            generate_3d_button = gr.Button("Generate 3D Model")
            status_3d_gen = gr.Textbox(label="3D Generation Status", lines=1)
            output_3d_model_link = gr.HTML(label="Generated 3D Model Link")

            # Event listeners
            _3d_gen_db.change(fn=update_collections_dropdown, inputs=[_3d_gen_db], outputs=[_3d_gen_collection])
            _3d_gen_collection.change(fn=update_biomes_dropdown, inputs=[_3d_gen_db, _3d_gen_collection], outputs=[existing_biome_dropdown_3d, biome_choices_list])
            biome_action_type_3d.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_3d,
                outputs=[new_biome_col_3d, existing_biome_col_3d]
            )
            generate_3d_button.click(
                fn=run_3d_generation,
                inputs=[_3d_gen_db, _3d_gen_collection, biome_action_type_3d, new_biome_name_3d, existing_biome_dropdown_3d, input_2d_image_for_3d],
                outputs=[task_id_3d_gen, status_3d_gen, gr.Json(), output_3d_model_link]
            )

        # Decimated 3D Tab
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

            task_id_decimate = gr.Textbox(label="Submitted Task ID", interactive=False)
            
            input_3d_file_decimate = gr.File(label="Upload 3D Model (GLB, OBJ, STL)", type="filepath")
            decimate_button = gr.Button("Decimate 3D Model")
            status_decimate = gr.Textbox(label="Decimation Status", lines=1)
            output_decimated_model_link = gr.HTML(label="Decimated 3D Model Link")
            
            # Event listeners
            decimate_db.change(fn=update_collections_dropdown, inputs=[decimate_db], outputs=[decimate_collection])
            decimate_collection.change(fn=update_biomes_dropdown, inputs=[decimate_db, decimate_collection], outputs=[existing_biome_dropdown_decimate, biome_choices_list])
            biome_action_type_decimate.change(
                fn=lambda x: (gr.Column(visible=x=="Create New Biome"), gr.Column(visible=x=="Select Existing Biome")),
                inputs=biome_action_type_decimate,
                outputs=[new_biome_col_decimate, existing_biome_col_decimate]
            )
            decimate_button.click(
                fn=run_decimation,
                inputs=[decimate_db, decimate_collection, biome_action_type_decimate, new_biome_name_decimate, existing_biome_dropdown_decimate, input_3d_file_decimate],
                outputs=[task_id_decimate, status_decimate, gr.Json(), output_decimated_model_link]
            )

# Launch the Gradio application
demo.launch(server_name="0.0.0.0", server_port=7860)









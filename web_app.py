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

def run_2d_generation_live(doc_id, database_name, collection_name):
    """
    Simulates running the 2D image generation task for a given biome.
    """
    if not doc_id:
        return "Please select a biome first.", {}, []
    
    print(f"Starting 2D generation for document ID: {doc_id}")
    
    # Simulate a brief delay for the task
    time.sleep(2)
    
    biome_doc = fetch_live_biome_details(database_name, collection_name, doc_id)
    biome_name = biome_doc.get("biome_name", "Unknown Biome")
    
    # Generate new mock data for the completed step
    new_images = [
        f"https://placehold.co/400x400/222222/EEEEEE?text={biome_name}+Image+1",
        f"https://placehold.co/400x400/222222/EEEEEE?text={biome_name}+Image+2",
    ]
    new_data = {
        "status": "COMPLETED",
        "prompt": "high-resolution photo of a {} environment".format(biome_name),
        "model_used": "Simulated AI Model",
        "generated_images": new_images,
        "timestamp": time.time()
    }

    # Update the live database
    update_live_biome_details(database_name, collection_name, doc_id, "image_generation_details", new_data)
    
    return "COMPLETED", json.dumps(new_data, indent=2), new_images

def run_3d_generation_live(doc_id, database_name, collection_name):
    """
    Simulates running the 3D model generation task.
    """
    if not doc_id:
        return "Please select a biome first.", {}, ""

    print(f"Starting 3D generation for document ID: {doc_id}")
    
    # Simulate a brief delay
    time.sleep(3)
    
    biome_doc = fetch_live_biome_details(database_name, collection_name, doc_id)
    images_count = len(biome_doc.get("image_generation_details", {}).get("generated_images", []))

    new_data = {
        "status": "COMPLETED",
        "input_images_count": images_count,
        "model_url": "https://placehold.co/400x200/50C878/000000?text=New+3D+Model",
        "timestamp": time.time()
    }
    
    update_live_biome_details(database_name, collection_name, doc_id, "3d_generation_details", new_data)

    model_link = f"<a href='{new_data['model_url']}' target='_blank'>Download 3D Model</a>"
    return "COMPLETED", json.dumps(new_data, indent=2), new_images

# --- GRADIO INTERFACE LAYOUT ---

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# 🌎 AI World Builder")
    
    # Hidden states to manage the live connection
    current_biome_id = gr.State(None)
    biome_choices_list = gr.State([])

    with gr.TabItem("Database Settings"):
        gr.Markdown("## ⚙️ MongoDB Connection")
        gr.Markdown("First, select your database and collection. Then, click 'Refresh Biomes' to populate the dropdown on the 'Asset Pipeline' tab.")
        
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

    with gr.TabItem("Asset Pipeline"):
        gr.Markdown("## 📦 Asset Generation Pipeline")
        gr.Markdown("Select a biome to view and manage its asset generation status.")
        
        biome_dropdown = gr.Dropdown(
            label="Select Biome", 
            choices=[],  # Initially empty
            interactive=True
        )
        
        status_2d_output = gr.Textbox(label="2D Status", show_label=False)
        json_2d_output = gr.Json(label="2D Generation Details")
        images_2d_gallery = gr.Gallery(label="Generated 2D Images", columns=2)
        
        status_3d_output = gr.Textbox(label="3D Status", show_label=False)
        json_3d_output = gr.Json(label="3D Generation Details")
        model_link_output = gr.HTML(label="3D Model Link")
        
        generate_2d_button = gr.Button("Generate 2D Images")

        with gr.Accordion("2D Image Generation", open=True):
            gr.Markdown("### Status")
            # We no longer need to call .render() here, Gradio handles it automatically
            gr.Markdown("### Details (JSON)")
            # We no longer need to call .render() here, Gradio handles it automatically
            
        generate_3d_button = gr.Button("Generate 3D Model")

        with gr.Accordion("3D Model Generation", open=False):
            gr.Markdown("### Status")
            # We no longer need to call .render() here, Gradio handles it automatically
            gr.Markdown("### Details (JSON)")
            # We no longer need to call .render() here, Gradio handles it automatically
            
    # Event listeners
    database_dropdown.change(
        fn=update_collections_dropdown,
        inputs=[database_dropdown],
        outputs=[collection_dropdown]
    )

    refresh_button.click(
        fn=update_biomes_dropdown,
        inputs=[database_dropdown, collection_dropdown],
        outputs=[biome_dropdown, biome_choices_list]
    )
    
    # Update the biome pipeline when a biome is selected.
    # Pass the database and collection names to the function.
    biome_dropdown.change(
        fn=load_biome_pipeline_live,
        inputs=[biome_dropdown, biome_choices_list, database_dropdown, collection_dropdown],
        outputs=[current_biome_id, status_2d_output, json_2d_output, images_2d_gallery,
                 status_3d_output, json_3d_output, model_link_output]
    )
    
    # Update the 2D generation button click event.
    # Pass the database and collection names.
    generate_2d_button.click(
        fn=run_2d_generation_live,
        inputs=[current_biome_id, database_dropdown, collection_dropdown],
        outputs=[status_2d_output, json_2d_output, images_2d_gallery]
    )
    
    # Update the 3D generation button click event.
    # Pass the database and collection names.
    generate_3d_button.click(
        fn=run_3d_generation_live,
        inputs=[current_biome_id, database_dropdown, collection_dropdown],
        outputs=[status_3d_output, json_3d_output, model_link_output]
    )


# Launch the Gradio application
demo.launch(server_name="0.0.0.0", server_port=7860)







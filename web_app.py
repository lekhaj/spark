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

# This block of code simulates a MongoDB connection and data.
# In your actual application, you would replace this with your
# PyMongo client and database operations.
try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    # Try to connect to a mock MongoDB client to check for installation
    _ = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=1)
    mongo_client_available = True
except ImportError:
    print("Warning: pymongo is not installed. Using mock database instead.")
    mongo_client_available = False
except ConnectionFailure:
    print("Warning: Could not connect to a local MongoDB instance. Using mock database instead.")
    mongo_client_available = False

# --- MOCK DATABASE AND FUNCTIONS ---
# This dictionary simulates a MongoDB collection. Each key is the document's
# unique ObjectId, and the value is the document itself.
# This structure is necessary to correctly model the user's request for tracking
# assets by their unique ID.
MOCK_BIOMES_COLLECTION = {
    # Biome 1: Partially completed pipeline
    str(ObjectId()): {
        "biome_name": "Forest Glade",
        "description": "A tranquil glade with ancient trees and a small pond.",
        "image_generation_details": {
            "status": "COMPLETED",
            "prompt": "high-resolution photo of a dense, sun-dappled forest glade, tranquil, fantasy art style",
            "model_used": "Stable Diffusion XL",
            "generated_images": [
                "https://placehold.co/400x400/000000/FFFFFF?text=Forest+Image+1",
                "https://placehold.co/400x400/000000/FFFFFF?text=Forest+Image+2"
            ],
            "timestamp": time.time()
        },
        "3d_generation_details": {
            "status": "NOT_STARTED"
        }
    },
    # Biome 2: Fully completed pipeline
    str(ObjectId()): {
        "biome_name": "Crystal Cave",
        "description": "A glowing cave filled with bioluminescent crystals.",
        "image_generation_details": {
            "status": "COMPLETED",
            "prompt": "cinematic shot of a massive, glowing crystal cave, bioluminescence, surreal, deep colors",
            "model_used": "Midjourney 6",
            "generated_images": [
                "https://placehold.co/400x400/000000/FFFFFF?text=Cave+Image+1",
                "https://placehold.co/400x400/000000/FFFFFF?text=Cave+Image+2"
            ],
            "timestamp": time.time()
        },
        "3d_generation_details": {
            "status": "COMPLETED",
            "input_images_count": 2,
            "model_url": "https://placehold.co/400x200/50C878/000000?text=3D+Model+Link",
            "timestamp": time.time()
        }
    },
    # Biome 3: New biome, not started
    str(ObjectId()): {
        "biome_name": "Desert Oasis",
        "description": "A verdant oasis surrounded by endless sand dunes.",
        "image_generation_details": {
            "status": "NOT_STARTED"
        },
        "3d_generation_details": {
            "status": "NOT_STARTED"
        }
    }
}

def get_biome_choices():
    """
    Simulates fetching all biome names and their IDs from a MongoDB collection.
    Returns a list of tuples: [(biome_name, doc_id), ...].
    """
    choices = [(doc["biome_name"], doc_id) 
               for doc_id, doc in MOCK_BIOMES_COLLECTION.items() 
               if "biome_name" in doc]
    return choices

def get_biome_id_by_name(biome_name):
    """
    Simulates looking up a biome's document ID by its name.
    """
    for doc_id, doc in MOCK_BIOMES_COLLECTION.items():
        if doc.get("biome_name") == biome_name:
            return doc_id
    return None

def fetch_biome_details(doc_id):
    """
    Simulates fetching a single document from the database by its ID.
    Returns the document or None if not found.
    """
    return MOCK_BIOMES_COLLECTION.get(doc_id, None)

def update_biome_details(doc_id, section, new_data):
    """
    Simulates updating a specific section of a document in the database.
    This is a critical function for our pipeline logic.
    """
    if doc_id in MOCK_BIOMES_COLLECTION:
        MOCK_BIOMES_COLLECTION[doc_id][section] = new_data
        return True
    return False

# --- GRADIO UI FUNCTIONS ---

def load_biome_pipeline(biome_name):
    """
    This function is triggered when a biome is selected from the dropdown.
    It fetches the biome's details and populates the UI.
    """
    doc_id = get_biome_id_by_name(biome_name)
    if not doc_id:
        return (None, "Biome not found.", "", [], "Biome not found.", "", "")

    biome_doc = fetch_biome_details(doc_id)
    
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

    # Return all the new values to update the UI
    return (
        doc_id,
        status_2d_text,
        json_2d_text,
        images_2d_list,
        status_3d_text,
        json_3d_text,
        model_link
    )

def run_2d_generation(doc_id):
    """
    Simulates running the 2D image generation task for a given biome.
    This function would contain your actual API calls.
    """
    if not doc_id:
        return "Please select a biome first.", {}, []
    
    print(f"Starting 2D generation for document ID: {doc_id}")
    
    # Simulate a brief delay for the task
    time.sleep(2)
    
    # Get the current biome details
    biome_doc = fetch_biome_details(doc_id)
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

    # Update the mock database
    update_biome_details(doc_id, "image_generation_details", new_data)
    
    # Return the updated data to the UI
    return "COMPLETED", json.dumps(new_data, indent=2), new_images

def run_3d_generation(doc_id):
    """
    Simulates running the 3D model generation task.
    This function would contain your actual API calls for 3D generation.
    """
    if not doc_id:
        return "Please select a biome first.", {}, ""

    print(f"Starting 3D generation for document ID: {doc_id}")
    
    # Simulate a brief delay
    time.sleep(3)
    
    # Get the current biome details
    biome_doc = fetch_biome_details(doc_id)
    
    # Get the number of images generated in the previous step
    images_count = len(biome_doc.get("image_generation_details", {}).get("generated_images", []))

    # Generate new mock data for the completed step
    new_data = {
        "status": "COMPLETED",
        "input_images_count": images_count,
        "model_url": "https://placehold.co/400x200/50C878/000000?text=New+3D+Model",
        "timestamp": time.time()
    }
    
    # Update the mock database
    update_biome_details(doc_id, "3d_generation_details", new_data)

    # Return the updated data to the UI
    model_link = f"<a href='{new_data['model_url']}' target='_blank'>Download 3D Model</a>"
    return "COMPLETED", json.dumps(new_data, indent=2), model_link

# --- GRADIO INTERFACE LAYOUT ---

with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# 🌎 AI World Builder")

    # This is a hidden state that will hold the unique ID of the selected biome.
    # We use this ID to query the database, not the biome name, as IDs are unique.
    current_biome_id = gr.State(None)

    # Use a TabItem for the new "Asset Pipeline" page.
    with gr.TabItem("Asset Pipeline"):
        gr.Markdown("## 📦 Asset Generation Pipeline")
        gr.Markdown("Select a biome to view and manage its asset generation status.")
        
        with gr.Row():
            biome_names = [name for name, _ in get_biome_choices()]
            biome_dropdown = gr.Dropdown(
                label="Select Biome", 
                choices=biome_names, 
                value=biome_names[0] if biome_names else None,
                interactive=True
            )
            
        # The outputs from `load_biome_pipeline` will be connected here.
        # It's important to set up the output components before the function call.
        status_2d_output = gr.Textbox(label="2D Status", show_label=False)
        json_2d_output = gr.Json(label="2D Generation Details")
        images_2d_gallery = gr.Gallery(label="Generated 2D Images", columns=2)
        
        status_3d_output = gr.Textbox(label="3D Status", show_label=False)
        json_3d_output = gr.Json(label="3D Generation Details")
        model_link_output = gr.HTML(label="3D Model Link")
        
        # This button is used to re-trigger the 2D generation task.
        generate_2d_button = gr.Button("Generate 2D Images")

        # Use collapsible accordions to make the page cleaner.
        with gr.Accordion("2D Image Generation", open=True):
            gr.Markdown("### Status")
            # We no longer need to call .render() here
            gr.Textbox.render(status_2d_output)
            gr.Markdown("### Details (JSON)")
            # We no longer need to call .render() here
            gr.Json.render(json_2d_output)
            gr.Gallery.render(images_2d_gallery)
            gr.Button.render(generate_2d_button)

        # This button is used to re-trigger the 3D generation task.
        generate_3d_button = gr.Button("Generate 3D Model")

        with gr.Accordion("3D Model Generation", open=False):
            gr.Markdown("### Status")
            # We no longer need to call .render() here
            gr.Textbox.render(status_3d_output)
            gr.Markdown("### Details (JSON)")
            # We no longer need to call .render() here
            gr.Json.render(json_3d_output)
            gr.HTML.render(model_link_output)
            gr.Button.render(generate_3d_button)
        
        # Now, connect the functions to the UI events.
        # This call runs once to load the initial biome on page load.
        demo.load(
            fn=load_biome_pipeline,
            inputs=[biome_dropdown],
            outputs=[current_biome_id, status_2d_output, json_2d_output, images_2d_gallery,
                     status_3d_output, json_3d_output, model_link_output]
        )

        # This event triggers every time the user selects a new biome from the dropdown.
        biome_dropdown.change(
            fn=load_biome_pipeline,
            inputs=[biome_dropdown],
            outputs=[current_biome_id, status_2d_output, json_2d_output, images_2d_gallery,
                     status_3d_output, json_3d_output, model_link_output]
        )

        # This event triggers when the "Generate 2D" button is clicked.
        # We use `gr.Button.click` and pass the hidden `current_biome_id` as input.
        generate_2d_button.click(
            fn=run_2d_generation,
            inputs=[current_biome_id],
            outputs=[status_2d_output, json_2d_output, images_2d_gallery]
        )
        
        # This event triggers when the "Generate 3D" button is clicked.
        generate_3d_button.click(
            fn=run_3d_generation,
            inputs=[current_biome_id],
            outputs=[status_3d_output, json_3d_output, model_link_output]
        )

    # You can add your other existing tabs here.
    with gr.TabItem("Your Existing Tabs"):
        gr.Markdown("### Your Existing Gradio UI will go here.")
        gr.Markdown("This shows how the new Asset Pipeline tab fits into your existing app.")

# Launch the Gradio application
demo.launch(server_name="0.0.0.0", server_port=7860)



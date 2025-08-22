# web_app.py
import os
import json
import time
from bson.objectid import ObjectId
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import gradio as gr
from celery import Celery

# ----------------- Config -----------------
REDIS_IP = os.getenv("REDIS_IP", "172.31.8.113")
CONNECTION_STRING = os.getenv(
    "MONGO_URI",
    "mongodb://sagar:KrSiDnSI9m8RgcHE@ec2-15-206-99-66.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin"
)

celery_app = Celery(
    "app",
    broker=f"redis://{REDIS_IP}:6379/0",
    backend=f"redis://{REDIS_IP}:6379/0",
)

db_client = None

# ----------------- Mongo helpers -----------------
def get_db_client():
    global db_client
    if db_client is None:
        try:
            db_client = MongoClient(CONNECTION_STRING, serverSelectionTimeoutMS=5000)
            db_client.admin.command("ping")
            print("Successfully connected to MongoDB.")
        except ConnectionFailure as e:
            print(f"Could not connect to MongoDB: {e}")
            return None
    return db_client

def get_database_names():
    client = get_db_client()
    if not client:
        return []
    try:
        return client.list_database_names()
    except OperationFailure as e:
        print(f"Failed to list databases: {e}")
        return []

def get_collection_names(database_name):
    client = get_db_client()
    if not client or not database_name:
        return []
    try:
        return client[database_name].list_collection_names()
    except OperationFailure as e:
        print(f"Failed to list collections in '{database_name}': {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []

def get_biome_choices_live(database_name, collection_name):
    client = get_db_client()
    if not client or not database_name or not collection_name:
        return []
    try:
        docs = list(client[database_name][collection_name].find({}, {"biome_name": 1}))
        return [(doc.get("biome_name", "Unknown Biome"), str(doc["_id"])) for doc in docs]
    except OperationFailure as e:
        print(f"Failed to fetch biomes from '{collection_name}': {e}")
        return []

def fetch_live_biome_details(database_name, collection_name, doc_id):
    client = get_db_client()
    if not client or not doc_id:
        return None
    try:
        return client[database_name][collection_name].find_one({"_id": ObjectId(doc_id)})
    except Exception as e:
        print(f"Failed to fetch biome details for ID {doc_id}: {e}")
        return None

def create_new_biome(database_name, collection_name, biome_name):
    client = get_db_client()
    if not client or not database_name or not collection_name or not biome_name:
        return (None, "Failed to create biome. Please check inputs.")
    try:
        col = client[database_name][collection_name]
        if col.find_one({"biome_name": biome_name}):
            return (None, f"Biome '{biome_name}' already exists.")
        result = col.insert_one({
            "biome_name": biome_name,
            "status": "created",
            "image_generation_details": {},
            "grid_generation_details": {},
            "3d_generation_details": {},
            "decimation_details": {},
            "timestamp": time.time(),
        })
        return (str(result.inserted_id), "Biome created successfully!")
    except Exception as e:
        print(f"Failed to create new biome: {e}")
        return (None, f"Error creating biome: {e}")

# ----------------- UI helpers -----------------
def update_collections_dropdown(database_name):
    cols = get_collection_names(database_name)
    return gr.update(choices=cols, value=(cols[0] if cols else None))

def update_biomes_dropdown(database_name, collection_name):
    biome_choices = get_biome_choices_live(database_name, collection_name)
    biome_names = [name for name, _ in biome_choices]
    return gr.update(choices=biome_names, value=(biome_names[0] if biome_names else None)), biome_choices

def load_biome_pipeline_live(biome_name, biome_choices, database_name, collection_name):
    doc_id = next((_id for name, _id in biome_choices if name == biome_name), None)
    if not doc_id:
        return None, "Not Started", "{}", [], "Not Started", "{}", ""

    biome_doc = fetch_live_biome_details(database_name, collection_name, doc_id)
    if not biome_doc:
        return doc_id, "Not Started", "{}", [], "Not Started", "{}", ""

    # 2D
    details_2d = biome_doc.get("image_generation_details", {}) or {}
    status_2d_text = details_2d.get("status", "Not Started")
    try:
        json_2d_text = json.dumps(details_2d, indent=2)
    except TypeError:
        json_2d_text = "Error: Invalid JSON data"
    images_2d_list = details_2d.get("generated_images", []) if isinstance(details_2d.get("generated_images", []), list) else []

    # 3D
    details_3d = biome_doc.get("3d_generation_details", {}) or {}
    status_3d_text = details_3d.get("status", "Not Started")
    try:
        json_3d_text = json.dumps(details_3d, indent=2)
    except TypeError:
        json_3d_text = "Error: Invalid JSON data"
    model_link = ""
    model_url = details_3d.get("model_url")
    if status_3d_text == "COMPLETED" and model_url:
        model_link = f"<a href='{model_url}' target='_blank'>Download 3D Model</a>"

    return doc_id, status_2d_text, json_2d_text, images_2d_list, status_3d_text, json_3d_text, model_link

def get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name):
    if biome_action_type == "Create New Biome":
        doc_id, msg = create_new_biome(database_name, collection_name, new_biome_name)
        if not doc_id:
            return None, msg
    else:
        doc_id = next((_id for name, _id in biome_choices if name == selected_biome_name), None)
        if not doc_id:
            return None, "Selected biome not found."
    return doc_id, None

# ----------------- Task starters -----------------
def _start_2d_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, prompt):
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return None, msg, {}

    initial_data = {
        "status": "PENDING",
        "prompt": prompt,
        "model_used": "runwayml/stable-diffusion-v1-5",
        "generated_images": [],
        "timestamp": time.time(),
    }
    client = get_db_client()
    client[database_name][collection_name].update_one({"_id": ObjectId(doc_id)}, {"$set": {"image_generation_details": initial_data}})

    celery_app.send_task("app.generate_2d_image_task", args=[CONNECTION_STRING, database_name, collection_name, doc_id, prompt, 512, 512, 1])
    return doc_id, "Task submitted: PENDING", initial_data

def run_2d_generation(task_id_input, database_name, collection_name):
    if not task_id_input:
        return {}, [], "No task to run."
    biome_doc = fetch_live_biome_details(database_name, collection_name, task_id_input)
    if not biome_doc:
        return {}, [], "Document not found."
    details_2d = biome_doc.get("image_generation_details", {}) or {}
    status_2d = details_2d.get("status", "Not Started")
    images_2d = details_2d.get("generated_images", [])
    return details_2d, images_2d, f"Task Status: {status_2d}"

def _start_grid_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, grid_data_str):
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return None, msg, {}
    try:
        parsed = json.loads(grid_data_str)
    except Exception as e:
        return doc_id, f"Invalid grid JSON: {e}", {}

    initial_data = {
        "status": "PENDING",
        "grid_data": parsed,
        "model_used": "grid-to-image",
        "generated_images": [],
        "timestamp": time.time(),
    }
    client = get_db_client()
    client[database_name][collection_name].update_one({"_id": ObjectId(doc_id)}, {"$set": {"grid_generation_details": initial_data}})

    celery_app.send_task("app.generate_image_from_grid_task", args=[CONNECTION_STRING, database_name, collection_name, doc_id, grid_data_str, 512, 512, 1])
    return doc_id, "Task submitted: PENDING", initial_data

def run_grid_generation(task_id_input, database_name, collection_name):
    if not task_id_input:
        return {}, [], "No task to run."
    biome_doc = fetch_live_biome_details(database_name, collection_name, task_id_input)
    if not biome_doc:
        return {}, [], "Document not found."
    details = biome_doc.get("grid_generation_details", {}) or {}
    status = details.get("status", "Not Started")
    imgs = details.get("generated_images", [])
    return details, imgs, f"Task Status: {status}"

def _start_3d_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices):
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return None, msg, ""
    initial_data = {
        "status": "PENDING",
        "input_images_count": 1,
        "model_url": "",
        "timestamp": time.time(),
    }
    client = get_db_client()
    client[database_name][collection_name].update_one({"_id": ObjectId(doc_id)}, {"$set": {"3d_generation_details": initial_data}})
    celery_app.send_task("app.generate_3d_from_2d_task", args=[CONNECTION_STRING, database_name, collection_name, doc_id])
    return doc_id, "Task submitted: PENDING", ""

def run_3d_generation(task_id_input, database_name, collection_name):
    if not task_id_input:
        return "", "No task to run."
    biome_doc = fetch_live_biome_details(database_name, collection_name, task_id_input)
    if not biome_doc:
        return "", "Document not found."
    details = biome_doc.get("3d_generation_details", {}) or {}
    status = details.get("status", "Not Started")
    model_url = details.get("model_url")
    link = f"<a href='{model_url}' target='_blank'>Download 3D Model</a>" if model_url else ""
    return link, f"Task Status: {status}"

def _start_decimation_task(database_name, collection_name, biome_action_type, new_biome_name, selected_biome_name, biome_choices, input_3d_file):
    doc_id, msg = get_or_create_biome_doc(biome_action_type, new_biome_name, selected_biome_name, biome_choices, database_name, collection_name)
    if not doc_id:
        return None, msg, ""
    if not input_3d_file:
        return None, "Please upload a 3D model.", ""

    initial_data = {
        "status": "PENDING",
        "input_file": os.path.basename(input_3d_file),
        "model_url": "",
        "timestamp": time.time(),
    }
    client = get_db_client()
    client[database_name][collection_name].update_one({"_id": ObjectId(doc_id)}, {"$set": {"decimation_details": initial_data}})

    # NOTE: input_3d_file is already a filepath (because type="filepath")
    with open(input_3d_file, "rb") as f:
        file_bytes = f.read()
    celery_app.send_task("app.decimate_3d_task", args=[CONNECTION_STRING, database_name, collection_name, doc_id, os.path.basename(input_3d_file), file_bytes])
    return doc_id, "Task submitted: PENDING", ""

def run_decimation(task_id_input, database_name, collection_name):
    if not task_id_input:
        return "", "No task to run."
    biome_doc = fetch_live_biome_details(database_name, collection_name, task_id_input)
    if not biome_doc:
        return "", "Document not found."
    details = biome_doc.get("decimation_details", {}) or {}
    status = details.get("status", "Not Started")
    model_url = details.get("model_url")
    link = f"<a href='{model_url}' target='_blank'>Download Decimated 3D Model</a>" if model_url else ""
    return link, f"Task Status: {status}"

# ----------------- UI -----------------
with gr.Blocks(title="AI-Powered 3D Asset Generator") as demo:
    gr.Markdown("# 🌎 AI World Builder")

    biome_choices_list = gr.State([])

    with gr.Tabs():
        with gr.TabItem("Asset Pipeline"):
            gr.Markdown("## ⚙️ MongoDB Connection")
            with gr.Row():
                database_dropdown = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                collection_dropdown = gr.Dropdown(label="Select Collection", choices=[], interactive=True)
                refresh_button = gr.Button("Refresh Biomes")

            gr.Markdown("## 📦 Asset Generation Pipeline")
            biome_dropdown = gr.Dropdown(label="Select Biome", choices=[], interactive=True)

            with gr.Accordion("2D Image Generation", open=True):
                status_2d_output = gr.Textbox(label="2D Status", show_label=False)
                json_2d_output = gr.Json(label="2D Generation Details")
                images_2d_gallery = gr.Gallery(label="Generated 2D Images", columns=2)

            with gr.Accordion("3D Model Generation", open=False):
                status_3d_output = gr.Textbox(label="3D Status", show_label=False)
                json_3d_output = gr.Json(label="3D Generation Details")
                model_link_output = gr.HTML(label="3D Model Link")

            database_dropdown.change(update_collections_dropdown, [database_dropdown], [collection_dropdown])
            collection_dropdown.change(update_biomes_dropdown, [database_dropdown, collection_dropdown], [biome_dropdown, biome_choices_list])
            refresh_button.click(update_biomes_dropdown, [database_dropdown, collection_dropdown], [biome_dropdown, biome_choices_list])

            biome_dropdown.change(
                load_biome_pipeline_live,
                [biome_dropdown, biome_choices_list, database_dropdown, collection_dropdown],
                [gr.State(), status_2d_output, json_2d_output, images_2d_gallery, status_3d_output, json_3d_output, model_link_output],
            )

        with gr.TabItem("Text to Image"):
            with gr.Row():
                text_to_image_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                text_to_image_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            biome_action_type_txt2img = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")
            with gr.Column(visible=True) as new_biome_col_txt2img:
                new_biome_name_txt2img = gr.Textbox(label="New Biome Name")
            with gr.Column(visible=False) as existing_biome_col_txt2img:
                existing_biome_dropdown_txt2img = gr.Dropdown(label="Select Biome", choices=[], interactive=True)

            task_status_2d = gr.Textbox(label="Task Status", interactive=False)
            task_id_2d = gr.State()

            text_to_image_prompt = gr.Textbox(label="Text Prompt", placeholder="Describe the biome...")
            generate_image_button = gr.Button("🚀 Generate Image from Text")
            check_2d_status_button = gr.Button("Run Task & Refresh")

            json_2d_results = gr.Json(label="Generation Results")
            images_2d_results = gr.Gallery(label="Generated Images")

            text_to_image_db.change(update_collections_dropdown, [text_to_image_db], [text_to_image_collection])
            text_to_image_collection.change(update_biomes_dropdown, [text_to_image_db, text_to_image_collection], [existing_biome_dropdown_txt2img, biome_choices_list])
            biome_action_type_txt2img.change(
                lambda x: (gr.update(visible=x == "Create New Biome"), gr.update(visible=x == "Select Existing Biome")),
                inputs=biome_action_type_txt2img,
                outputs=[new_biome_col_txt2img, existing_biome_col_txt2img],
            )

            generate_image_button.click(
                _start_2d_task,
                [text_to_image_db, text_to_image_collection, biome_action_type_txt2img, new_biome_name_txt2img, existing_biome_dropdown_txt2img, biome_choices_list, text_to_image_prompt],
                [task_id_2d, task_status_2d, json_2d_results],
            )
            check_2d_status_button.click(
                run_2d_generation,
                [task_id_2d, text_to_image_db, text_to_image_collection],
                [json_2d_results, images_2d_results, task_status_2d],
            )

        with gr.TabItem("Grid to Image"):
            with gr.Row():
                grid_to_image_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                grid_to_image_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            biome_action_type_grid2img = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")
            with gr.Column(visible=True) as new_biome_col_grid2img:
                new_biome_name_grid2img = gr.Textbox(label="New Biome Name")
            with gr.Column(visible=False) as existing_biome_col_grid2img:
                existing_biome_dropdown_grid2img = gr.Dropdown(label="Select Biome", choices=[], interactive=True)

            task_status_grid = gr.Textbox(label="Task Status", interactive=False)
            task_id_grid = gr.State()

            grid_data_input = gr.Textbox(label="Grid Data (JSON array of arrays)", lines=10, placeholder="Example: [[0,0,1,1],[0,1,1,0]]")
            generate_grid_image_button = gr.Button("Generate Image from Grid")
            check_grid_status_button = gr.Button("Run Task & Refresh")

            json_grid_results = gr.Json(label="Generation Results")
            images_grid_results = gr.Gallery(label="Generated Images")

            grid_to_image_db.change(update_collections_dropdown, [grid_to_image_db], [grid_to_image_collection])
            grid_to_image_collection.change(update_biomes_dropdown, [grid_to_image_db, grid_to_image_collection], [existing_biome_dropdown_grid2img, biome_choices_list])
            biome_action_type_grid2img.change(
                lambda x: (gr.update(visible=x == "Create New Biome"), gr.update(visible=x == "Select Existing Biome")),
                inputs=biome_action_type_grid2img,
                outputs=[new_biome_col_grid2img, existing_biome_col_grid2img],
            )

            generate_grid_image_button.click(
                _start_grid_task,
                [grid_to_image_db, grid_to_image_collection, biome_action_type_grid2img, new_biome_name_grid2img, existing_biome_dropdown_grid2img, biome_choices_list, grid_data_input],
                [task_id_grid, task_status_grid, json_grid_results],
            )
            check_grid_status_button.click(
                run_grid_generation,
                [task_id_grid, grid_to_image_db, grid_to_image_collection],
                [json_grid_results, images_grid_results, task_status_grid],
            )

        with gr.TabItem("3D Generation"):
            with gr.Row():
                _3d_gen_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                _3d_gen_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            biome_action_type_3d = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")
            with gr.Column(visible=True) as new_biome_col_3d:
                new_biome_name_3d = gr.Textbox(label="New Biome Name")
            with gr.Column(visible=False) as existing_biome_col_3d:
                existing_biome_dropdown_3d = gr.Dropdown(label="Select Biome", choices=[], interactive=True)

            task_status_3d = gr.Textbox(label="Task Status", interactive=False)
            task_id_3d = gr.State()

            # (UI allows upload, but the provided backend 3D task is a stub)
            input_2d_image_for_3d = gr.Image(label="Upload 2D Image", type="pil")
            generate_3d_button = gr.Button("Generate 3D Model")
            check_3d_status_button = gr.Button("Run Task & Refresh")
            model_link_3d_results = gr.HTML(label="3D Model Link")

            _3d_gen_db.change(update_collections_dropdown, [_3d_gen_db], [_3d_gen_collection])
            _3d_gen_collection.change(update_biomes_dropdown, [_3d_gen_db, _3d_gen_collection], [existing_biome_dropdown_3d, biome_choices_list])
            biome_action_type_3d.change(
                lambda x: (gr.update(visible=x == "Create New Biome"), gr.update(visible=x == "Select Existing Biome")),
                inputs=biome_action_type_3d,
                outputs=[new_biome_col_3d, existing_biome_col_3d],
            )
            generate_3d_button.click(
                _start_3d_task,
                [_3d_gen_db, _3d_gen_collection, biome_action_type_3d, new_biome_name_3d, existing_biome_dropdown_3d, biome_choices_list],
                [task_id_3d, task_status_3d, model_link_3d_results],
            )
            check_3d_status_button.click(
                run_3d_generation,
                [task_id_3d, _3d_gen_db, _3d_gen_collection],
                [model_link_3d_results, task_status_3d],
            )

        with gr.TabItem("Decimated 3D"):
            with gr.Row():
                decimate_db = gr.Dropdown(label="Select Database", choices=get_database_names(), interactive=True)
                decimate_collection = gr.Dropdown(label="Select Collection", choices=[], interactive=True)

            biome_action_type_decimate = gr.Radio(choices=["Create New Biome", "Select Existing Biome"], value="Create New Biome", label="Biome Action")
            with gr.Column(visible=True) as new_biome_col_decimate:
                new_biome_name_decimate = gr.Textbox(label="New Biome Name")
            with gr.Column(visible=False) as existing_biome_col_decimate:
                existing_biome_dropdown_decimate = gr.Dropdown(label="Select Biome", choices=[], interactive=True)

            task_status_decimate = gr.Textbox(label="Task Status", interactive=False)
            task_id_decimate = gr.State()

            input_3d_file_decimate = gr.File(label="Upload 3D Model (GLB, OBJ, STL)", type="filepath")
            decimate_button = gr.Button("Decimate 3D Model")
            check_decimate_status_button = gr.Button("Run Task & Refresh")
            model_link_decimate_results = gr.HTML(label="Decimated 3D Model Link")

            decimate_db.change(update_collections_dropdown, [decimate_db], [decimate_collection])
            decimate_collection.change(update_biomes_dropdown, [decimate_db, decimate_collection], [existing_biome_dropdown_decimate, biome_choices_list])
            biome_action_type_decimate.change(
                lambda x: (gr.update(visible=x == "Create New Biome"), gr.update(visible=x == "Select Existing Biome")),
                inputs=biome_action_type_decimate,
                outputs=[new_biome_col_decimate, existing_biome_col_decimate],
            )

            decimate_button.click(
                _start_decimation_task,
                [decimate_db, decimate_collection, biome_action_type_decimate, new_biome_name_decimate, existing_biome_dropdown_decimate, biome_choices_list, input_3d_file_decimate],
                [task_id_decimate, task_status_decimate, model_link_decimate_results],
            )
            check_decimate_status_button.click(
                run_decimation,
                [task_id_decimate, decimate_db, decimate_collection],
                [model_link_decimate_results, task_status_decimate],
            )

demo.launch(server_name="0.0.0.0", server_port=7860)

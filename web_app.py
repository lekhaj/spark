import gradio as gr
import pymongo
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import uuid
import time
import os
import random

# --- MongoDB Connection and Operations ---
# It's a good practice to use environment variables for sensitive data like connection strings.
# For this example, we'll use a placeholder.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = "gradio_app_db"
COLLECTION_NAME = "generation_tasks"

# Function to check MongoDB connection and return a status
def get_mongo_status():
    """Checks the MongoDB connection and returns a status message."""
    try:
        # Use a client with a timeout to prevent the app from hanging
        client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return "✅ Connected to MongoDB"
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        return f"❌ Connection to MongoDB failed: {e}"
    except Exception as e:
        return f"❌ An error occurred: {e}"

def insert_task(task_id, task_type, file_path):
    """
    Inserts a new task document into the MongoDB collection.
    
    Args:
        task_id (str): The unique ID for the task.
        task_type (str): The type of generation task (e.g., '2D', '3D').
        file_path (str): The path to the generated file.
    """
    try:
        client = pymongo.MongoClient(MONGO_URI)
        db = client[DB_NAME]
        collection = db[COLLECTION_NAME]
        
        task_data = {
            "task_id": task_id,
            "task_type": task_type,
            "file_path": file_path,
            "status": "completed",
            "timestamp": time.time()
        }
        collection.insert_one(task_data)
    except Exception as e:
        # In a real app, you might log this error instead of printing
        print(f"Failed to insert task into MongoDB: {e}")

# --- Core Generation Function (Mock) ---
def generate_file(generation_type):
    """
    A mock function to simulate a file generation task.
    In a real app, this would trigger your actual generation process.
    """
    # Simulate a long-running process
    time.sleep(random.uniform(2, 5))
    
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Create a dummy file path
    file_name = f"{task_id}_{generation_type}_output.txt"
    file_path = os.path.join("generated_files", file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, "w") as f:
        f.write(f"This is a generated file for task ID: {task_id}\n")
        f.write(f"Type: {generation_type}\n")
    
    # Insert task into MongoDB
    insert_task(task_id, generation_type, file_path)
    
    # Return the task ID and a success message
    return f"Task completed successfully! Your Task ID is:", task_id

# --- Gradio UI Layout with gr.Blocks ---
with gr.Blocks(title="Generative Application") as demo:
    # Header Section
    gr.Markdown(
        """
        # 🤖 Generative Art & Model App
        Welcome! Use the options below to generate a new file and track your tasks.
        """
    )

    # MongoDB Status and Generation Section
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### MongoDB Status")
            # This Markdown component will be updated with the connection status
            mongo_status_text = gr.Markdown(get_mongo_status())
            gr.Markdown("---")
            gr.Markdown("### Generation Options")
            generation_type = gr.Radio(
                ["2D", "3D", "Decimated"],
                label="Select Generation Type",
                value="2D"
            )
            generate_btn = gr.Button("Generate File", variant="primary")

        with gr.Column(scale=2):
            gr.Markdown("### Task Output & Status")
            with gr.Column():
                # A place to display the loading spinner and messages
                status_message_md = gr.Markdown("Ready to generate...", label="Status")
                
                # These components will display the task ID after completion
                task_id_label = gr.Markdown(visible=False)
                task_id_output = gr.Textbox(
                    label="Generated Task ID",
                    interactive=False,
                    visible=False
                )

    # Event Handlers
    @generate_btn.click(
        fn=generate_file,
        inputs=generation_type,
        outputs=[task_id_label, task_id_output]
    )
    def update_ui(generation_type):
        """
        Wrapper function to handle UI updates before and after the main function.
        """
        # Show loading message
        yield gr.Markdown.update(value="Generating file, please wait...", visible=True), gr.Markdown.update(visible=False), gr.Textbox.update(visible=False)
        
        # Run the generation function
        message, task_id = generate_file(generation_type)
        
        # Update UI with the final output
        yield gr.Markdown.update(value=message, visible=True), gr.Markdown.update(value="Your Task ID is:", visible=True), gr.Textbox.update(value=task_id, visible=True)

# Launch the app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

import gradio as gr
import json
from app.services.mongo_service import get_biome_choices_live, get_biome, update_or_add_biome_asset
from app.config import settings

def biome_editor_ui():
    """Simple biome editor tab for fetching and updating biome JSON documents"""
    
    with gr.Blocks() as biome_editor:
        gr.Markdown("##  Biome Editor")
        gr.Markdown("Fetch biome documents from MongoDB, edit the JSON, and update them back to the database.")
        
        # Biome selection
        with gr.Row():
            biome_dropdown = gr.Dropdown(
                label="Select Biome",
                choices=[],
                value=None,
                interactive=True
            )
            
        # Update controls
        with gr.Row():
            refresh_biomes_btn = gr.Button("Refresh Biomes")
            fetch_btn = gr.Button("Fetch Biome JSON")
            update_btn = gr.Button("Update Biome in MongoDB")
        
        # JSON editing area
        biome_json_editor = gr.Textbox(
            label="Biome JSON (Editable)",
            lines=20,
            interactive=True,
            placeholder="Select a biome to load its JSON document...",
            show_copy_button=True
        )
        
        
        
        # Status output
        status_output = gr.Textbox(
            label="Status",
            interactive=False,
            placeholder="Operation status will appear here..."
        )
        
        # Helper functions
        def refresh_biome_choices():
            """Refresh the biome dropdown choices"""
            try:
                biome_choices = get_biome_choices_live(settings.MONGODB_DB_NAME, "biomes")
                if not biome_choices:
                    return gr.update(choices=[("No biomes available", "none")])
                return gr.update(
                    choices=[(name, _id) for name, _id in biome_choices],
                    value=biome_choices[0][1] if biome_choices else None
                )
            except Exception as e:
                print(f"Error refreshing biome choices: {e}")
                return gr.update(choices=[("Error loading biomes", "none")])
        
        def fetch_biome_json(biome_id):
            """Fetch and return biome JSON as editable string"""
            if not biome_id or biome_id == "none":
                return "", "Please select a valid biome"
            
            try:
                biome_doc = get_biome(biome_id)
                if not biome_doc:
                    return "", "Biome not found in database"
                
                # Convert to formatted JSON string
                json_str = json.dumps(biome_doc, indent=2, ensure_ascii=False)
                return json_str, "✅ Biome JSON loaded successfully"
                
            except Exception as e:
                error_msg = f"Error fetching biome: {e}"
                print(error_msg)
                return "", error_msg
        
        def update_biome_json(biome_id, json_text):
            """Update the biome document with edited JSON"""
            if not biome_id or biome_id == "none":
                return "Please select a biome first"
            
            if not json_text.strip():
                return "No JSON content to update"
            
            try:
                # Parse and validate JSON
                updated_doc = json.loads(json_text)
                
                # Update the entire biome document
                # Since update_or_add_biome_asset is for specific assets, we need a different approach
                # We'll update the entire document by replacing it
                from app.services.mongo_service import get_db
                
                db = get_db()
                if db is None:
                    return "❌ Database connection unavailable"
                
                collection = db["biomes"]
                
                # Replace the entire document
                result = collection.replace_one(
                    {"_id": biome_id},
                    updated_doc
                )
                
                if result.modified_count > 0:
                    return "✅ Biome updated successfully in MongoDB"
                else:
                    return "⚠️ No changes made (document may be identical)"
                
            except json.JSONDecodeError as e:
                return f"❌ Invalid JSON format: {e}"
            except Exception as e:
                error_msg = f"❌ Error updating biome: {e}"
                print(error_msg)
                return error_msg
        
        # Event handlers
        refresh_biomes_btn.click(
            fn=refresh_biome_choices,
            inputs=[],
            outputs=[biome_dropdown]
        )
        
        fetch_btn.click(
            fn=fetch_biome_json,
            inputs=[biome_dropdown],
            outputs=[biome_json_editor, status_output]
        )
        
        update_btn.click(
            fn=update_biome_json,
            inputs=[biome_dropdown, biome_json_editor],
            outputs=[status_output]
        )
        
        # Load initial biome choices
        biome_editor.load(
            fn=refresh_biome_choices,
            inputs=[],
            outputs=[biome_dropdown]
        )
    
    return biome_editor
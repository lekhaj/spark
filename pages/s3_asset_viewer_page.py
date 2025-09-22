
import gradio as gr
from bson.objectid import ObjectId
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from db_utils import get_db_client, MONGO_DB


def s3_asset_viewer_ui():

    with gr.Blocks() as s3_asset_viewer:
        gr.Markdown("## 🔍 S3 Asset Image Viewer")
        gr.Markdown("Select a biome to view all images, or enter an asset name to filter images for that asset.")


        # Import directly from db_utils to avoid circular import
        from db_utils import get_biome_choices_live

        try:
            biome_choices = get_biome_choices_live(MONGO_DB, "biomes") if get_biome_choices_live else []
        except Exception as e:
            print(f"[S3 Asset Viewer] Error fetching biomes: {e}")
            biome_choices = []

        # Only show dropdown if there are selectable biomes
        if biome_choices:
            biome_dropdown = gr.Dropdown(choices=[(name, _id) for name, _id in biome_choices], label="Biome Name", interactive=True, value=biome_choices[0][1])
        else:
            biome_dropdown = gr.Dropdown(choices=[], label="Biome Name", interactive=False, visible=False)
        refresh_biomes_button = gr.Button("Refresh Biome List")
        asset_name_box = gr.Textbox(label="Asset Name (optional)", placeholder="Enter asset name (e.g., steppe_watchtower)")
        fetch_s3_button = gr.Button("Show Images")
        s3_image_uri_output = gr.Textbox(label="Image URI(s)", interactive=False)
        s3_image_display = gr.Gallery(label="Image Gallery", show_label=True, allow_preview=True, columns=3, rows=2, height=400)

        def s3_to_https(uri):
            if isinstance(uri, str) and uri.startswith("s3://"):
                parts = uri[5:].split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    return f"https://{bucket}.s3.amazonaws.com/{key}"
            return uri

        def find_all_s3_uris(obj, asset_name=None, parent_key=None):
            uris = []
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # If asset_name is provided, only match keys that match asset_name
                    if asset_name and asset_name.strip() and str(k) != asset_name.strip():
                        # But still search deeper in case asset_name is nested
                        uris.extend(find_all_s3_uris(v, asset_name, k))
                        continue
                    if k == "s3_image_uri" and isinstance(v, str) and v:
                        uris.append(s3_to_https(v))
                    else:
                        uris.extend(find_all_s3_uris(v, asset_name, k))
            elif isinstance(obj, list):
                for item in obj:
                    uris.extend(find_all_s3_uris(item, asset_name, parent_key))
            return uris

        def fetch_images(biome_id, asset_name):
            try:
                client = get_db_client()
                if not client or not biome_id or biome_id == "none":
                    return ("No biome selected or DB unavailable.", None)
                db = client[MONGO_DB]
                collection = db["biomes"]
                biome_doc = collection.find_one({"_id": biome_id})
                if not biome_doc:
                    return ("Biome not found", None)
                # Recursively find all s3_image_uri in the document
                all_uris = find_all_s3_uris(biome_doc, asset_name)
                if all_uris:
                    return ("\n".join(all_uris), all_uris)
                else:
                    return ("No images found for this biome/asset.", None)
            except Exception as e:
                print(f"[S3 Asset Viewer] Error fetching images: {e}")
                return (f"Error: {e}", None)

        fetch_s3_button.click(
            fn=fetch_images,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[s3_image_uri_output, s3_image_display]
        )

        def refresh_biome_choices():
            from db_utils import get_biome_choices_live
            biome_choices = get_biome_choices_live(MONGO_DB, "biomes")
            if not biome_choices:
                biome_choices = [("No biomes available", "none")]
            return gr.update(choices=[(name, _id) for name, _id in biome_choices], value=biome_choices[0][1] if biome_choices else None)

        refresh_biomes_button.click(
            fn=refresh_biome_choices,
            inputs=[],
            outputs=[biome_dropdown]
        )
    return s3_asset_viewer


import gradio as gr
from bson.objectid import ObjectId
from app.services.mongo_service import get_db, get_biome_choices_live, get_biome_asset_update_key, get_biome
try:
    from app.config import settings
except ModuleNotFoundError:
    import sys, os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from app.config import settings


def s3_asset_viewer_ui():

    with gr.Blocks() as s3_asset_viewer:
        gr.Markdown("## 🔍 S3 Asset Image Viewer")
        gr.Markdown("Select a biome to view all images, or enter an asset name to filter images for that asset.")


        try:
            biome_choices = get_biome_choices_live(settings.MONGODB_DB_NAME, "biomes") if get_biome_choices_live else []
        except Exception as e:
            print(f"[S3 Asset Viewer] Error fetching biomes: {e}")
            biome_choices = []

        # print("[DEBUG] biome_choices:", biome_choices)
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
            keys = ["s3_image_uri", "image_url"]
            def collect_all_images(subtree):
                images = []
                if isinstance(subtree, dict):
                    for k, v in subtree.items():
                        if k in keys and isinstance(v, str) and v:
                            images.append(s3_to_https(v))
                        else:
                            images.extend(collect_all_images(v))
                elif isinstance(subtree, list):
                    for item in subtree:
                        images.extend(collect_all_images(item))
                return images

            if isinstance(obj, dict):
                for k, v in obj.items():
                    # If asset_name is provided and matches this key (case-insensitive, stripped), collect all images under this dict (recursively)
                    if asset_name and asset_name.strip() and str(k).strip().lower() == asset_name.strip().lower():
                        uris.extend(collect_all_images(v))
                        continue
                    # Otherwise, keep searching
                    if k in keys and isinstance(v, str) and v:
                        uris.append(s3_to_https(v))
                    else:
                        uris.extend(find_all_s3_uris(v, asset_name, k))
            elif isinstance(obj, list):
                for item in obj:
                    uris.extend(find_all_s3_uris(item, asset_name, parent_key))
            return uris

        def fetch_images(biome_id, asset_name):
            try:
                db = get_db()
                if db is None or not biome_id or biome_id == "none":
                    return ("No biome selected or DB unavailable.", None)
                # If asset_name is provided, try to get only that asset dict
                if asset_name and asset_name.strip():
                    update_key = get_biome_asset_update_key(biome_id, asset_name.strip())
                    if not update_key:
                        return (f"Asset '{asset_name}' not found in biome.", None)
                    # Fetch the asset dict directly
                    biome_doc = get_biome(biome_id)
                    # Traverse to the asset dict using the update_key
                    def get_nested(d, path):
                        for k in path:
                            if isinstance(d, dict):
                                d = d.get(k, {})
                            else:
                                return {}
                        return d if isinstance(d, dict) else {}
                    asset_path = update_key.split('.')
                    asset_dict = get_nested(biome_doc, asset_path)
                    if not asset_dict:
                        return (f"Asset '{asset_name}' not found in biome.", None)
                    all_uris = find_all_s3_uris(asset_dict)
                else:
                    # No asset name: search whole biome
                    db = get_db()
                    collection = db["biomes"]
                    biome_doc = collection.find_one({"_id": biome_id})
                    if not biome_doc:
                        return ("Biome not found", None)
                    all_uris = find_all_s3_uris(biome_doc)
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
            biome_choices = get_biome_choices_live(settings.MONGODB_DB_NAME, "biomes")
            if not biome_choices:
                biome_choices = [("No biomes available", "none")]
            return gr.update(choices=[(name, _id) for name, _id in biome_choices], value=biome_choices[0][1] if biome_choices else None)

        refresh_biomes_button.click(
            fn=refresh_biome_choices,
            inputs=[],
            outputs=[biome_dropdown]
        )
    return s3_asset_viewer

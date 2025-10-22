
import gradio as gr
import json
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
        with gr.Row():
            if biome_choices:
                biome_dropdown = gr.Dropdown(choices=[(name, _id) for name, _id in biome_choices], label="Biome Name", interactive=True, value=biome_choices[0][1])
            else:
                biome_dropdown = gr.Dropdown(choices=[], label="Biome Name", interactive=False, visible=False)
            asset_name_box = gr.Textbox(label="Asset Name (optional)", placeholder="Enter asset name (steppe_watchtower)")
            
        refresh_biomes_button = gr.Button("Refresh Biome List")

        # Image URL Lookup Accordion
        with gr.Accordion("Image URL Lookup", open=True):
            asset_name_box = gr.Textbox(label="Asset Name (optional)", placeholder="Enter asset name (e.g., steppe_watchtower)")
            fetch_s3_button = gr.Button("Show Images")
            s3_image_uri_output = gr.Textbox(label="Image URI(s)", interactive=False)
            s3_image_display = gr.Gallery(label="Image Gallery", show_label=True, allow_preview=True, columns=3, rows=2, height=400)

        # Biome JSON Accordion (structured/collapsible)
        with gr.Accordion("Biome JSON", open=False):
            fetch_biome_json_btn = gr.Button("Fetch Biome JSON")
            biome_json_structured = gr.Json(label="Biome JSON (structured)")

        # 3D Assets Accordion
        with gr.Accordion("3D Assets (status = 3d_generated)", open=False):
            fetch_3d_btn = gr.Button("Fetch 3D Asset URLs")
            three_d_list_output = gr.Textbox(label="3D Asset URLs", interactive=False)
            # three_d_gallery = gr.Gallery(label="3D Previews", show_label=True, allow_preview=True, columns=3, rows=2, height=400)

        def s3_to_https(uri):
            if isinstance(uri, str) and uri.startswith("s3://"):
                parts = uri[5:].split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    return f"https://{bucket}.s3.amazonaws.com/{key}"
            return uri

        def find_all_s3_uris(obj, asset_name=None, parent_key=None):
            uris = []
            # consider multiple common key names for image URLs stored in documents
            keys = [
                "image_s3_url",
                "image_url",
                "s3_image_url",
            ]

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

        def fetch_biome_json(biome_id):
            try:
                if not biome_id:
                    return {"error": "No biome selected"}
                doc = get_biome(biome_id)
                if not doc:
                    return {"error": "Biome not found"}
                # Return the parsed document (dict) so gr.Json renders a collapsible JSON
                return doc
            except Exception as e:
                return {"error": f"Error fetching biome JSON: {e}"}

        def fetch_3d_assets(biome_id, asset_name=None):
            """Fetch 3D model URLs for the biome or a specific asset. Returns (text_list, gallery_list).
            Reuses the biome/asset retrieval strategy from fetch_images to find the asset dict if an asset_name is provided.
            """
            try:
                if not biome_id or biome_id == "none":
                    return ("No biome selected", [])

                # Get the biome document
                biome_doc = get_biome(biome_id)
                if not biome_doc:
                    return ("Biome not found", [])

                # If asset_name provided, try to resolve the specific asset dict similar to fetch_images
                target_assets = []
                if asset_name and asset_name.strip():
                    update_key = get_biome_asset_update_key(biome_id, asset_name.strip())
                    if not update_key:
                        return (f"Asset '{asset_name}' not found in biome.", [])

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
                        return (f"Asset '{asset_name}' not found in biome.", [])

                    # asset_dict could be a dict of assets or a single asset
                    if isinstance(asset_dict, dict):
                        # If it looks like a single asset (has status or attributes), treat it directly
                        if any(k in asset_dict for k in ("status", "attributes", "image_s3_url", "model_url")):
                            target_assets.append((asset_name, asset_dict))
                        else:
                            # treat as mapping of many assets
                            for n, a in asset_dict.items():
                                target_assets.append((n, a))
                else:
                    # No asset filter: scan possible_structures categories
                    possible_structures = biome_doc.get("possible_structures", {})
                    for category in ["buildings", "creatures", "props", "terrain"]:
                        assets = possible_structures.get(category, {})
                        for name, asset in assets.items():
                            target_assets.append((name, asset))

                keys_3d = ["s3_3d_url", "3d_url", "model_url", "s3_model_url"]
                urls = []
                gallery_previews = []
                for name, asset in target_assets:
                    if not isinstance(asset, dict):
                        continue
                    # only consider assets marked 3d_generated
                    if asset.get("status") != "3d_generated":
                        continue
                    found = None
                    for k in keys_3d:
                        if k in asset and asset.get(k):
                            found = asset.get(k)
                            break
                    if not found and isinstance(asset.get("attributes"), dict):
                        for k in keys_3d:
                            if k in asset["attributes"] and asset["attributes"].get(k):
                                found = asset["attributes"].get(k)
                                break
                    if found:
                        urls.append(s3_to_https(found))
                        # preview image
                        preview = None
                        for img_key in ["image_s3_url", "image_url", "s3_image_url"]:
                            if img_key in asset and asset.get(img_key):
                                preview = s3_to_https(asset.get(img_key))
                                break
                        if not preview and isinstance(asset.get("attributes"), dict):
                            for img_key in ["image_s3_url", "image_url", "s3_image_url"]:
                                if img_key in asset["attributes"] and asset["attributes"].get(img_key):
                                    preview = s3_to_https(asset["attributes"].get(img_key))
                                    break
                        if preview:
                            gallery_previews.append(preview)

                if urls:
                    gallery = gallery_previews if gallery_previews else urls
                    return ("\n".join(urls), gallery)
                return ("No 3D assets with status '3d_generated' found.", [])
            except Exception as e:
                return (f"Error: {e}", [])

        fetch_s3_button.click(
            fn=fetch_images,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[s3_image_uri_output, s3_image_display]
        )

        fetch_biome_json_btn.click(
            fn=fetch_biome_json,
            inputs=[biome_dropdown],
            outputs=[biome_json_structured]
        )

        fetch_3d_btn.click(
            fn=fetch_3d_assets,
            inputs=[biome_dropdown],
            outputs=[three_d_list_output]
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

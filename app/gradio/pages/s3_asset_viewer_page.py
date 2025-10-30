
import gradio as gr
import json
from bson.objectid import ObjectId
from app.services.mongo_service import (
    get_db,
    get_biome_choices_live,
    get_biome_asset_update_key,
    get_biome,
    biome_assets_for_task,
)
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
            three_d_list_output = gr.Markdown("#### 3D Asset URLs")
            # three_d_gallery = gr.Gallery(label="3D Previews", show_label=True, allow_preview=True, columns=3, rows=2, height=400)

        def s3_to_https(uri):
            if isinstance(uri, str) and uri.startswith("s3://"):
                parts = uri[5:].split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    return f"https://{bucket}.s3.amazonaws.com/{key}"
            return uri

        # Centralized key lists and small helpers to extract urls from asset dicts.
        IMAGE_KEYS = ("image_s3_url", "image_url", "s3_image_url", "image_s3_uri")
        MODEL_KEYS = ("s3_3d_url", "s3_3d_uri", "3d_s3_url", "s3_model_url", "3d_url", "model_url", "mesh_url", "painted_url", "glb_url")

        def _recursive_find_by_ext(o, exts):
            if isinstance(o, dict):
                for vk, vv in o.items():
                    if isinstance(vv, str) and vv.lower().endswith(exts):
                        return vv
                    res = _recursive_find_by_ext(vv, exts)
                    if res:
                        return res
            elif isinstance(o, list):
                for item in o:
                    res = _recursive_find_by_ext(item, exts)
                    if res:
                        return res
            return None

        def get_image_url_from_asset(asset):
            if not isinstance(asset, dict):
                return None
            # top-level
            for k in IMAGE_KEYS:
                if k in asset and asset.get(k):
                    return s3_to_https(asset.get(k))
            # attributes
            attrs = asset.get("attributes") if isinstance(asset.get("attributes"), dict) else {}
            for k in IMAGE_KEYS:
                if k in attrs and attrs.get(k):
                    return s3_to_https(attrs.get(k))
            # fallback: find any image-like url
            img = _recursive_find_by_ext(asset, (".png", ".jpg", ".jpeg", ".webp"))
            return s3_to_https(img) if img else None

        def get_model_url_from_asset(asset):
            if not isinstance(asset, dict):
                return None
            # check canonical model keys first
            for k in MODEL_KEYS:
                if k in asset and asset.get(k):
                    return s3_to_https(asset.get(k))
            # check attributes block
            attrs = asset.get("attributes") if isinstance(asset.get("attributes"), dict) else {}
            for k in MODEL_KEYS:
                if k in attrs and attrs.get(k):
                    return s3_to_https(attrs.get(k))

            # check some other common keys that might store outputs
            EXTRA_MODEL_KEYS = ("output_key", "output", "output_url", "s3_path", "s3_obj", "s3_key", "result_url", "url")
            for k in EXTRA_MODEL_KEYS:
                if k in asset and asset.get(k):
                    val = asset.get(k)
                    # if it's an s3 path like generated-images/... or starts with s3://, normalize
                    if isinstance(val, str):
                        return s3_to_https(val)
                    # if it's a dict with a url field, try that
                    if isinstance(val, dict) and val.get("url"):
                        return s3_to_https(val.get("url"))
            for k in EXTRA_MODEL_KEYS:
                if k in attrs and attrs.get(k):
                    val = attrs.get(k)
                    if isinstance(val, str):
                        return s3_to_https(val)
                    if isinstance(val, dict) and val.get("url"):
                        return s3_to_https(val.get("url"))

            # check decimated profiles for a glb/obj url (prefer finished decimation)
            dec_obj = asset.get("decimated_assets") or asset.get("decimated")
            if isinstance(dec_obj, dict):
                # prefer profiles with explicit 'url' or 'glb_url'
                for prof, prof_val in dec_obj.items():
                    if isinstance(prof_val, dict):
                        durl = prof_val.get("url") or prof_val.get("glb_url") or prof_val.get("model_url")
                        if durl:
                            return s3_to_https(durl)

            # fallback: search recursively for any string value that ends with a 3D extension
            found = _recursive_find_by_ext(asset, (".glb", ".gltf", ".fbx", ".obj"))
            if found:
                return s3_to_https(found)

            # lastly, if any nested dict has a 'url' that looks like an s3 path, use it
            def _find_url_field(o):
                if isinstance(o, dict):
                    for vk, vv in o.items():
                        if isinstance(vv, str) and (vv.startswith("s3://") or vv.startswith("http")) and any(vv.lower().endswith(ext) for ext in (".glb", ".gltf", ".fbx", ".obj")):
                            return vv
                        if isinstance(vv, dict) and vv.get("url"):
                            u = vv.get("url")
                            if isinstance(u, str) and any(u.lower().endswith(ext) for ext in (".glb", ".gltf", ".fbx", ".obj")):
                                return u
                        res = _find_url_field(vv)
                        if res:
                            return res
                elif isinstance(o, list):
                    for item in o:
                        res = _find_url_field(item)
                        if res:
                            return res
                return None

            last = _find_url_field(asset)
            return s3_to_https(last) if last else None

        def get_decimated_info_from_asset(asset):
            dec_obj = asset.get("decimated_assets") or asset.get("decimated")
            if not isinstance(dec_obj, dict):
                return None
            # determine if decimation is done (top-level flag or any profile marked complete)
            decimation_done = False
            top_status = asset.get("decimation_status")
            if isinstance(top_status, str) and top_status.lower() in ("complete", "completed", "done"):
                decimation_done = True
            for prof_val in dec_obj.values():
                if isinstance(prof_val, dict) and prof_val.get("status") and str(prof_val.get("status")).lower() in ("complete", "completed", "done"):
                    decimation_done = True
                    break
            if not decimation_done:
                return None
            out = {}
            for prof_name, prof_val in dec_obj.items():
                if not isinstance(prof_val, dict):
                    continue
                durl = prof_val.get("url") or prof_val.get("glb_url")
                if not durl:
                    continue
                out[prof_name] = {
                    "url": s3_to_https(durl),
                    "poly_before": prof_val.get("poly_before"),
                    "poly_after": prof_val.get("poly_after"),
                    "reduction_ratio": prof_val.get("reduction_ratio"),
                }
            return out if out else None

        # legacy recursive search removed — use get_image_url_from_asset / get_model_url_from_asset helpers instead

        def collect_possible_structure_assets(biome_doc):
            """Collect all assets from possible_structures categories and return list of (name, asset)."""
            target = []
            if not isinstance(biome_doc, dict):
                return target
            possible_structures = biome_doc.get("possible_structures", {})
            for category in ["buildings", "creatures", "props", "terrain"]:
                assets = possible_structures.get(category, {})
                if isinstance(assets, dict):
                    for name, asset in assets.items():
                        target.append((name, asset))
            return target

        def fetch_images(biome_id, asset_name):
            try:
                db = get_db()
                if db is None or not biome_id or biome_id == "none":
                    return ("No biome selected or DB unavailable.", None)
                all_uris = []
                biome_doc = None
                # If asset_name is provided, try to get only that asset dict
                if asset_name and asset_name.strip():
                    update_key = get_biome_asset_update_key(biome_id, asset_name.strip())
                    if not update_key:
                        return (f"Asset '{asset_name}' not found in biome.", None)
                    biome_doc = get_biome(biome_id)
                    if not biome_doc:
                        return ("Biome not found", None)
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
                    # asset_dict may be a mapping of assets or single asset
                    if isinstance(asset_dict, dict) and any(k in asset_dict for k in ("status", "attributes", "image_url", "image_s3_url")):
                        url = get_image_url_from_asset(asset_dict)
                        if url:
                            all_uris.append(url)
                    elif isinstance(asset_dict, dict):
                        for n, a in asset_dict.items():
                            if not isinstance(a, dict):
                                continue
                            url = get_image_url_from_asset(a)
                            if url:
                                all_uris.append(url)
                else:
                    # No asset name: prefer using helper that returns normalized asset map
                    try:
                        assets_map = biome_assets_for_task(biome_id, status_filter="")
                    except Exception:
                        assets_map = {}
                    if assets_map and isinstance(assets_map, dict):
                        for a in assets_map.values():
                            url = get_image_url_from_asset(a)
                            if url:
                                all_uris.append(url)
                    else:
                        biome_doc = get_biome(biome_id)
                        if not biome_doc:
                            return ("Biome not found", None)
                        target_assets = collect_possible_structure_assets(biome_doc)
                        for name, a in target_assets:
                            if not isinstance(a, dict):
                                continue
                            url = get_image_url_from_asset(a)
                            if url:
                                all_uris.append(url)
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

                # Resolve target assets (either single asset or whole biome)
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

                    if isinstance(asset_dict, dict) and any(k in asset_dict for k in ("status", "attributes", "image_url", "model_url")):
                        target_assets.append((asset_name, asset_dict))
                    elif isinstance(asset_dict, dict):
                        for n, a in asset_dict.items():
                            target_assets.append((n, a))
                else:
                    try:
                        # fetch all assets (don't filter by status) so we can return every model URL
                        assets_map = biome_assets_for_task(biome_id, status_filter="")
                    except Exception:
                        assets_map = {}
                    if assets_map and isinstance(assets_map, dict):
                        for name, a in assets_map.items():
                            target_assets.append((name, a))
                    else:
                        target_assets = collect_possible_structure_assets(biome_doc)
                urls = []
                for name, asset in target_assets:
                    if not isinstance(asset, dict):
                        continue
                    # primary model url (mesh / glb / model_url)
                    model_url = get_model_url_from_asset(asset)
                    if model_url:
                        urls.append(model_url)

                    # painted mesh (commonly in attributes.paint* or painted_url)
                    painted = None
                    if isinstance(asset.get("attributes"), dict):
                        painted = asset["attributes"].get("painted_url") or asset["attributes"].get("painted_s3_url")
                    painted = painted or asset.get("painted_url") or asset.get("painted_s3_url")
                    if painted:
                        urls.append(s3_to_https(painted))

                    # include any decimated profile urls (only if present)
                    dec_info = get_decimated_info_from_asset(asset)
                    if isinstance(dec_info, dict):
                        for prof, info in dec_info.items():
                            if isinstance(info, dict) and info.get("url"):
                                urls.append(info.get("url"))

                # deduplicate while preserving order
                seen = set()
                unique_urls = []
                for u in urls:
                    if not u:
                        continue
                    if u not in seen:
                        seen.add(u)
                        unique_urls.append(u)

                # Build a Markdown-formatted, labeled list of links for nicer display
                import os
                def label_for_url(u):
                    ln = os.path.basename(u)
                    lower = u.lower()
                    if "painted" in lower:
                        return f"Painted — {ln}"
                    if "decimated" in lower or "decimate" in lower:
                        return f"Decimated — {ln}"
                    return f"Model — {ln}"

                md_lines = [f"### 3D Assets ({len(unique_urls)})\n"]
                for u in unique_urls:
                    label = label_for_url(u)
                    # Markdown link
                    md_lines.append(f"- [{label}]({u})")

                md = "\n".join(md_lines)
                return md
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

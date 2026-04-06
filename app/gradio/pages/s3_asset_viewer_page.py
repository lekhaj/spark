import gradio as gr
import json
import urllib.parse
from bson.objectid import ObjectId
from app.services.mongo_service import (
    get_db,
    get_biome_choices_live,
    get_biome_asset_update_key,
    get_biome,
    biome_assets_for_task,
    update_or_add_biome_asset,
)
try:
    from app.config import settings
except ModuleNotFoundError:
    import sys
    import os
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__), '../..')))
    from app.config import settings


def s3_asset_viewer_ui():

    with gr.Blocks() as s3_asset_viewer:
        gr.Markdown("## 🔍 S3 Asset Image Viewer")
        gr.Markdown(
            "Select a biome to view all images, or enter an asset name to filter images for that asset.")

        # Initialize biome choices
        try:
            biome_choices = get_biome_choices_live(
                settings.MONGODB_DB_NAME, "biomes") if get_biome_choices_live else []
        except Exception as e:
            print(f"[S3 Asset Viewer] Error fetching biomes: {e}")
            biome_choices = []

        # UI Components
        with gr.Row():
            if biome_choices:
                biome_dropdown = gr.Dropdown(
                    choices=[(name, _id) for name, _id in biome_choices],
                    label="Biome Name",
                    interactive=True,
                    value=biome_choices[0][1]
                )
            else:
                biome_dropdown = gr.Dropdown(
                    choices=[], label="Biome Name", interactive=False, visible=False)

            asset_name_box = gr.Textbox(
                label="Asset Name (optional)",
                placeholder="Enter asset name (e.g., steppe_watchtower)"
            )

        refresh_biomes_button = gr.Button("Refresh Biome List")

        # Image URL Lookup Accordion
        with gr.Accordion("Image URL Lookup", open=True):
            fetch_s3_button = gr.Button("Show Images")
            s3_image_uri_output = gr.Textbox(label="Image URI(s)", interactive=False)

            # Image selection mechanism - now dynamic for any number of images
            with gr.Row():
                with gr.Column(scale=1,variant="compact"):
                    image_selector = gr.Radio(
                        label="Select an Image to Rate",
                        choices=[],
                        type="value",
                        interactive=True,
                        visible=False
                    )
                with gr.Column(scale=1):
                    selected_image_display = gr.Image(
                        label="Selected Image Preview",
                        interactive=False,
                        visible=False
                    )

        # Asset Quality Rating Accordion
        with gr.Accordion("Asset Quality Rating", open=False):
            gr.Markdown("Rate the quality of the selected asset")
            with gr.Row():
                quality_good_btn = gr.Button("👍 Good", variant="primary")
                quality_bad_btn = gr.Button("👎 Bad", variant="secondary")
            quality_status = gr.Textbox(label="Quality Rating Status", interactive=False)

        # Biome JSON Accordion
        with gr.Accordion("Biome JSON", open=False):
            fetch_biome_json_btn = gr.Button("Fetch Biome JSON")
            biome_json_structured = gr.Json(label="Biome JSON (structured)")

        # 3D Assets Accordion
        with gr.Accordion("3D Assets (status = 3d_generated)", open=False):
            fetch_3d_btn = gr.Button("Fetch 3D Asset URLs")
            three_d_list_output = gr.Markdown("#### 3D Asset URLs")
            model_select = gr.Dropdown(
                label="Select 3D Model to Preview",
                choices=[],
                value=None,
                interactive=True
            )
            model_action_html = gr.HTML(value="", elem_id="model-action-area")

        # Decimated Models Accordion
        with gr.Accordion("Decimated Models", open=False):
            gr.Markdown("View per-LOD decimated `.glb` models stored in S3.")
            fetch_decimated_btn = gr.Button("Fetch Decimated Models")
            decimated_status_md = gr.Markdown("#### Decimated Model URLs")
            decimated_model_select = gr.Dropdown(
                label="Select Decimated Model",
                choices=[],
                value=None,
                interactive=True
            )
            decimated_action_html = gr.HTML(value="", elem_id="decimated-action-area")

        # TRELLIS Models Accordion
        with gr.Accordion("TRELLIS Models", open=False):
            gr.Markdown("View TRELLIS.2-4B generated `.glb` models stored in S3.")
            fetch_trellis_btn = gr.Button("Fetch TRELLIS Models")
            trellis_status_md = gr.Markdown("#### TRELLIS Model URLs")
            trellis_model_select = gr.Dropdown(
                label="Select TRELLIS Model",
                choices=[],
                value=None,
                interactive=True
            )
            trellis_action_html = gr.HTML(value="", elem_id="trellis-action-area")

        # Rigged Models Accordion
        with gr.Accordion("Rigged Models", open=False):
            gr.Markdown("View rigged `.glb` models stored in S3 after UniRig processing.")
            fetch_rigged_btn = gr.Button("Fetch Rigged Models")
            rigged_status_md = gr.Markdown("#### Rigged Model URLs")
            rigged_model_select = gr.Dropdown(
                label="Select Rigged Model",
                choices=[],
                value=None,
                interactive=True
            )
            rigged_action_html = gr.HTML(value="", elem_id="rigged-action-area")
            with gr.Row():
                rigged_url_box = gr.Textbox(
                    label="Direct S3 URL",
                    interactive=False,
                    placeholder="Select a rigged model to see its URL"
                )

        # Core Helper Functions
        def s3_to_https(uri):
            """Convert S3 URI to HTTPS URL"""
            if isinstance(uri, str) and uri.startswith("s3://"):
                parts = uri[5:].split("/", 1)
                if len(parts) == 2:
                    bucket, key = parts
                    return f"https://{bucket}.s3.amazonaws.com/{key}"
            return uri

        def _recursive_find_url_by_ext(obj, extensions):
            """Recursively find URL ending with specified extensions"""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.lower().endswith(extensions):
                        return value
                    result = _recursive_find_url_by_ext(value, extensions)
                    if result:
                        return result
            elif isinstance(obj, list):
                for item in obj:
                    result = _recursive_find_url_by_ext(item, extensions)
                    if result:
                        return result
            return None

        def _get_url_from_asset(asset, keys, extensions=None):
            """Generic function to extract URL from asset using keys and optional extensions"""
            if not isinstance(asset, dict):
                return None

            # Check direct keys
            for key in keys:
                if key in asset and asset.get(key):
                    return s3_to_https(asset.get(key))

            # Check attributes
            attrs = asset.get("attributes", {})
            if isinstance(attrs, dict):
                for key in keys:
                    if key in attrs and attrs.get(key):
                        return s3_to_https(attrs.get(key))

            # Fallback to recursive search by extensions
            if extensions:
                found_url = _recursive_find_url_by_ext(asset, extensions)
                if found_url:
                    return s3_to_https(found_url)

            return None

        def get_image_url_from_asset(asset):
            """Extract image URL from asset"""
            image_keys = ("image_s3_url", "image_url", "s3_image_url", "image_s3_uri")
            image_extensions = (".png", ".jpg", ".jpeg", ".webp")
            return _get_url_from_asset(asset, image_keys, image_extensions)

        def get_model_url_from_asset(asset):
            """Extract 3D model URL from asset"""
            model_keys = ("s3_3d_url", "s3_3d_uri", "3d_s3_url", "s3_model_url",
                         "3d_url", "model_url", "mesh_url", "painted_url", "glb_url")
            model_extensions = (".glb", ".gltf", ".fbx", ".obj")

            url = _get_url_from_asset(asset, model_keys, model_extensions)
            if url:
                return url

            # Additional model URL search logic
            extra_keys = ("output_url", "s3_path", "s3_key", "result_url", "url")
            for key in extra_keys:
                if key in asset and asset.get(key):
                    val = asset.get(key)
                    if isinstance(val, str):
                        return s3_to_https(val)
                    if isinstance(val, dict) and val.get("url"):
                        return s3_to_https(val.get("url"))

            # Check decimated assets
            decimated_info = get_decimated_info_from_asset(asset)
            if decimated_info:
                for profile_info in decimated_info.values():
                    if isinstance(profile_info, dict) and profile_info.get("url"):
                        return profile_info.get("url")

            return None

        def get_decimated_info_from_asset(asset):
            """Extract decimated asset information"""
            dec_obj = asset.get("decimated_assets") or asset.get("decimated")
            if not isinstance(dec_obj, dict):
                return None

            # Check decimation completion
            top_status = asset.get("decimation_status", "").lower()
            status_checks = ["complete", "completed", "done"]
            decimation_done = top_status in status_checks

            if not decimation_done:
                for prof_val in dec_obj.values():
                    if isinstance(prof_val, dict):
                        prof_status = str(prof_val.get("status", "")).lower()
                        if prof_status in status_checks:
                            decimation_done = True
                            break

            if not decimation_done:
                return None

            # Extract decimated profile information
            decimated_info = {}
            for profile_name, profile_data in dec_obj.items():
                if not isinstance(profile_data, dict):
                    continue

                profile_url = profile_data.get("url") or profile_data.get("glb_url")
                if profile_url:
                    decimated_info[profile_name] = {
                        "url": s3_to_https(profile_url),
                        "poly_before": profile_data.get("poly_before"),
                        "poly_after": profile_data.get("poly_after"),
                        "reduction_ratio": profile_data.get("reduction_ratio"),
                    }

            return decimated_info if decimated_info else None

        def get_biome_assets(biome_id, asset_name=None):
            """Unified function to get assets for a biome with optional filtering"""
            if not biome_id:
                return {}

            try:
                if asset_name and asset_name.strip():
                    # Get specific asset
                    biome_doc = get_biome(biome_id)
                    if not biome_doc:
                        return {}

                    update_key = get_biome_asset_update_key(biome_id, asset_name.strip())
                    if not update_key:
                        return {}

                    # Navigate to the asset using the update key path
                    asset_path = update_key.split('.')
                    current_level = biome_doc
                    for key in asset_path:
                        if isinstance(current_level, dict):
                            current_level = current_level.get(key, {})
                        else:
                            return {}

                    if isinstance(current_level, dict):
                        if any(k in current_level for k in ("status", "attributes", "image_url")):
                            return {asset_name: current_level}
                        else:
                            return current_level
                    return {}
                else:
                    # Get all assets
                    try:
                        assets_map = biome_assets_for_task(biome_id, status_filter="")
                        if assets_map and isinstance(assets_map, dict):
                            return assets_map
                    except Exception:
                        pass

                    # Fallback to collecting from possible_structures
                    biome_doc = get_biome(biome_id)
                    if not biome_doc:
                        return {}

                    possible_structures = biome_doc.get("possible_structures", {})
                    assets = {}
                    categories_to_search = list(possible_structures.keys())
                    for category in categories_to_search:
                        category_assets = possible_structures.get(category, {})
                        if isinstance(category_assets, dict):
                            assets.update(category_assets)
                    return assets
            except Exception as e:
                print(f"Error getting biome assets: {e}")
                return {}

        def fetch_images(biome_id, asset_name):
            """Fetch and display images for the selected biome/asset - handles any number of images"""
            try:
                if not biome_id or biome_id == "none":
                    return ("No biome selected or DB unavailable.",
                            gr.update(visible=False),
                            gr.update(visible=False))

                assets_map = get_biome_assets(biome_id, asset_name)
                all_uris = []
                image_choices = []

                # Dynamically process all assets and extract image URLs
                for asset_name_key, asset_data in assets_map.items():
                    image_url = get_image_url_from_asset(asset_data)
                    if image_url:
                        all_uris.append(image_url)
                        # Create descriptive label with truncation for long descriptions
                        description = asset_data.get('description', 'No description')
                        truncated_desc = description[:80] + "..." if len(description) > 80 else description
                        label = f"{asset_name_key} - {truncated_desc}"
                        image_choices.append((label, image_url))

                if all_uris:
                    # Return text output and update selector with dynamic choices
                    uri_text_output = "\n".join(all_uris)

                    # Update radio selector with all found images
                    selector_update = gr.update(
                        visible=True,
                        choices=image_choices,
                        value=image_choices[0][1] if image_choices else None
                    )

                    # Auto-select first image for preview
                    preview_update = gr.update(
                        visible=True,
                        value=image_choices[0][1] if image_choices else None
                    )

                    return (uri_text_output, selector_update, preview_update)
                else:
                    return ("No images found for this biome/asset.",
                            gr.update(visible=False),
                            gr.update(visible=False))

            except Exception as e:
                print(f"[S3 Asset Viewer] Error fetching images: {e}")
                return (f"Error: {e}",
                        gr.update(visible=False),
                        gr.update(visible=False))

        def rate_asset_quality(selected_image_url, biome_id, asset_name, quality):
            """Update quality rating for an asset"""
            if not selected_image_url or not biome_id:
                return "Error: No image selected or biome ID missing"

            assets_map = get_biome_assets(biome_id, asset_name)
            selected_asset_name = None

            # Find asset matching the selected image URL
            for asset_name_key, asset_data in assets_map.items():
                asset_image_url = get_image_url_from_asset(asset_data)
                if asset_image_url == selected_image_url:
                    selected_asset_name = asset_name_key
                    break

            if not selected_asset_name:
                return "Error: Could not find asset for selected image"

            update_result = update_or_add_biome_asset(
                biome_id,
                get_biome_asset_update_key(biome_id, selected_asset_name),
                {"quality": quality}
            )

            return f"✅ Rated '{selected_asset_name}' as {quality} quality"

        def rate_good(selected_image_url, biome_id, asset_name):
            return rate_asset_quality(selected_image_url, biome_id, asset_name, "good")

        def rate_bad(selected_image_url, biome_id, asset_name):
            return rate_asset_quality(selected_image_url, biome_id, asset_name, "bad")

        def fetch_biome_json(biome_id):
            """Fetch and return biome JSON data"""
            try:
                if not biome_id:
                    return {"error": "No biome selected"}
                doc = get_biome(biome_id)
                return doc if doc else {"error": "Biome not found"}
            except Exception as e:
                return {"error": f"Error fetching biome JSON: {e}"}

        def fetch_3d_assets(biome_id, asset_name=None):
            """Fetch 3D model URLs for biome or specific asset - handles any number of models"""
            try:
                if not biome_id or biome_id == "none":
                    return ("No biome selected", gr.update(choices=[]))

                assets_map = get_biome_assets(biome_id, asset_name)
                if not assets_map:
                    return ("No assets found", gr.update(choices=[]))

                urls = []
                for asset_name_key, asset_data in assets_map.items():
                    if not isinstance(asset_data, dict):
                        continue

                    # Primary model URL
                    model_url = get_model_url_from_asset(asset_data)
                    if model_url:
                        urls.append(model_url)

                    # Painted mesh URLs
                    painted_url = None
                    attrs = asset_data.get("attributes", {})
                    if isinstance(attrs, dict):
                        painted_url = attrs.get("painted_url") or attrs.get("painted_s3_url")
                    painted_url = painted_url or asset_data.get("painted_url") or asset_data.get("painted_s3_url")
                    if painted_url:
                        urls.append(s3_to_https(painted_url))

                    # Decimated assets
                    decimated_info = get_decimated_info_from_asset(asset_data)
                    if isinstance(decimated_info, dict):
                        for profile_info in decimated_info.values():
                            if isinstance(profile_info, dict) and profile_info.get("url"):
                                urls.append(profile_info.get("url"))

                    # TRELLIS model
                    trellis_url = asset_data.get("trellis_glb_url")
                    if trellis_url:
                        urls.append(s3_to_https(trellis_url))

                    # Rigged models: rigged_lod_* (Hunyuan) and rigged_trellis_* (TRELLIS)
                    for k, v in asset_data.items():
                        if (k.startswith("rigged_lod_") or k.startswith("rigged_trellis_")) and isinstance(v, str) and v:
                            urls.append(s3_to_https(v))

                # Deduplicate URLs
                seen_urls = set()
                unique_urls = []
                for url in urls:
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        unique_urls.append(url)

                # Create labeled choices and markdown output - dynamic for any number of models
                import os

                def create_url_label(url, hint=""):
                    filename = os.path.basename(url)
                    url_lower = url.lower()
                    if hint:
                        return f"{hint} — {filename}"
                    if "rigging" in url_lower and "trellis" in filename.lower():
                        return f"Rigged [TRELLIS] — {filename}"
                    if "rigging" in url_lower:
                        return f"Rigged [Hunyuan] — {filename}"
                    if "trellis" in url_lower:
                        return f"TRELLIS — {filename}"
                    if "painted" in url_lower:
                        return f"Painted — {filename}"
                    if "decimated" in url_lower or "decimate" in url_lower:
                        return f"Decimated — {filename}"
                    return f"Model — {filename}"

                md_lines = [f"### 3D Assets ({len(unique_urls)})\n"]
                choices = []
                for url in unique_urls:
                    label = create_url_label(url)
                    md_lines.append(f"- [{label}]({url})")
                    choices.append((label, url))

                markdown_output = "\n".join(md_lines)
                default_value = choices[0][1] if choices else None

                return markdown_output, gr.update(choices=choices, value=default_value)

            except Exception as e:
                return (f"Error: {e}", gr.update(choices=[]))

        def fetch_rigged_models(biome_id, asset_name=None):
            """Fetch rigged model URLs for a biome — checks rigged_model_url, rigged_url,
            and per-variant rigged_url inside decimated_assets."""
            try:
                if not biome_id or biome_id == "none":
                    return "No biome selected", gr.update(choices=[])

                assets_map = get_biome_assets(biome_id, asset_name)
                if not assets_map:
                    return "No assets found", gr.update(choices=[])

                rigged_keys = ("rigged_model_url", "rigged_url", "rigged_glb_url")
                found = []

                for asset_name_key, asset_data in assets_map.items():
                    if not isinstance(asset_data, dict):
                        continue

                    # Top-level rigged URL (fixed keys)
                    for k in rigged_keys:
                        url = asset_data.get(k)
                        if url:
                            found.append((asset_name_key, s3_to_https(url), "rigged"))

                    # Dynamic rigged_lod_* keys (Hunyuan): rigged_lod_300k, rigged_lod_100k, etc.
                    for k, v in asset_data.items():
                        if k.startswith("rigged_lod_") and isinstance(v, str) and v:
                            label = f"{asset_name_key} [Hunyuan {k}]"
                            found.append((label, s3_to_https(v), "hunyuan-rigged"))

                    # Dynamic rigged_trellis_* keys: rigged_trellis_lod_300k, etc.
                    for k, v in asset_data.items():
                        if k.startswith("rigged_trellis_") and isinstance(v, str) and v:
                            label = f"{asset_name_key} [TRELLIS {k}]"
                            found.append((label, s3_to_https(v), "trellis-rigged"))

                    # Per-variant inside decimated_assets
                    dec_obj = asset_data.get("decimated_assets") or {}
                    for variant_key, variant_val in dec_obj.items():
                        if not isinstance(variant_val, dict):
                            continue
                        for k in rigged_keys:
                            url = variant_val.get(k)
                            if url:
                                found.append((
                                    f"{asset_name_key} / {variant_key}",
                                    s3_to_https(url), "rigged-decimated"
                                ))

                if not found:
                    return "No rigged models found. Run rigging first.", gr.update(choices=[])

                import os
                md_lines = [f"### Rigged Models ({len(found)})\n"]
                choices = []
                for asset_lbl, url, kind in found:
                    fname = os.path.basename(url)
                    label = f"{asset_lbl} — {fname}"
                    md_lines.append(f"- [{label}]({url})")
                    choices.append((label, url))

                return "\n".join(md_lines), gr.update(choices=choices, value=choices[0][1] if choices else None)

            except Exception as e:
                return f"Error: {e}", gr.update(choices=[])

        def fetch_decimated_models(biome_id, asset_name=None):
            """Fetch per-LOD decimated models from decimated_assets.{lod}.url"""
            try:
                if not biome_id or biome_id == "none":
                    return "No biome selected", gr.update(choices=[])

                assets_map = get_biome_assets(biome_id, asset_name)
                if not assets_map:
                    return "No assets found", gr.update(choices=[])

                found = []
                for asset_name_key, asset_data in assets_map.items():
                    if not isinstance(asset_data, dict):
                        continue
                    dec_obj = asset_data.get("decimated_assets") or {}
                    for lod_key, lod_data in dec_obj.items():
                        if not isinstance(lod_data, dict):
                            continue
                        url = lod_data.get("url") or lod_data.get("glb_url")
                        if url:
                            found.append((f"{asset_name_key} / {lod_key}", s3_to_https(url)))

                if not found:
                    return "No decimated models found. Run decimation first.", gr.update(choices=[])

                import os
                md_lines = [f"### Decimated Models ({len(found)} LOD variants)\n"]
                choices = []
                for label, url in found:
                    fname = os.path.basename(url)
                    md_lines.append(f"- [{label} — {fname}]({url})")
                    choices.append((label, url))

                return "\n".join(md_lines), gr.update(choices=choices, value=choices[0][1] if choices else None)

            except Exception as e:
                return f"Error: {e}", gr.update(choices=[])

        def fetch_trellis_models(biome_id, asset_name=None):
            """Fetch TRELLIS.2-4B generated models from trellis_glb_url field."""
            try:
                if not biome_id or biome_id == "none":
                    return "No biome selected", gr.update(choices=[])

                assets_map = get_biome_assets(biome_id, asset_name)
                if not assets_map:
                    return "No assets found", gr.update(choices=[])

                found = []
                for asset_name_key, asset_data in assets_map.items():
                    if not isinstance(asset_data, dict):
                        continue
                    # TRELLIS primary output
                    url = asset_data.get("trellis_glb_url")
                    if url:
                        found.append((f"{asset_name_key} (TRELLIS)", s3_to_https(url)))
                    # TRELLIS decimated variants (future: trellis_decimated_assets)
                    trellis_dec = asset_data.get("trellis_decimated_assets") or {}
                    for lod_key, lod_data in trellis_dec.items():
                        if isinstance(lod_data, dict):
                            dec_url = lod_data.get("url") or lod_data.get("glb_url")
                            if dec_url:
                                found.append((f"{asset_name_key} / TRELLIS-{lod_key}", s3_to_https(dec_url)))

                if not found:
                    return "No TRELLIS models found.", gr.update(choices=[])

                import os
                md_lines = [f"### TRELLIS Models ({len(found)})\n"]
                choices = []
                for label, url in found:
                    fname = os.path.basename(url)
                    md_lines.append(f"- [{label} — {fname}]({url})")
                    choices.append((label, url))

                return "\n".join(md_lines), gr.update(choices=choices, value=choices[0][1] if choices else None)

            except Exception as e:
                return f"Error: {e}", gr.update(choices=[])

        def show_model_viewer_html(model_url, label_prefix="Model"):
            """Generic GLB viewer HTML — works for decimated, TRELLIS, rigged."""
            if not model_url:
                return ""
            encoded_url = urllib.parse.quote(model_url, safe=':/')
            viewer_url = f"https://3dviewer.net/#model={encoded_url}"
            return (
                f'<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">'
                f'<a href="{viewer_url}" target="_blank" rel="noopener noreferrer" '
                f'style="padding:8px 14px;background:#0b5fff;color:#fff;border-radius:6px;'
                f'text-decoration:none;font-weight:600;">🔍 Open in 3D Viewer</a>'
                f'<a href="{model_url}" target="_blank" rel="noopener noreferrer" '
                f'style="padding:8px 14px;background:#1f7f46;color:#fff;border-radius:6px;'
                f'text-decoration:none;font-weight:600;">⬇️ Download GLB</a>'
                f'</div>'
            )

        def show_rigged_model_actions(model_url):
            """Generate viewer HTML for a rigged model URL."""
            if not model_url:
                return "", ""

            import urllib.parse
            encoded_url = urllib.parse.quote(model_url, safe=':/')
            viewer_url = f"https://3dviewer.net/#model={encoded_url}"

            html = (
                f'<div style="display:flex;gap:12px;align-items:center;flex-wrap:wrap">'
                f'<a href="{viewer_url}" target="_blank" rel="noopener noreferrer" '
                f'style="padding:8px 14px;background:#7c3aed;color:#fff;border-radius:6px;'
                f'text-decoration:none;font-weight:600;">🦴 Open Rigged Model in 3D Viewer</a>'
                f'<a href="{model_url}" target="_blank" rel="noopener noreferrer" '
                f'style="padding:8px 14px;background:#1f7f46;color:#fff;border-radius:6px;'
                f'text-decoration:none;font-weight:600;">⬇️ Download GLB</a>'
                f'</div>'
                f'<p style="margin-top:8px;font-size:12px;color:#666">'
                f'Tip: 3DViewer.net supports GLB files with bones/armature from UniRig.</p>'
            )
            return html, model_url

        def update_image_preview(selected_image_url):
            """Update image preview when selection changes - works with any image"""
            return gr.update(visible=bool(selected_image_url), value=selected_image_url)

        def show_model_actions(model_url):
            """Generate HTML for model action buttons"""
            if not model_url:
                return ""

            encoded_url = urllib.parse.quote(model_url, safe=':/')
            viewer_url = f"https://3dviewer.net/#model={encoded_url}"

            html = (
                f'<div style="display:flex;gap:12px;align-items:center">'
                f'<a href="{viewer_url}" target="_blank" rel="noopener noreferrer" '
                f'style="padding:8px 12px;background:#0b5fff;color:#fff;border-radius:6px;text-decoration:none;">🔍 Open in 3D Viewer</a>'
                f'<a href="{model_url}" target="_blank" rel="noopener noreferrer" download '
                f'style="padding:8px 12px;background:#1f7f46;color:#fff;border-radius:6px;text-decoration:none;">⬇️ Download</a>'
                f'</div>'
            )
            return html

        def refresh_biome_choices():
            """Refresh the biome dropdown choices"""
            try:
                biome_choices = get_biome_choices_live(settings.MONGODB_DB_NAME, "biomes")
                if not biome_choices:
                    biome_choices = [("No biomes available", "none")]
                return gr.update(
                    choices=[(name, _id) for name, _id in biome_choices],
                    value=biome_choices[0][1] if biome_choices else None
                )
            except Exception as e:
                print(f"Error refreshing biome choices: {e}")
                return gr.update(choices=[])

        # Event Handlers
        quality_good_btn.click(
            fn=rate_good,
            inputs=[image_selector, biome_dropdown, asset_name_box],
            outputs=[quality_status]
        )

        quality_bad_btn.click(
            fn=rate_bad,
            inputs=[image_selector, biome_dropdown, asset_name_box],
            outputs=[quality_status]
        )

        fetch_s3_button.click(
            fn=fetch_images,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[s3_image_uri_output, image_selector, selected_image_display]
        )

        fetch_biome_json_btn.click(
            fn=fetch_biome_json,
            inputs=[biome_dropdown],
            outputs=[biome_json_structured]
        )

        image_selector.change(
            fn=update_image_preview,
            inputs=[image_selector],
            outputs=[selected_image_display]
        )

        fetch_3d_btn.click(
            fn=fetch_3d_assets,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[three_d_list_output, model_select]
        )

        model_select.change(
            fn=show_model_actions,
            inputs=[model_select],
            outputs=[model_action_html]
        )

        refresh_biomes_button.click(
            fn=refresh_biome_choices,
            inputs=[],
            outputs=[biome_dropdown]
        )

        # Rigged model event handlers
        fetch_rigged_btn.click(
            fn=fetch_rigged_models,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[rigged_status_md, rigged_model_select]
        )

        rigged_model_select.change(
            fn=show_rigged_model_actions,
            inputs=[rigged_model_select],
            outputs=[rigged_action_html, rigged_url_box]
        )

        # Decimated model event handlers
        fetch_decimated_btn.click(
            fn=fetch_decimated_models,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[decimated_status_md, decimated_model_select]
        )

        decimated_model_select.change(
            fn=show_model_viewer_html,
            inputs=[decimated_model_select],
            outputs=[decimated_action_html]
        )

        # TRELLIS model event handlers
        fetch_trellis_btn.click(
            fn=fetch_trellis_models,
            inputs=[biome_dropdown, asset_name_box],
            outputs=[trellis_status_md, trellis_model_select]
        )

        trellis_model_select.change(
            fn=show_model_viewer_html,
            inputs=[trellis_model_select],
            outputs=[trellis_action_html]
        )

    return s3_asset_viewer

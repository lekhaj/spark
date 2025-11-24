from urllib.parse import urlparse
import os
import gradio as gr
import time
from app.config import settings
from app.services.mongo_service import get_db, get_biome_choices_live
from app.services.mongo_service import get_biome, biome_assets_for_task, update_or_add_biome_asset, get_biome_asset_update_key


from app.services.aws_service import upload_to_s3, download_from_s3

DECIMATION_PROFILES = [
    ("5k", 5000, "COLLAPSE", None),
    ("6k", 6000, "COLLAPSE", None),
    ("8k", 8000, "COLLAPSE", None),
    ("10k", 10000, "COLLAPSE", None),
]

# Debug / behaviour flag (tweak for local runs)
# SAVE_LOCAL: when True, keep downloaded input files and decimated outputs locally.
#            when False, downloaded input files and decimated outputs are removed after upload/Mongo update.
SAVE_LOCAL = False


def _cleanup_files(files):
    """Attempt to remove each path in files.
    Returns a list of log messages (strings) describing actions or warnings.
    The caller may iterate and yield each message to the Gradio stream.
    """
    msgs = []
    for f in files or []:
        try:
            if os.path.exists(f):
                os.remove(f)
            else:
                # file not present; nothing to do
                pass
        except Exception as e:
            msgs.append(f"[WARN] Could not remove {f}: {e}")
    return msgs



def start_decimation_process(database_name: str, collection_name: str, biome_name: str, asset_name: str, biome_choices: list) -> str:
    """
    Start the decimation process for selected biome and asset.
    """
    if not biome_name:
        return "Please select a biome first."
    biome_id = None
    for name, _id in biome_choices:
        if name == biome_name:
            biome_id = _id
            break
    if not biome_id:
        return "Selected biome not found."
    try:
        if asset_name == "All Assets":
            result = f"Starting decimation for ALL assets in biome '{biome_name}' (ID: {biome_id})\n"
            result += "This would normally trigger the full biome decimation process.\n"
            result += "Note: Full biome processing should be run on the GPU server with Blender."
        else:
            result = f"Starting decimation for asset '{asset_name}' in biome '{biome_name}' (ID: {biome_id})\n"
            result += "This would normally trigger the single asset decimation process.\n"
            result += "Note: Asset processing should be run on the GPU server with Blender."
        return result + f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        return f"Error starting decimation process: {str(e)}"

def get_decimation_status(database_name: str, collection_name: str, biome_name: str, asset_name: str, biome_choices: list) -> str:
    """
    Get the current decimation status for the selected asset.
    """
    if not biome_name or asset_name == "All Assets":
        return "Please select a specific asset to check status."
    biome_id = next((_id for name, _id in biome_choices if name == biome_name), None)
    if not biome_id:
        return "Selected biome not found."
    try:
        biome_data = get_biome(biome_id)
        if not biome_data:
            return "Biome data not found."
        possible_structures = biome_data.get("possible_structures", {})
        asset_data = None
        for category in ["buildings", "creatures", "props", "terrain"]:
            assets = possible_structures.get(category, {})
            if asset_name in assets:
                asset_data = assets[asset_name]
                break
        if not asset_data:
            return f"Asset '{asset_name}' not found in biome."
        decimation_status = asset_data.get("decimation_status", "Not started")
        decimated_assets = asset_data.get("decimated_assets", {})
        status_text = f"Decimation Status: {decimation_status}\n"
        if decimated_assets:
            status_text += "\nDecimated Versions Available:\n"
            for profile, data in decimated_assets.items():
                if isinstance(data, dict) and "poly_after" in data:
                    status_text += f"- {profile}: {data.get('poly_after', 'N/A')} polygons\n"
                elif isinstance(data, dict) and "error" in data:
                    status_text += f"- {profile}: Error - {data['error']}\n"
        timestamp = asset_data.get("decimation_timestamp")
        if timestamp:
            status_text += f"\nLast Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))}"
        return status_text
    except Exception as e:
        return f"Error getting status: {str(e)}"

# --- Gradio UI Function ---
def decimation_page_ui():
    with gr.Blocks() as decimation_ui:
        gr.Markdown("# 🔧 3D Asset Decimation")
        gr.Markdown("Reduce polygon count of 3D assets for optimization. Select a biome and optionally a specific asset to decimate.")

        db_name = settings.MONGODB_DB_NAME
        db = get_db()
        collections = []
        if db is not None and db_name:
            collections = db.list_collection_names()
            collections = [str(c) for c in collections] if collections else []
        default_collection = collections[0] if collections else None

        decimation_collection = gr.Dropdown(label="Select Collection", choices=collections, value=default_collection, interactive=True)

        biome_choices = get_biome_choices_live(db_name, default_collection) if default_collection else []
        biome_dropdown_choices = [(name, _id) for name, _id in biome_choices]
        default_biome = biome_dropdown_choices[0][1] if biome_dropdown_choices else None
        biome_dropdown_decim = gr.Dropdown(label="Select Biome", choices=biome_dropdown_choices, value=default_biome, interactive=True)


        # Use backend helper for asset list (status filtered by backend)
        asset_choices = [("All Assets", "all")]
        if default_biome:
            # biome_assets_for_task returns dict of asset_name: asset_dict
            # Use the normalized status token '3d_generated' (matches asset.status)
            assets_dict = biome_assets_for_task(default_biome, status_filter="3d_generated")
            if isinstance(assets_dict, dict):
                asset_choices += [
                    (f"{v.get('type','').title() if isinstance(v, dict) else 'Asset'}: {k}", k)
                    for k, v in assets_dict.items()
                ]
        asset_dropdown_decim = gr.Dropdown(label="Select Asset (only those ready for decimation)", choices=asset_choices, value="all", interactive=True)

        biome_choices_decim = gr.State(biome_choices)

        with gr.Row():
            start_decimation_btn = gr.Button("🚀 Start Decimation", variant="primary")
            check_status_btn = gr.Button("📊 Check Status")
            refresh_assets_btn = gr.Button("🔄 Refresh Assets")
        decimation_status_output = gr.Textbox(label="Decimation Status", interactive=False, lines=10)

        # --- Event Handlers ---
        def update_biomes_dropdown_decim(collection_name):
            db_name_local = settings.MONGODB_DB_NAME
            if not db_name_local or not collection_name:
                return gr.Dropdown(choices=[], value=None), []
            biome_choices = get_biome_choices_live(db_name_local, collection_name)
            biome_dropdown_choices = [(name, _id) for name, _id in biome_choices]
            return gr.Dropdown(choices=biome_dropdown_choices, value=biome_dropdown_choices[0][1] if biome_dropdown_choices else None), biome_choices

        def update_assets_dropdown_decim(collection_name, biome_id, biome_choices):
            """
            Update asset dropdown for the currently selected biome using backend helper (status filtered by backend).
            """
            if not biome_id:
                return gr.Dropdown(choices=[("All Assets", "all")], value="all")
            try:
                # Use normalized status token so assets with status == '3d_generated' are returned
                assets_dict = biome_assets_for_task(biome_id, status_filter="3d_generated")
                asset_choices = [("All Assets", "all")]
                if isinstance(assets_dict, dict):
                    asset_choices += [
                        (f"{v.get('type','').title() if isinstance(v, dict) else 'Asset'}: {k}", k)
                        for k, v in assets_dict.items()
                    ]
                return gr.Dropdown(choices=asset_choices, value="all")
            except Exception as e:
                print(f"Error updating asset dropdown for biome {biome_id}: {e}")
                return gr.Dropdown(choices=[("All Assets", "all")], value="all")


        def handle_start_decimation(coll_name, biome_id, asset_id, biome_choices):
            import os
            import subprocess, json
            db_name = settings.MONGODB_DB_NAME
            biome_name_display = next((name for name, _id in biome_choices if _id == biome_id), biome_id)
            assets_dict = biome_assets_for_task(biome_id, status_filter="3d_generated")
            if not isinstance(assets_dict, dict) or not assets_dict:
                yield "No assets ready for decimation."
                return
            assets_to_decimate = []
            if asset_id == "all":
                for asset_name, asset in assets_dict.items():
                    assets_to_decimate.append((asset.get('type', 'Asset'), asset_name, asset))
            else:
                if asset_id in assets_dict:
                    asset = assets_dict[asset_id]
                    assets_to_decimate.append((asset.get('type', 'Asset'), asset_id, asset))
            if not assets_to_decimate:
                yield "No assets ready for decimation."
                return
            bucket_name = settings.AWS_S3_BUCKET if hasattr(settings, 'AWS_S3_BUCKET') else 'dummy-bucket'
            s3_prefix = "3d_assets"

            # Single base path for this page: the parent 'app/gradio' directory.
            # Use this for all local S3-like downloads and decimation outputs.
            base_gradio_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            # Download folder: ../s3_downloads (i.e. app/gradio/s3_downloads)
            models_folder = os.path.join(base_gradio_dir, "s3_downloads")
            # Save model folder: ../s3_downloads/decimation_output
            output_folder = os.path.join(models_folder, "decimation_output")
            os.makedirs(models_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            # Clear the UI output box at the start of a run so previous runs don't remain visible
            yield ""
            # Report SAVE_LOCAL so it's easy to see whether local files will be retained
            yield f"[INFO] SAVE_LOCAL={SAVE_LOCAL} (if False, downloaded and decimated files will be removed after successful Mongo update)"
            # Create a session log file (append-only) in app/gradio/logs
            try:
                import uuid
                logs_dir = os.path.join(base_gradio_dir, "logs")
                os.makedirs(logs_dir, exist_ok=True)
                session_log_filename = f"decimation_{int(time.time())}_{uuid.uuid4().hex}.log"
                session_log_path = os.path.join(logs_dir, session_log_filename)
                session_log = open(session_log_path, 'a', encoding='utf-8')
            except Exception:
                session_log = None

            def _write_session_log(msg: str):
                """Write a single line to the session log (if available) and flush.
                Keep session log independent of the `log_accum` variable used for Gradio streaming.
                """
                if not session_log:
                    return
                try:
                    # Ensure we write one physical line per call
                    session_log.write(msg.rstrip('\n') + '\n')
                    session_log.flush()
                except Exception:
                    # best-effort logging; don't raise
                    pass
            # log accumulator for the realtime output in the gradio Ui
            log_accum = ""
            # Track downloaded input files and decimated outputs for optional cleanup
            downloaded_files = []
            decimated_local_files = []
            for category, asset_name, asset in assets_to_decimate:
                update_key = get_biome_asset_update_key(biome_id, asset_name)
                # Inline priority selection: painted -> mesh -> s3_3d_url (only if status == '3d_generated')
                input_url = None
                if isinstance(asset, dict):
                    attrs = asset.get("attributes") if isinstance(asset.get("attributes"), dict) else {}
                    # painted first
                    if asset.get("painted_url"):
                        input_url = asset.get("painted_url")
                    elif attrs.get("painted_url"):
                        input_url = attrs.get("painted_url")
                    # then mesh
                    if not input_url:
                        # Track downloaded input files for optional cleanup
                        downloaded_files = []
                        if asset.get("mesh_url"):
                            input_url = asset.get("mesh_url")
                        elif attrs.get("mesh_url"):
                            input_url = attrs.get("mesh_url")
                    # finally s3_3d_url only if status indicates generated
                    if not input_url and asset.get("status") == "3d_generated":
                        if attrs.get("s3_3d_url"):
                            input_url = attrs.get("s3_3d_url")
                        elif asset.get("s3_3d_url"):
                            input_url = asset.get("s3_3d_url")

                if not input_url:
                    yield f"[ERROR] Asset '{asset_name}' has no usable input URL (painted/mesh/s3_3d_url with status=3d_generated), skipping."
                    # If we have an update_key, mark the asset as errored in Mongo
                    if update_key:
                        try:
                            update_or_add_biome_asset(biome_id, update_key, {"decimation_status": "error", "decimation_error": "no input_url"})
                        except Exception as _e:
                            yield f"[WARN] Could not write 'no input_url' error to Mongo for {asset_name}: {_e}"
                    else:
                        yield f"[WARN] No update_key available for {asset_name}; cannot write error state to Mongo"
                    continue
                try:
                    parsed = urlparse(input_url)
                    bucket = parsed.netloc.split('.')[0] if parsed.netloc else bucket_name
                    key = parsed.path.lstrip('/')
                    s3_filename = os.path.basename(key)
                    local_file = os.path.abspath(os.path.join(models_folder, s3_filename))
                    download_from_s3(bucket, key, local_file)
                    downloaded_files.append(local_file)
                    # keep original detailed download message
                    yield f"[S3] Downloaded {input_url} to {local_file}"
                except Exception as e:
                    yield f"[ERROR] Failed to download s3_3d_url for {asset_name}: {e}"
                    if update_key:
                        update_or_add_biome_asset(biome_id, update_key, {"decimation_status": "error", "decimation_error": str(e)})
                    continue
                if update_key:
                    try:
                        update_or_add_biome_asset(biome_id, update_key, {"decimation_status": "queued"})
                    except Exception as _e:
                        yield f"[WARN] Failed to set 'queued' in Mongo for {asset_name}: {_e}"
                else:
                    yield f"[WARN] No update_key available for {asset_name}; could not set 'queued' status in Mongo"
                decimated_assets = {}
                success_count = 0
                # Resolve blender binary using config, PATH, or common locations
                import shutil
                blender_path = getattr(settings, "BLENDER_PATH", None)
                if not blender_path:
                    p = "/usr/bin/blender"
                    if os.path.exists(p) and os.access(p, os.X_OK):
                        blender_path = p
                if not blender_path:
                    p = "/usr/bin/blender"
                    if os.path.exists(p) and os.access(p, os.X_OK):
                        blender_path = p
                if not blender_path:
                    # Inform user and mark asset as error in Mongo if possible
                    yield f"[ERROR] Blender executable not found. Set BLENDER_PATH in .env or install blender on PATH on this host."
                    if update_key:
                        update_or_add_biome_asset(biome_id, update_key, {"decimation_status": "error", "decimation_error": "Blender executable not found"})
                    continue
                decimation_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'decimate_only.py'))
                blender_cmd = [
                    blender_path,
                    "--background",
                    "--python", decimation_script,
                    "--",
                    asset_name,
                    local_file
                ]
                yield f"[INFO] Using Blender binary: {blender_path}\n[INFO] Running Blender command: {' '.join(blender_cmd)}"
### --- IGNORE --- USE_THIS_ONLY_WHEN_CPU_RESOURCES_ARE_CONSTRAINED
                try:
                    # Prepare a constrained environment for Blender to reduce CPU/RAM/IO pressure
                    env = os.environ.copy()
                    # Allow overrides from settings, default to 2 threads for BLAS/OMP
                    env.update({
                        "OMP_NUM_THREADS": str(getattr(settings, "BLENDER_OMP_THREADS", 2)),
                        "MKL_NUM_THREADS": str(getattr(settings, "BLENDER_MKL_THREADS", 2)),
                        "OPENBLAS_NUM_THREADS": str(getattr(settings, "BLENDER_OPENBLAS_THREADS", 2)),
                        "BLIS_NUM_THREADS": str(getattr(settings, "BLENDER_BLIS_THREADS", 2)),
                    })
                    # On POSIX, lower niceness and ignore SIGINT in the child via preexec_fn
                    preexec = None
                    try:
                        if os.name != 'nt':
                            def _preexec():
                                try:
                                    # lower CPU priority (use configured niceness)
                                    os.nice(getattr(settings, "BLENDER_NICE", 10))
                                except Exception:
                                    pass
                                # ignore Ctrl-C in child so parent handles termination
                                try:
                                    import signal
                                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                                except Exception:
                                    pass
                            preexec = _preexec
                    except Exception:
                        preexec = None
### --- IGNORE --- USE_THIS_ONLY_WHEN_CPU_RESOURCES_ARE_CONSTRAINED

# running of command to initiate decimation
                    process = subprocess.Popen(
                        blender_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=env
                        #preexec_fn=preexec
                    )
                    output_lines = []
                    output_json = None
                    # Accumulate and yield all output lines for Gradio
                    for line in process.stdout:
                        output_lines.append(line)
                        # Append to local accumulator for UI streaming
                        log_accum += line.rstrip() + "\n"
                        # Also write the raw blender output line to the session log
                        try:
                            _write_session_log(line.rstrip())
                        except Exception:
                            pass
                        yield log_accum
                        # try quick per-line JSON parse (some blender scripts may print compact JSON on one line)
                        try:
                            maybe_json = json.loads(line)
                            output_json = maybe_json
                        except Exception:
                            pass
                    process.wait(timeout=1200)
                    full_text = ''.join(output_lines)
                    if process.returncode != 0:
                        msg = f"[ERROR] Blender decimation failed for {asset_name}: Return code {process.returncode}\nSee Blender output above."
                        log_accum += msg + "\n"
                        _write_session_log(msg)
                        yield log_accum
                        # close session log on error for this asset and continue to next asset
                        continue
                    # If we didn't get JSON per-line, try parsing the full stdout as JSON
                    if not output_json:
                        try:
                            output_json = json.loads(full_text)
                        except Exception:
                            # Attempt to extract a JSON object from within a larger log blob
                            first = full_text.find('{')
                            last = full_text.rfind('}')
                            if first != -1 and last != -1 and last > first:
                                try:
                                    candidate = full_text[first:last+1]
                                    output_json = json.loads(candidate)
                                except Exception:
                                    output_json = None
                    if not output_json:
                        # show a short tail of Blender output to help debugging without duplicating the whole stream
                        tail = full_text[-2000:] if len(full_text) > 2000 else full_text
                        msg = f"[ERROR] No valid JSON output from Blender script for {asset_name}.\nLast part of Blender output:\n{tail}"
                        log_accum += msg + "\n"
                        _write_session_log(msg)
                        yield log_accum
                        continue
                    for suffix, data in output_json.items():
                        if "error" in data:
                            decimated_assets[f"decimated_{suffix}"] = {"error": data["error"]}
                            msg = f"[ERROR] Decimation failed for {asset_name} profile {suffix}: {data['error']}"
                            log_accum += msg + "\n"
                            _write_session_log(msg)
                            yield log_accum
                            continue
                        local_fbx = data["local_file"]
                        # remember decimated output to delete later if SAVE_LOCAL is False
                        decimated_local_files.append(local_fbx)
                        if not os.path.exists(local_fbx):
                            log_accum += f"[ERROR] Output file not found: {local_fbx} for {asset_name} profile {suffix}.\n"
                            decimated_assets[f"decimated_{suffix}"] = {"error": f"Output file not found: {local_fbx}"}
                            yield log_accum
                            continue
                        try:
                            s3_dest = f"{s3_prefix}/decimated/{os.path.basename(local_fbx)}"
                            msg = f"[DEBUG] Uploading {local_fbx} to S3 bucket '{bucket_name}' at key '{s3_dest}'"
                            log_accum += msg + "\n"
                            _write_session_log(msg)
                            yield log_accum
                            upload_to_s3(bucket_name, s3_dest, local_fbx)
                            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_dest}"
                            decimated_assets[f"decimated_{suffix}"] = {
                                "url": s3_url,
                                "poly_before": data.get("poly_before"),
                                "poly_after": data.get("poly_after"),
                                "reduction_ratio": data.get("reduction_ratio"),
                                "status": "complete"
                            }
                            msg = f"[SUCCESS] Uploaded {local_fbx} to S3 bucket '{bucket_name}' at key '{s3_dest}'"
                            log_accum += msg + "\n"
                            _write_session_log(msg)
                            msg2 = f"[SUCCESS] {suffix}: {data.get('poly_before')} → {data.get('poly_after')} polygons for {asset_name}"
                            log_accum += msg2 + "\n"
                            _write_session_log(msg2)
                            success_count += 1
                            yield log_accum
                        except Exception as e:
                            import traceback
                            tb = traceback.format_exc()
                            decimated_assets[f"decimated_{suffix}"] = {"error": f"S3 upload failed: {e}"}
                            msg = f"[ERROR] S3 upload failed for {local_fbx}: {e}\n{tb}"
                            log_accum += msg + "\n"
                            _write_session_log(msg)
                            yield log_accum
                        # decimated_local_files appended above; cleanup will happen after Mongo update
                except Exception as e:
                    import traceback
                    tb = traceback.format_exc()
                    msg = f"[ERROR] Blender decimation failed for {asset_name}: {e}\n{tb}"
                    _write_session_log(msg)
                    yield msg
                update_data = {
                    "decimation_status": "completed" if success_count > 0 else "failed",
                    "decimated_assets": decimated_assets,
                    "decimation_timestamp": int(time.time()),
                    "decimation_profiles_processed": success_count,
                    "decimation_profiles_total": len(DECIMATION_PROFILES)
                }
                # Remove decimation_error if decimation succeeded
                if success_count > 0:
                    update_data["decimation_error"] = None
                mongo_write_ok = False
                if update_key:
                    try:
                        # If decimation_error is None, remove the field from Mongo
                        if "decimation_error" in update_data and update_data["decimation_error"] is None:
                            update_or_add_biome_asset(biome_id, update_key, {**update_data, "$unset": {"decimation_error": ""}})
                        else:
                            update_or_add_biome_asset(biome_id, update_key, update_data)
                        mongo_write_ok = True
                    except Exception as _e:
                        msg = f"[WARN] Failed to write decimation results to Mongo for {asset_name}: {_e}"
                        try:
                            _write_session_log(msg)
                        except Exception:
                            pass
                        yield msg
                        mongo_write_ok = False
                else:
                    msg = f"[WARN] No update_key available for {asset_name}; decimation results not written to Mongo"
                    _write_session_log(msg)
                    yield msg
                    mongo_write_ok = False

                # Cleanup downloaded input file(s) and decimated outputs depending on SAVE_LOCAL flag.
                try:
                    if not SAVE_LOCAL:
                        if mongo_write_ok:
                            to_cleanup = []
                            to_cleanup.extend(downloaded_files or [])
                            to_cleanup.extend(decimated_local_files or [])
                            # ensure local input file is also cleaned
                            to_cleanup.append(local_file)
                            msgs = _cleanup_files(to_cleanup)
                            for m in msgs:
                                _write_session_log(m)
                                yield m
                            downloaded_files = []
                            decimated_local_files = []
                        else:
                            msg = f"[WARN] Skipping local cleanup for {asset_name} because Mongo update did not complete. Set SAVE_LOCAL=True to keep files for debugging."
                            _write_session_log(msg)
                            yield msg
                except Exception as e:
                    msg = f"[WARN] Cleanup step failed: {e}"
                    _write_session_log(msg)
                    yield msg
            final_ts = f"\nTimestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            log_accum += final_ts
            _write_session_log(final_ts)
            yield log_accum
            # Close session log for this run
            try:
                if session_log:
                    session_log.close()
            except Exception:
                pass

        def handle_check_status(coll_name, biome_id, asset_id, biome_choices):
            db_name = settings.MONGODB_DB_NAME
            biome_name_display = next((name for name, _id in biome_choices if _id == biome_id), biome_id)
            asset_name_display = asset_id
            if asset_id == "all":
                asset_name_display = "All Assets"
            return get_decimation_status(db_name, coll_name, biome_name_display, asset_name_display, biome_choices)

        # --- Wiring ---
        decimation_collection.change(
            fn=update_biomes_dropdown_decim,
            inputs=[decimation_collection],
            outputs=[biome_dropdown_decim, biome_choices_decim]
        )
        biome_dropdown_decim.change(
            fn=update_assets_dropdown_decim,
            inputs=[decimation_collection, biome_dropdown_decim, biome_choices_decim],
            outputs=[asset_dropdown_decim]
        )
        refresh_assets_btn.click(
            fn=update_assets_dropdown_decim,
            inputs=[decimation_collection, biome_dropdown_decim, biome_choices_decim],
            outputs=[asset_dropdown_decim]
        )
        start_decimation_btn.click(
            fn=handle_start_decimation,
            inputs=[decimation_collection, biome_dropdown_decim, asset_dropdown_decim, biome_choices_decim],
            outputs=[decimation_status_output],
            api_name=None,
            queue=True,
            show_progress=True
        )
        check_status_btn.click(
            fn=handle_check_status,
            inputs=[decimation_collection, biome_dropdown_decim, asset_dropdown_decim, biome_choices_decim],
            outputs=[decimation_status_output]
        )

    return decimation_ui
from urllib.parse import urlparse
import os
import gradio as gr
import time
import threading
import queue
import uuid
from app.config import settings
from app.services.mongo_service import get_db, get_biome_choices_live
from app.services.mongo_service import (
    get_biome,
    biome_assets_for_task,
    update_or_add_biome_asset,
    get_biome_asset_update_key,
)
from app.services.aws_service import upload_to_s3, download_from_s3

# ──────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────

DECIMATION_PROFILES = [
    ("100k", 100_000, "COLLAPSE", None),
    ("300k", 300_000, "COLLAPSE", None),
    ("50k",   50_000, "COLLAPSE", None),
    ("40k",   40_000, "COLLAPSE", None),
]

SAVE_LOCAL = False

# Timeout for how long the Gradio generator polls before giving up
# (the background thread continues regardless)
POLL_TIMEOUT_SECONDS = 3600  # 1 hour max display time


# ──────────────────────────────────────────────────────────────────
#  GLOBAL JOB TRACKER
#  Survives client disconnects. Each job has:
#    - thread: the background worker
#    - msg_queue: queue.Queue of log lines
#    - status: "running" | "completed" | "failed"
#    - result_summary: final summary string
# ──────────────────────────────────────────────────────────────────

_jobs = {}  # job_id -> {thread, msg_queue, status, result_summary}
_jobs_lock = threading.Lock()


def _cleanup_files(files):
    msgs = []
    for f in files or []:
        try:
            if os.path.exists(f):
                os.remove(f)
        except Exception as e:
            msgs.append(f"[WARN] Could not remove {f}: {e}")
    return msgs


# ──────────────────────────────────────────────────────────────────
#  BACKGROUND WORKER
#  Runs in its own thread. Blender stdout goes to a FILE (not pipe)
#  so Blender never hangs even if nobody reads the output.
# ──────────────────────────────────────────────────────────────────

def _decimation_worker(
    job_id: str,
    biome_id: str,
    assets_to_decimate: list,
    bucket_name: str,
    s3_prefix: str,
    models_folder: str,
    output_folder: str,
    base_gradio_dir: str,
):
    """
    Runs entirely in a background thread.
    Posts log messages to _jobs[job_id]['msg_queue'].
    Completes even if the Gradio client disconnects.
    """
    import subprocess, json, traceback, shutil

    msg_q = _jobs[job_id]["msg_queue"]

    def _log(msg: str):
        """Thread-safe log: push to queue + write to session log."""
        msg_q.put(msg)
        if session_log:
            try:
                session_log.write(msg.rstrip("\n") + "\n")
                session_log.flush()
            except Exception:
                pass

    # Session log file
    session_log = None
    try:
        logs_dir = os.path.join(base_gradio_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        session_log_path = os.path.join(logs_dir, f"decimation_{job_id}.log")
        session_log = open(session_log_path, "a", encoding="utf-8")
    except Exception:
        pass

    _log(f"[INFO] Job {job_id} started in background thread")
    _log(
        f"[INFO] SAVE_LOCAL={SAVE_LOCAL} | "
        f"Processing {len(assets_to_decimate)} asset(s)"
    )

    total_success = 0
    total_failed = 0

    for category, asset_name, asset in assets_to_decimate:
        update_key = get_biome_asset_update_key(biome_id, asset_name)
        downloaded_files = []
        decimated_local_files = []
        local_file = None

        # ── Resolve input URL ────────────────────────────────────
        input_url = None
        if isinstance(asset, dict):
            attrs = (
                asset.get("attributes")
                if isinstance(asset.get("attributes"), dict)
                else {}
            )
            input_url = asset.get("painted_url") or attrs.get("painted_url")
            if not input_url:
                input_url = asset.get("mesh_url") or attrs.get("mesh_url")
            if not input_url and asset.get("status") == "3d_generated":
                input_url = attrs.get("s3_3d_url") or asset.get("s3_3d_url")

        if not input_url:
            _log(f"[ERROR] '{asset_name}': no usable input URL, skipping.")
            if update_key:
                try:
                    update_or_add_biome_asset(
                        biome_id,
                        update_key,
                        {
                            "decimation_status": "error",
                            "decimation_error": "no input_url",
                        },
                    )
                except Exception as _e:
                    _log(f"[WARN] Mongo error-write failed: {_e}")
            total_failed += 1
            continue

        # ── Download from S3 ─────────────────────────────────────
        try:
            parsed = urlparse(input_url)
            bucket = (
                parsed.netloc.split(".")[0] if parsed.netloc else bucket_name
            )
            key = parsed.path.lstrip("/")
            s3_filename = os.path.basename(key)
            local_file = os.path.abspath(
                os.path.join(models_folder, s3_filename)
            )
            download_from_s3(bucket, key, local_file)
            downloaded_files.append(local_file)
            _log(f"[S3] Downloaded {input_url} → {local_file}")
        except Exception as e:
            _log(f"[ERROR] S3 download failed for {asset_name}: {e}")
            if update_key:
                try:
                    update_or_add_biome_asset(
                        biome_id,
                        update_key,
                        {
                            "decimation_status": "error",
                            "decimation_error": f"download: {e}",
                        },
                    )
                except Exception:
                    pass
            total_failed += 1
            continue

        # ── Mark queued ───────────────────────────────────────────
        if update_key:
            try:
                update_or_add_biome_asset(
                    biome_id, update_key, {"decimation_status": "processing"}
                )
            except Exception as _e:
                _log(f"[WARN] Could not set 'processing': {_e}")

        # ── Find Blender ──────────────────────────────────────────
        blender_path = getattr(settings, "BLENDER_PATH", None)
        if not blender_path:
            for candidate in ["/usr/bin/blender", "/snap/bin/blender"]:
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    blender_path = candidate
                    break
        if not blender_path:
            blender_path = shutil.which("blender")
        if not blender_path:
            _log("[ERROR] Blender not found. Set BLENDER_PATH in .env.")
            if update_key:
                try:
                    update_or_add_biome_asset(
                        biome_id,
                        update_key,
                        {
                            "decimation_status": "error",
                            "decimation_error": "Blender not found",
                        },
                    )
                except Exception:
                    pass
            total_failed += 1
            continue

        decimation_script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "decimate_only.py")
        )
        blender_cmd = [
            blender_path,
            "--background",
            "--python",
            decimation_script,
            "--",
            asset_name,
            local_file,
            output_folder,
        ]

        _log(f"[INFO] Running: {' '.join(blender_cmd)}")

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        #  KEY FIX: Write Blender stdout to a FILE, not a pipe.
        #  This means Blender NEVER blocks on write, even if
        #  nobody is reading. The background thread reads the
        #  file after Blender completes.
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        blender_log_path = os.path.join(
            output_folder, f"{asset_name}_{job_id}_blender.log"
        )
        output_json = None
        blender_ok = False

        try:
            env = os.environ.copy()
            env.update(
                {
                    "OMP_NUM_THREADS": str(
                        getattr(settings, "BLENDER_OMP_THREADS", 2)
                    ),
                    "MKL_NUM_THREADS": str(
                        getattr(settings, "BLENDER_MKL_THREADS", 2)
                    ),
                    "OPENBLAS_NUM_THREADS": str(
                        getattr(settings, "BLENDER_OPENBLAS_THREADS", 2)
                    ),
                }
            )

            with open(blender_log_path, "w") as blender_log_f:
                process = subprocess.Popen(
                    blender_cmd,
                    stdout=blender_log_f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=env,
                )

                _log(
                    f"[INFO] Blender PID {process.pid} started, "
                    f"log → {blender_log_path}"
                )

                # Wait for completion (no pipe to read, so just wait)
                # Post periodic heartbeat so the UI shows it's alive
                while True:
                    try:
                        retcode = process.wait(timeout=10)
                        break  # process finished
                    except subprocess.TimeoutExpired:
                        _log(f"[INFO] Blender still running (PID {process.pid})…")

            # ── Read the log file ─────────────────────────────────
            full_text = ""
            try:
                with open(blender_log_path, "r") as f:
                    full_text = f.read()
            except Exception as e:
                _log(f"[WARN] Could not read Blender log: {e}")

            # Stream relevant lines to the message queue
            for line in full_text.splitlines():
                # Only forward lines that are useful (skip Blender boilerplate)
                if any(
                    kw in line
                    for kw in [
                        "[LOD",
                        "ERROR",
                        "WARN",
                        "SUCCESS",
                        "Exported",
                        "Baking",
                        "Decimated",
                        "Auto-cage",
                        "saved",
                    ]
                ):
                    _log(line)

            if retcode != 0:
                _log(
                    f"[ERROR] Blender exited with code {retcode} "
                    f"for {asset_name}"
                )
            else:
                _log(f"[INFO] Blender completed successfully for {asset_name}")

            # ── Parse JSON from output ────────────────────────────
            try:
                output_json = json.loads(full_text)
            except Exception:
                first = full_text.find("{")
                last = full_text.rfind("}")
                if first != -1 and last > first:
                    try:
                        output_json = json.loads(full_text[first : last + 1])
                    except Exception:
                        pass

            if output_json and isinstance(output_json, dict):
                blender_ok = True
            else:
                _log(f"[ERROR] No valid JSON in Blender output for {asset_name}")

        except Exception as e:
            tb = traceback.format_exc()
            _log(f"[ERROR] Blender process failed for {asset_name}: {e}\n{tb}")

        # ── Process LOD results ───────────────────────────────────
        decimated_assets = {}
        success_count = 0

        if output_json and isinstance(output_json, dict):
            for suffix, data in output_json.items():
                if not isinstance(data, dict):
                    continue

                if "error" in data:
                    decimated_assets[f"decimated_{suffix}"] = {
                        "error": data["error"]
                    }
                    _log(
                        f"[ERROR] {asset_name}/{suffix}: {data['error']}"
                    )
                    continue

                try:
                    local_output = data.get("file") or data.get("local_file") or ""
                    poly_before = data.get("faces_before") or data.get("poly_before")
                    poly_after = data.get("faces_after") or data.get("poly_after")
                    reduction = data.get("reduction_pct") or data.get("reduction_ratio")

                    if not local_output:
                        _log(
                            f"[ERROR] No output path for "
                            f"{asset_name}/{suffix}. Keys: {list(data.keys())}"
                        )
                        decimated_assets[f"decimated_{suffix}"] = {
                            "error": "no output file path"
                        }
                        continue

                    decimated_local_files.append(local_output)

                    if not os.path.exists(local_output):
                        _log(f"[ERROR] File not found: {local_output}")
                        decimated_assets[f"decimated_{suffix}"] = {
                            "error": f"file not found: {local_output}"
                        }
                        continue

                    # Upload to S3
                    s3_dest = (
                        f"{s3_prefix}/decimated/{os.path.basename(local_output)}"
                    )
                    _log(f"[S3] Uploading {os.path.basename(local_output)}…")

                    upload_to_s3(bucket_name, s3_dest, local_output)
                    s3_url = (
                        f"https://{bucket_name}.s3.amazonaws.com/{s3_dest}"
                    )

                    decimated_assets[f"decimated_{suffix}"] = {
                        "url": s3_url,
                        "poly_before": poly_before,
                        "poly_after": poly_after,
                        "reduction_ratio": reduction,
                        "status": "complete",
                    }
                    success_count += 1
                    _log(
                        f"[SUCCESS] {suffix}: {poly_before} → {poly_after} "
                        f"polys for {asset_name}"
                    )

                except Exception as e:
                    tb = traceback.format_exc()
                    decimated_assets[f"decimated_{suffix}"] = {
                        "error": f"processing failed: {e}"
                    }
                    _log(f"[ERROR] {asset_name}/{suffix}: {e}\n{tb}")

        # ── Write to MongoDB ──────────────────────────────────────
        # THIS CODE NOW ALWAYS RUNS regardless of client connection
        update_data = {
            "decimation_status": "completed" if success_count > 0 else "failed",
            "decimated_assets": decimated_assets,
            "decimation_timestamp": int(time.time()),
            "decimation_profiles_processed": success_count,
            "decimation_profiles_total": len(DECIMATION_PROFILES),
        }
        if success_count > 0:
            update_data["decimation_error"] = None
        elif not blender_ok:
            update_data["decimation_error"] = "Blender produced no JSON output"

        mongo_write_ok = False
        if update_key:
            try:
                update_or_add_biome_asset(biome_id, update_key, update_data)
                mongo_write_ok = True
                _log(
                    f"[MONGO] ✓ Updated {asset_name}: "
                    f"status={update_data['decimation_status']}, "
                    f"{success_count}/{len(DECIMATION_PROFILES)} profiles"
                )
                if success_count > 0:
                    total_success += 1
                else:
                    total_failed += 1
            except Exception as _e:
                _log(f"[ERROR] Mongo update FAILED for {asset_name}: {_e}")
                total_failed += 1
        else:
            _log(f"[WARN] No update_key for {asset_name}")
            total_failed += 1

        # ── Cleanup ───────────────────────────────────────────────
        try:
            if not SAVE_LOCAL and mongo_write_ok:
                to_clean = list(
                    set(
                        downloaded_files
                        + decimated_local_files
                        + ([local_file] if local_file else [])
                    )
                )
                # Also clean up the Blender log file
                if os.path.exists(blender_log_path):
                    to_clean.append(blender_log_path)
                msgs = _cleanup_files(to_clean)
                for m in msgs:
                    _log(m)
            elif not mongo_write_ok:
                _log(f"[WARN] Keeping files for {asset_name} (Mongo write failed)")
        except Exception as e:
            _log(f"[WARN] Cleanup error: {e}")

    # ── Job complete ──────────────────────────────────────────────
    summary = (
        f"\n{'━'*50}\n"
        f"✅ Job {job_id} complete\n"
        f"   Success: {total_success} | Failed: {total_failed}\n"
        f"   Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"{'━'*50}"
    )
    _log(summary)

    with _jobs_lock:
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result_summary"] = summary

    # Signal the queue that we're done
    msg_q.put(None)  # sentinel

    if session_log:
        try:
            session_log.close()
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────
#  STATUS HELPERS
# ──────────────────────────────────────────────────────────────────

def get_decimation_status(
    database_name, collection_name, biome_name, asset_name, biome_choices
):
    if not biome_name or asset_name == "All Assets":
        return "Please select a specific asset to check status."
    biome_id = next(
        (_id for name, _id in biome_choices if name == biome_name), None
    )
    if not biome_id:
        return "Selected biome not found."
    try:
        biome_data = get_biome(biome_id)
        if not biome_data:
            return "Biome data not found."

        possible_structures = biome_data.get("possible_structures", {})
        asset_data = None
        for cat in ["buildings", "creatures", "props", "terrain"]:
            assets = possible_structures.get(cat, {})
            if asset_name in assets:
                asset_data = assets[asset_name]
                break
        if not asset_data:
            return f"Asset '{asset_name}' not found in biome."

        decimation_status = asset_data.get("decimation_status", "Not started")
        decimated_assets = asset_data.get("decimated_assets", {})

        txt = f"Decimation Status: {decimation_status}\n"
        if decimated_assets:
            txt += "\nDecimated Versions:\n"
            for profile, data in decimated_assets.items():
                if not isinstance(data, dict):
                    continue
                poly = data.get("poly_after") or data.get("faces_after") or "N/A"
                if "error" in data:
                    txt += f"  - {profile}: Error - {data['error']}\n"
                else:
                    txt += f"  - {profile}: {poly} polygons\n"

        ts = asset_data.get("decimation_timestamp")
        if ts:
            txt += f"\nLast Updated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))}"

        # Show active background jobs
        with _jobs_lock:
            active = [
                jid for jid, j in _jobs.items() if j["status"] == "running"
            ]
        if active:
            txt += f"\n\n🔄 Active background jobs: {len(active)}"

        return txt
    except Exception as e:
        return f"Error getting status: {str(e)}"


# ──────────────────────────────────────────────────────────────────
#  GRADIO UI
# ──────────────────────────────────────────────────────────────────

def decimation_page_ui():
    with gr.Blocks() as decimation_ui:
        gr.Markdown("# 🔧 3D Asset Decimation")
        gr.Markdown(
            "Reduce polygon count of 3D assets. "
            "**Work continues in background even if you close this tab.**"
        )

        db_name = settings.MONGODB_DB_NAME
        db = get_db()
        collections = []
        if db is not None and db_name:
            collections = db.list_collection_names()
            collections = [str(c) for c in collections] if collections else []
        default_collection = collections[0] if collections else None

        decimation_collection = gr.Dropdown(
            label="Select Collection",
            choices=collections,
            value=default_collection,
            interactive=True,
        )

        biome_choices = (
            get_biome_choices_live(db_name, default_collection)
            if default_collection
            else []
        )
        biome_dd_choices = [(name, _id) for name, _id in biome_choices]
        default_biome = biome_dd_choices[0][1] if biome_dd_choices else None

        biome_dropdown_decim = gr.Dropdown(
            label="Select Biome",
            choices=biome_dd_choices,
            value=default_biome,
            interactive=True,
        )

        asset_choices = [("All Assets", "all")]
        if default_biome:
            ad = biome_assets_for_task(default_biome, status_filter="3d_generated")
            if isinstance(ad, dict):
                asset_choices += [
                    (
                        f"{v.get('type','').title() if isinstance(v, dict) else 'Asset'}: {k}",
                        k,
                    )
                    for k, v in ad.items()
                ]

        asset_dropdown_decim = gr.Dropdown(
            label="Select Asset (only those ready for decimation)",
            choices=asset_choices,
            value="all",
            interactive=True,
        )

        biome_choices_decim = gr.State(biome_choices)

        with gr.Row():
            start_btn = gr.Button("🚀 Start Decimation", variant="primary")
            check_btn = gr.Button("📊 Check Status")
            refresh_btn = gr.Button("🔄 Refresh Assets")

        decimation_output = gr.Textbox(
            label="Decimation Status", interactive=False, lines=15
        )

        # ── Event handlers ─────────────────────────────────────────

        def update_biomes_dropdown(collection_name):
            db_name_local = settings.MONGODB_DB_NAME
            if not db_name_local or not collection_name:
                return gr.Dropdown(choices=[], value=None), []
            bc = get_biome_choices_live(db_name_local, collection_name)
            dd = [(name, _id) for name, _id in bc]
            return (
                gr.Dropdown(choices=dd, value=dd[0][1] if dd else None),
                bc,
            )

        def update_assets_dropdown(collection_name, biome_id, biome_choices):
            if not biome_id:
                return gr.Dropdown(choices=[("All Assets", "all")], value="all")
            try:
                ad = biome_assets_for_task(
                    biome_id, status_filter="3d_generated"
                )
                ac = [("All Assets", "all")]
                if isinstance(ad, dict):
                    ac += [
                        (
                            f"{v.get('type','').title() if isinstance(v, dict) else 'Asset'}: {k}",
                            k,
                        )
                        for k, v in ad.items()
                    ]
                return gr.Dropdown(choices=ac, value="all")
            except Exception:
                return gr.Dropdown(choices=[("All Assets", "all")], value="all")

        def handle_start_decimation(coll_name, biome_id, asset_id, biome_choices):
            """
            Launches the decimation in a BACKGROUND THREAD.
            The Gradio generator polls the message queue for updates.
            If the client disconnects, the thread continues to completion.
            """
            assets_dict = biome_assets_for_task(
                biome_id, status_filter="3d_generated"
            )
            if not isinstance(assets_dict, dict) or not assets_dict:
                yield "No assets ready for decimation."
                return

            assets_to_decimate = []
            if asset_id == "all":
                for aname, aval in assets_dict.items():
                    assets_to_decimate.append(
                        (aval.get("type", "Asset"), aname, aval)
                    )
            else:
                if asset_id in assets_dict:
                    a = assets_dict[asset_id]
                    assets_to_decimate.append(
                        (a.get("type", "Asset"), asset_id, a)
                    )

            if not assets_to_decimate:
                yield "No assets ready for decimation."
                return

            bucket_name = (
                settings.AWS_S3_BUCKET
                if hasattr(settings, "AWS_S3_BUCKET")
                else "dummy-bucket"
            )
            s3_prefix = "3d_assets"

            base_gradio_dir = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..")
            )
            models_folder = os.path.join(base_gradio_dir, "s3_downloads")
            output_folder = os.path.join(models_folder, "decimation_output")
            os.makedirs(models_folder, exist_ok=True)
            os.makedirs(output_folder, exist_ok=True)

            # ── Create job ────────────────────────────────────────
            job_id = uuid.uuid4().hex[:12]
            msg_q = queue.Queue()

            with _jobs_lock:
                _jobs[job_id] = {
                    "thread": None,
                    "msg_queue": msg_q,
                    "status": "running",
                    "result_summary": "",
                    "started_at": time.time(),
                }

            worker = threading.Thread(
                target=_decimation_worker,
                args=(
                    job_id,
                    biome_id,
                    assets_to_decimate,
                    bucket_name,
                    s3_prefix,
                    models_folder,
                    output_folder,
                    base_gradio_dir,
                ),
                daemon=True,  # daemon so it doesn't prevent shutdown
                name=f"decimate-{job_id}",
            )

            with _jobs_lock:
                _jobs[job_id]["thread"] = worker

            worker.start()

            # ── Poll the message queue and yield to Gradio ────────
            # If client disconnects, this generator dies but the
            # worker thread continues to completion.
            log_accum = (
                f"[INFO] Job {job_id} launched in background thread.\n"
                f"[INFO] Work continues even if you close this tab.\n"
                f"[INFO] Use 'Check Status' to see results later.\n\n"
            )
            yield log_accum

            deadline = time.time() + POLL_TIMEOUT_SECONDS

            while time.time() < deadline:
                try:
                    # Block for up to 2 seconds waiting for a message
                    msg = msg_q.get(timeout=2.0)

                    if msg is None:
                        # Sentinel: worker is done
                        # Drain any remaining messages
                        while not msg_q.empty():
                            try:
                                remaining = msg_q.get_nowait()
                                if remaining:
                                    log_accum += remaining.rstrip("\n") + "\n"
                            except queue.Empty:
                                break
                        yield log_accum
                        return

                    log_accum += msg.rstrip("\n") + "\n"
                    yield log_accum

                except queue.Empty:
                    # No message in 2s — check if worker is still alive
                    if not worker.is_alive():
                        # Worker finished but we missed the sentinel
                        while not msg_q.empty():
                            try:
                                remaining = msg_q.get_nowait()
                                if remaining:
                                    log_accum += remaining.rstrip("\n") + "\n"
                            except queue.Empty:
                                break
                        yield log_accum
                        return
                    # Otherwise just continue polling
                    continue

            # If we reach here, poll timeout exceeded
            log_accum += (
                f"\n[INFO] UI poll timeout ({POLL_TIMEOUT_SECONDS}s) reached.\n"
                f"[INFO] Background job {job_id} is still running.\n"
                f"[INFO] Use 'Check Status' to see results.\n"
            )
            yield log_accum

        def handle_check_status(coll_name, biome_id, asset_id, biome_choices):
            db_name = settings.MONGODB_DB_NAME
            biome_name_display = next(
                (name for name, _id in biome_choices if _id == biome_id),
                biome_id,
            )
            asset_name_display = "All Assets" if asset_id == "all" else asset_id
            return get_decimation_status(
                db_name,
                coll_name,
                biome_name_display,
                asset_name_display,
                biome_choices,
            )

        # ── Wiring ─────────────────────────────────────────────────

        decimation_collection.change(
            fn=update_biomes_dropdown,
            inputs=[decimation_collection],
            outputs=[biome_dropdown_decim, biome_choices_decim],
        )
        biome_dropdown_decim.change(
            fn=update_assets_dropdown,
            inputs=[
                decimation_collection,
                biome_dropdown_decim,
                biome_choices_decim,
            ],
            outputs=[asset_dropdown_decim],
        )
        refresh_btn.click(
            fn=update_assets_dropdown,
            inputs=[
                decimation_collection,
                biome_dropdown_decim,
                biome_choices_decim,
            ],
            outputs=[asset_dropdown_decim],
        )
        start_btn.click(
            fn=handle_start_decimation,
            inputs=[
                decimation_collection,
                biome_dropdown_decim,
                asset_dropdown_decim,
                biome_choices_decim,
            ],
            outputs=[decimation_output],
            api_name=None,
            queue=True,
            show_progress=True,
        )
        check_btn.click(
            fn=handle_check_status,
            inputs=[
                decimation_collection,
                biome_dropdown_decim,
                asset_dropdown_decim,
                biome_choices_decim,
            ],
            outputs=[decimation_output],
        )

    return decimation_ui

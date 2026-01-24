"""Standalone Unirig worker.

- Peeks head of Redis list `rig_model` and processes item in-place.
- Keeps item in Redis while processing; on success removes the head occurrence
  (single LREM 1), on failure requeues the payload to tail (RPUSH) then removes
  the head occurrence so the failed job moves to the tail.
- Does NOT write intermediate rigging status fields. On failure writes only
  `rigging_error`. On success sets `rigged_status` = 'rigged' and `rigged_timestamp`.

Configuration via .env (loaded from cwd):
- MONGODB_URL, MONGODB_DB_NAME
- CELERY_BROKER_URL or REDIS_URL (or REDIS_HOST/REDIS_PORT)
- AWS_S3_BUCKET (and AWS creds in env)
- UNIRIG_CMD (or use per-job `unirig_cmd` or `use_pipeline`)
- UNIRIG_MAX_RETRIES (optional)
"""
from __future__ import annotations
import os
import sys
import time
import json
import shutil
import logging
import subprocess
import traceback
from urllib.parse import urlparse

from dotenv import load_dotenv

# load .env from cwd so running on remote host is simple
load_dotenv(os.path.join(os.getcwd(), ".env"))

import redis
import requests
from pymongo import MongoClient
from pymongo.errors import PyMongoError

LOG = logging.getLogger("unirig_worker")
# stdout-only logging
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s"))
root_logger = logging.getLogger()
for h in list(root_logger.handlers):
    root_logger.removeHandler(h)
root_logger.addHandler(stream_handler)
root_logger.setLevel(logging.INFO)
LOG.setLevel(logging.INFO)
LOG.propagate = True


# --- S3 helpers ---
def download_from_s3(bucket: str, key: str, download_path: str):
    import boto3
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, key, download_path)
    LOG.info("[S3] Downloaded: %s/%s -> %s", bucket, key, download_path)


def upload_to_s3(bucket: str, key: str, file_path: str):
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(file_path, bucket, key)
    LOG.info("[S3] Uploaded: %s -> %s/%s", file_path, bucket, key)


# --- Mongo helpers ---
_mongo_client: MongoClient | None = None
_mongo_db = None


def get_db():
    global _mongo_client, _mongo_db
    if _mongo_db is not None:
        return _mongo_db
    try:
        mongo_url = os.getenv("MONGODB_URL")
        mongo_db_name = os.getenv("MONGODB_DB_NAME")
        if not mongo_url or not mongo_db_name:
            LOG.error("MONGODB_URL or MONGODB_DB_NAME not set in environment")
            return None
        _mongo_client = MongoClient(mongo_url)
        # quick ping
        _mongo_client.admin.command("ismaster")
        _mongo_db = _mongo_client[mongo_db_name]
        return _mongo_db
    except PyMongoError as e:
        LOG.error("Failed to connect to MongoDB: %s", e)
        return None


def get_biome(biome_id: str):
    db = get_db()
    if db is None:
        return None
    col = db["biomes"]
    doc = col.find_one({"_id": biome_id})
    return doc

def update_or_add_biome_asset(biome_id: str, update_key: str, update_dict: dict):
    db = get_db()
    if db is None:
        return {"error": "db not connected"}
    col = db["biomes"]
    filt = {"_id": biome_id}
    try:
        projection = {update_key: 1}
        current_doc = col.find_one(filt, projection)
        # traverse nested dict at update_key
        parts = update_key.split('.') if update_key else []
        cur = current_doc or {}
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur.get(p)
            else:
                cur = {}
                break
        if not isinstance(cur, dict):
            cur = {}
        merged = {**cur, **update_dict}
        result = col.update_one(filt, {"$set": {update_key: merged}})
        return {"matched": getattr(result, 'matched_count', 0), "modified": getattr(result, 'modified_count', 0)}
    except Exception as e:
        LOG.error("Failed to update biome asset: %s", e)
        return {"error": str(e)}


def sweep_old_tmpdirs(parent: str, keep: int = 5):
    try:
        entries = [os.path.join(parent, p) for p in os.listdir(parent)]
    except Exception:
        return
    dirs = [p for p in entries if os.path.isdir(p)]
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    for d in dirs[keep:]:
        try:
            shutil.rmtree(d)
        except Exception:
            pass



# --- download helpers ---
def download_http_to_local(url: str, dest_path: str):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_s3_or_http(url: str, dest_folder: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme == 's3':
        bucket = parsed.netloc
        key = parsed.path.lstrip('/')
        local_name = os.path.basename(key)
        local_path = os.path.join(dest_folder, local_name)
        download_from_s3(bucket, key, local_path)
        return local_path
    else:
        local_name = os.path.basename(parsed.path) or f"model_{int(time.time())}"
        local_path = os.path.join(dest_folder, local_name)
        download_http_to_local(url, local_path)
        return local_path


# --- Unirig invocation helpers ---
def run_and_stream(cmd_list: list, cwd: str | None = None) -> tuple[int, str]:
    outputs = []
    try:
        proc = subprocess.Popen(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, cwd=cwd)
    except Exception as e:
        LOG.error('Failed to start process %s: %s', cmd_list, e)
        return 1, str(e)
    try:
        for line in proc.stdout:
            if line is None:
                continue
            print(line)
            outputs.append(line)
    except Exception:
        pass
    rc = proc.wait()
    return rc, ''.join(outputs)


def run_unirig(unirig_cmd_template: str, input_path: str, output_path: str, asset_name: str) -> tuple[int, str]:
    import shlex
    cmd_str = unirig_cmd_template.format(input=input_path, output=output_path, asset_name=asset_name)
    cmd_list = shlex.split(cmd_str)
    LOG.info("Running Unirig: %s", cmd_str)
    return run_and_stream(cmd_list)


def run_unirig_pipeline(tmpdir: str, local_model: str, final_output_path: str, asset_name: str, step2_cmd: str | None = None) -> tuple[int, str]:
    temp_dir = os.path.join(tmpdir, 'temp')
    temp_skel_dir = os.path.join(temp_dir, 'skeleton')
    results_dir = os.path.join(tmpdir, 'results')
    os.makedirs(temp_skel_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    temp_glb = os.path.join(temp_dir, f"{asset_name}.glb")
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(local_model, temp_glb)

    outputs = []

    def run_cmd(cmd_list, cwd_override: str | None = None):
        LOG.info('Running: %s', ' '.join(cmd_list))
        run_cwd = cwd_override if cwd_override is not None else tmpdir
        rc, out = run_and_stream(cmd_list, cwd=run_cwd)
        outputs.append(out)
        return rc

    gen_skel_out = os.path.join(results_dir, f"{asset_name}_skeleton.fbx")
    gen_skel_script = os.path.join(os.getcwd(), 'launch', 'inference', 'generate_skeleton.sh')
    rc = run_cmd(['bash', gen_skel_script, '--input', os.path.abspath(temp_glb), '--output', os.path.abspath(gen_skel_out)], cwd_override=os.getcwd())
    if rc != 0:
        return rc, '\n'.join(outputs)

    examples_skel_fbx = os.path.join(temp_skel_dir, f"{asset_name}.fbx")
    shutil.copy(gen_skel_out, examples_skel_fbx)

    if step2_cmd:
        cmd_rendered = step2_cmd.format(asset=asset_name, tmpdir=tmpdir)
        rc = run_cmd(['bash', '-lc', cmd_rendered], cwd_override=os.getcwd())
        if rc != 0:
            return rc, '\n'.join(outputs)

    gen_skin_out = os.path.join(results_dir, f"{asset_name}_skin.fbx")
    gen_skin_script = os.path.join(os.getcwd(), 'launch', 'inference', 'generate_skin.sh')
    rc = run_cmd(['bash', gen_skin_script, '--input', os.path.abspath(examples_skel_fbx), '--output', os.path.abspath(gen_skin_out)], cwd_override=os.getcwd())
    if rc != 0:
        return rc, '\n'.join(outputs)

    final_rigged = os.path.join(results_dir, f"{asset_name}_rigged.glb")
    merge_script = os.path.join(os.getcwd(), 'launch', 'inference', 'merge.sh')
    rc = run_cmd(['bash', merge_script, '--source', os.path.abspath(gen_skel_out), '--target', os.path.abspath(temp_glb), '--output', os.path.abspath(final_rigged)], cwd_override=os.getcwd())
    if rc != 0:
        return rc, '\n'.join(outputs)

    if os.path.exists(final_rigged):
        shutil.copy(final_rigged, final_output_path)
        return 0, '\n'.join(outputs)
    return 1, '\n'.join(outputs)


def upload_result_to_s3(local_output: str, biome_id: str, asset_name: str) -> str:
    bucket = os.getenv('AWS_S3_BUCKET')
    if not bucket:
        return local_output
    key = f"3d_assets/rigging/{biome_id}/{asset_name}/{os.path.basename(local_output)}"
    upload_to_s3(bucket, key, local_output)
    return f"https://{bucket}.s3.amazonaws.com/{key}"


def resolve_variant_from_asset(biome_id: str, asset_name: str, variant_key: str | None = None):
    biome = get_biome(biome_id)
    if not biome:
        return None
    possible_structures = biome.get('possible_structures', {})
    for category in ['buildings', 'creatures', 'props', 'terrain']:
        assets = possible_structures.get(category, {})
        if asset_name in assets:
            asset = assets[asset_name]
            dec = asset.get('decimated_assets') or {}
            if variant_key and variant_key in dec and isinstance(dec[variant_key], dict):
                return dec[variant_key].get('url')
            if isinstance(dec, dict) and dec:
                for k, v in dec.items():
                    if isinstance(v, dict) and v.get('url'):
                        return v.get('url'), k
            painted = asset.get('painted_url') or (asset.get('attributes') or {}).get('painted_url')
            if painted:
                return painted, 'painted_raw'
            raw = asset.get('s3_3d_url') or (asset.get('attributes') or {}).get('s3_3d_url')
            if raw:
                return raw, 'raw_3d'
    return None


def process_job(job: dict, unirig_cmd: str, max_retries: int = 2):
    biome_id = job.get('biome_id')
    asset_name = job.get('asset_name')
    variant_key = job.get('variant_key')
    model_url = job.get('model_url')
    update_key = job.get('update_key')
    rigged_asset_name = job.get('rigged_asset_name') or asset_name

    if not biome_id or not asset_name:
        LOG.error('Invalid job payload, missing biome_id or asset_name: %s', job)
        return False

    if not model_url:
        resolved = resolve_variant_from_asset(biome_id, asset_name, variant_key)
        if isinstance(resolved, tuple):
            model_url, derived_variant = resolved
            if not variant_key:
                variant_key = derived_variant
        else:
            model_url = resolved

    if not model_url:
        LOG.error('No model URL found for %s in biome %s', asset_name, biome_id)
        # Only write an error if the producer supplied an update_key
        if update_key:
            update_or_add_biome_asset(biome_id, update_key, {'rigging_error': 'no_model_url'})
        return False

    # Use repo-root `temp/{asset_name}` so intermediate files are easy to find
    base_tmp_parent = os.path.join(os.getcwd(), 'temp')
    os.makedirs(base_tmp_parent, exist_ok=True)
    # Recreate a fresh folder named by the asset for easy access (overwritten each run)
    safe_name = str(asset_name)
    tmpdir = os.path.join(base_tmp_parent, safe_name)
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir, ignore_errors=True)
    os.makedirs(tmpdir, exist_ok=True)
    try:
        local_model = download_s3_or_http(model_url, tmpdir)
    except Exception as e:
        LOG.error('Failed to download model %s: %s', model_url, e)
        if update_key:
            update_or_add_biome_asset(biome_id, update_key, {'rigging_error': str(e)})
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            # remove parent if empty
            os.rmdir(base_tmp_parent)
        except Exception:
            pass
        return False

    out_name = f"{rigged_asset_name}_{variant_key or 'rigged'}.glb"
    local_out = os.path.join(tmpdir, out_name)

    attempt = 0
    success = False
    last_err = None
    while attempt <= max_retries and not success:
        attempt += 1
        try:
            if (isinstance(unirig_cmd, str) and unirig_cmd.strip().lower() == 'pipeline') or job.get('use_pipeline'):
                rc, out = run_unirig_pipeline(tmpdir, local_model, local_out, rigged_asset_name, job.get('step2_cmd'))
            else:
                rc, out = run_unirig(unirig_cmd, local_model, local_out, rigged_asset_name)
            if rc == 0 and os.path.exists(local_out):
                success = True
                break
            last_err = f'Unirig failed rc={rc} output={out}'
            LOG.warning(last_err)
            time.sleep(2)
        except Exception as e:
            last_err = traceback.format_exc()
            LOG.error('Exception running unirig: %s', e)
            time.sleep(2)

    if not success:
        LOG.error('Unirig failed for %s after %d attempts: %s', asset_name, attempt, last_err)
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            os.rmdir(base_tmp_parent)
        except Exception:
            pass
        return False

    # upload
    try:
        s3_url = upload_result_to_s3(local_out, biome_id, rigged_asset_name)
        if update_key:
            
            if variant_key and str(variant_key).startswith('decimated_'):
                update_or_add_biome_asset(biome_id, update_key, {'decimated_assets': {variant_key: {'rigged_url': s3_url, 'rigged_timestamp': int(time.time())}}})
            else:
                update_or_add_biome_asset(biome_id, update_key, {'rigged_model_url': s3_url, 'rigged_timestamp': int(time.time())})
    except Exception as e:
        LOG.error('Upload or DB update failed: %s', e)
        
        shutil.rmtree(tmpdir, ignore_errors=True)
        try:
            os.rmdir(base_tmp_parent)
        except Exception:
            pass
        return False

    shutil.rmtree(tmpdir, ignore_errors=True)
    try:
        os.rmdir(base_tmp_parent)
    except Exception:
        pass
    LOG.info('Job complete: %s %s -> %s', biome_id, asset_name, s3_url)
    return True


def worker_loop(unirig_cmd: str, redis_host: str, redis_port: int, max_retries: int = 2):
    try:
        if isinstance(redis_host, str) and (redis_host.startswith('redis://') or redis_host.startswith('rediss://')):
            r = redis.from_url(redis_host)
            LOG.info('Connected to Redis via URL')
        else:
            r = redis.Redis(host=redis_host, port=redis_port, db=0)
            LOG.info('Connected to Redis %s:%s', redis_host, redis_port)
    except Exception as e:
        LOG.error('Failed to connect to Redis: %s', e)
        time.sleep(5)
        return
    while True:
        try:
            try:
                head = r.lindex('rig_model', 0)
            except Exception as e:
                LOG.error('Error reading rig_model head: %s', e)
                time.sleep(2)
                continue
            if not head:
                time.sleep(1)
                continue
            payload = head.decode('utf-8') if isinstance(head, (bytes, bytearray)) else head
            try:
                job = json.loads(payload)
            except Exception:
                LOG.error('Invalid JSON payload in rig_model: %s', payload)
                try:
                    r.lrem('rig_model', 1, payload)
                except Exception:
                    LOG.exception('Failed to remove malformed payload')
                continue
            LOG.info('Processing job (in-place): %s', job)
            # Skip assets that live under buildings, props, or terrain
            try:
                biome_id_local = job.get('biome_id')
                asset_name_local = job.get('asset_name')
                if biome_id_local and asset_name_local:
                    try:
                        # avoid heavy DB calls per tick when possible; reuse get_biome
                        possible = get_biome(biome_id_local)
                        if possible:
                            ps = possible.get('possible_structures', {})
                            for skip_cat in ('buildings', 'props', 'terrain'):
                                assets_map = ps.get(skip_cat, {})
                                if asset_name_local in assets_map:
                                    LOG.info('Skipping non-riggable asset in category %s: %s', skip_cat, asset_name_local)
                                    try:
                                        r.lrem('rig_model', 1, payload)
                                    except Exception:
                                        LOG.exception('Failed to remove skipped payload from rig_model')
                                    raise StopIteration
                    except StopIteration:
                        continue
            except Exception:
                LOG.exception('Error while checking non-riggable categories')
            try:
                unirig_cmd_job = job.get('unirig_cmd') or unirig_cmd
                ok = process_job(job, unirig_cmd_job, max_retries=max_retries)
                if ok:
                    try:
                        r.lrem('rig_model', 1, payload)
                    except Exception:
                        LOG.exception('Failed to remove processed payload from rig_model')
                else:
                    LOG.warning('Job failed, requeueing to tail of rig_model')
                    try:
                        r.rpush('rig_model', payload)
                    except Exception:
                        LOG.exception('Failed to rpush failed job back to rig_model')
                    try:
                        r.lrem('rig_model', 1, payload)
                    except Exception:
                        LOG.exception('Failed to remove failed payload from rig_model')
            except Exception as e:
                LOG.error('Unhandled error processing job: %s', e)
                try:
                    r.rpush('rig_model', payload)
                except Exception:
                    LOG.exception('Failed to rpush job back to rig_model after unhandled error')
                try:
                    r.lrem('rig_model', 1, payload)
                except Exception:
                    LOG.exception('Failed to remove payload after unhandled error')
        except Exception as e:
            LOG.error('Redis loop error: %s', e)
            time.sleep(5)


def main():
    unirig_cmd = os.environ.get('UNIRIG_CMD')
    if not unirig_cmd:
        LOG.error('UNIRIG_CMD not configured; set env UNIRIG_CMD')
        sys.exit(2)

    broker_url = os.environ.get('CELERY_BROKER_URL') or os.environ.get('REDIS_URL')
    max_retries = int(os.getenv('UNIRIG_MAX_RETRIES', 2))
    if broker_url:
        worker_loop(unirig_cmd, broker_url, None, max_retries=max_retries)
    else:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', 6379))
        worker_loop(unirig_cmd, redis_host, redis_port, max_retries=max_retries)


if __name__ == '__main__':
    main()

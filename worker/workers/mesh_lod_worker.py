"""
mesh_lod_worker.py — CPU worker that generates LOD GLBs from a source mesh.

Receives tasks on Redis queue ``mesh_lod_tasks``. Each task generates up to
4 LOD profiles (a/b/c/d) by invoking Blender headless + optionally gltfpack.

Task payload (Redis JSON):
{
  "type":          "mesh_lod",
  "session_id":    "<uuid>",
  "stage":         "mesh_lod",
  "char_name":     "salaryman_test",
  "major":         1,
  "minor":         0,
  "source_url":    "https://sparkassets-us.s3.amazonaws.com/.../<>.glb",
  "profiles":      ["a", "b", "c", "d"],          # subset OK
  "params":        {                              # optional overrides
      "ratio_b":         0.4,
      "ratio_c":         0.15,
      "voxel_size_d":    0.030,
      "quadriflow":      false,
      "quadriflow_target": 5000,
      "bake_resolution": 1024,
      "gltfpack_si_b":   1.0,    # 1.0 = no further reduction (B is decimate-only)
      "gltfpack_si_c":   0.85,
      "gltfpack_si_d":   0.85,
  },
}

Per-profile result fields are written back to manual_gen_stage_runs:
  lod_a_url / lod_a_tris / lod_a_bytes
  lod_b_url / lod_b_tris / lod_b_bytes
  lod_c_url / lod_c_tris / lod_c_bytes
  lod_d_url / lod_d_tris / lod_d_bytes
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Optional

import boto3
import redis

# Local imports — these resolve because run_mesh_lod_worker.py puts worker/ on sys.path
from lib.manual_gen_schema import (
    get_db, get_run, mark_running, mark_done, mark_error, update_run,
)

logger = logging.getLogger("MeshLodWorker")

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", "6379"))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD") or None
QUEUE          = os.getenv("MESH_LOD_QUEUE", "mesh_lod_tasks")

MONGO_URI      = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB       = os.getenv("MONGO_DB",  "World_builder")

S3_BUCKET      = os.getenv("AWS_S3_BUCKET", "sparkassets-us")
S3_REGION      = os.getenv("AWS_REGION",    "us-east-1")

BLENDER_BIN    = os.getenv("BLENDER_BIN", "/usr/bin/blender")
GLTFPACK_BIN   = os.getenv("GLTFPACK_BIN", "/usr/local/bin/gltfpack")
BLENDER_SCRIPT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "blender_scripts", "mesh_lod.py"
)

PROFILE_DEFAULTS = {
    "a": {"gltfpack_si": 1.0},
    "b": {"gltfpack_si": 1.0,  "ratio": 0.4},
    "c": {"gltfpack_si": 0.85, "ratio": 0.15},
    "d": {"gltfpack_si": 0.85, "voxel_size": 0.030},
}

_RESULT_RX = re.compile(r"MESH_LOD_RESULT:\s+profile=(\w+)\s+tris=(\d+)\s+seconds=([\d.]+)")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _redis():
    return redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=0, password=REDIS_PASSWORD,
        socket_connect_timeout=10, socket_timeout=30,
    )


def _s3():
    return boto3.client("s3", region_name=S3_REGION)


def _download(url: str, dst: str) -> None:
    import urllib.request
    with urllib.request.urlopen(url) as r, open(dst, "wb") as f:
        shutil.copyfileobj(r, f)


def _upload(local_path: str, key: str) -> str:
    _s3().upload_file(
        Filename=local_path, Bucket=S3_BUCKET, Key=key,
        ExtraArgs={"ContentType": "model/gltf-binary"},
    )
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{key}"


def _slug(s: str) -> str:
    return (s or "").replace(" ", "_").lower()


def _s3_key_for(task: dict, profile: str) -> str:
    char = _slug(task.get("char_name", "unknown"))
    maj  = int(task.get("major", 1))
    mnr  = int(task.get("minor", 0))
    return f"chars/{char}/v{maj}.{mnr}/{char}_{maj}_{mnr}_lod{profile}.glb"


def _run_blender(profile: str, in_glb: str, out_glb: str, args: dict) -> tuple[int, str, str]:
    args_json = json.dumps(args)
    cmd = [
        BLENDER_BIN, "--background", "--python", BLENDER_SCRIPT, "--",
        profile, in_glb, out_glb, args_json,
    ]
    # Blender needs HOME/USER to know where to put its config & cache dirs.
    # Under systemd EnvironmentFile= these aren't auto-set, causing
    # `mkdir: missing operand` and sometimes a SIGTERM during startup.
    env = {
        **os.environ,
        "HOME":            os.environ.get("HOME", "/home/ubuntu"),
        "USER":            os.environ.get("USER", "ubuntu"),
        "TMPDIR":          os.environ.get("TMPDIR", "/tmp"),
        "BLENDER_USER_CONFIG":  os.environ.get("BLENDER_USER_CONFIG", "/tmp/blender_cfg"),
        "BLENDER_USER_SCRIPTS": os.environ.get("BLENDER_USER_SCRIPTS", "/tmp/blender_cfg"),
    }
    os.makedirs(env["BLENDER_USER_CONFIG"], exist_ok=True)
    logger.info(f"[lod {profile}] blender → {' '.join(cmd[:4])} … out={out_glb}")
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1800, env=env,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_gltfpack(in_glb: str, out_glb: str, si: float) -> tuple[int, str]:
    if not shutil.which(GLTFPACK_BIN) and not os.path.isfile(GLTFPACK_BIN):
        # gltfpack not installed — fall back to copying the blender output
        shutil.copyfile(in_glb, out_glb)
        return 0, "(gltfpack not installed — passthrough)"
    cmd = [GLTFPACK_BIN, "-i", in_glb, "-o", out_glb, "-tc", "-cc"]
    if si and si < 1.0:
        cmd += ["-si", str(si)]
    logger.info(f"[lod gltfpack] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    return proc.returncode, proc.stderr or proc.stdout


def _process_profile(profile: str, in_glb: str, work_dir: str, task: dict) -> dict:
    """Run blender + gltfpack for one profile. Returns dict of result fields."""
    params  = task.get("params") or {}
    extras  = {
        "ratio_b":           params.get("ratio_b", PROFILE_DEFAULTS["b"]["ratio"]),
        "ratio_c":           params.get("ratio_c", PROFILE_DEFAULTS["c"]["ratio"]),
        "voxel_size_d":      params.get("voxel_size_d", PROFILE_DEFAULTS["d"]["voxel_size"]),
        "quadriflow":        params.get("quadriflow", False),
        "quadriflow_target": params.get("quadriflow_target", 5000),
        "bake_resolution":   params.get("bake_resolution", 1024),
    }
    si = params.get(f"gltfpack_si_{profile}", PROFILE_DEFAULTS[profile]["gltfpack_si"])

    blender_out = os.path.join(work_dir, f"lod_{profile}_raw.glb")
    final_out   = os.path.join(work_dir, f"lod_{profile}.glb")

    rc, stdout, stderr = _run_blender(profile, in_glb, blender_out, extras)
    if rc != 0 or not os.path.isfile(blender_out):
        tail = (stderr or stdout or "")[-1500:]
        raise RuntimeError(f"blender failed for profile {profile} (rc={rc}):\n{tail}")

    # Extract triangle count + time from blender stdout
    tris    = -1
    seconds = -1.0
    for line in stdout.splitlines():
        m = _RESULT_RX.search(line)
        if m and m.group(1) == profile:
            tris    = int(m.group(2))
            seconds = float(m.group(3))
            break

    rc2, gltfpack_log = _run_gltfpack(blender_out, final_out, si)
    if rc2 != 0 or not os.path.isfile(final_out):
        logger.warning(f"[lod {profile}] gltfpack rc={rc2}; using blender output as final")
        shutil.copyfile(blender_out, final_out)

    s3_key  = _s3_key_for(task, profile)
    url     = _upload(final_out, s3_key)
    size    = os.path.getsize(final_out)

    logger.info(f"[lod {profile}] done tris={tris} bytes={size} url={url}")
    return {
        f"lod_{profile}_url":     url,
        f"lod_{profile}_tris":    tris,
        f"lod_{profile}_bytes":   size,
        f"lod_{profile}_seconds": seconds,
    }


# ── Main worker loop ──────────────────────────────────────────────────────────

def _process_task(task: dict):
    sid     = task["session_id"]
    src_url = task["source_url"]
    profiles = [p.lower() for p in (task.get("profiles") or ["a", "b", "c", "d"])]

    db = get_db(MONGO_URI, MONGO_DB)
    mark_running(db, sid, stage="mesh_lod")

    with tempfile.TemporaryDirectory(prefix="mesh_lod_") as work:
        in_glb = os.path.join(work, "input.glb")
        _download(src_url, in_glb)
        in_size = os.path.getsize(in_glb)
        logger.info(f"[{sid[:8]}] downloaded source {in_size} bytes → {in_glb}")

        result_fields: dict = {"source_url": src_url, "source_bytes": in_size}
        for profile in profiles:
            try:
                fields = _process_profile(profile, in_glb, work, task)
                result_fields.update(fields)
                update_run(db, sid, fields, coll="manual_gen_stage_runs")
            except Exception as e:
                logger.exception(f"[{sid[:8]}] profile {profile} failed: {e}")
                result_fields[f"lod_{profile}_error"] = str(e)[:300]
                update_run(db, sid, {f"lod_{profile}_error": str(e)[:300]},
                           coll="manual_gen_stage_runs")

        # Primary image_url for downstream stages → highest-quality LOD that succeeded
        primary = None
        for p in ("a", "b", "c", "d"):
            if result_fields.get(f"lod_{p}_url"):
                primary = result_fields[f"lod_{p}_url"]
                break
        if primary:
            mark_done(db, sid, stage="mesh_lod", image_url=primary)
        else:
            mark_error(db, sid, stage="mesh_lod", error="no profile succeeded")


def main():
    logger.info(f"MeshLodWorker starting — queue={QUEUE} redis={REDIS_HOST}:{REDIS_PORT}")
    r = _redis()
    while True:
        try:
            item = r.blpop(QUEUE, timeout=30)
            if item is None:
                continue
            _, raw = item
            try:
                task = json.loads(raw)
            except Exception as e:
                logger.error(f"bad task JSON: {e} — payload: {raw[:200]}")
                continue
            sid = task.get("session_id", "?")
            logger.info(f"[{sid[:8]}] picked up mesh_lod task")
            try:
                _process_task(task)
            except Exception as e:
                logger.exception(f"task failed: {e}")
                try:
                    mark_error(get_db(MONGO_URI, MONGO_DB), sid,
                               stage="mesh_lod", error=str(e)[:500])
                except Exception:
                    pass
        except redis.exceptions.RedisError as e:
            logger.warning(f"redis error: {e} — reconnect in 5s")
            time.sleep(5)
            r = _redis()
        except KeyboardInterrupt:
            logger.info("shutting down")
            return


if __name__ == "__main__":
    main()

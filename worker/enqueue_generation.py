#!/usr/bin/env python3
"""
enqueue_generation.py — Generic Character Generation Queue Handler
==================================================================
Reads any biome from MongoDB and queues all its characters for SD1.5
image generation (Stage 1 ControlNet + Stage 2 img2img).

Works for ALL biomes — not tied to any specific biome name.
All ML workloads run on the GPU server; this script runs on CPU.

Usage:
  python worker/enqueue_generation.py --biome-id <biome_id>
  python worker/enqueue_generation.py --biome-id claudetest002 --stage1-only
  python worker/enqueue_generation.py --biome-id claudetest002 --chars cultivation_youth
  python worker/enqueue_generation.py --list-biomes
  python worker/enqueue_generation.py --biome-id <id> --dry-run

Options:
  --biome-id        Target biome _id in MongoDB biomes collection
  --stage1-only     Queue tasks with stage1_only=True (stop after ControlNet pass)
  --chars           Comma-separated list of character names to queue (default: all)
  --no-auto-start   Skip auto-starting the GPU EC2 instance
  --list-biomes     List all available biomes and exit
  --dry-run         Show tasks that would be queued without writing to Redis
"""

import argparse
import json
import os
import sys
import time
import uuid

import boto3
import pymongo
import redis
from bson import ObjectId
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Config ────────────────────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "18.207.13.85")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
MONGO_URI  = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or ""
MONGO_DB   = os.getenv("MONGO_DB", "World_builder")
SD15_QUEUE = "sd15_tasks"


# ── GPU Auto-Start ────────────────────────────────────────────────────────────

def start_gpu_if_needed() -> bool:
    """
    Start the GPU EC2 instance if it's stopped.
    Returns True if the instance is running (or was just started), False on error.
    All SD1.5 / TRELLIS / rig workloads run exclusively on the GPU — never CPU.
    """
    instance_id = os.getenv("AWS_GPU_INSTANCE_ID", "i-0d6b9d6d34ccc053d")
    region      = os.getenv("AWS_REGION", "us-east-1")

    try:
        ec2 = boto3.client(
            "ec2",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )

        resp  = ec2.describe_instances(InstanceIds=[instance_id])
        inst  = resp["Reservations"][0]["Instances"][0]
        state = inst["State"]["Name"]
        print(f"[EC2] GPU instance {instance_id} state: {state}")

        if state == "running":
            ip = inst.get("PublicIpAddress", "unknown")
            print(f"[EC2] Already running at {ip} — no action needed.")
            return True

        if state in ("stopped", "stopping"):
            print(f"[EC2] Starting GPU instance {instance_id}...")
            ec2.start_instances(InstanceIds=[instance_id])
            print("[EC2] Waiting for instance to reach 'running'...")
            waiter = ec2.get_waiter("instance_running")
            waiter.wait(
                InstanceIds=[instance_id],
                WaiterConfig={"Delay": 10, "MaxAttempts": 30},
            )
            resp2 = ec2.describe_instances(InstanceIds=[instance_id])
            inst2 = resp2["Reservations"][0]["Instances"][0]
            ip2   = inst2.get("PublicIpAddress", "unknown")
            print(f"[EC2] GPU instance is now RUNNING at {ip2}")
            print("[EC2] NOTE: GPU workers take ~2-3 min to start after OS boot.")
            print(f"[EC2] SSH:  ssh -i /Users/lekhaj/Documents/us_cpu_key.pem ec2-user@{ip2}")
            print(f"[EC2] Then: screen -r workers  (or check /tmp/gpu_workers.log)")
            return True

        print(f"[EC2] GPU instance state is '{state}' — cannot auto-start. Manage manually.")
        return False

    except Exception as e:
        print(f"[EC2] Warning: could not check/start GPU instance: {e}")
        print("[EC2] Tasks queued anyway — start GPU manually if needed.")
        return False


# ── MongoDB helpers ───────────────────────────────────────────────────────────

def get_db():
    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=8000)
    return client[MONGO_DB]


def find_biome(db, biome_id: str):
    """Try string _id first, then ObjectId fallback."""
    doc = db.biomes.find_one({"_id": biome_id})
    if doc is None:
        try:
            doc = db.biomes.find_one({"_id": ObjectId(biome_id)})
        except Exception:
            pass
    return doc


def list_biomes(db):
    docs = list(db.biomes.find({}, {"_id": 1, "biome_name": 1, "biome_type": 1,
                                     "possible_structures": 1}))
    print(f"\n{'─'*70}")
    print(f"  {'BIOME ID':<30}  {'TYPE':<20}  {'CHARS':<5}  NAME")
    print(f"{'─'*70}")
    for d in docs:
        bid   = str(d["_id"])
        name  = d.get("biome_name", "—")
        btype = d.get("biome_type", "—")
        chars = d.get("possible_structures", {}).get("characters", {})
        n     = len(chars) if isinstance(chars, dict) else 0
        print(f"  {bid:<30}  {btype:<20}  {n:<5}  {name}")
    print(f"{'─'*70}\n")


# ── Task builder ──────────────────────────────────────────────────────────────

def build_tasks(biome_id: str, chars: dict, stage1_only: bool,
                only_chars: list | None = None) -> list[dict]:
    """
    Build task payloads from MongoDB character data.

    Prompt priority per character:
      stage1.prompt / stage2.prompt (stored in nested stage dict)
      → stage1_prompt / stage2_prompt (top-level legacy fields)
      → empty string fallback (worker templates used)
    """
    tasks = []
    for char_name, cd in chars.items():
        if not isinstance(cd, dict):
            continue
        if only_chars and char_name not in only_chars:
            continue

        stage1 = cd.get("stage1") or {}
        stage2 = cd.get("stage2") or {}

        s1_pos = stage1.get("prompt") or cd.get("stage1_prompt") or ""
        s1_neg = stage1.get("negative") or cd.get("stage1_negative") or ""
        s2_pos = stage2.get("prompt") or cd.get("stage2_prompt") or cd.get("prompt") or ""
        s2_neg = stage2.get("negative") or cd.get("stage2_negative") or ""

        tasks.append({
            "task_id":         str(uuid.uuid4()),
            "biome_id":        biome_id,
            "character_name":  char_name,
            "character_type":  cd.get("character_type") or cd.get("creature_category") or "humanoid",
            "stage1_prompt":   s1_pos,
            "stage1_negative": s1_neg,
            "stage2_prompt":   s2_pos,
            "stage2_negative": s2_neg,
            "stage1_only":     stage1_only,
            "timestamp":       time.time(),
        })
    return tasks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Queue character generation tasks for any biome.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--biome-id",       help="MongoDB biome _id to queue")
    parser.add_argument("--stage1-only",    action="store_true",
                        help="Stop after ControlNet Stage 1 (skip img2img Stage 2)")
    parser.add_argument("--chars",          help="Comma-separated character names (default: all)")
    parser.add_argument("--no-auto-start",  action="store_true",
                        help="Skip auto-starting the GPU EC2 instance")
    parser.add_argument("--list-biomes",    action="store_true",
                        help="List all biomes in MongoDB and exit")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Show tasks without pushing to Redis")
    args = parser.parse_args()

    db = get_db()

    if args.list_biomes:
        list_biomes(db)
        return

    if not args.biome_id:
        parser.error("--biome-id is required (or use --list-biomes to see available biomes)")

    biome_id = args.biome_id.strip()
    doc = find_biome(db, biome_id)
    if not doc:
        print(f"[ERROR] Biome '{biome_id}' not found in MongoDB.")
        print("        Use --list-biomes to see available biomes.")
        sys.exit(1)

    name  = doc.get("biome_name", biome_id)
    btype = doc.get("biome_type", "—")
    chars = doc.get("possible_structures", {}).get("characters", {})

    print(f"\n[Biome]  {name}  (type={btype}, id={biome_id})")
    print(f"         {len(chars)} character(s): {', '.join(chars.keys()) or '(none)'}")

    if not chars:
        print("[ERROR] No characters found under possible_structures.characters.")
        sys.exit(1)

    only_chars = [c.strip() for c in args.chars.split(",")] if args.chars else None
    tasks = build_tasks(biome_id, chars, args.stage1_only, only_chars)

    if not tasks:
        print("[ERROR] No tasks to queue (check --chars filter or MongoDB data).")
        sys.exit(1)

    mode = "Stage 1 ONLY (ControlNet)" if args.stage1_only else "Full 2-stage pipeline"
    print(f"\n[Queue]  {mode}")
    print(f"         {len(tasks)} task(s) → {SD15_QUEUE}\n")

    for t in tasks:
        s1_words = len(t["stage1_prompt"].split())
        s2_words = len(t["stage2_prompt"].split())
        flag = "⚠ OVER LIMIT" if s1_words * 1.3 > 68 else "✓"
        print(f"  {t['character_name']:<25}  type={t['character_type']:<12}  "
              f"s1_tokens≈{round(s1_words*1.3):<4} {flag}  s2_tokens≈{round(s2_words*1.3)}")
        if not t["stage1_prompt"]:
            print(f"    ⚠ stage1_prompt empty — worker will use built-in template")

    if args.dry_run:
        print("\n[DRY RUN] No tasks pushed to Redis. Re-run without --dry-run to queue.")
        return

    # ── Auto-start GPU FIRST ──────────────────────────────────────────────────
    # Check + start GPU before pushing tasks so workers are already booting
    # when tasks land in the queue.  Tasks sitting in Redis with workers not
    # yet running is fine (workers block-pop), but starting early reduces
    # total wall-clock time from "enqueue" to "first image done".
    if not args.no_auto_start:
        print()
        gpu_ready = start_gpu_if_needed()
        if not gpu_ready:
            print("[WARNING] Could not confirm GPU is running — tasks will be queued anyway.")
            print("          Start GPU manually and workers will pick up from Redis.")
        print()

    # ── Push to Redis ─────────────────────────────────────────────────────────
    r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_connect_timeout=10)
    try:
        r.ping()
    except Exception as e:
        print(f"[ERROR] Cannot connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
        sys.exit(1)

    for t in tasks:
        r.rpush(SD15_QUEUE, json.dumps(t))
        label = "(Stage 1 only)" if t["stage1_only"] else "(full pipeline)"
        print(f"[Redis] Queued: {t['character_name']} {label}  (task_id={t['task_id'][:8]}...)")

    depth = r.llen(SD15_QUEUE)
    print(f"[Redis] {SD15_QUEUE} depth: {depth}")

    print(f"\n✓  {len(tasks)} task(s) queued for biome '{biome_id}'")
    print(f"   Monitor: ssh ec2-user@<gpu-ip> 'tail -f /tmp/gpu_workers.log'")
    print(f"   MongoDB: db.biomes.findOne({{_id: '{biome_id}'}})")


if __name__ == "__main__":
    main()

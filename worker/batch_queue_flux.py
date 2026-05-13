#!/usr/bin/env python3
"""
batch_queue_flux.py — Create sessions + queue Flux concept generation for all 6 characters.

Run BEFORE starting the GPU. Tasks wait in Redis; GPU worker picks them up automatically.

Usage:
  python worker/batch_queue_flux.py [--dry-run]

Characters queued:
  cultivation_youth  → flux_humanoid_tpose  (768x1024)
  suanni_lion        → flux_quadruped_side   (1024x768)
  qingmao_spirit     → flux_humanoid_tpose  (768x1024)
  tielong_boar       → flux_quadruped_side   (1024x768)
  baihe_crane        → flux_humanoid_tpose  (768x1024, bipedal)
  jinyuan_ape        → flux_quadruped_side   (1024x768, knuckle-walker)
"""
import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(__file__))
from lib.spec_schema import get_db as get_spec_db, assemble_prompt, assemble_negative
from lib.manual_gen_schema import get_db, COLLECTION, create_session, save_stage_prompts, mark_queued

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT  = int(os.getenv("REDIS_PORT", "6379"))
REDIS_QUEUE = "manual_gen_tasks"

MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")

CHARACTERS = [
    # (char_id, template_id, width, height, version)
    ("cultivation_youth", "flux_humanoid_tpose",  768, 1024, "v3"),
    ("suanni_lion",       "flux_quadruped_side",  1024,  768, "v3"),
    ("qingmao_spirit",    "flux_humanoid_tpose",  768, 1024, "v1"),
    ("tielong_boar",      "flux_quadruped_side",  1024,  768, "v1"),
    ("baihe_crane",       "flux_humanoid_tpose",  768, 1024, "v1"),  # bipedal — humanoid template
    ("jinyuan_ape",       "flux_quadruped_side",  1024,  768, "v1"),  # knuckle-walker — quadruped template
]


def _get_or_create_session(db, char_id: str, version: str) -> str:
    docs = list(db[COLLECTION].find(
        {"char_label": char_id, "version": version},
        {"_id": 1},
    ).sort("created_at", -1).limit(1))
    if docs:
        sid = docs[0]["_id"]
        print(f"  [session] reusing {sid[:8]}… for {char_id}/{version}")
        return sid
    sid = create_session(db, char_id, version)
    print(f"  [session] created  {sid[:8]}… for {char_id}/{version}")
    return sid


def run(dry_run: bool = False):
    db      = get_db(MONGO_URI)
    spec_db = get_spec_db(MONGO_URI)

    if not dry_run:
        import redis
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()

    print(f"\n{'DRY RUN — ' if dry_run else ''}Queuing Flux for {len(CHARACTERS)} characters\n")

    queued = []
    for char_id, template_id, width, height, version in CHARACTERS:
        print(f"\n▶ {char_id}")
        try:
            prompt   = assemble_prompt(spec_db, char_id, template_id)
            negative = assemble_negative(spec_db, char_id, template_id)
        except Exception as e:
            print(f"  [ERROR] prompt assembly failed: {e}")
            continue

        # Truncate for display
        print(f"  prompt  : {prompt[:120]}…")

        session_id = _get_or_create_session(db, char_id, version)

        task_id = str(uuid.uuid4())
        payload = {
            "type":       "flux",
            "session_id": session_id,
            "stage":      "flux",
            "task_id":    task_id,
            "timestamp":  time.time(),
            "prompt":     prompt,
            "negative":   negative,
            "params": {
                "width":          width,
                "height":         height,
                "steps":          4,
                "guidance_scale": 0.0,
            },
        }

        if dry_run:
            print(f"  [DRY RUN] would push task_id={task_id[:8]}…")
        else:
            save_stage_prompts(db, session_id, "flux", prompt, negative,
                               {"width": width, "height": height, "steps": 4, "guidance_scale": 0.0})
            r.rpush(REDIS_QUEUE, json.dumps(payload))
            mark_queued(db, session_id, "flux", task_id)
            print(f"  [queued] task_id={task_id[:8]}…  queue_len={r.llen(REDIS_QUEUE)}")

        queued.append(char_id)

    print(f"\n{'Would queue' if dry_run else 'Queued'} {len(queued)}/{len(CHARACTERS)} characters: {', '.join(queued)}")
    if not dry_run:
        print(f"Start GPU (i-0d6b9d6d34ccc053d) — worker will pick up tasks automatically.\n")


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    run(dry_run=dry)

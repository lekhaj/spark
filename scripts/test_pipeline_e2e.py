#!/usr/bin/env python3
"""
End-to-end pipeline test driver — acts as a headless UI tester.

Calls the exact same backend functions that each Gradio Generation Studio
button triggers, in the same sequence a manual user would click. Verifies
each stage in MongoDB before proceeding.

Run from the CPU server:
    cd /home/ubuntu/spark
    python3 scripts/test_pipeline_e2e.py human_warrior_test "human warrior in simple chainmail attire" humanoid
    python3 scripts/test_pipeline_e2e.py fantasy_bull_test "fantasy bull, large quadruped creature with curved horns" quadruped
"""
import os, sys, time, argparse, json
from datetime import datetime

# Make the studio page importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "gradio", "pages"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "worker"))

# Same imports as the page
from generation_studio_page import (
    _db, _get_or_create_session, _refresh_stage,
    _queue_flux, _run_normalize_cpu, _queue_sd, _queue_multiview,
    _queue_trellis, _push_task,
)
from lib.manual_gen_schema import create_session, COLLECTION


# ── Test prompts ──────────────────────────────────────────────────────────────
PRESETS = {
    "humanoid": dict(
        flux="full body human warrior, simple chainmail attire, leather boots, "
             "T-pose, arms extended horizontally, front view, white background, "
             "character sheet, orthographic, photorealistic",
        sd1="same human warrior, T-pose, front view, orthographic, white background",
        sd2="human warrior, detailed chainmail, leather, realistic textures, "
            "front view, white background",
        mv_side="same human warrior, side view profile, T-pose, white background",
        mv_back="same human warrior, back view, T-pose, white background",
    ),
    "quadruped": dict(
        flux="full body fantasy bull, large quadruped creature, curved horns, "
             "muscular body, four legs visible, side view, white background, "
             "character sheet, orthographic, photorealistic",
        sd1="same fantasy bull, four legs visible, side view, white background, orthographic",
        sd2="fantasy bull, detailed fur and horns, realistic textures, "
            "side view, white background",
        mv_side="same fantasy bull, opposite side view, four legs, white background",
        mv_back="same fantasy bull, rear view, tail visible, white background",
    ),
}

NEG_FLUX = "deformed, extra limbs, text, watermark, blurry, nsfw"
NEG_SD1  = "deformed, extra limbs, text, watermark, background, shadows, blurry, nsfw"
NEG_SD2  = "background, blurry, extra limbs, text, watermark, deformed, ugly, nsfw"


# ── Helpers ──────────────────────────────────────────────────────────────────

def log(msg, indent=0):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"  {ts} {' '*indent}{msg}", flush=True)

def wait_for(sid, stage, timeout=600, label=None):
    """Poll MongoDB until stage is done/error or timeout."""
    label = label or stage
    log(f"⏳ Waiting for {label} (max {timeout}s)…", indent=2)
    started = time.time()
    last_status = None
    while time.time() - started < timeout:
        status, url = _refresh_stage(sid, stage)
        if status != last_status:
            log(f"   status: {status}", indent=2)
            last_status = status
        if status == "done" or "error" in (status or "").lower():
            elapsed = int(time.time() - started)
            log(f"   ✅ {label}: {status}  url={url[:80] if url else '(none)'}  ({elapsed}s)", indent=2)
            return status, url
        time.sleep(5)
    log(f"   ❌ {label}: TIMEOUT after {timeout}s", indent=2)
    return "timeout", ""


def stage_info(sid, stage):
    """Return raw stage dict from MongoDB."""
    doc = _db()[COLLECTION].find_one({"_id": sid}, {f"stages.{stage}": 1})
    return ((doc or {}).get("stages") or {}).get(stage, {})


# ── Pipeline driver ──────────────────────────────────────────────────────────

def run_pipeline(char, prompt_set, category):
    """Walk the full Generation Studio flow for one character."""
    print()
    print("═" * 70)
    print(f"  CHARACTER: {char}  (category={category})")
    print("═" * 70)

    # ─── Stage: Create asset (UI: New Asset → Create) ─────────────────────────
    log("▶ Create asset (UI: New Asset → Create)")
    create_session(_db(), char, "v1")
    sid, info = _get_or_create_session(char, "v1")
    if not sid:
        log(f"❌ failed to create session: {info}")
        return False
    log(f"   session_id = {sid}", indent=2)

    p = PRESETS[prompt_set]

    # ─── Stage 0: Flux (UI: Stage 0 → Queue Flux) ─────────────────────────────
    log("▶ Stage 0 — Queue Flux")
    task_id, status = _queue_flux(
        sid, p["flux"], NEG_FLUX,
        width=768, height=1024, steps=4, guidance=0.0,
    )
    log(f"   queued: {status}", indent=2)
    s, url = wait_for(sid, "flux", timeout=300, label="Stage 0 Flux")
    if s != "done":
        log("❌ Flux failed — aborting pipeline")
        return False

    # ─── Stage 1: Normalize (UI: Stage 1 → Run Normalize) ─────────────────────
    log("▶ Stage 1 — Run Normalize (CPU)")
    norm_status, norm_url, norm_info = _run_normalize_cpu(sid, 512, 512, "flux")
    log(f"   normalize: {norm_status}  url={norm_url[:80] if norm_url else '(none)'}", indent=2)
    log(f"   info: {norm_info[:120]}", indent=2)
    if norm_status != "done":
        log("❌ Normalize failed — aborting")
        return False

    # ─── Stage 2: SD1 (UI: Stage 2 → Queue SD Stage 1) ────────────────────────
    log(f"▶ Stage 2 — Queue SD1 (category={category})")
    sd1_params = {
        "denoise": 0.20, "cfg": 5.5, "steps": 20,
        "openpose_weight": 0.85, "canny_weight": 0.55,
        "category": category,
    }
    task_id, status = _queue_sd(sid, "sd_stage1", p["sd1"], NEG_SD1, sd1_params,
                                input_source="normalize")
    log(f"   queued: {status}", indent=2)
    s, url = wait_for(sid, "sd_stage1", timeout=600, label="Stage 2 SD1")
    if s != "done":
        log("⚠️ SD1 failed — continuing to test other stages")

    # ─── Stage 3: SD2 (UI: Stage 3 → Queue SD Stage 2) ────────────────────────
    log("▶ Stage 3 — Queue SD2")
    sd2_params = {"denoise": 0.35, "cfg": 7.0, "steps": 20}
    task_id, status = _queue_sd(sid, "sd_stage2", p["sd2"], NEG_SD2, sd2_params,
                                input_source="sd_stage1")
    log(f"   queued: {status}", indent=2)
    s, url = wait_for(sid, "sd_stage2", timeout=600, label="Stage 3 SD2")
    if s != "done":
        log("⚠️ SD2 failed — continuing")

    # ─── Stage 4: Multi-view side + back ──────────────────────────────────────
    log("▶ Stage 4 — Queue Multi-view (side + back)")
    _queue_multiview(sid, "side", p["mv_side"], "", 0.45, 7.0, "flux")
    _queue_multiview(sid, "back", p["mv_back"], "", 0.45, 7.0, "flux")
    s_side, _ = wait_for(sid, "multiview_side", timeout=600, label="Stage 4 side")
    s_back, _ = wait_for(sid, "multiview_back", timeout=600, label="Stage 4 back")
    if s_side != "done" or s_back != "done":
        log("⚠️ Multi-view incomplete — continuing")

    # ─── Stage 5: TRELLIS ─────────────────────────────────────────────────────
    log("▶ Stage 5 — Queue TRELLIS")
    task_id, status = _queue_trellis(sid, "sd_stage2", "multiview_side", "multiview_back")
    log(f"   queued: {status}", indent=2)
    s, url = wait_for(sid, "trellis", timeout=1200, label="Stage 5 TRELLIS")

    # ─── Final report ─────────────────────────────────────────────────────────
    print()
    print("═" * 70)
    print(f"  RESULT for {char}")
    print("═" * 70)
    for stage in ("flux", "normalize", "sd_stage1", "sd_stage2",
                  "multiview_side", "multiview_back", "trellis"):
        sd = stage_info(sid, stage)
        st = sd.get("status", "—")
        url = sd.get("image_url") or sd.get("mesh_url") or ""
        emoji = "✅" if st == "done" else ("❌" if "error" in str(st) else "⚠️ ")
        print(f"  {emoji} {stage:18} {st:15} {url[:80]}")
    print()
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("char", help="character label (e.g. human_warrior_test)")
    ap.add_argument("prompt_set", help="preset key (humanoid|quadruped) — also used for SD1 category")
    ap.add_argument("category", choices=["humanoid", "quadruped"],
                    help="ControlNet category for SD1")
    args = ap.parse_args()

    if args.prompt_set not in PRESETS:
        print(f"unknown prompt_set: {args.prompt_set}; choose from {list(PRESETS)}")
        sys.exit(2)

    ok = run_pipeline(args.char, args.prompt_set, args.category)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

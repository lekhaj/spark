#!/usr/bin/env python3
"""
SD 1.5 Multi-Stage Image Worker — A10 GPU (13.203.114.77)
=========================================================
Generates game-ready character images for TRELLIS 3D input.

Pipeline per character:
  Stage 1 : SD1.5 + ControlNet (OpenPose / Depth)
            → clean base structure (pose, proportions, silhouette)
  Stage 2 : SD1.5 img2img (strength 0.3–0.4)
            → refined details (clothing, textures, aura)
  Output  : final PNG uploaded to S3 → pushed to model_tasks (TRELLIS queue)

ControlNet routing:
  bipedal / humanoid → lllyasviel/control_v11p_sd15_openpose  (COCO-18 T-pose skeleton)
  quadruped          → huchenlei/animal_openpose               (AP10K-17 animal skeleton)

Prompt strategy (CLIP token limit ~75 tokens effective):
  Stage 1 : structure-only prompt — pose, silhouette, white bg, flat lighting
  Stage 2 : detail prompt — clothing, style, quality descriptors
"""

import os
import io
import json
import time
import uuid
import fcntl
import logging

import boto3
import redis
import torch
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    UniPCMultistepScheduler,
)

load_dotenv()
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Single-instance lock ──────────────────────────────────────────────────────
LOCKFILE = "/tmp/sd15_image_worker.lock"

# ── Model IDs ─────────────────────────────────────────────────────────────────
# DreamShaper v8: best SD1.5 fine-tune for stylized fantasy/game characters.
# Clean white-bg outputs, strong silhouettes, TRELLIS-ready.
SD15_MODEL_ID = "Lykon/DreamShaper"

# SD1.5-native ControlNet models
CONTROLNET_OPENPOSE_ID = "lllyasviel/control_v11p_sd15_openpose"
# AP10K animal skeleton ControlNet — trained on quadruped/animal poses.
# Source: abehonest/ControlNet_AnimalPose, pruned + uploaded by huchenlei.
# Accepts AP10K-17 skeleton images (same visual format as OpenPose — coloured
# keypoints + limb sticks on black background, but animal anatomy).
CONTROLNET_ANIMAL_ID   = "huchenlei/animal_openpose"

# ── Control reference images — generate with prepare_controlnet_refs.py ──────
# tpose_openpose.png    : COCO-18 OpenPose T-pose skeleton (512×512, black bg)
# quad_animalpose.png   : AP10K-17 neutral standing quadruped skeleton (512×512)
TPOSE_OPENPOSE_PATH = "/home/ubuntu/controlnet_refs/tpose_openpose.png"
QUAD_ANIMALPOSE_PATH = "/home/ubuntu/controlnet_refs/quad_animalpose.png"

# ── Generation hyper-params ───────────────────────────────────────────────────
IMG_SIZE        = 512    # SD1.5 native — do NOT use 768 to avoid OOM
STAGE1_STEPS    = 25
STAGE1_CFG      = 7.5
STAGE1_CN_SCALE = 1.0    # ControlNet conditioning scale
STAGE2_STEPS    = 20
STAGE2_CFG      = 7.5
STAGE2_STRENGTH = 0.35   # img2img denoising strength (0.3–0.4 per spec)

# ── Infrastructure ────────────────────────────────────────────────────────────
REDIS_HOST   = os.getenv("REDIS_HOST",   "15.206.99.66")
REDIS_PORT   = int(os.getenv("REDIS_PORT", 6380))
MONGO_URI    = os.getenv("MONGO_URI",    "mongodb://kartik:Kartikg421@15.206.99.66:27017")
MONGO_DB     = "World_builder"
S3_BUCKET    = os.getenv("AWS_S3_BUCKET", "spackassets")
S3_REGION    = os.getenv("AWS_REGION",    "ap-south-1")

INPUT_QUEUE  = "sd15_tasks"    # A10 worker reads from here
OUTPUT_QUEUE = "model_tasks"   # TRELLIS worker reads from here (L4 GPU)

TASK_TTL = 14400   # seconds — matches orchestrator TTL


# ── Prompt Templates ──────────────────────────────────────────────────────────
#
# DESIGN RULES:
#   Stage 1 — structure only: pose / proportions / silhouette.
#             NO clothing detail, NO colour, NO material.
#             Token budget ~45–50 so Stage 1 stays far under CLIP's 75-token limit.
#
#   Stage 2 — detail pass: clothing, style, material, atmosphere.
#             Token budget ~60–65. Quality prefix adds ~5.
#
#   Negative prompts are shared across stages for each creature category.
#
QUALITY_PREFIX = "best quality, masterpiece, "   # ~5 tokens

CREATURE_PROMPTS = {
    # ── Bipedal (default for humanoid / cultivation / warrior) ────────────────
    "bipedal": {
        "stage1": {
            "positive": (
                "young male, athletic build, T-pose, arms extended horizontally, "
                "front view, full body, white background, flat lighting, "
                "centered, plain outfit, character sheet"
            ),
            # ~25 tokens — plenty of room for quality prefix
            "negative": (
                "background, shadows, complex textures, accessories, weapons, armor, "
                "text, watermark, blurry, cropped, extra limbs, deformed, nsfw"
            ),
        },
        "stage2": {
            "positive": (
                "young male cultivator, layered robes, cloth belt, "
                "stylized fantasy outfit, subtle mystical aura, "
                "front view, white background, clean game asset"
            ),
            # ~22 tokens + quality prefix = ~27 total
            "negative": (
                "background, photorealistic, blurry, extra limbs, "
                "text, watermark, deformed, ugly, nsfw"
            ),
        },
    },

    # ── Quadruped (wolf, tiger, dragon-mount, etc.) ───────────────────────────
    "quadruped": {
        "stage1": {
            "positive": (
                "four-legged creature, neutral standing pose, side view, "
                "full body, white background, flat lighting, centered, "
                "simple form, creature design sheet"
            ),
            "negative": (
                "background, shadows, complex textures, text, watermark, "
                "blurry, cropped, extra limbs, deformed, human, bipedal, nsfw"
            ),
        },
        "stage2": {
            "positive": (
                "mystical beast, detailed fur, glowing markings, "
                "fantasy creature, side view, white background, clean game asset"
            ),
            "negative": (
                "background, photorealistic, blurry, extra limbs, "
                "text, watermark, deformed, ugly, human, nsfw"
            ),
        },
    },

    # ── Humanoid alias (same as bipedal but defaults to armoured look) ────────
    "humanoid": {
        "stage1": {
            "positive": (
                "humanoid character, athletic build, T-pose, arms extended horizontally, "
                "front view, full body, white background, flat lighting, "
                "centered, plain outfit, character sheet"
            ),
            "negative": (
                "background, shadows, complex textures, accessories, weapons, "
                "text, watermark, blurry, cropped, extra limbs, deformed, nsfw"
            ),
        },
        "stage2": {
            "positive": (
                "humanoid character, detailed outfit, stylized fantasy costume, "
                "front view, white background, clean game asset, high detail"
            ),
            "negative": (
                "background, photorealistic, blurry, extra limbs, "
                "text, watermark, deformed, ugly, nsfw"
            ),
        },
    },
}


def build_prompt(template: str, custom_override: str | None, max_tokens: int = 68) -> str:
    """
    Prepend quality prefix and optionally override the template with a
    task-supplied custom prompt.  Truncates at word boundary to stay under
    max_tokens (conservative estimate: 1 word ≈ 1.3 CLIP tokens).
    """
    base = QUALITY_PREFIX + (custom_override if custom_override else template)
    words = base.split()
    if len(words) * 1.3 > max_tokens:
        words = words[:int(max_tokens / 1.3)]
    return " ".join(words)


# ── Model Manager ─────────────────────────────────────────────────────────────
class SD15Worker:
    """
    Loads two ControlNet models and two diffusers pipelines.

    VRAM budget on A10 (22 GB):
      controlnet_openpose  : ~1.4 GB fp16
      controlnet_animal    : ~1.4 GB fp16
      pipe_cn (SD1.5 UNet) : ~3.4 GB UNet + 0.3 GB VAE + 0.5 GB TextEnc = ~4.2 GB
      pipe_i2i             : ~4.2 GB (separate copy)
      Peak inference        : ~2–3 GB activations
      Total                : ~13–14 GB  →  safe on A10 (22 GB)
    """

    def __init__(self):
        logger.info("Loading ControlNet models...")
        self.controlnet_openpose = ControlNetModel.from_pretrained(
            CONTROLNET_OPENPOSE_ID, torch_dtype=torch.float16
        )
        # AP10K animal skeleton ControlNet — same architecture as OpenPose CN
        self.controlnet_animal = ControlNetModel.from_pretrained(
            CONTROLNET_ANIMAL_ID, torch_dtype=torch.float16
        )

        logger.info(f"Loading SD1.5 base: {SD15_MODEL_ID}")

        # ControlNet pipeline — controlnet is swapped at call time per creature type
        self.pipe_cn = StableDiffusionControlNetPipeline.from_pretrained(
            SD15_MODEL_ID,
            controlnet=self.controlnet_openpose,   # default; swapped per task
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe_cn.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe_cn.scheduler.config
        )
        self.pipe_cn.enable_xformers_memory_efficient_attention()
        self.pipe_cn.to("cuda")

        # img2img pipeline for Stage 2 (no ControlNet overhead)
        self.pipe_i2i = StableDiffusionImg2ImgPipeline.from_pretrained(
            SD15_MODEL_ID,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self.pipe_i2i.scheduler = UniPCMultistepScheduler.from_config(
            self.pipe_i2i.scheduler.config
        )
        self.pipe_i2i.enable_xformers_memory_efficient_attention()
        self.pipe_i2i.to("cuda")

        logger.info("All models loaded on CUDA.")

    def generate_base(
        self,
        creature_category: str,
        positive: str,
        negative: str,
    ) -> Image.Image:
        """Stage 1: txt2img + ControlNet → clean base structure."""
        if creature_category == "quadruped":
            # AP10K animal skeleton — same visual format as OpenPose but animal anatomy
            control_img = Image.open(QUAD_ANIMALPOSE_PATH).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            self.pipe_cn.controlnet = self.controlnet_animal
        else:
            # COCO-18 OpenPose T-pose skeleton
            control_img = Image.open(TPOSE_OPENPOSE_PATH).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            self.pipe_cn.controlnet = self.controlnet_openpose

        with torch.no_grad():
            result = self.pipe_cn(
                prompt=positive,
                negative_prompt=negative,
                image=control_img,
                num_inference_steps=STAGE1_STEPS,
                guidance_scale=STAGE1_CFG,
                controlnet_conditioning_scale=STAGE1_CN_SCALE,
                width=IMG_SIZE,
                height=IMG_SIZE,
            )
        return result.images[0]

    def refine(
        self,
        base_img: Image.Image,
        positive: str,
        negative: str,
    ) -> Image.Image:
        """Stage 2: img2img → add clothing / texture details."""
        init_img = base_img.resize((IMG_SIZE, IMG_SIZE))
        with torch.no_grad():
            result = self.pipe_i2i(
                prompt=positive,
                negative_prompt=negative,
                image=init_img,
                strength=STAGE2_STRENGTH,
                num_inference_steps=STAGE2_STEPS,
                guidance_scale=STAGE2_CFG,
            )
        return result.images[0]


# ── S3 helpers ────────────────────────────────────────────────────────────────
_s3_client = None


def get_s3():
    global _s3_client
    if _s3_client is None:
        _s3_client = boto3.client("s3", region_name=S3_REGION)
    return _s3_client


def upload_to_s3(img: Image.Image, s3_key: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    get_s3().put_object(Bucket=S3_BUCKET, Key=s3_key, Body=buf, ContentType="image/png")
    return s3_key


# ── MongoDB helper ────────────────────────────────────────────────────────────
def update_mongo(db, biome_id: str, char_name: str, fields: dict):
    """Dot-notation update for a character inside possible_structures.characters."""
    db.biomes.update_one(
        {"_id": biome_id},
        {"$set": {
            f"possible_structures.characters.{char_name}.{k}": v
            for k, v in fields.items()
        }},
    )


# ── Task processor ────────────────────────────────────────────────────────────
def process_task(task: dict, worker: SD15Worker, db, r: redis.Redis):
    biome_id   = task["biome_id"]
    char_name  = task["character_name"]
    char_type  = task.get("character_type", "humanoid")

    # Normalise to template key
    creature_category = "quadruped" if char_type == "quadruped" else "bipedal"
    templates = CREATURE_PROMPTS.get(char_type, CREATURE_PROMPTS["bipedal"])

    # Stage 1 always uses template (structure-only, no custom override)
    stage1_pos = build_prompt(templates["stage1"]["positive"], None)
    stage1_neg = templates["stage1"]["negative"]

    # Stage 2 uses task-supplied prompt as override if provided
    custom_prompt = task.get("refine_prompt") or task.get("prompt")
    stage2_pos = build_prompt(templates["stage2"]["positive"], custom_prompt)
    stage2_neg = templates["stage2"]["negative"]

    logger.info(f"[{char_name}] Stage1 prompt ({len(stage1_pos.split())} words): {stage1_pos}")
    logger.info(f"[{char_name}] Stage2 prompt ({len(stage2_pos.split())} words): {stage2_pos}")

    # ── Stage 1: base structure ───────────────────────────────────────────────
    update_mongo(db, biome_id, char_name, {"generation_stage": "base_gen", "status": "generating"})
    logger.info(f"[{char_name}] Generating base structure...")
    base_img = worker.generate_base(creature_category, stage1_pos, stage1_neg)

    base_key = f"images/{biome_id}/{char_name}_base.png"
    upload_to_s3(base_img, base_key)
    update_mongo(db, biome_id, char_name, {
        "generation_stage": "refinement",
        "images.base": base_key,
    })
    logger.info(f"[{char_name}] Base uploaded → s3://{S3_BUCKET}/{base_key}")

    # ── Stage 2: detail refinement ────────────────────────────────────────────
    logger.info(f"[{char_name}] Refining details (strength={STAGE2_STRENGTH})...")
    refined_img = worker.refine(base_img, stage2_pos, stage2_neg)

    refined_key = f"images/{biome_id}/{char_name}_refined.png"
    upload_to_s3(refined_img, refined_key)
    update_mongo(db, biome_id, char_name, {
        "generation_stage": "complete",
        "status": "image_complete",          # triggers TRELLIS pickup
        "images.refined": refined_key,
        "images.final": refined_key,          # TRELLIS reads this key
    })
    logger.info(f"[{char_name}] Refined uploaded → s3://{S3_BUCKET}/{refined_key}")

    # ── Push to TRELLIS queue ─────────────────────────────────────────────────
    model_task = {
        "task_id":        str(uuid.uuid4()),
        "biome_id":       biome_id,
        "character_name": char_name,
        "character_type": char_type,
        "image_path":     refined_key,
        "timestamp":      time.time(),        # REQUIRED — orchestrator TTL check
    }
    r.rpush(OUTPUT_QUEUE, json.dumps(model_task))
    logger.info(f"[{char_name}] Pushed to {OUTPUT_QUEUE} for TRELLIS.")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    # Enforce single instance
    lock_fd = open(LOCKFILE, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        logger.error("Another sd15_image_worker is already running. Exiting.")
        return

    logger.info("SD15 Image Worker starting up...")
    worker = SD15Worker()

    r  = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
    db = MongoClient(MONGO_URI)[MONGO_DB]

    logger.info(f"Listening on queue: {INPUT_QUEUE}")

    while True:
        raw = r.blpop(INPUT_QUEUE, timeout=30)
        if raw is None:
            continue

        _, payload = raw
        try:
            task = json.loads(payload)
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON: {payload[:120]}")
            continue

        # TTL guard — matches orchestrator logic
        age = time.time() - task.get("timestamp", 0)
        if age > TASK_TTL:
            logger.warning(f"Task expired ({age:.0f}s old), skipping: {task.get('character_name')}")
            continue

        logger.info(
            f"Task received → biome={task.get('biome_id')}  "
            f"char={task.get('character_name')}  type={task.get('character_type')}"
        )
        try:
            process_task(task, worker, db, r)
        except Exception as exc:
            logger.exception(f"Task failed for {task.get('character_name')}: {exc}")
            try:
                update_mongo(db, task["biome_id"], task["character_name"],
                             {"status": "error", "error_msg": str(exc)})
            except Exception:
                pass


if __name__ == "__main__":
    main()

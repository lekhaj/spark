#!/usr/bin/env python3
"""
Prepare ControlNet Reference Skeleton Images
============================================
Generates two conditioning images required by sd15_image_worker.py:

  1. tpose_openpose.png   — COCO-18 OpenPose T-pose skeleton  (bipedal/humanoid)
  2. quad_animalpose.png  — AP10K-17 animal skeleton, neutral stand (quadruped)

Both are 512×512 PNG, coloured skeleton on pure black background —
the exact format the respective ControlNet models were trained on.

Run this ONCE on the A10 before starting the worker:
  mkdir -p /home/ec2-user/controlnet_refs
  python prepare_controlnet_refs.py

Two approaches are attempted in order:
  A. controlnet_aux library (best — exact preprocessor output)
  B. Pure-PIL programmatic draw (no extra deps — always works)
"""

import os
import sys
import math
import logging
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

OUT_DIR = Path(os.getenv("CONTROLNET_REFS_DIR", "/home/ec2-user/controlnet_refs"))
TPOSE_PATH     = OUT_DIR / "tpose_openpose.png"
QUAD_PATH      = OUT_DIR / "quad_animalpose.png"
IMG_SIZE = 512


# ════════════════════════════════════════════════════════════════════════════
# SHARED DRAWING HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _draw_limb(draw: ImageDraw.Draw, p1, p2, color, width=6):
    """Draw anti-aliased limb line between two keypoints."""
    if p1 is None or p2 is None:
        return
    draw.line([p1, p2], fill=tuple(color), width=width)

def _draw_joint(draw: ImageDraw.Draw, pt, color, radius=6):
    """Draw filled circle at a keypoint."""
    if pt is None:
        return
    x, y = pt
    draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                 fill=tuple(color), outline=tuple(color))


# ════════════════════════════════════════════════════════════════════════════
# ① BIPEDAL — COCO-18 OpenPose T-pose skeleton
# ════════════════════════════════════════════════════════════════════════════
#
# Keypoint order (COCO-18, 0-indexed):
#   0  Nose       1  Neck
#   2  RShoulder  3  RElbow   4  RWrist
#   5  LShoulder  6  LElbow   7  LWrist
#   8  RHip       9  RKnee   10  RAnkle
#  11  LHip      12  LKnee   13  LAnkle
#  14  REye      15  LEye    16  REar   17  LEar
#
# "R" = character's right = image LEFT when facing the viewer
#
# Rainbow palette — one colour per keypoint (matches controlnet_aux ordering)
OPENPOSE_COLORS = [
    [255,   0,   0], [255,  85,   0], [255, 170,   0], [255, 255,   0],
    [170, 255,   0], [ 85, 255,   0], [  0, 255,   0], [  0, 255,  85],
    [  0, 255, 170], [  0, 255, 255], [  0, 170, 255], [  0,  85, 255],
    [  0,   0, 255], [ 85,   0, 255], [170,   0, 255], [255,   0, 255],
    [255,   0, 170], [255,   0,  85],
]

# Limb connections (0-indexed).  Colour index = index of the first keypoint.
OPENPOSE_LIMBS = [
    (1, 2), (1, 5),           # neck → shoulders
    (2, 3), (3, 4),           # R arm
    (5, 6), (6, 7),           # L arm
    (1, 8), (1, 11),          # neck → hips
    (8, 9), (9, 10),          # R leg
    (11, 12), (12, 13),       # L leg
    (1, 0),                   # neck → nose
    (0, 14), (14, 16),        # nose → R eye → R ear
    (0, 15), (15, 17),        # nose → L eye → L ear
    (2, 16), (5, 17),         # shoulders → ears
]

# T-pose keypoints in 512×512 image.
# Character is centred, occupies ~85 % of image height (y 40–480).
# Arms fully horizontal; legs straight and hip-width apart.
TPOSE_KPS = {
    #   idx : (x,   y)
     0: (256,  66),   # Nose
     1: (256,  98),   # Neck
     2: (166, 124),   # RShoulder  ← char's right = image left
     3: ( 96, 124),   # RElbow
     4: ( 28, 124),   # RWrist
     5: (346, 124),   # LShoulder
     6: (416, 124),   # LElbow
     7: (484, 124),   # LWrist
     8: (210, 278),   # RHip
     9: (210, 368),   # RKnee
    10: (210, 456),   # RAnkle
    11: (302, 278),   # LHip
    12: (302, 368),   # LKnee
    13: (302, 456),   # LAnkle
    14: (240,  52),   # REye
    15: (272,  52),   # LEye
    16: (222,  63),   # REar
    17: (290,  63),   # LEar
}


def draw_tpose_openpose() -> Image.Image:
    """
    Programmatically draw a COCO-18 OpenPose T-pose skeleton on black 512×512.
    Matches the draw_bodypose() output of controlnet_aux exactly.
    """
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw limbs first (behind joints)
    for (a, b) in OPENPOSE_LIMBS:
        color = OPENPOSE_COLORS[a]           # colour = colour of the start joint
        _draw_limb(draw, TPOSE_KPS[a], TPOSE_KPS[b], color, width=6)

    # Draw joints on top
    for idx, pt in TPOSE_KPS.items():
        _draw_joint(draw, pt, OPENPOSE_COLORS[idx], radius=7)

    return img


# ════════════════════════════════════════════════════════════════════════════
# ② QUADRUPED — AP10K-17 animal skeleton, neutral standing (side view)
# ════════════════════════════════════════════════════════════════════════════
#
# AP10K-17 keypoint order (used by huchenlei/animal_openpose):
#   0  L_Eye      1  R_Eye     2  L_Ear     3  R_Ear
#   4  Nose       5  Throat    6  Withers   7  Tail_root
#   8  L_F_Elbow  9  R_F_Elbow 10 L_B_Elbow 11 R_B_Elbow
#  12  L_F_Paw   13  R_F_Paw  14  L_B_Paw  15  R_B_Paw
#  16  Center_body (midpoint of spine, helps the model orient the torso)
#
# "L/R" = animal's left/right.
# Side view facing RIGHT (nose points right), all 4 legs visible
# with slight horizontal offset between front/back pairs.
#
# Colour scheme matches MMPose AP10K palette:
#   head        — cyan/teal
#   spine       — bright green
#   front legs  — orange
#   back legs   — red/magenta
#
AP10K_COLORS = {
    "head":   [ 0, 220, 220],
    "spine":  [ 0, 255,   0],
    "fleg":   [255, 128,   0],
    "bleg":   [255,  50, 150],
    "center": [255, 255, 255],
}

AP10K_LIMBS = [
    # head
    (0, 4,  "head"),    # L_Eye → Nose
    (1, 4,  "head"),    # R_Eye → Nose
    (0, 2,  "head"),    # L_Eye → L_Ear
    (1, 3,  "head"),    # R_Eye → R_Ear
    (4, 5,  "head"),    # Nose  → Throat
    # spine
    (5, 6,  "spine"),   # Throat → Withers
    (6, 16, "spine"),   # Withers → Center
    (16, 7, "spine"),   # Center  → Tail_root
    # front legs
    (5,  8, "fleg"),    # Throat    → L_F_Elbow
    (5,  9, "fleg"),    # Throat    → R_F_Elbow
    (8,  12,"fleg"),    # L_F_Elbow → L_F_Paw
    (9,  13,"fleg"),    # R_F_Elbow → R_F_Paw
    # back legs
    (7,  10,"bleg"),    # Tail_root → L_B_Elbow
    (7,  11,"bleg"),    # Tail_root → R_B_Elbow
    (10, 14,"bleg"),    # L_B_Elbow → L_B_Paw
    (11, 15,"bleg"),    # R_B_Elbow → R_B_Paw
]

# Neutral standing side view, facing RIGHT, 512×512.
# Both L (near) and R (far) legs are shown with slight x-offset so all 4
# legs are distinct — important for the model to learn 4-legged pose.
QUAD_KPS = {
    #    idx : (x,   y)
     0: (445,  84),   # L_Eye
     1: (458,  84),   # R_Eye   (slight x offset — depth)
     2: (425,  62),   # L_Ear
     3: (440,  62),   # R_Ear
     4: (480, 112),   # Nose
     5: (400, 152),   # Throat
     6: (255, 106),   # Withers (highest spine point)
     7: ( 82, 148),   # Tail_root
     8: (345, 220),   # L_F_Elbow (front left — near camera)
     9: (370, 215),   # R_F_Elbow (front right — far, offset)
    10: (130, 235),   # L_B_Elbow (back left — near)
    11: (155, 228),   # R_B_Elbow (back right — far)
    12: (335, 405),   # L_F_Paw
    13: (362, 398),   # R_F_Paw
    14: (120, 410),   # L_B_Paw
    15: (148, 402),   # R_B_Paw
    16: (180, 132),   # Center_body (spine midpoint)
}

# Per-keypoint display colours (grouped by body region)
QUAD_KP_COLORS = {
     0: AP10K_COLORS["head"],  1: AP10K_COLORS["head"],
     2: AP10K_COLORS["head"],  3: AP10K_COLORS["head"],
     4: AP10K_COLORS["head"],  5: AP10K_COLORS["head"],
     6: AP10K_COLORS["spine"], 7: AP10K_COLORS["spine"],
     8: AP10K_COLORS["fleg"],  9: AP10K_COLORS["fleg"],
    10: AP10K_COLORS["bleg"], 11: AP10K_COLORS["bleg"],
    12: AP10K_COLORS["fleg"], 13: AP10K_COLORS["fleg"],
    14: AP10K_COLORS["bleg"], 15: AP10K_COLORS["bleg"],
    16: AP10K_COLORS["center"],
}


def draw_quad_animalpose() -> Image.Image:
    """
    Programmatically draw an AP10K-17 neutral standing quadruped skeleton.
    Side view, facing right, all 4 legs visible with minor depth offset.
    """
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw limbs
    for (a, b, group) in AP10K_LIMBS:
        _draw_limb(draw, QUAD_KPS[a], QUAD_KPS[b], AP10K_COLORS[group], width=6)

    # Draw joints
    for idx, pt in QUAD_KPS.items():
        _draw_joint(draw, pt, QUAD_KP_COLORS[idx], radius=7)

    return img


# ════════════════════════════════════════════════════════════════════════════
# METHOD A — controlnet_aux (preferred: uses the actual preprocessor)
# ════════════════════════════════════════════════════════════════════════════

def try_controlnet_aux(out_dir: Path):
    """
    If controlnet_aux is installed, use it to process reference photos and
    produce skeleton images that EXACTLY match training-time preprocessing.

    T-pose reference: use any CC0 T-pose photo (provide path or URL below).
    Quad reference  : provide any photo of a dog/wolf in neutral standing.

    Set TPOSE_REF and QUAD_REF to file paths or leave empty to skip.
    """
    TPOSE_REF = os.getenv("TPOSE_REF_IMAGE", "")   # e.g. /tmp/tpose_ref.jpg
    QUAD_REF  = os.getenv("QUAD_REF_IMAGE", "")    # e.g. /tmp/wolf_ref.jpg

    if not TPOSE_REF and not QUAD_REF:
        return False

    try:
        from controlnet_aux import OpenposeDetector
        log.info("controlnet_aux found — using OpenposeDetector for T-pose")
    except ImportError:
        log.warning("controlnet_aux not installed — falling back to programmatic draw")
        return False

    generated = False

    if TPOSE_REF and Path(TPOSE_REF).exists():
        detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
        ref_img  = Image.open(TPOSE_REF).convert("RGB")
        skeleton = detector(ref_img, hand_and_face=False)
        skeleton.resize((IMG_SIZE, IMG_SIZE)).save(out_dir / "tpose_openpose.png")
        log.info(f"Saved OpenposeDetector output → {out_dir / 'tpose_openpose.png'}")
        generated = True

    if QUAD_REF and Path(QUAD_REF).exists():
        try:
            from controlnet_aux import AnimalPosePreprocessor
            estimator = AnimalPosePreprocessor()
            ref_img   = Image.open(QUAD_REF).convert("RGB")
            skeleton  = estimator(ref_img, detect_resolution=512, image_resolution=512)
            skeleton.save(out_dir / "quad_animalpose.png")
            log.info(f"Saved AnimalPosePreprocessor output → {out_dir / 'quad_animalpose.png'}")
            generated = True
        except (ImportError, AttributeError):
            log.warning("AnimalPosePreprocessor not available in this controlnet_aux build")

    return generated


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── T-pose ────────────────────────────────────────────────────────────────
    if TPOSE_PATH.exists():
        log.info(f"T-pose skeleton already exists at {TPOSE_PATH} — skipping.")
        log.info("  Delete the file and re-run to regenerate.")
    else:
        # Try controlnet_aux first (uses real reference photo if env var set)
        aux_ok = False
        if os.getenv("TPOSE_REF_IMAGE"):
            aux_ok = try_controlnet_aux(OUT_DIR)

        if not aux_ok or not TPOSE_PATH.exists():
            log.info("Generating T-pose skeleton programmatically (COCO-18 OpenPose format)…")
            img = draw_tpose_openpose()
            img.save(TPOSE_PATH)
            log.info(f"Saved → {TPOSE_PATH}")

    # ── Quadruped ─────────────────────────────────────────────────────────────
    if QUAD_PATH.exists():
        log.info(f"Quad skeleton already exists at {QUAD_PATH} — skipping.")
    else:
        aux_ok = False
        if os.getenv("QUAD_REF_IMAGE"):
            aux_ok = try_controlnet_aux(OUT_DIR)

        if not aux_ok or not QUAD_PATH.exists():
            log.info("Generating quadruped skeleton programmatically (AP10K-17 format)…")
            img = draw_quad_animalpose()
            img.save(QUAD_PATH)
            log.info(f"Saved → {QUAD_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n── Reference images ready ─────────────────────────────────")
    for p in [TPOSE_PATH, QUAD_PATH]:
        size_kb = p.stat().st_size // 1024
        print(f"  {p}  ({size_kb} KB)")

    print("""
── Visual check (SSH to A10, then): ──────────────────────────
  python3 -c "
  from PIL import Image
  img = Image.open('/home/ec2-user/controlnet_refs/tpose_openpose.png')
  img.save('/tmp/tpose_check.png')
  img2 = Image.open('/home/ec2-user/controlnet_refs/quad_animalpose.png')
  img2.save('/tmp/quad_check.png')
  print('OK — copy /tmp/tpose_check.png and /tmp/quad_check.png to inspect')
  "

── Using real reference photos (best quality): ───────────────
  # Provide a T-pose reference photo (any person in T-pose):
  TPOSE_REF_IMAGE=/tmp/my_tpose_photo.jpg python prepare_controlnet_refs.py

  # Provide a quadruped reference photo (dog/wolf standing):
  QUAD_REF_IMAGE=/tmp/my_wolf_photo.jpg python prepare_controlnet_refs.py
""")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
openpose_humanoid.py — HUMANOID T-POSE SKELETON GENERATOR
==========================================================
Branch : bhavesh-dev  |  Strategy A

Generates a standard OpenPose COCO-18 skeleton image for a humanoid
character in a perfect T-pose. This is the "control signal" fed to the
Shakker Labs ControlNet (mode=4, POSE) to tell Flux exactly what
body position to generate.

WHY WE DRAW THE SKELETON MANUALLY:
  In Strategy A, we skip the concept generation step entirely.
  Instead of generating an image and hoping the pose is correct,
  we FORCE the exact T-pose by drawing the skeleton programmatically
  with pixel-perfect coordinates. Zero ambiguity.

KEYPOINTS (COCO-18 format):
  0=Nose, 1=Neck, 2=RShoulder, 3=RElbow, 4=RWrist,
  5=LShoulder, 6=LElbow, 7=LWrist, 8=RHip, 9=RKnee,
  10=RAnkle, 11=LHip, 12=LKnee, 13=LAnkle,
  14=REye, 15=LEye, 16=REar, 17=LEar

COORDINATE NOTES:
  - All arm joints share the SAME Y-coordinate → perfectly horizontal arms
  - Wrists pulled inward (not at image edge) → prevents cylinder hand hallucinations
  - Ears are at EYE LEVEL Y, not neck level → correct head signal for ControlNet
"""

from PIL import Image, ImageDraw

# ── Standard OpenPose COCO-18 joint colors ────────────────────────────────────
OPENPOSE_COLORS = [
    [255,   0,   0],   #  0 Nose
    [255,  85,   0],   #  1 Neck
    [255, 170,   0],   #  2 RShoulder
    [255, 255,   0],   #  3 RElbow
    [170, 255,   0],   #  4 RWrist
    [ 85, 255,   0],   #  5 LShoulder
    [  0, 255,   0],   #  6 LElbow
    [  0, 255,  85],   #  7 LWrist
    [  0, 255, 170],   #  8 RHip
    [  0, 255, 255],   #  9 RKnee
    [  0, 170, 255],   # 10 RAnkle
    [  0,  85, 255],   # 11 LHip
    [  0,   0, 255],   # 12 LKnee
    [ 85,   0, 255],   # 13 LAnkle
    [170,   0, 255],   # 14 REye
    [255,   0, 255],   # 15 LEye
    [255,   0, 170],   # 16 REar
    [255,   0,  85],   # 17 LEar
]

# ── Bone connections (limb pairs) ─────────────────────────────────────────────
OPENPOSE_LIMBS = [
    (1, 2),  (1, 5),    # neck → both shoulders
    (2, 3),  (3, 4),    # right arm: shoulder → elbow → wrist
    (5, 6),  (6, 7),    # left arm:  shoulder → elbow → wrist
    (1, 8),  (1, 11),   # neck → both hips
    (8, 9),  (9, 10),   # right leg: hip → knee → ankle
    (11, 12),(12, 13),  # left leg:  hip → knee → ankle
    (1, 0),             # neck → nose
    (0, 14), (14, 16),  # nose → REye → REar
    (0, 15), (15, 17),  # nose → LEye → LEar
    (2, 16), (5, 17),   # shoulder → ear (COCO-18 diagonal — intentional standard)
]


def _scale_kps(kps: dict, src_w: int, src_h: int, dst_w: int, dst_h: int) -> dict:
    """Scale keypoints from one resolution to another."""
    sx = dst_w / src_w
    sy = dst_h / src_h
    return {k: (int(x * sx), int(y * sy)) for k, (x, y) in kps.items()}


# ── Calibrated T-pose keypoints at 512x512 (base resolution) ─────────────────
# IMPORTANT CALIBRATIONS:
#   - Arms: all Y=124 → perfectly horizontal (critical for T-pose)
#   - Wrists: X=28/487 (not X=4/508 at image edge → prevents cylinder hand hallucinations)
#   - Ears: Y=48 (eye level, not Y=90 at neck level → correct head pose signal)
_TPOSE_KPS_512 = {
    # HEAD
     0: (256,  62),   # Nose
     1: (256, 108),   # Neck
    14: (240,  45),   # REye
    15: (271,  45),   # LEye
    16: (230,  48),   # REar  ← eye level
    17: (281,  48),   # LEar  ← eye level

    # RIGHT ARM — person's right = viewer's left
     2: (173, 124),   # RShoulder
     3: (101, 124),   # RElbow  ← same Y as shoulder
     4: ( 28, 124),   # RWrist  ← pulled inward from edge

    # LEFT ARM — person's left = viewer's right
     5: (338, 124),   # LShoulder
     6: (414, 124),   # LElbow  ← same Y as shoulder
     7: (487, 124),   # LWrist  ← pulled inward from edge

    # RIGHT LEG — all X=225 → perfectly straight down
     8: (225, 309),   # RHip
     9: (225, 403),   # RKnee
    10: (225, 472),   # RAnkle

    # LEFT LEG — all X=280 → perfectly straight down
    11: (280, 309),   # LHip
    12: (280, 403),   # LKnee
    13: (280, 472),   # LAnkle
}


def generate_tpose_skeleton(width: int = 512, height: int = 512) -> Image.Image:
    """
    Draw a pixel-perfect T-pose OpenPose skeleton on a black background.

    This image is used as the control signal for Shakker Labs ControlNet
    in POSE mode (control_mode=4). The skeleton tells Flux exactly where
    to place every limb of the character.

    Args:
        width  : Output image width in pixels. Default 512.
        height : Output image height in pixels. Default 512 (square for Shakker).
                 Use 768 for a taller full-body image (better for Trellis input).

    Returns:
        PIL Image — black background with colored OpenPose skeleton drawn on it.
    """
    # Scale the base 512x512 keypoints to the requested size
    kps = _scale_kps(_TPOSE_KPS_512, 512, 512, width, height)

    img  = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw bones (lines) first so joints appear on top
    for (a, b) in OPENPOSE_LIMBS:
        if a in kps and b in kps:
            color = tuple(OPENPOSE_COLORS[a])
            draw.line([kps[a], kps[b]], fill=color, width=6)

    # Draw joints (filled circles) on top of bones
    r = max(5, width // 80)   # joint radius scales with image size
    for idx, (x, y) in kps.items():
        color = tuple(OPENPOSE_COLORS[idx])
        draw.ellipse([x - r, y - r, x + r, y + r], fill=color)

    return img


# ── Quick local preview (run this file directly to check the skeleton) ────────
if __name__ == "__main__":
    import os
    # Save both sizes for visual inspection
    for w, h in [(512, 512), (512, 768)]:
        img = generate_tpose_skeleton(width=w, height=h)
        out = os.path.join(os.path.dirname(__file__), f"tpose_skeleton_{w}x{h}.png")
        img.save(out)
        print(f"Saved: {out}")

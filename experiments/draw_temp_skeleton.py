"""
draw_temp_skeleton.py
─────────────────────
LOCAL PREVIEW TOOL — runs on Windows laptop only.
Draws the corrected COCO-18 T-pose skeleton for visual comparison.
Does NOT connect to GPU or AI.

Coordinates calibrated from pixel-level analysis of the reference image.
Key fixes vs old version:
  - Ears at eye level Y=48 (not neck level Y=90)
  - Wrists pulled inward (X=28/487 not X=4/508) → fixes cylinder hand hallucination
  - Nose/neck lowered for better head proportions
  - Hip height and width corrected
"""
from PIL import Image, ImageDraw
import os

IMG_SIZE = 512

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

OPENPOSE_LIMBS = [
    (1, 2),  (1, 5),    # neck → shoulders
    (2, 3),  (3, 4),    # right arm
    (5, 6),  (6, 7),    # left arm
    (1, 8),  (1, 11),   # neck → hips
    (8, 9),  (9, 10),   # right leg
    (11, 12),(12, 13),  # left leg
    (1, 0),             # neck → nose
    (0, 14), (14, 16),  # nose → REye → REar
    (0, 15), (15, 17),  # nose → LEye → LEar
    (2, 16), (5, 17),   # shoulder → ear (COCO-18 standard diagonal — intentional)
]

# CORRECTED T-POSE KEYPOINTS
# All arm joints Y=124 → perfectly horizontal arms
# Wrists at X=28/487 (not edge X=4/508) → prevents cylinder hand hallucination
# Ears at Y=48 (eye level, not neck level) → correct head pose signal for ControlNet
TPOSE_KPS = {
    # HEAD
     0: (256,  62),   # Nose
     1: (256, 108),   # Neck
    14: (240,  45),   # REye
    15: (271,  45),   # LEye
    16: (230,  48),   # REar  ← eye level (was Y=90 at neck — WRONG)
    17: (281,  48),   # LEar  ← eye level (was Y=90 at neck — WRONG)

    # RIGHT ARM (person's right = viewer's left)
     2: (173, 124),   # RShoulder
     3: (101, 124),   # RElbow    ← same Y=124
     4: ( 28, 124),   # RWrist    ← pulled inward (was X=4 at edge → cylinder hands)

    # LEFT ARM (person's left = viewer's right)
     5: (338, 124),   # LShoulder
     6: (414, 124),   # LElbow    ← same Y=124
     7: (487, 124),   # LWrist    ← pulled inward (was X=508 at edge → cylinder hands)

    # RIGHT LEG — all X=225 → perfectly straight down
     8: (225, 309),   # RHip
     9: (225, 403),   # RKnee   ← same X as hip
    10: (225, 472),   # RAnkle  ← same X as hip

    # LEFT LEG — all X=280 → perfectly straight down
    11: (280, 309),   # LHip
    12: (280, 403),   # LKnee   ← same X as hip
    13: (280, 472),   # LAnkle  ← same X as hip
}

img  = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (0, 0, 0))
draw = ImageDraw.Draw(img)

for (a, b) in OPENPOSE_LIMBS:
    draw.line([TPOSE_KPS[a], TPOSE_KPS[b]], fill=tuple(OPENPOSE_COLORS[a]), width=6)

for idx, (x, y) in TPOSE_KPS.items():
    draw.ellipse([x-7, y-7, x+7, y+7], fill=tuple(OPENPOSE_COLORS[idx]))

out = os.path.expandvars(r"%USERPROFILE%\Desktop\tpose_skeleton_corrected.png")
img.save(out)
print("Saved:", out)

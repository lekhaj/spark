"""
generate_apose.py — render a canonical OpenPose A-pose skeleton.

A-pose = arms angled ~35° below horizontal. FLUX's training distribution
contains far more A-poses than T-poses (character sheets, fashion refs,
portraits), so an A-pose skeleton gives much better aesthetic quality
under ControlNet conditioning while still providing the clean limb
separation required for image-to-3D + auto-rigging downstream.

Both Auto-Rig Pro and Mixamo accept A-pose natively.

Output: worker/controlnet_refs/apose_canonical.png

Run from anywhere:
    python worker/controlnet_refs/generate_apose.py [--size 1024] [--out path.png]
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

# ── COCO-18 keypoints — same order as T-pose; only arm joints differ ──────────
# 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist, 5 l_shoulder, 6 l_elbow,
# 7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee, 13 l_ankle,
# 14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear

# Arm angle below horizontal, in degrees. 35° gives a clear silhouette gap
# between arms and torso while still reading as a relaxed natural pose.
_ARM_DEG = 35.0
_ARM_RAD = math.radians(_ARM_DEG)
_C, _S = math.cos(_ARM_RAD), math.sin(_ARM_RAD)

# Arm-segment length (shoulder→elbow, same as elbow→wrist) — matches T-pose
# proportions so the bones don't shrink, only rotate.
_ARM_SEG = 0.165
_SHOULDER_Y = 0.235
_R_SHO_X, _L_SHO_X = 0.430, 0.570

APOSE_NORMALIZED: list[tuple[float, float]] = [
    (0.500, 0.140),  # 0  nose
    (0.500, _SHOULDER_Y),  # 1  neck
    # Right arm — rotates down-and-out from shoulder by _ARM_DEG
    (_R_SHO_X, _SHOULDER_Y),                                             # 2 r_shoulder
    (_R_SHO_X - _ARM_SEG * _C,       _SHOULDER_Y + _ARM_SEG * _S),       # 3 r_elbow
    (_R_SHO_X - 2 * _ARM_SEG * _C,   _SHOULDER_Y + 2 * _ARM_SEG * _S),   # 4 r_wrist
    # Left arm — mirrored
    (_L_SHO_X, _SHOULDER_Y),                                             # 5 l_shoulder
    (_L_SHO_X + _ARM_SEG * _C,       _SHOULDER_Y + _ARM_SEG * _S),       # 6 l_elbow
    (_L_SHO_X + 2 * _ARM_SEG * _C,   _SHOULDER_Y + 2 * _ARM_SEG * _S),   # 7 l_wrist
    # Right leg — same as T-pose
    (0.435, 0.520),  # 8  r_hip
    (0.435, 0.730),  # 9  r_knee
    (0.435, 0.940),  # 10 r_ankle
    # Left leg — same as T-pose
    (0.565, 0.520),  # 11 l_hip
    (0.565, 0.730),  # 12 l_knee
    (0.565, 0.940),  # 13 l_ankle
    # Face — same as T-pose
    (0.483, 0.128),  # 14 r_eye
    (0.517, 0.128),  # 15 l_eye
    (0.465, 0.140),  # 16 r_ear
    (0.535, 0.140),  # 17 l_ear
]

LIMB_PAIRS: list[tuple[int, int]] = [
    (1, 2), (2, 3), (3, 4),         # right arm
    (1, 5), (5, 6), (6, 7),         # left arm
    (1, 8), (8, 9), (9, 10),        # right leg
    (1, 11), (11, 12), (12, 13),    # left leg
    (0, 1),                          # neck → nose
    (0, 14), (14, 16),               # right eye/ear
    (0, 15), (15, 17),               # left eye/ear
]

LIMB_COLORS: list[tuple[int, int, int]] = [
    (153,   0,   0), (153,  51,   0), (153, 102,   0),
    (153, 153,   0), (102, 153,   0), ( 51, 153,   0),
    (  0, 153,   0), (  0, 153,  51), (  0, 153, 102),
    (  0, 153, 153), (  0, 102, 153), (  0,  51, 153),
    (  0,   0, 153),
    (153,   0, 153), (153,   0, 102),
    (102,   0, 153), ( 51,   0, 153),
]

JOINT_COLORS: list[tuple[int, int, int]] = [
    (255,   0,   0), (255,  85,   0), (255, 170,   0),
    (255, 255,   0), (170, 255,   0), ( 85, 255,   0),
    (  0, 255,   0), (  0, 255,  85), (  0, 255, 170),
    (  0, 255, 255), (  0, 170, 255), (  0,  85, 255),
    (  0,   0, 255), ( 85,   0, 255), (170,   0, 255),
    (255,   0, 255), (255,   0, 170), (255,   0,  85),
]


def render_apose(size: int = 1024) -> Image.Image:
    img  = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    pts  = [(int(x * size), int(y * size)) for x, y in APOSE_NORMALIZED]
    bone_w  = max(4, size // 110)
    joint_r = max(3, size // 200)
    for (a, b), color in zip(LIMB_PAIRS, LIMB_COLORS):
        draw.line([pts[a], pts[b]], fill=(*color, 200), width=bone_w)
    for (x, y), color in zip(pts, JOINT_COLORS):
        draw.ellipse((x - joint_r, y - joint_r, x + joint_r, y + joint_r),
                     fill=(*color, 255))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "apose_canonical.png")
    args = ap.parse_args()
    img = render_apose(args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}  ({args.size}x{args.size})")


if __name__ == "__main__":
    main()

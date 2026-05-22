"""
generate_tpose.py — render a canonical OpenPose T-pose skeleton.

Uses the COCO-18 keypoint layout and standard OpenPose colors/connections.
The skeleton this produces is the same format that any OpenPose detector
would emit if it processed a real photo of someone in T-pose — meaning
it's the most authoritative T-pose conditioning we can give a ControlNet
that was trained on OpenPose annotations.

Output: worker/controlnet_refs/tpose_canonical.png

Run from anywhere:
    python worker/controlnet_refs/generate_tpose.py [--size 1024] [--out path.png]
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw

# ── COCO-18 keypoint indices (OpenPose body order) ────────────────────────────
# 0 nose, 1 neck, 2 r_shoulder, 3 r_elbow, 4 r_wrist, 5 l_shoulder, 6 l_elbow,
# 7 l_wrist, 8 r_hip, 9 r_knee, 10 r_ankle, 11 l_hip, 12 l_knee, 13 l_ankle,
# 14 r_eye, 15 l_eye, 16 r_ear, 17 l_ear

# Canonical T-pose in normalized image coords (x ∈ [0,1], y ∈ [0,1], y down).
# Hand-tuned anthropometric proportions: head 12% of frame height, total span
# of arms ≈ 80% width, feet just above bottom margin.
TPOSE_NORMALIZED: list[tuple[float, float]] = [
    # Head — sits above the shoulder line, eyes/ears at face height
    (0.500, 0.140),  # 0  nose
    (0.500, 0.235),  # 1  neck (shoulder line y)
    # Right arm — perfectly horizontal at neck y
    (0.430, 0.235),  # 2  r_shoulder
    (0.265, 0.235),  # 3  r_elbow
    (0.100, 0.235),  # 4  r_wrist
    # Left arm — symmetric
    (0.570, 0.235),  # 5  l_shoulder
    (0.735, 0.235),  # 6  l_elbow
    (0.900, 0.235),  # 7  l_wrist
    # Right leg — shoulder-width stance, straight
    (0.435, 0.520),  # 8  r_hip
    (0.435, 0.730),  # 9  r_knee
    (0.435, 0.940),  # 10 r_ankle
    # Left leg — symmetric
    (0.565, 0.520),  # 11 l_hip
    (0.565, 0.730),  # 12 l_knee
    (0.565, 0.940),  # 13 l_ankle
    # Face keypoints
    (0.483, 0.128),  # 14 r_eye
    (0.517, 0.128),  # 15 l_eye
    (0.465, 0.140),  # 16 r_ear
    (0.535, 0.140),  # 17 l_ear
]

# OpenPose limb connections — pairs of keypoint indices.
LIMB_PAIRS: list[tuple[int, int]] = [
    (1, 2), (2, 3), (3, 4),         # right arm
    (1, 5), (5, 6), (6, 7),         # left arm
    (1, 8), (8, 9), (9, 10),        # right leg
    (1, 11), (11, 12), (12, 13),    # left leg
    (0, 1),                          # neck → nose
    (0, 14), (14, 16),               # right eye/ear
    (0, 15), (15, 17),               # left eye/ear
]

# Standard OpenPose limb colors (R,G,B). Matches the rendering produced by
# controlnet_aux.OpenposeDetector — important so ControlNets recognize it.
LIMB_COLORS: list[tuple[int, int, int]] = [
    (153,   0,   0),  # 1→2  r_shoulder
    (153,  51,   0),  # 2→3  r_upperarm
    (153, 102,   0),  # 3→4  r_forearm
    (153, 153,   0),  # 1→5  l_shoulder
    (102, 153,   0),  # 5→6  l_upperarm
    ( 51, 153,   0),  # 6→7  l_forearm
    (  0, 153,   0),  # 1→8  r_pelvis
    (  0, 153,  51),  # 8→9  r_thigh
    (  0, 153, 102),  # 9→10 r_shin
    (  0, 153, 153),  # 1→11 l_pelvis
    (  0, 102, 153),  # 11→12 l_thigh
    (  0,  51, 153),  # 12→13 l_shin
    (  0,   0, 153),  # 0→1  neck→nose
    (153,   0, 153),  # 0→14 r_eye
    (153,   0, 102),  # 14→16 r_ear
    (102,   0, 153),  # 0→15 l_eye
    ( 51,   0, 153),  # 15→17 l_ear
]

# Keypoint dot colors — match OpenPose convention (one color per keypoint).
JOINT_COLORS: list[tuple[int, int, int]] = [
    (255,   0,   0), (255,  85,   0), (255, 170,   0),
    (255, 255,   0), (170, 255,   0), ( 85, 255,   0),
    (  0, 255,   0), (  0, 255,  85), (  0, 255, 170),
    (  0, 255, 255), (  0, 170, 255), (  0,  85, 255),
    (  0,   0, 255), ( 85,   0, 255), (170,   0, 255),
    (255,   0, 255), (255,   0, 170), (255,   0,  85),
]


def render_tpose(size: int = 1024) -> Image.Image:
    """Render the canonical T-pose skeleton at *size* × *size*."""
    img = Image.new("RGB", (size, size), (0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    pts = [(int(x * size), int(y * size)) for x, y in TPOSE_NORMALIZED]

    # Bone width scales with image size. OpenPose renders use ~1% of dim.
    bone_w = max(4, size // 110)
    joint_r = max(3, size // 200)

    # Limbs first (semi-transparent so overlaps blend like real OpenPose output)
    for (a, b), color in zip(LIMB_PAIRS, LIMB_COLORS):
        rgba = (*color, 200)
        # Draw a thick line as a rotated rectangle so the alpha blends correctly
        x1, y1 = pts[a]
        x2, y2 = pts[b]
        # Use PIL's line — alpha works on RGBA canvas
        draw.line([(x1, y1), (x2, y2)], fill=rgba, width=bone_w)

    # Joints on top
    for (x, y), color in zip(pts, JOINT_COLORS):
        draw.ellipse((x - joint_r, y - joint_r, x + joint_r, y + joint_r),
                     fill=(*color, 255))

    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "tpose_canonical.png")
    args = ap.parse_args()

    img = render_tpose(args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out)
    print(f"wrote {args.out}  ({args.size}x{args.size})")


if __name__ == "__main__":
    main()

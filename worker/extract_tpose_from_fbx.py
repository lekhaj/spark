"""
extract_tpose_from_fbx.py
=========================
Extracts the canonical T-pose from a Mixamo FBX file (binary format, v7.x)
and renders a COCO-18 OpenPose skeleton PNG for use as ControlNet conditioning.

Usage:
    python extract_tpose_from_fbx.py --fbx /path/to/character.fbx [--out /path/to/output.png] [--upload-s3]

Output:
    512x512 RGB PNG — black background, rainbow-colored COCO-18 skeleton.
    Optionally uploads to s3://sparkassets-us/controlnet_refs/tpose_openpose.png
    and overwrites the active reference used by sd_model.py.

How it works:
    1. Parses the binary FBX (pure Python, no external deps except Pillow).
    2. Reads the BindPose section → exact global 4x4 world matrices for every bone.
    3. Maps Mixamo bone names → COCO-18 joint indices.
    4. Projects 3D world coords → 2D front-view (X horizontal, Y vertical, Z ignored).
    5. Fits skeleton to 512×512 with configurable padding, renders OpenPose colors.

Requirements: Pillow (already in requirements_gpu.txt)
Optional:     boto3 (for --upload-s3)
"""

import argparse
import os
import struct
import zlib
from pathlib import Path
from PIL import Image, ImageDraw


# ── COCO-18 index → Mixamo bone name ────────────────────────────────────────
# Note: OpenPose uses the *joint start* of each segment.
# RightArm = shoulder joint position, RightForeArm = elbow position, etc.
COCO_TO_MIXAMO = {
    0:  "mixamorig:Head",          # Nose   — approximated from Head + offset
    1:  "mixamorig:Neck",          # Neck
    2:  "mixamorig:RightArm",      # RShoulder
    3:  "mixamorig:RightForeArm",  # RElbow
    4:  "mixamorig:RightHand",     # RWrist
    5:  "mixamorig:LeftArm",       # LShoulder
    6:  "mixamorig:LeftForeArm",   # LElbow
    7:  "mixamorig:LeftHand",      # LWrist
    8:  "mixamorig:RightUpLeg",    # RHip
    9:  "mixamorig:RightLeg",      # RKnee
    10: "mixamorig:RightFoot",     # RAnkle
    11: "mixamorig:LeftUpLeg",     # LHip
    12: "mixamorig:LeftLeg",       # LKnee
    13: "mixamorig:LeftFoot",      # LAnkle
    # 14-17: eyes/ears — synthesized from Head position below
}

# COCO-18 limb pairs (joint_a, joint_b)
LIMBS = [
    (1, 2), (2, 3), (3, 4),          # right arm: neck→shoulder→elbow→wrist
    (1, 5), (5, 6), (6, 7),          # left arm
    (1, 8), (8, 9), (9, 10),         # right leg: neck→hip→knee→ankle
    (1, 11), (11, 12), (12, 13),     # left leg
    (1, 0), (0, 14), (14, 16),       # head: neck→nose→reye→rear
    (0, 15), (15, 17),               # head: nose→leye→lear
]

# Rainbow colours matching standard OpenPose COCO-18 palette
LIMB_COLORS = [
    (255,   0,   0), (255,  85,   0), (255, 170,   0),
    (255, 255,   0), (170, 255,   0), ( 85, 255,   0),
    (  0, 255,   0), (  0, 255,  85), (  0, 255, 170),
    (  0, 255, 255), (  0, 170, 255), (  0,  85, 255),
    (  0,   0, 255), ( 85,   0, 255), (170,   0, 255),
    (255,   0, 255), (255,   0, 170), (255,   0,  85),
]

IMG_SIZE  = 512
PAD       = 30   # pixels kept clear on each edge
DOT_R     = 5    # joint dot radius
LINE_W    = 4    # limb line width

# S3 target (matches sd_model.py _S3_TPOSE_URL)
S3_BUCKET = "sparkassets-us"
S3_KEY    = "controlnet_refs/tpose_openpose.png"

# Local path on GPU where worker expects the file
GPU_LOCAL_PATH = "/home/ec2-user/spark/worker/controlnet_refs/tpose_openpose.png"


# ── Binary FBX parser ────────────────────────────────────────────────────────

def _parse_binary_fbx(data: bytes) -> list:
    """
    Parse a Kaydara binary FBX file.
    Returns a list of top-level node dicts: {name, props, children}.
    Supports FBX versions 7.4 (4-byte offsets) and 7.5+ (8-byte offsets).
    """
    assert data[:21] == b"Kaydara FBX Binary  \x00", "Not a binary FBX file"
    version   = struct.unpack_from("<I", data, 23)[0]
    is_v75    = version >= 7500
    null_size = 25 if is_v75 else 13
    pos       = [27]

    def rb(n):
        v = data[pos[0]: pos[0] + n]
        pos[0] += n
        return v

    def rf(fmt):
        sz = struct.calcsize(fmt)
        v  = struct.unpack_from(fmt, data, pos[0])[0]
        pos[0] += sz
        return v

    def read_prop():
        tc = chr(rb(1)[0])
        if tc == "Y": return rf("<h")
        if tc == "C": return bool(rf("<B"))
        if tc == "I": return rf("<i")
        if tc == "F": return rf("<f")
        if tc == "D": return rf("<d")
        if tc == "L": return rf("<q")
        if tc in "fdlibi":
            count = rf("<I"); enc = rf("<I"); clen = rf("<I")
            raw   = rb(clen)
            if enc == 1:
                raw = zlib.decompress(raw)
            fmt_map = {"f": "<f", "d": "<d", "l": "<q", "i": "<i", "b": "<B"}
            elem_fmt  = fmt_map[tc]
            elem_size = struct.calcsize(elem_fmt)
            return [struct.unpack_from(elem_fmt, raw, i * elem_size)[0]
                    for i in range(count)]
        if tc == "S":
            n = rf("<I"); return rb(n).decode("utf-8", "replace")
        if tc == "R":
            n = rf("<I"); rb(n); return f"<raw:{n}>"
        return f"<unk:{tc}>"

    def read_node():
        off_fmt = "<Q" if is_v75 else "<I"
        end_off = rf(off_fmt)
        num_p   = rf(off_fmt)
        rf(off_fmt)                            # property_list_len (unused)
        name_len = rf("<B")
        name     = rb(name_len).decode("utf-8", "replace")
        if end_off == 0:                       # null sentinel record
            return None
        props    = [read_prop() for _ in range(num_p)]
        children = []
        while pos[0] < end_off - null_size:
            child = read_node()
            if child:
                children.append(child)
        pos[0] = end_off
        return {"name": name, "props": props, "children": children}

    top_nodes = []
    file_end  = len(data) - null_size
    while pos[0] < file_end:
        node = read_node()
        if node is None:
            break
        top_nodes.append(node)
    return top_nodes


# ── BindPose extraction ──────────────────────────────────────────────────────

def extract_world_positions(fbx_path: str) -> dict:
    """
    Parse FBX and return {bone_name: (world_x, world_y, world_z)}.

    Uses the BindPose section which stores exact global 4×4 matrices.
    Matrix layout: column-major → translation = elements [12, 13, 14].
    """
    data  = Path(fbx_path).read_bytes()
    nodes = _parse_binary_fbx(data)

    objects_node = next(n for n in nodes if n["name"] == "Objects")

    # Build node-id → clean bone name map
    id_to_name: dict[int, str] = {}
    for child in objects_node["children"]:
        if child["name"] != "Model" or not child["props"]:
            continue
        node_id  = child["props"][0]
        raw_name = child["props"][1] if len(child["props"]) > 1 else ""
        if isinstance(raw_name, str):
            id_to_name[node_id] = raw_name.split("\x00")[0]

    # Find Pose (BindPose) node
    pose_node = next(
        (c for c in objects_node["children"] if c["name"] == "Pose"), None
    )
    if pose_node is None:
        raise ValueError("No BindPose node found in FBX Objects section.")

    world_pos: dict[str, tuple] = {}
    for pc in pose_node["children"]:
        if pc["name"] != "PoseNode":
            continue
        node_id = None
        matrix  = None
        for cc in pc["children"]:
            if cc["name"] == "Node":   node_id = cc["props"][0]
            if cc["name"] == "Matrix": matrix  = cc["props"][0]  # list of 16 doubles
        if node_id is None or matrix is None:
            continue
        name = id_to_name.get(node_id, "")
        if not name:
            continue
        # Column-major 4×4: translation = last column → indices 12, 13, 14
        world_pos[name] = (matrix[12], matrix[13], matrix[14])

    return world_pos


# ── COCO-18 keypoint assembly ────────────────────────────────────────────────

def build_coco18_keypoints(world_pos: dict) -> dict:
    """
    Map world positions → COCO-18 dict {index: (world_x, world_y)}.
    Head sub-keypoints (eyes/ears/nose) are synthesised from the Head bone.
    """
    kps: dict[int, tuple] = {}

    # Joints that map 1:1 from FBX bone (indices 1–13)
    for coco_idx, bone_name in COCO_TO_MIXAMO.items():
        if bone_name in world_pos:
            wx, wy, _ = world_pos[bone_name]
            kps[coco_idx] = (wx, wy)

    # Synthesise nose/eyes/ears from Head bone position
    if "mixamorig:Head" in world_pos:
        hx, hy, _ = world_pos["mixamorig:Head"]
        # Offsets tuned to Mixamo "X Bot" proportions (head height ≈ 20 world units)
        kps[0]  = (hx +  0.0, hy + 5.0)   # Nose
        kps[14] = (hx -  3.5, hy + 4.0)   # REye
        kps[15] = (hx +  3.5, hy + 4.0)   # LEye
        kps[16] = (hx -  6.5, hy + 1.5)   # REar
        kps[17] = (hx +  6.5, hy + 1.5)   # LEar

    return kps


# ── Skeleton renderer ────────────────────────────────────────────────────────

def render_openpose_skeleton(
    kps: dict,
    img_size: int = IMG_SIZE,
    pad: int      = PAD,
) -> Image.Image:
    """
    Render COCO-18 OpenPose skeleton onto a black background.

    Args:
        kps:      {coco_index: (world_x, world_y)}
        img_size: output image size (square)
        pad:      padding in pixels kept clear on each edge

    Returns:
        PIL RGB image, img_size × img_size.
    """
    all_x = [v[0] for v in kps.values()]
    all_y = [v[1] for v in kps.values()]
    wx_min, wx_max = min(all_x), max(all_x)
    wy_min, wy_max = min(all_y), max(all_y)

    draw_area = img_size - 2 * pad

    def to_px(wx: float, wy: float) -> tuple:
        px = pad + (wx - wx_min) / (wx_max - wx_min) * draw_area
        py = pad + (1.0 - (wy - wy_min) / (wy_max - wy_min)) * draw_area  # flip Y
        return (int(round(px)), int(round(py)))

    px_kps = {idx: to_px(*wxy) for idx, wxy in kps.items()}

    img  = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw limbs
    for i, (a, b) in enumerate(LIMBS):
        if a in px_kps and b in px_kps:
            color = LIMB_COLORS[i % len(LIMB_COLORS)]
            draw.line([px_kps[a], px_kps[b]], fill=color, width=LINE_W)

    # Draw joint dots
    for idx, (px, py) in px_kps.items():
        draw.ellipse(
            [px - DOT_R, py - DOT_R, px + DOT_R, py + DOT_R],
            fill=(255, 255, 255),
        )

    return img


# ── S3 upload helper ─────────────────────────────────────────────────────────

def upload_to_s3(local_path: str, bucket: str, key: str) -> str:
    """Upload local file to S3 and return the public URL."""
    import boto3
    s3 = boto3.client("s3")
    s3.upload_file(
        local_path, bucket, key,
        ExtraArgs={"ContentType": "image/png"},
    )
    url = f"https://{bucket}.s3.us-east-1.amazonaws.com/{key}"
    print(f"[fbx→tpose] Uploaded to S3: {url}")
    return url


# ── Main entry point ─────────────────────────────────────────────────────────

def extract_tpose(
    fbx_path: str,
    out_path:  str  = "/tmp/tpose_openpose.png",
    upload_s3: bool = False,
    img_size:  int  = IMG_SIZE,
    pad:       int  = PAD,
) -> str:
    """
    Full pipeline: FBX → BindPose → COCO-18 → OpenPose PNG.

    Returns the path to the saved PNG.
    """
    print(f"[fbx→tpose] Parsing FBX: {fbx_path}")
    world_pos = extract_world_positions(fbx_path)
    print(f"[fbx→tpose] Found {len(world_pos)} bones in BindPose")

    kps = build_coco18_keypoints(world_pos)
    print(f"[fbx→tpose] Built {len(kps)}/18 COCO keypoints")

    img = render_openpose_skeleton(kps, img_size=img_size, pad=pad)
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    img.save(out_path)
    print(f"[fbx→tpose] Skeleton saved: {out_path}  ({img_size}×{img_size})")

    if upload_s3:
        upload_to_s3(out_path, S3_BUCKET, S3_KEY)

    return out_path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract T-pose OpenPose skeleton from a Mixamo FBX file."
    )
    parser.add_argument("--fbx",       required=True,  help="Path to input FBX file")
    parser.add_argument("--out",       default="/tmp/tpose_openpose.png",
                        help="Output PNG path (default: /tmp/tpose_openpose.png)")
    parser.add_argument("--upload-s3", action="store_true",
                        help=f"Upload result to s3://{S3_BUCKET}/{S3_KEY}")
    parser.add_argument("--size",      type=int, default=IMG_SIZE,
                        help=f"Output image size in pixels (default: {IMG_SIZE})")
    parser.add_argument("--pad",       type=int, default=PAD,
                        help=f"Edge padding in pixels (default: {PAD})")
    args = parser.parse_args()

    out = extract_tpose(
        fbx_path  = args.fbx,
        out_path  = args.out,
        upload_s3 = args.upload_s3,
        img_size  = args.size,
        pad       = args.pad,
    )
    print(f"[fbx→tpose] Done → {out}")

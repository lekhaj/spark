import bpy
import os
import sys
import glob
from mathutils import Vector
sys.path.append(os.path.dirname(__file__))

from io_helper_connect import (
    download_from_s3,
    upload_to_s3,
    get_mongo_collection,
    update_asset_status,
    get_latest_glb_from_s3
)

# --- CONFIG ---
input_folder = "/home/ubuntu/sarthak/input"
output_folder = "/home/ubuntu/sarthak/output"
template_blend = "/home/ubuntu/sarthak/logs/royal1.blend"
arm_name = "metarig.001"
template_mesh_name = "villager"
bucket = "sparkassets"
mongo_uri = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"

# Per-bone bias factors (0.0 = no move, 1.0 = full snap to centroid)
bias_map = {
    'neck1': 0.75,
    'neck2': 0.75,
    'shoulder.L': 0.6,
    'shoulder.R': 0.6,
    # you can tweak other bones here if desired
}
def get_bias(bone_name, default=0.5):
    return bias_map.get(bone_name, default)

# --- BLENDER UTILS ---
...  # [unchanged utilities: clear_scene, import_glb, append_object, etc.]

# in the main rigging section, after transfer_weights:
    # edit bones
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')

    # mapping: include elbow and wrist bones
    mapping = {
        "neck1": "neck1",
        "neck2": "neck2",
        "shoulder.L": "shoulder.L",
        "upper_arm.L": "upper_arm.L",
        "elbow.L": "forearm.L",
        "wrist.L": "hand.L",
        "shoulder.R": "shoulder.R",
        "upper_arm.R": "upper_arm.R",
        "elbow.R": "forearm.R",
        "wrist.R": "hand.R",
        # add more mappings as needed
    }

    for bone in arm.data.edit_bones:
        # determine which vertex group to use
        vgroup = mapping.get(bone.name, bone.name)
        cen = get_vgroup_centroid(target, vgroup)
        if not cen:
            print(f"[Align] No centroid for {vgroup} → skipping {bone.name}")
            continue

        # drop a visual marker
        debug_marker(cen, f"centroid_{bone.name}")

        # convert to armature-local space
        local_centroid = arm.matrix_world.inverted() @ cen
        orig_head = bone.head.copy()
        orig_tail = bone.tail.copy()

        # apply per-bone bias
        bias = get_bias(bone.name)
        bone.head = orig_head + (local_centroid - orig_head) * bias
        bone.tail = orig_tail + (local_centroid - orig_tail) * bias

        print(f"[Align] Bone {bone.name} moved with bias {bias} toward {vgroup}")
        print(f"[Debug] Original Head: {orig_head}, New Head: {bone.head}")

    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'

    parent_to_armature(target, arm)

    # ... remaining export + upload logic unchanged

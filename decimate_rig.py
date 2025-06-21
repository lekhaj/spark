import bpy
import os
import sys
import glob
import bmesh
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

# Per-bone bias factors (0.0 = no move, 1.0 = full snap)
bias_map = {
    'neck1': 0.75,
    'neck2': 0.75,
    'shoulder.L': 0.6,
    'shoulder.R': 0.6,
    'upper_arm.L': 0.7,
    'upper_arm.R': 0.7,
    'forearm.L': 0.65,
    'forearm.R': 0.65,
    'thigh.L': 0.6,
    'thigh.R': 0.6,
    'shin.L': 0.65,
    'shin.R': 0.65,
}

def get_bias(name, default=0.5):
    return bias_map.get(name, default)

# --- BLENDER HELPERS ---
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if hasattr(bpy.ops.outliner, "orphans_purge"):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def import_glb(fp):
    bpy.ops.import_scene.gltf(filepath=fp)
    for o in bpy.context.selected_objects:
        if o.type == "MESH":
            bpy.context.view_layer.objects.active = o
            return o
    return None

def append_object(blend, name):
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        if name in src.objects:
            dst.objects = [name]
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.context.collection.objects.link(obj)
        obj.select_set(True)
    return obj

def decimate_mesh(obj, tf, mode, param):
    faces = len(obj.data.polygons)
    if faces <= tf:
        print("[Decimate] Skip; under threshold")
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    m = obj.modifiers.new("Decimate", "DECIMATE")
    if mode == "COLLAPSE":
        m.decimate_type, m.ratio = "COLLAPSE", min(1.0, tf/faces )
    elif mode == "UNSUBDIV":
        m.decimate_type, m.iterations = "UNSUBDIV", int(param)
    else:
        m.decimate_type, m.angle_limit = "PLANAR", param
    bpy.ops.object.modifier_apply(modifier=m.name)
    print(f"[Decimate] {faces} → {len(obj.data.polygons)} faces")

def transfer_weights(src, dst):
    bpy.ops.object.select_all(action='DESELECT')
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object, dt.use_vert_data = src, True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'NEAREST'
    bpy.ops.object.modifier_apply(modifier=dt.name)
    print("[Weights] Transferred template → target")

def debug_marker(at, name):
    empty = bpy.data.objects.new(name, None)
    empty.location = at
    bpy.context.collection.objects.link(empty)

def get_vgroup_centroid(mesh, name):
    vg = mesh.vertex_groups.get(name)
    if not vg:
        return None
    total, pos = 0.0, Vector()
    for v in mesh.data.vertices:
        for g in v.groups:
            if mesh.vertex_groups[g.group].name == name and g.weight > 0:
                pos += mesh.matrix_world @ v.co * g.weight
                total += g.weight
    return pos / total if total > 0 else None

def get_bounding_box_corners(obj):
    return [obj.matrix_world @ Vector(c) for c in obj.bound_box]

def get_bounding_box_bottom_center(corners):
    bottom_z = min(c.z for c in corners)
    bottom = [c for c in corners if abs(c.z - bottom_z) < 1e-6]
    cx = sum(c.x for c in bottom) / len(bottom)
    cy = sum(c.y for c in bottom) / len(bottom)
    return Vector((cx, cy, bottom_z))

def align_bones(target, arm):
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    mapping = {
        "neck1": "neck1", "neck2": "neck2",
        "shoulder.L": "shoulder.L", "upper_arm.L": "upper_arm.L", "forearm.L": "forearm.L", "hand.L": "hand.L",
        "shoulder.R": "shoulder.R", "upper_arm.R": "upper_arm.R", "forearm.R": "forearm.R", "hand.R": "hand.R",
        "thigh.L": "thigh.L", "shin.L": "shin.L", "foot.L": "foot.L",
        "thigh.R": "thigh.R", "shin.R": "shin.R", "foot.R": "foot.R",
    }
    for bone in arm.data.edit_bones:
        vgroup = mapping.get(bone.name, bone.name)
        cen = get_vgroup_centroid(target, vgroup)
        if not cen:
            continue
        debug_marker(cen, f"centroid_{bone.name}")
        local_cen = arm.matrix_world.inverted() @ cen
        h0, t0 = bone.head.copy(), bone.tail.copy()
        bias = get_bias(bone.name)
        bone.head = h0 + (local_cen - h0) * bias
        bone.tail = t0 + (local_cen - t0) * bias
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'

def parent_to_armature(mesh, arm):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    mod = mesh.modifiers.get("Armature") or mesh.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    print("[Parent] Mesh → Armature")

def export_fbx(path, mesh, arm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nm = os.path.splitext(os.path.basename(path))[0]
    mesh.name = nm
    mesh.data.name = nm
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.export_scene.fbx(
        filepath=path,
        use_selection=True,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        mesh_smooth_type='FACE',
        bake_space_transform=True,
        path_mode='COPY',
        embed_textures=True,
        axis_forward='-Z',
        axis_up='Y'
    )
    print(f"[Export] {path}")

def main():
    collection = get_mongo_collection(mongo_uri)
    latest = get_latest_glb_from_s3(bucket, prefix="3d_assets/")
    if not latest:
        print("[Error] No GLB found")
        return
    lid = os.path.basename(latest)
    local = os.path.join(input_folder, lid)
    download_from_s3(bucket, latest, local)

    clear_scene()
    template = append_object(template_blend, template_mesh_name)
    arm = append_object(template_blend, arm_name)
    target = import_glb(local)
    if not all([template, arm, target]):
        print("[Error] Missing template, armature, or target mesh")
        return

    for o in (template, arm):
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    translation = get_bounding_box_bottom_center(get_bounding_box_corners(template)) - \
                  get_bounding_box_bottom_center(get_bounding_box_corners(target))
    target.location += translation
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if len(argv) < 3:
        print("Usage: script.py <TF> <Mode> <Param>")
        return
    tf, mode, param = int(argv[0]), argv[1].upper(), float(argv[2])
    decimate_mesh(target, tf, mode, param)

    for vg in template.vertex_groups:
        if vg.name not in target.vertex_groups:
            target.vertex_groups.new(name=vg.name)
    transfer_weights(template, target)
    align_bones(target, arm)
    parent_to_armature(target, arm)

    out = os.path.join(output_folder, f"{os.path.splitext(lid)[0]}_rigged.fbx")
    export_fbx(out, target, arm)

    url = upload_to_s3(bucket, f"processed/{os.path.splitext(lid)[0]}_rigged.fbx", out)
    update_asset_status(collection, os.path.splitext(lid)[0], status="completed", output_url=url)

if __name__ == "__main__":
    main()

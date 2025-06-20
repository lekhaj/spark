import bpy
import bmesh
import os
import sys
import glob
from mathutils import Vector

# Bring in S3 and Mongo helpers
sys.path.append(os.path.dirname(__file__))
from io_helper_connect import (
    download_from_s3,
    upload_to_s3,
    get_mongo_collection,
    update_asset_status,
    get_latest_glb_from_s3
)

# --- CONFIG ---
input_folder        = "/home/ubuntu/sarthak/input"
output_folder       = "/home/ubuntu/sarthak/output"
template_blend      = "/home/ubuntu/sarthak/logs/royal1.blend"
arm_name            = "metarig.001"
template_mesh_name  = "villager"
bucket              = "sparkassets"
mongo_uri           = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"

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

def get_bias(bone_name, default=0.5):
    return bias_map.get(bone_name, default)

# --- BLENDER UTILS ---
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if hasattr(bpy.ops.outliner, "orphans_purge"):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)
    for o in bpy.context.selected_objects:
        if o.type == "MESH":
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

def import_template_mesh():
    return append_object(template_blend, template_mesh_name)

def decimate_mesh(obj, tf, mode, p):
    faces = len(obj.data.polygons)
    if faces <= tf:
        print(f"[Decimate] Skip (faces {faces} ≤ threshold {tf})")
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    if mode == "COLLAPSE":
        mod.decimate_type = "COLLAPSE"
        mod.ratio = min(1.0, tf / faces * 1.5)
    elif mode == "UNSUBDIV":
        mod.decimate_type = "UNSUBDIV"
        mod.iterations = int(p)
    else:
        mod.decimate_type = "PLANAR"
        mod.angle_limit = p
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[Decimate] {faces} → {len(obj.data.polygons)} faces")

def transfer_weights(src, dst):
    bpy.ops.object.select_all(action='DESELECT')
    src.select_set(True)
    dst.select_set(True)
    bpy.context.view_layer.objects.active = dst
    dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = src
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'NEAREST'
    bpy.ops.object.modifier_apply(modifier=dt.name)
    print("[Weights] Transferred template → target")

def parent_to_armature(mesh, arm):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
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

def get_vgroup_centroid(mesh, group_name):
    vg = mesh.vertex_groups.get(group_name)
    if not vg:
        return None
    total_w, sum_pos = 0.0, Vector()
    for v in mesh.data.vertices:
        for g in v.groups:
            if mesh.vertex_groups[g.group].name == group_name and g.weight > 0:
                sum_pos += (mesh.matrix_world @ v.co) * g.weight
                total_w += g.weight
    return (sum_pos / total_w) if total_w else None

def get_bounding_box_corners(obj):
    return [obj.matrix_world @ Vector(c) for c in obj.bound_box]

def get_bounding_box_bottom_center(corners):
    bottom_z = min(c.z for c in corners)
    bottom_pts = [c for c in corners if abs(c.z - bottom_z) < 1e-6]
    cx = sum(c.x for c in bottom_pts) / len(bottom_pts)
    cy = sum(c.y for c in bottom_pts) / len(bottom_pts)
    return Vector((cx, cy, bottom_z))

def align_bones(target, arm):
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
    for bone in arm.data.edit_bones:
        cen = get_vgroup_centroid(target, bone.name)
        if not cen:
            continue
        local_cen = arm.matrix_world.inverted() @ cen
        h0, t0 = bone.head.copy(), bone.tail.copy()
        bias = get_bias(bone.name)
        bone.head = h0 + (local_cen - h0) * bias
        bone.tail = t0 + (local_cen - t0) * bias
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'
    print("[Bones] Aligned via centroid bias")

# --- MAIN PIPELINE ---
def main():
    # 1) Fetch latest from S3
    collection = get_mongo_collection(mongo_uri)
    latest = get_latest_glb_from_s3(bucket, prefix="3d_assets/")
    if not latest:
        print("[Error] No asset found in S3"); return
    lid = os.path.basename(latest)
    local_fp = os.path.join(input_folder, lid)
    download_from_s3(bucket, latest, local_fp)

    # 2) Clear & import
    clear_scene()
    template = import_template_mesh()
    arm      = append_object(template_blend, arm_name)
    target   = import_glb(local_fp)
    if not (template and arm and target):
        print("[Error] Missing template, armature or target"); return

    # 3) Freeze transforms
    for o in (template, arm):
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 4) Align by bounding-box bottom-center
    t_corners = get_bounding_box_corners(template)
    o_corners = get_bounding_box_corners(target)
    t_bb = get_bounding_box_bottom_center(t_corners)
    o_bb = get_bounding_box_bottom_center(o_corners)
    delta = t_bb - o_bb
    target.location += delta
    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # Apply same to armature
    arm.location += delta
    bpy.ops.object.select_all(action='DESELECT')
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # 5) Decimate
    argv = sys.argv[sys.argv.index("--") + 1:]
    if len(argv) < 3:
        print("Usage: decimate_rig.py <threshold> <MODE> <param>"); return
    tf, mode, param = int(argv[0]), argv[1].upper(), float(argv[2])
    decimate_mesh(target, tf, mode, param)

    # 6) Transfer weights
    for vg in template.vertex_groups:
        if vg.name not in target.vertex_groups:
            target.vertex_groups.new(name=vg.name)
    transfer_weights(template, target)

    # 7) Align bones & parent
    align_bones(target, arm)
    parent_to_armature(target, arm)

    # 8) Set origin & export
    # (same as local: origin→bottom face)
    set_origin_to_bottom_face(target)

    out_fp = os.path.join(output_folder, f"{os.path.splitext(lid)[0]}_rigged.fbx")
    export_fbx(out_fp, target, arm)

    # 9) Upload & update
    url = upload_to_s3(bucket, f"processed/{os.path.splitext(lid)[0]}_rigged.fbx", out_fp)
    update_asset_status(collection, os.path.splitext(lid)[0],
                        status="completed", output_url=url)

if __name__ == "__main__":
    main()

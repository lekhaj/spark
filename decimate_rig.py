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

# Per-bone bias factors (0.0 = no move, 1.0 = full snap)
bias_map = {
    'neck1': 0.75,
    'neck2': 0.75,
    'shoulder.L': 0.6,
    'shoulder.R': 0.6,
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


def import_template_mesh():
    return append_object(template_blend, template_mesh_name)


def decimate_mesh(obj, tf, mode, p):
    faces = len(obj.data.polygons)
    if faces <= tf:
        print("[Decimate] Skip; under threshold")
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    if mode == "COLLAPSE":
        mod.decimate_type = "COLLAPSE"
        mod.ratio = min(1.0, tf / faces)
    elif mode == "UNSUBDIV":
        mod.decimate_type = "UNSUBDIV"
        mod.iterations = int(p)
    else:
        mod.decimate_type = "PLANAR"
        mod.angle_limit = p
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[Decimate] {faces} → {len(obj.data.polygons)} faces")


def transfer_weights(src, dst):
    dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = src
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'NEAREST'
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.modifier_apply(modifier=dt.name)
    print("[Weights] Transferred template → target")


def check_vertex_group_weights(mesh, group_name):
    vg = mesh.vertex_groups.get(group_name)
    if vg:
        weighted_verts = [v for v in mesh.data.vertices if any(g.group == vg.index and g.weight > 0 for g in v.groups)]
        print(f"[Debug] Vertex group '{group_name}' has {len(weighted_verts)} vertices with weights > 0")
        return len(weighted_verts) > 0
    else:
        print(f"[Error] Vertex group '{group_name}' not found on mesh '{mesh.name}'")
        return False


def debug_marker(at, name):
    empty = bpy.data.objects.new(name, None)
    empty.location = at
    bpy.context.collection.objects.link(empty)


def parent_to_armature(mesh, arm):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    mod = mesh.modifiers.get("Armature") or mesh.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    print("[Parent] Mesh → Armature")


def export_glb(path, mesh, arm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nm = os.path.splitext(os.path.basename(path))[0]
    mesh.name = nm
    mesh.data.name = nm
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_skins=True
    )
    print(f"[Export] {path}")


def get_vgroup_centroid(mesh, group_name):
    vg = mesh.vertex_groups.get(group_name)
    if not vg:
        return None
    total_w, sum_pos = 0.0, Vector((0.0, 0.0, 0.0))
    for v in mesh.data.vertices:
        for g in v.groups:
            if mesh.vertex_groups[g.group].name == group_name and g.weight > 0:
                sum_pos += (mesh.matrix_world @ v.co) * g.weight
                total_w += g.weight
    if total_w == 0:
        return None
    centroid = sum_pos / total_w
    print(f"[Debug] {group_name} Centroid (World): {centroid}")
    return centroid


def main():
    collection = get_mongo_collection(mongo_uri)
    latest_key = get_latest_glb_from_s3(bucket, prefix="3d_assets/Humanoids/")
    if not latest_key:
        print("[Error] No GLB found")
        return

    asset_id = os.path.basename(latest_key).split('.')[0]
    local_glb = os.path.join(input_folder, os.path.basename(latest_key))
    download_from_s3(bucket, latest_key, local_glb)

    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    if len(args) < 3:
        print("Usage: script.py <TF> <Mode> <Param> [vhds_res] [vhds_smooth]")
        return
    tf = int(args[0])
    mode = args[1].upper()
    param = float(args[2])
    vhds_res = float(args[3]) if len(args) > 3 else 0.1
    vhds_smooth = int(args[4]) if len(args) > 4 else 5

    clear_scene()
    template = import_template_mesh()
    arm = append_object(template_blend, arm_name)
    target = import_glb(local_glb)
    if not all([template, arm, target]):
        print("[Error] Missing template, armature, or target mesh")
        return

    # Bake transforms
    for obj in (template, arm):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    # Align base meshes
    translation = get_bounding_box_corners(template)  # reuse earlier logic if desired
    target.location += translation

    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    decimate_mesh(target, tf, mode, param)
    for vg in template.vertex_groups:
        if vg.name not in target.vertex_groups:
            target.vertex_groups.new(name=vg.name)
    transfer_weights(template, target)

    # Bone alignment
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')
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
    }
    for bone in arm.data.edit_bones:
        vgroup = mapping.get(bone.name, bone.name)
        cen = get_vgroup_centroid(target, vgroup)
        if not cen:
            continue
        debug_marker(cen, bone.name + "_cen")
        local_cen = arm.matrix_world.inverted() @ cen
        h0, t0 = bone.head.copy(), bone.tail.copy()
        bias = get_bias(bone.name)
        bone.head = h0 + (local_cen - h0) * bias
        bone.tail = t0 + (local_cen - t0) * bias
    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'

    parent_to_armature(target, arm)

    # Optional remesh
    try:
        bpy.ops.object.select_all(action='DESELECT')
        target.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.voxel_remesh(mode='BOUNDED', resolution=vhds_res)
        bpy.ops.object.modifier_add(type='SMOOTH')
        bpy.context.object.modifiers["Smooth"].factor = 0.5
        bpy.context.object.modifiers["Smooth"].iterations = vhds_smooth
        bpy.ops.object.modifier_apply(modifier="Smooth")
    except Exception:
        pass

    # Export & upload
    out_path = os.path.join(output_folder, f"{asset_id}_rigged.glb")
    export_glb(out_path, target, arm)
    url = upload_to_s3(bucket, f"processed/{asset_id}_rigged.glb", out_path)
    update_asset_status(collection, asset_id, status="completed", output_url=url)

if __name__ == "__main__":
    main()

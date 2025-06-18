import bpy
import os
import sys
import glob
from mathutils import Vector
sys.path.append(os.path.dirname(__file__))

from io_helper_connect import download_from_s3, upload_to_s3, get_mongo_collection, update_asset_status

# --- CONFIG ---
input_folder = "/home/ubuntu/sarthak/input"
output_folder = "/home/ubuntu/sarthak/output"
template_blend = "/home/ubuntu/sarthak/logs/royal1.blend"
arm_name = "metarig.001"
template_mesh_name = "villager"
bucket = "sparkassets"
mongo_uri = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"

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

def transfer_weights(src, dst):
    dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = src
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'NEAREST'
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.modifier_apply(modifier=dt.name)

def parent_to_armature(mesh, arm):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    mod = mesh.modifiers.get("Armature") or mesh.modifiers.new("Armature", "ARMATURE")
    mod.object = arm

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
    bpy.ops.export_scene.gltf(filepath=path, export_format='GLB', use_selection=True, export_apply=True, export_skins=True)

def get_bounding_box_corners(obj):
    return [obj.matrix_world @ Vector(b) for b in obj.bound_box]

def get_bounding_box_bottom_center(corners):
    bottom_z = min(c.z for c in corners)
    bottom = [c for c in corners if c.z == bottom_z]
    cx = sum(c.x for c in bottom) / len(bottom)
    cy = sum(c.y for c in bottom) / len(bottom)
    return Vector((cx, cy, bottom_z))

def align_meshes(target, template):
    translation = get_bounding_box_bottom_center(get_bounding_box_corners(template)) - get_bounding_box_bottom_center(get_bounding_box_corners(target))
    target.location += translation

def get_vgroup_centroid(mesh, group_name):
    vg = mesh.vertex_groups.get(group_name)
    if not vg:
        return None
    total_w, sum_pos = 0.0, Vector((0.0, 0.0, 0.0))
    for v in mesh.data.vertices:
        for g in v.groups:
            if mesh.vertex_groups[g.group].name == group_name:
                sum_pos += (mesh.matrix_world @ v.co) * g.weight
                total_w += g.weight
                break
    return (sum_pos / total_w) if total_w > 0 else None

def main():
    collection = get_mongo_collection(mongo_uri)
    latest_key = get_latest_glb_from_s3(bucket)
    if not latest_key:
        print("[Error] No GLB found")
        return

    asset_id = os.path.basename(latest_key).split('.')[0]
    local_glb = os.path.join(input_folder, os.path.basename(latest_key))
    download_from_s3(bucket, latest_key, local_glb)

    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    tf = int(args[0]) if len(args) > 0 else 3000
    mode = args[1].upper() if len(args) > 1 else "COLLAPSE"
    param = float(args[2]) if len(args) > 2 else 0.5

    clear_scene()

    template = import_template_mesh()
    arm = append_object(template_blend, arm_name)
    target = import_glb(local_glb)
    if not all([template, arm, target]):
        print("[Error] Template/Armature/Target missing")
        return

    align_meshes(target, template)
    decimate_mesh(target, tf, mode, param)

    for vg in template.vertex_groups:
        if vg.name not in target.vertex_groups:
            target.vertex_groups.new(name=vg.name)

    transfer_weights(template, target)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')

    mapping = {"hand.L": "wrist.L", "hand.R": "wrist.R", "forearm.L": "elbow.L",
               "forearm.R": "elbow.R", "upperarm.L": "upper_arm.L", "upperarm.R": "upper_arm.R",
               "neck": ["neck.001", "neck.002"]}

    for bone in arm.data.edit_bones:
        vgroup = bone.name
        for k, v in mapping.items():
            if isinstance(v, list) and bone.name in v:
                vgroup = k
            elif bone.name.startswith(k):
                vgroup = k

        cen = get_vgroup_centroid(target, vgroup)
        if cen:
            local = arm.matrix_world.inverted() @ cen
            bone.head = local
            bone.tail = local + Vector((0, 0.05, 0))

    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'
    parent_to_armature(target, arm)

    try:
        bpy.ops.object.select_all(action='DESELECT')
        target.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = arm
        bpy.ops.object.modifier_set_active(modifier="Armature")
        print("[VHDS] Simulated - Blender GUI required")
    except Exception as e:
        print(f"[VHDS] Error: {e}")

    out_file = f"{asset_id}_rigged.glb"
    out_path = os.path.join(output_folder, out_file)
    export_glb(out_path, target, arm)
    upload_key = f"processed/{out_file}"
    final_url = upload_to_s3(bucket, upload_key, out_path)

    update_asset_status(collection, asset_id, status="completed", output_url=final_url)

if __name__ == "__main__":
    main()





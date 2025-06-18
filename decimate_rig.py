
import bpy
import os
import sys
import glob
from mathutils import Vector
from io_helper_connect import download_from_s3, upload_to_s3, get_mongo_collection, update_asset_status

# ——— USER SETTINGS ———
input_folder = "/home/ubuntu/sarthak/input"
output_folder = "/home/ubuntu/sarthak/output"
template_blend = "/home/ubuntu/logs/royal1.blend"
arm_name = "metarig.001"
template_mesh_name = "villager"
bucket = "sparkassets"
mongo_uri ="mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"

# ——— FUNCTIONS ———
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    for block in bpy.data.meshes:
        bpy.data.meshes.remove(block)

def append_object(blendfile, obj_name):
    with bpy.data.libraries.load(blendfile, link=False) as (data_from, data_to):
        if obj_name in data_from.objects:
            data_to.objects.append(obj_name)
    obj = data_to.objects[0] if data_to.objects else None
    if obj:
        bpy.context.collection.objects.link(obj)
    return obj

def import_glb(glb_path):
    bpy.ops.import_scene.gltf(filepath=glb_path)
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            return obj
    return None

def align_meshes(source, target):
    source.location = target.location
    source.rotation_euler = target.rotation_euler
    source.scale = target.scale

def decimate_mesh(obj, target_faces, mode, ratio):
    mod = obj.modifiers.new(name="Decimate", type='DECIMATE')
    if mode == "COLLAPSE":
        mod.ratio = ratio
        mod.use_collapse_triangulate = True
    elif mode == "UNSUBDIV":
        mod.iterations = int(ratio)
    elif mode == "PLANAR":
        mod.angle_limit = ratio
    mod.decimate_type = mode
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)

def transfer_weights(source, target):
    bpy.context.view_layer.objects.active = target
    source.select_set(True)
    target.select_set(True)
    bpy.ops.object.data_transfer(use_reverse_transfer=True,
                                 data_type='VGROUP_WEIGHTS',
                                 vert_mapping='NEAREST',
                                 layers_select_src='ALL',
                                 layers_select_dst='ALL')

def get_vgroup_centroid(obj, vgroup_name):
    if vgroup_name not in obj.vertex_groups:
        return None
    group = obj.vertex_groups[vgroup_name]
    verts = [v.co for v in obj.data.vertices if any(g.group == group.index for g in v.groups)]
    if not verts:
        return None
    return sum(verts, Vector()) / len(verts)

def parent_to_armature(mesh, armature):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

def export_glb(filepath, mesh, armature):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.export_scene.gltf(filepath=filepath, export_format='GLB', use_selection=True)

# ——— MAIN ENTRY ———
def main():
    argv = sys.argv
    if "--" not in argv or len(argv[argv.index("--")+1:]) < 6:
        print("Usage: script.py asset_id s3_key TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM align_method vhds_res vhds_smooth")
        return

    args = argv[argv.index("--")+1:]
    asset_id, s3_key = args[0], args[1]
    tf, mode, param = int(args[2]), args[3].upper(), float(args[4])
    align_method = int(args[5])
    vhds_res = float(args[6])
    vhds_smooth = int(args[7])

    collection = get_mongo_collection(mongo_uri)
    update_asset_status(collection, asset_id, "processing")

    clear_scene()

    local_glb_path = os.path.join(input_folder, os.path.basename(s3_key))
    download_from_s3(bucket, s3_key, local_glb_path)

    template = append_object(template_blend, template_mesh_name)
    arm = append_object(template_blend, arm_name)
    if not template or not arm:
        update_asset_status(collection, asset_id, "error")
        return

    target = import_glb(local_glb_path)
    if not target:
        update_asset_status(collection, asset_id, "error")
        return

    align_meshes(target, template)
    decimate_mesh(target, tf, mode, param)

    for vg in template.vertex_groups:
        if vg.name not in target.vertex_groups:
            target.vertex_groups.new(name=vg.name)

    transfer_weights(template, target)

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')

    vgroup_to_bone = {
        "hand.L": "wrist.L",
        "hand.R": "wrist.R",
        "forearm.L": "elbow.L",
        "forearm.R": "elbow.R",
        "upperarm.L": "upper_arm.L",
        "upperarm.R": "upper_arm.R",
        "neck": ["neck.001", "neck.002"]
    }

    for bone in arm.data.edit_bones:
        vgroup_name = bone.name
        for vg_prefix, mapped_bones in vgroup_to_bone.items():
            if isinstance(mapped_bones, list) and bone.name in mapped_bones:
                vgroup_name = vg_prefix
                break
            elif bone.name.startswith(vg_prefix):
                vgroup_name = vg_prefix
                break

        cen = get_vgroup_centroid(target, vgroup_name)
        if cen:
            local_centroid = arm.matrix_world.inverted() @ cen
            if align_method == 0:
                bone.head = local_centroid
                bone.tail = local_centroid + (bone.tail - bone.head)
            elif align_method == 1:
                bone.head += (local_centroid - bone.head) * 0.5
                bone.tail += (local_centroid - bone.tail) * 0.5

    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'

    parent_to_armature(target, arm)

    bpy.ops.object.select_all(action='DESELECT')
    target.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    try:
        bpy.ops.object.modifier_set_active(modifier="Armature")
        bpy.ops.object.voxel_remesh(mode='BOUNDED', resolution=vhds_res)
        bpy.ops.object.modifier_add(type='SMOOTH')
        bpy.context.object.modifiers["Smooth"].factor = 0.5
        bpy.context.object.modifiers["Smooth"].iterations = vhds_smooth
        bpy.ops.object.modifier_apply(modifier="Smooth")
        print(f"[VHDS] Applied: Res={vhds_res}, Smooth={vhds_smooth}")
    except Exception as e:
        print(f"[VHDS] Error: {e}")

    output_glb = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(s3_key))[0]}_rigged.glb")
    export_glb(output_glb, target, arm)

    s3_output_key = f"output/{os.path.basename(output_glb)}"
    s3_output_url = upload_to_s3(bucket, s3_output_key, output_glb)

    update_asset_status(collection, asset_id, "done", s3_output_url)


if __name__ == "__main__":
    main()


































import bpy
import os
import sys
import glob
import bmesh
from mathutils import Vector

# allow import of helpers
sys.path.append(os.path.dirname(__file__))
from io_helper_connect import (
    get_latest_glb_from_s3,
    download_from_s3,
    upload_to_s3,
    get_mongo_collection,
    update_asset_status
)

bucket_name = "sparkassets"
s3_prefix = "3d_assets"
models_folder = "/home/ubuntu/input"
output_folder = "/home/ubuntu/output"
mongo_uri = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"

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

def decimate_mesh(obj, tf, mode, p):
    fcount = len(obj.data.polygons)
    if fcount <= tf: return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    m = obj.modifiers.new("Decimate", "DECIMATE")
    if mode == "COLLAPSE":
        m.decimate_type, m.ratio = "COLLAPSE", min(1.0, tf/fcount)
    elif mode == "UNSUBDIV":
        m.decimate_type, m.iterations = "UNSUBDIV", int(p)
    else:
        m.decimate_type, m.angle_limit = "PLANAR", p
    bpy.ops.object.modifier_apply(modifier=m.name)

def set_origin_to_bottom_face_cursor(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bottom = min(bm.faces, key=lambda f: sum((obj.matrix_world@v.co).z for v in f.verts)/len(f.verts))
    center = sum(((obj.matrix_world@v.co) for v in bottom.verts), Vector())/len(bottom.verts)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.cursor.location = center
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    obj.location = (0, 0, 0)

def export_fbx(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
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

def main():
    argv = sys.argv
    if "--" not in argv: return
    args = argv[argv.index("--")+1:]
    if len(args) < 3: return
    tf, mode, param = int(args[0]), args[1].upper(), float(args[2])

    coll = get_mongo_collection(mongo_uri)

    clear_scene()
    latest_key = get_latest_glb_from_s3(bucket_name, prefix=s3_prefix)
    if not latest_key:
        update_asset_status(coll, asset_id=None, status="error", message="No GLB in S3")
        return

    asset_id = os.path.splitext(os.path.basename(latest_key))[0]
    local_glb = os.path.join(models_folder, os.path.basename(latest_key))
    download_from_s3(bucket_name, latest_key, local_glb)

    obj = import_glb(local_glb)
    if not obj:
        update_asset_status(coll, asset_id, "error", message="Import failed")
        return

    update_asset_status(coll, asset_id, "processing")

    decimate_mesh(obj, tf, mode, param)
    set_origin_to_bottom_face_cursor(obj)

    out_name = f"{asset_id}_decimated.fbx"
    local_fbx = os.path.join(output_folder, out_name)
    export_fbx(local_fbx, obj)

    s3_dest = f"{s3_prefix}/{out_name}"
    upload_to_s3(bucket_name, s3_dest, local_fbx)

    final_url = f"s3://{bucket_name}/{s3_dest}"
    update_asset_status(coll, asset_id, "completed", output_url=final_url)

if __name__=="__main__":
    main()

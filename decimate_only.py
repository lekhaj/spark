import bpy
import os
import sys
import bmesh
import boto3
import pymongo
from mathutils import Vector
from urllib.parse import urlparse

# ---------------- USER CONFIGURATION ----------------
bucket_name       = "sparkassets"
s3_prefix         = "3d_assets"       # where processed FBXs go: e.g. "3d_assets/generated/"
models_folder     = "/home/ubuntu/sarthak/input"
output_folder     = "/home/ubuntu/sarthak/output"
mongo_uri         = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"
db_name           = "World_builder"
collection_name   = "biomes"

# ---------------- MONGODB CONNECTION ----------------
def get_mongo_collection(uri, db_name, collection_name):
    client = pymongo.MongoClient(uri)
    return client[db_name][collection_name]

# ---------------- S3 HELPERS ----------------
def download_from_s3(bucket, key, download_path):
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3.download_file(bucket, key, download_path)
    print(f"[S3] Downloaded: {bucket}/{key} → {download_path}")

def upload_to_s3(bucket, key, file_path):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    print(f"[S3] Uploaded: {file_path} → {bucket}/{key}")

# ---------------- BLENDER OPERATIONS ----------------
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if hasattr(bpy.ops.outliner, "orphans_purge"):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def import_glb(filepath):
    bpy.ops.import_scene.gltf(filepath=filepath)
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            return obj
    return None

def decimate_mesh(obj, threshold, mode, param):
    face_count = len(obj.data.polygons)
    if face_count <= threshold:
        return
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    mod = obj.modifiers.new("Decimate", "DECIMATE")
    if mode == 'COLLAPSE':
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = min(1.0, threshold/face_count)
    elif mode == 'UNSUBDIV':
        mod.decimate_type = 'UNSUBDIV'
        mod.iterations = int(param)
    else:
        mod.decimate_type = 'PLANAR'
        mod.angle_limit = param
    bpy.ops.object.modifier_apply(modifier=mod.name)

def set_origin_to_bottom_face_cursor(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.object.mode_set(mode='EDIT')
    bm = bmesh.from_edit_mesh(obj.data)
    bottom_face = min(
        bm.faces,
        key=lambda f: sum((obj.matrix_world @ v.co).z for v in f.verts) / len(f.verts)
    )
    center = sum(
        ((obj.matrix_world @ v.co) for v in bottom_face.verts), Vector()
    ) / len(bottom_face.verts)
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.scene.cursor.location = center
    bpy.ops.object.origin_set(type='ORIGIN_CURSOR')
    obj.location = (0, 0, 0)

def export_fbx(filepath, obj):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.fbx(
        filepath=filepath,
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

# ---------------- MAIN PROCESSING ----------------
def main():
    argv = sys.argv
    if "--" not in argv:
        return
    args = argv[argv.index("--") + 1:]
    if len(args) < 3:
        print("Usage: blender --background --python this_script.py -- <threshold> <mode> <param>")
        return

    threshold = int(args[0])
    mode      = args[1].upper()
    param     = float(args[2])

    coll = get_mongo_collection(mongo_uri, db_name, collection_name)
    clear_scene()

    for doc in coll.find({}):
        buildings = doc.get('possible_structures', {}).get('buildings', {})
        for bldg_key, building in buildings.items():
            status = building.get('status', '').lower()
            # only process those ready
            if status != '3d asset generated':
                print(f"[Skip] {bldg_key} status = {status}")
                continue

            # read full S3 URL from MongoDB
            asset_3d_url = building.get('asset_3d_url')
            if not asset_3d_url:
                print(f"[Warning] Missing 'asset_3d_url' for building {bldg_key}")
                continue

            # parse bucket & key from URL
            parsed = urlparse(asset_3d_url)
            bucket = parsed.netloc.split('.')[0]
            key    = parsed.path.lstrip('/')

            local_glb = os.path.join(models_folder, f"{bldg_key}.glb")
            try:
                download_from_s3(bucket, key, local_glb)
            except Exception as e:
                print(f"[Download Error] {bldg_key}: {e}")
                coll.update_one(
                    {'_id': doc['_id']},
                    {'$set': {f"possible_structures.buildings.{bldg_key}.status": 'error'}}
                )
                continue

            obj = import_glb(local_glb)
            if not obj:
                print(f"[Import Error] {bldg_key}")
                coll.update_one(
                    {'_id': doc['_id']},
                    {'$set': {f"possible_structures.buildings.{bldg_key}.status": 'error'}}
                )
                continue

            # mark as decimating
            coll.update_one(
                {'_id': doc['_id']},
                {'$set': {f"possible_structures.buildings.{bldg_key}.status": 'decimating'}}
            )

            poly_before = len(obj.data.polygons)
            decimate_mesh(obj, threshold, mode, param)
            poly_after  = len(obj.data.polygons)
            set_origin_to_bottom_face_cursor(obj)

            # export & upload
            out_name = f"{bldg_key}_decimated.fbx"
            local_fbx = os.path.join(output_folder, out_name)
            export_fbx(local_fbx, obj)

            s3_dest = f"{s3_prefix}/generated/{out_name}"
            upload_to_s3(bucket_name, s3_dest, local_fbx)
            new_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_dest}"

            # update MongoDB
            updates = {
                f"possible_structures.buildings.{bldg_key}.status": 'Decimated',
                f"possible_structures.buildings.{bldg_key}.decimated_3d_asset": new_url,
                f"possible_structures.buildings.{bldg_key}.poly_before": poly_before,
                f"possible_structures.buildings.{bldg_key}.poly_after": poly_after,
            }
            coll.update_one({'_id': doc['_id']}, {'$set': updates})

            print(f"[Done] {bldg_key}: {poly_before} → {poly_after}")
            clear_scene()

if __name__ == "__main__":
    main()

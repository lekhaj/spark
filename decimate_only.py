import bpy
import os
import sys
import glob
import bmesh
import boto3
import pymongo
from mathutils import Vector

# ---------------- USER CONFIGURATION ----------------
bucket_name = "sparkassets"
s3_prefix = "3d_assets"
# Updated local folders inside 'sarthak'
models_folder = "/home/ubuntu/sarthak/input"
output_folder = "/home/ubuntu/sarthak/output"
mongo_uri = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017"

# ---------------- MONGODB CONNECTION ----------------
def get_mongo_collection(uri, db_name="World_builder", collection_name="biomes"):
    client = pymongo.MongoClient(uri)
    return client[db_name][collection_name]

# ---------------- S3 HELPERS ----------------
def download_from_s3(bucket, key, download_path):
    s3 = boto3.client('s3')
    os.makedirs(os.path.dirname(download_path), exist_ok=True)
    s3.download_file(bucket, key, download_path)
    print(f"[S3] Downloaded: {key} → {download_path}")


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
        return
    threshold = int(args[0])
    mode = args[1].upper()
    param = float(args[2])

    coll = get_mongo_collection(mongo_uri, db_name="World_builder", collection_name="biomes")
    clear_scene()

    for doc in coll.find({}):
        buildings = doc.get('possible_structures', {}).get('buildings', {})
        for bldg_key, building in buildings.items():
            # Use structureId and lowercase status
            asset_id = building.get('structureId')
            if not asset_id:
                print(f"[Warning] Missing 'structureId' for building {bldg_key}")
                continue

            status = building.get('status', '').lower()
            # Process only when 3D model is generated
            if status != '3D Model Generated':
                print(f"[Skip] {asset_id} status = {status}")
                continue

            s3_key = f"{s3_prefix}/{asset_id}.glb"
            local_glb = os.path.join(models_folder, f"{asset_id}.glb")

            try:
                download_from_s3(bucket_name, s3_key, local_glb)
            except Exception as e:
                print(f"[Download Error] {asset_id}: {e}")
                coll.update_one(
                    {'_id': doc['_id']},
                    {'$set': {f"possible_structures.buildings.{bldg_key}.status": 'error'}}
                )
                continue

            obj = import_glb(local_glb)
            if not obj:
                print(f"[Import Error] {asset_id}")
                coll.update_one(
                    {'_id': doc['_id']},
                    {'$set': {f"possible_structures.buildings.{bldg_key}.status": 'error'}}
                )
                continue

            # Mark as Decimating
            coll.update_one(
                {'_id': doc['_id']},
                {'$set': {f"possible_structures.buildings.{bldg_key}.status": 'decimating'}}
            )

            poly_before = len(obj.data.polygons)
            decimate_mesh(obj, threshold, mode, param)
            poly_after = len(obj.data.polygons)
            set_origin_to_bottom_face_cursor(obj)

            out_name = f"{asset_id}_decimated.fbx"
            local_fbx = os.path.join(output_folder, out_name)
            export_fbx(local_fbx, obj)

            # Upload under '3d_assets/generated/' folder
            s3_dest = f"{s3_prefix}/generated/{out_name}"
            upload_to_s3(bucket_name, s3_dest, local_fbx)
            s3_url = f"https://{bucket_name}.s3.amazonaws.com/{s3_dest}"

            updates = {
                f"possible_structures.buildings.{bldg_key}.status": 'Decimated',
                f"possible_structures.buildings.{bldg_key}.model3dUrl": s3_url,
                f"possible_structures.buildings.{bldg_key}.poly_before": poly_before,
                f"possible_structures.buildings.{bldg_key}.poly_after": poly_after,
            }
            coll.update_one({'_id': doc['_id']}, {'$set': updates})

            print(f"[Done] {asset_id}: {poly_before}→{poly_after}")
            clear_scene()

if __name__ == '__main__':
    main()

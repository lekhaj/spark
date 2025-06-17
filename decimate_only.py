import bpy
import os
import sys
import glob
from mathutils import Vector

# Add import for S3 helper
sys.path.append(os.path.dirname(__file__))
from io_helper_connect import get_latest_glb_from_s3, download_from_s3, upload_to_s3

# --- CONFIGURATION ---
bucket_name = "sparkassets"
s3_prefix = "/3d_assets"
models_folder = "/home/ubuntu/sarthak/input"
output_folder = "/home/ubuntu/sarthak/output"

# ——— UTILS ———
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
            print(f"[Import] GLB mesh: {o.name}")
            return o
    return None

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

def export_glb(path, mesh):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nm = os.path.splitext(os.path.basename(path))[0]
    mesh.name = nm
    mesh.data.name = nm
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_apply=True)
    print(f"[Export] {path}")

# ——— MAIN ———
def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM")
        return
    args = argv[argv.index("--") + 1:]
    if len(args) < 3:
        print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM")
        return

    tf = int(args[0])
    mode = args[1].upper()
    param = float(args[2])

    clear_scene()

    # Step 1: Get latest S3 .glb key
    latest_key = get_latest_glb_from_s3(bucket_name, prefix=s3_prefix)
    if not latest_key:
        print("[Error] No GLB found in S3")
        return

    # Step 2: Download it to EC2 input folder
    local_path = os.path.join(models_folder, os.path.basename(latest_key))
    download_from_s3(bucket_name, latest_key, local_path)

    # Step 3: Import the GLB
    target = import_glb(local_path)
    if not target:
        print("[Error] Failed to import GLB")
        return

    # Step 4: Decimate it
    decimate_mesh(target, tf, mode, param)

    # Step 5: Export it
    output_filename = f"{os.path.splitext(os.path.basename(latest_key))[0]}_decimated.glb"
    output_path = os.path.join(output_folder, output_filename)
    export_glb(output_path, target)

if __name__ == "__main__":
    main()

















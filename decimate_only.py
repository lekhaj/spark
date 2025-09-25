import bpy
import os
import sys
import bmesh

import json
from mathutils import Vector





DECIMATION_PROFILES = [
    ("5k", 5000, "COLLAPSE", None),
    ("6k", 6000, "COLLAPSE", None),
    ("8k", 8000, "COLLAPSE", None),
    ("10k", 10000, "COLLAPSE", None),
]

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
        mod.ratio = min(1.0, threshold / face_count)
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
        object_types={'MESH'},
        bake_space_transform=True
    )

# ---------------- MAIN PROCESSING ----------------

def main():
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--") + 2:
        print("Usage: blender --background --python decimate_only.py -- <asset_name> <input_file> [<output_folder>]")
        sys.exit(1)
    args = argv[argv.index("--") + 1:]
    # Asset name: if not provided, use input file name
    input_file = args[1] if len(args) > 1 else None
    if not input_file:
        print("Input file required.")
        sys.exit(1)
    # Determine asset_name from input file if not provided
    if len(args) > 0 and args[0]:
        asset_name = args[0]
    else:
        asset_name = os.path.splitext(os.path.basename(input_file))[0]
    # Default extension is .fbx
    file_ext = os.path.splitext(input_file)[1].lower() or '.fbx'
    # Output folder: use third arg or sibling to input_file called 'decimated_output'
    if len(args) > 2 and args[2]:
        output_folder = args[2]
    else:
        output_folder = os.path.join(os.path.dirname(input_file), 'decimated_output')
    os.makedirs(output_folder, exist_ok=True)
    def import_model(filepath):
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.glb' or ext == '.gltf':
            return import_glb(filepath)
        # Add more importers if needed
        else:
            return import_glb(filepath)
    results = {}
    for suffix, threshold, mode, param in DECIMATION_PROFILES:
        clear_scene()
        obj = import_model(input_file)
        if not obj:
            results[suffix] = {"error": f"Import failed for {input_file}"}
            continue
        poly_before = len(obj.data.polygons)
        decimate_mesh(obj, threshold, mode, param)
        poly_after = len(obj.data.polygons)
        set_origin_to_bottom_face_cursor(obj)
        out_name = f"{asset_name}_{suffix}_decimated.fbx"
        local_fbx = os.path.join(output_folder, out_name)
        export_fbx(local_fbx, obj)
        results[suffix] = {
            "local_file": local_fbx,
            "poly_before": poly_before,
            "poly_after": poly_after,
            "reduction_ratio": round((poly_before - poly_after) / poly_before, 3) if poly_before > 0 else None
        }
    print(json.dumps(results))

if __name__ == "__main__":
    main()

import bpy
import os
import sys
import glob

# --- USER SETTINGS ---
models_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
output_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"

# --- DECIMATION FUNCTION ---
def decimate_object(obj, method, param):
    mod = obj.modifiers.new(name="Decimate", type="DECIMATE")
    if method == "COLLAPSE":
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = param
    elif method == "UNSUBDIV": 
        mod.decimate_type = 'UNSUBDIVIDE'
        mod.iterations = int(param)
    elif method == "PLANAR":
        mod.decimate_type = 'DISSOLVE'
        mod.angle_limit = param
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    print(f"[Decimate] {method} applied with param {param}")

# --- MAIN ---
def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM")
        return
    args = argv[argv.index("--") + 1:]
    tf     = int(args[0])
    mode   = args[1].upper()
    param  = float(args[2])

    # Clear the scene
    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Import latest GLB
    glbs = glob.glob(os.path.join(models_folder, "*.glb"))
    if not glbs:
        print("[Error] No GLB file found")
        return
    latest = max(glbs, key=os.path.getmtime)
    bpy.ops.import_scene.gltf(filepath=latest)
    print(f"[Import] {latest}")

    # Get mesh object
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not mesh_objs:
        print("[Error] No mesh object found")
        return
    target = mesh_objs[0]

    # Apply decimation
    decimate_object(target, mode, param)

    # Export decimated mesh
    name = os.path.splitext(os.path.basename(latest))[0]
    output_path = os.path.join(output_folder, f"{name}_decimated.glb")
    os.makedirs(output_folder, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format='GLB',
        use_selection=False,
        export_apply=True,
    )
    print(f"[Export] {output_path}")

if __name__ == "__main__":
    main()

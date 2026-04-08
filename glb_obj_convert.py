import bpy
import os
import sys
import shutil
import glob

input_glb  = r"C:\Users\sarthak mohapatra\Downloads\mehses\phinox.glb"
output_dir = r"C:\Users\sarthak mohapatra\Downloads\mehses\OBJ"
obj_name   = os.path.splitext(os.path.basename(input_glb))[0]
output_obj = os.path.join(output_dir, obj_name + ".obj")

os.makedirs(output_dir, exist_ok=True)

for old in glob.glob(os.path.join(output_dir, f"{obj_name}_*.png")):
    os.remove(old)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=input_glb)

bpy.ops.file.unpack_all(method='WRITE_LOCAL')
for img in bpy.data.images:
    img.reload()

mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
bpy.ops.object.select_all(action='DESELECT')
for o in mesh_objs:
    o.select_set(True)
if mesh_objs:
    bpy.context.view_layer.objects.active = mesh_objs[0]
else:
    sys.exit("No mesh to export")

for img in bpy.data.images:
    abs_path = bpy.path.abspath(img.filepath)
    if not os.path.exists(abs_path):
        continue
    base = os.path.basename(abs_path)
    pref = f"{obj_name}_{base}"
    dst  = os.path.join(output_dir, pref)
    shutil.copy2(abs_path, dst)
    img.filepath     = "//" + pref
    img.filepath_raw = "//" + pref

bpy.ops.wm.obj_export(
    filepath=output_obj,
    export_selected_objects=True,
    export_materials=True,
    path_mode='COPY',
    export_uv=True,
    export_normals=True,
    export_colors=False,
    export_triangulated_mesh=True
)

print("Done")
print("OBJ:", output_obj)
print("MTL:", output_obj.replace('.obj', '.mtl'))




# === Blender Command to Run This Script ===
# & "C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe" `
#   --background `
#   --python "C:\Users\sarthak mohapatra\Downloads\mehses\glb_obj_convert.py"

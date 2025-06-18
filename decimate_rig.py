import bpy
import os
import sys
import glob
from mathutils import Vector

# --- USER SETTINGS ---
models_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
output_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"
template_blend = r"C:\Users\sarthak mohapatra\Downloads\mehses\royal1.blend"
arm_name = "metarig.001"
template_mesh_name = "villager"

# --- SCENE UTILITIES ---
def clear_scene():
    print("[Step] Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if hasattr(bpy.ops.outliner, "orphans_purge"):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    print("[Success] Scene cleared.")

def import_glb(filepath):
    print(f"[Step] Importing GLB from {filepath}")
    bpy.ops.import_scene.gltf(filepath=filepath)
    for o in bpy.context.selected_objects:
        if o.type == "MESH":
            bpy.context.view_layer.objects.active = o
            print(f"[Success] GLB mesh imported: {o.name}")
            return o
    print("[Error] No mesh found in GLB")
    return None

def append_object(blend, name):
    print(f"[Step] Appending object '{name}' from blend file: {blend}")
    if not os.path.exists(blend):
        print(f"[Error] Blend file not found: {blend}")
        return None
    with bpy.data.libraries.load(blend, link=False) as (src, dst):
        if name in src.objects:
            dst.objects = [name]
    obj = bpy.data.objects.get(name)
    if obj:
        bpy.context.collection.objects.link(obj)
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        print(f"[Success] Appended: {name}")
    else:
        print(f"[Error] Failed to append: {name}")
    return obj

def import_template_mesh():
    return append_object(template_blend, template_mesh_name)

# --- MESH PROCESSING ---
def decimate_mesh(obj, tf, mode, p):
    print(f"[Step] Decimating mesh '{obj.name}'")
    faces = len(obj.data.polygons)
    print(f"[Info] Original face count: {faces}")
    if faces <= tf:
        print("[Skip] Mesh is already under the threshold.")
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
    print(f"[Success] Decimated to {len(obj.data.polygons)} faces")

def transfer_weights(src, dst):
    print(f"[Step] Transferring weights from '{src.name}' to '{dst.name}'")
    dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = src
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'NEAREST'
    bpy.context.view_layer.objects.active = dst
    bpy.ops.object.modifier_apply(modifier=dt.name)
    print("[Success] Weights transferred")

def check_vertex_group(mesh, group_name):
    vg = mesh.vertex_groups.get(group_name)
    if vg:
        weighted_verts = sum(1 for v in mesh.data.vertices for g in v.groups if mesh.vertex_groups[g.group].name == group_name and g.weight > 0)
        print(f"[Debug] Vertex group '{group_name}' has {weighted_verts} vertices with weight > 0")
    else:
        print(f"[Error] Vertex group '{group_name}' not found on '{mesh.name}'")

def parent_to_armature(mesh, arm):
    print(f"[Step] Parenting mesh '{mesh.name}' to armature '{arm.name}'")
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    mod = mesh.modifiers.get("Armature")
    if not mod:
        mod = mesh.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    print("[Success] Parent set and Armature modifier applied")

def export_glb(path, mesh, arm):
    print(f"[Step] Exporting GLB to {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    nm = os.path.splitext(os.path.basename(path))[0]
    mesh.name = nm
    mesh.data.name = nm
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    mesh.select_set(True)
    arm.select_set(True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_skins=True)
    print("[Success] Export complete")

# --- ALIGNMENT HELPERS ---
def get_bounding_box_corners(obj):
    return [obj.matrix_world @ Vector(b) for b in obj.bound_box]

def get_bounding_box_bottom_center(corners):
    bottom_z_values = sorted(list(set(c.z for c in corners)))[:1]
    min_x = min(c.x for c in corners)
    max_x = max(c.x for c in corners)
    min_y = min(c.y for c in corners)
    max_y = max(c.y for c in corners)
    return Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, bottom_z_values[0]))

def align_meshes(target, template):
    print("[Step] Aligning meshes based on bottom-center of bounding boxes")
    target_corners = get_bounding_box_corners(target)
    template_corners = get_bounding_box_corners(template)
    translation = get_bounding_box_bottom_center(template_corners) - get_bounding_box_bottom_center(target_corners)
    target.location += translation
    print("[Success] Target aligned to template")

def get_vgroup_centroid(mesh, group_name):
    vg = mesh.vertex_groups.get(group_name)
    if not vg:
        print(f"[Warning] Vertex group '{group_name}' not found on mesh '{mesh.name}'.")
        return None
    total_w = 0.0
    sum_pos = Vector((0.0, 0.0, 0.0))
    for v in mesh.data.vertices:
        for g in v.groups:
            if mesh.vertex_groups[g.group].name == group_name:
                w = g.weight
                sum_pos += (mesh.matrix_world @ v.co) * w
                total_w += w
                break
    if total_w > 0:
        centroid = sum_pos / total_w
        print(f"[Debug] {group_name} Centroid (World): {centroid}")
        return centroid
    else:
        print(f"[Warning] Vertex group '{group_name}' has no weighted vertices.")
        return None

# --- MAIN ---
def main():
    print("[Start] Rigging script initiated")

    argv = sys.argv
    if "--" not in argv:
        print("[Error] Script requires arguments after '--'")
        print("Usage: blender --background --python script.py -- <target_face_count> <COLLAPSE/UNSUBDIV/PLANAR> <param> [align_method] [vhds_res] [vhds_smooth] [skip_vhds]")
        return

    args = argv[argv.index("--") + 1:]
    if len(args) < 3:
        print("[Error] Not enough arguments. Expected at least 3 (TF, Mode, Param).")
        print("Usage: blender --background --python script.py -- <target_face_count> <COLLAPSE/UNSUBDIV/PLANAR> <param> [align_method] [vhds_res] [vhds_smooth] [skip_vhds]")
        return

    tf = int(args[0])
    mode = args[1].upper()
    param = float(args[2])
    align_method = int(args[3]) if len(args) > 3 else 0
    vhds_res = float(args[4]) if len(args) > 4 else 0.1
    vhds_smooth = int(args[5]) if len(args) > 5 else 5
    skip_vhds = len(args) > 6 and args[6].lower() == "skip"

    clear_scene()

    template = import_template_mesh()
    arm = append_object(template_blend, arm_name)
    if not template or not arm:
        print("[Error] Template or Armature not found.")
        return

    # Bake transforms
    for obj in (template, arm):
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        print(f"[TransformApply] Baked transforms on {obj.name}")

    glbs = glob.glob(os.path.join(models_folder, "*.glb"))
    if not glbs:
        print(f"[Error] No GLB files found in {models_folder}.")
        return

    latest = max(glbs, key=os.path.getmtime)
    print(f"[Info] Using latest GLB: {latest}")
    target = import_glb(latest)
    if not target:
        print("[Error] Could not import GLB")
        return

    align_meshes(target, template)
    decimate_mesh(target, tf, mode, param)

    for vg in template.vertex_groups:
        if vg.name not in target.vertex_groups:
            target.vertex_groups.new(name=vg.name)

    transfer_weights(template, target)
    check_vertex_group(target, "neck")  # Debug neck vertex group

    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.mode_set(mode='EDIT')

    print(f"[Debug] Armature World Matrix: {arm.matrix_world}")

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
        vgroup_name_for_centroid = bone.name
        for vg_prefix, mapped_bones in vgroup_to_bone.items():
            if isinstance(mapped_bones, list) and bone.name in mapped_bones:
                vgroup_name_for_centroid = vg_prefix
                break
            elif isinstance(mapped_bones, str) and bone.name == mapped_bones:
                vgroup_name_for_centroid = vg_prefix
                break
        if vgroup_name_for_centroid == bone.name:
            for vg_prefix in vgroup_to_bone.keys():
                if bone.name.startswith(vg_prefix):
                    vgroup_name_for_centroid = vg_prefix
                    break

        cen = get_vgroup_centroid(target, vgroup_name_for_centroid)
        if cen:
            local_centroid = arm.matrix_world.inverted() @ cen
            original_head = bone.head.copy()
            original_tail = bone.tail.copy()
            if align_method == 0:
                bone.head = local_centroid
                bone.tail = local_centroid + (bone.tail - original_head)
            elif align_method == 1:
                bone.head = original_head + (local_centroid - original_head) * 0.5
                bone.tail = original_tail + (local_centroid - original_tail) * 0.5
            print(f"[Align] Bone {bone.name} to centroid of {vgroup_name_for_centroid}")
            print(f"[Debug] Bone {bone.name}, World Centroid: {cen}, Local Centroid: {local_centroid}, Original Head: {original_head}, New Head: {bone.head}, Original Tail: {original_tail}, New Tail: {bone.tail}")
        else:
            print(f"[Skip] No centroid found for {vgroup_name_for_centroid} for bone {bone.name}")

    bpy.ops.object.mode_set(mode='OBJECT')
    arm.data.display_type = 'STICK'

    parent_to_armature(target, arm)

    if not skip_vhds:
        print("[Step] Applying Voxel Heat Diffuse Skinning...")
        bpy.ops.object.select_all(action='DESELECT')
        target.select_set(True)
        arm.select_set(True)
        bpy.context.view_layer.objects.active = target
        try:
            bpy.ops.object.modifier_set_active(modifier="Armature")
            bpy.ops.object.voxel_remesh(mode='BOUNDED', resolution=vhds_res)
            bpy.ops.object.modifier_add(type='SMOOTH')
            bpy.context.object.modifiers["Smooth"].factor = 0.5
            bpy.context.object.modifiers["Smooth"].iterations = vhds_smooth
            bpy.ops.object.modifier_apply(modifier="Smooth")
            print(f"[Success] VHDS applied: res={vhds_res}, smooth={vhds_smooth}")
        except Exception as e:
            print(f"[Error] VHDS failed: {e}")

    out_path = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(latest))[0]}_rigged.glb")
    export_glb(out_path, target, arm)

    print("[Done] Script finished successfully.")

if __name__ == "__main__":
    main()

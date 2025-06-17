
import bpy
import os
import sys
import glob
from mathutils import Vector
from io_helper_connect import download_from_s3, upload_to_s3, get_mongo_collection, update_asset_status

# ——— USER SETTINGS ———
input_folder = "/home/ubuntu/sarthak/input"
output_folder = "/home/ubuntu/sarthak/output"
template_blend = "/home/ubuntu/log/royal1.blend"
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


































"""
# import bpy
# import os
# import sys
# import glob
# from mathutils import Vector

# # ——— USER SETTINGS ———
# models_folder         = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
# output_folder         = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"
# template_blend        = r"C:\Users\sarthak mohapatra\Downloads\mehses\royal2.blend"
# arm_name              = "metarig.001"
# template_mesh_name    = "reptile"
# template_glb_path     = os.path.join(models_folder, "reptile.glb")
# template_ply_path     = os.path.join(models_folder, "reptile.ply")

# # ——— UTILITIES ———
# def clear_scene():
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete(use_global=False)
#     if hasattr(bpy.ops.outliner, "orphans_purge"):
#         bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

# def import_glb(filepath):
#     bpy.ops.import_scene.gltf(filepath=filepath)
#     for o in bpy.context.selected_objects:
#         if o.type=="MESH":
#             bpy.context.view_layer.objects.active = o
#             print(f"Imported GLB mesh: {o.name}")
#             return o
#     return None

# def decimate_mesh(obj, tf, mode, p):
#     faces = len(obj.data.polygons)
#     if faces <= tf:
#         print("Skip decimate; already under target faces.")
#         return
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     bpy.context.view_layer.objects.active = obj
#     m = obj.modifiers.new("Decimate","DECIMATE")
#     if mode=="COLLAPSE":
#         m.decimate_type="COLLAPSE"; m.ratio=min(1.0,tf/faces)
#     elif mode=="UNSUBDIV":
#         m.decimate_type="UNSUBDIV"; m.iterations=int(p)
#     else:
#         m.decimate_type="PLANAR"; m.angle_limit=p
#     bpy.ops.object.modifier_apply(modifier=m.name)
#     print(f"Decimated from {faces}→{len(obj.data.polygons)} faces")

# def append_object(blend, name):
#     if not os.path.exists(blend):
#         print("Blend not found:", blend); return None
#     with bpy.data.libraries.load(blend, link=False) as (src,dst):
#         if name in src.objects:
#             dst.objects=[name]
#     obj=bpy.data.objects.get(name)
#     if obj:
#         bpy.context.collection.objects.link(obj)
#         obj.select_set(True)
#         bpy.context.view_layer.objects.active=obj
#         print(f"Appended `{name}` from {blend}`")
#     return obj

# def import_template_mesh():
#     tm = append_object(template_blend, template_mesh_name)
#     if tm and tm.type=="MESH":
#         return tm
#     if os.path.isfile(template_glb_path):
#         return import_glb(template_glb_path)
#     if os.path.isfile(template_ply_path):
#         bpy.ops.preferences.addon_enable(module="io_mesh_ply")
#         bpy.ops.import_mesh.ply(filepath=template_ply_path)
#         for o in bpy.context.selected_objects:
#             if o.type=="MESH":
#                 return o
#     for ply in glob.glob(os.path.join(models_folder,"*.ply")):
#         bpy.ops.preferences.addon_enable(module="io_mesh_ply")
#         bpy.ops.import_mesh.ply(filepath=ply)
#         for o in bpy.context.selected_objects:
#             if o.type=="MESH":
#                 return o
#     return None

# def transfer_weights(src,dst):
#     dt=dst.modifiers.new("WeightTransfer","DATA_TRANSFER")
#     dt.object=src; dt.use_vert_data=True
#     dt.data_types_verts={'VGROUP_WEIGHTS'}; dt.vert_mapping='NEAREST'
#     bpy.context.view_layer.objects.active=dst
#     bpy.ops.object.modifier_apply(modifier=dt.name)
#     print("Weights transferred")

# def parent_to_armature(mesh,arm):
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True); arm.select_set(True)
#     bpy.context.view_layer.objects.active=arm
#     bpy.ops.object.parent_set(type='ARMATURE_NAME')
#     mod=mesh.modifiers.get("Armature")
#     if not mod: mod=mesh.modifiers.new("Armature","ARMATURE")
#     mod.object=arm
#     print("Parented + Armature modifier set")

# def export_glb(path,mesh,arm):
#     os.makedirs(os.path.dirname(path),exist_ok=True)
#     nm=os.path.splitext(os.path.basename(path))[0]
#     mesh.name=data_name=nm; mesh.data.name=nm
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     bpy.context.view_layer.objects.active=mesh
#     bpy.ops.object.transform_apply(location=True,rotation=True,scale=True)
#     bpy.context.view_layer.update()
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True); arm.select_set(True)
#     bpy.context.view_layer.objects.active=arm
#     bpy.ops.export_scene.gltf(filepath=path,export_format='GLB',
#                                  use_selection=True,export_apply=True,export_skins=True)
#     print("Exported GLB:",path)

# # ——— NEW: Auto‑fit Rig to Mesh by Landmark Vertex‑Groups ———
# def align_rig_to_mesh(mesh, arm):
#     """Move each bone head to the position of the assigned vertices in its vertex-group."""
#     # Switch to Object Mode
#     bpy.ops.object.mode_set(mode='OBJECT')
#     bpy.context.view_layer.objects.active = mesh

#     # Create vertex groups based on armature bone names
#     arm_obj = bpy.data.objects.get(arm.name)
#     if arm_obj and arm_obj.type == 'ARMATURE':
#         bpy.context.view_layer.objects.active = arm_obj
#         bpy.ops.object.mode_set(mode='EDIT')
#         for bone in arm_obj.data.edit_bones:
#             if bone.name not in mesh.vertex_groups:
#                 mesh.vertex_groups.new(name=bone.name)
#         bpy.ops.object.mode_set(mode='OBJECT')
#         print("Vertex groups created on mesh based on armature bone names.")
#     else:
#         print(f"Error: Armature object '{arm.name}' not found or is not an armature.")
#         return

#     # Switch to Edit Mode on armature
#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.mode_set(mode='EDIT')
#     ebones = arm.data.edit_bones

#     # For each bone in the armature
#     for bone_name in ebones.keys():
#         if bone_name in mesh.vertex_groups:
#             vgroup = mesh.vertex_groups[bone_name]
#             # Assuming you have manually assigned landmark vertices *before* running this script
#             group_indices = [v.index for v in mesh.data.vertices if vgroup.index in [g.group for g in v.groups]]

#             if group_indices:
#                 # Get the world coordinates of the vertices in this group
#                 world_coords = [mesh.matrix_world @ mesh.data.vertices[i].co for i in group_indices]

#                 # Calculate the average position (will be the landmark if only one vertex is assigned)
#                 avg_world = sum(world_coords, Vector()) / len(world_coords)

#                 # Convert the average world position to armature local space
#                 local_avg = arm.matrix_world.inverted() @ avg_world

#                 bone = ebones[bone_name]
#                 delta = local_avg - bone.head
#                 bone.head = local_avg
#                 bone.tail += delta
#                 print(f"Bone '{bone_name}': Head moved to landmark.")
#             else:
#                 print(f"Warning: Vertex group '{bone_name}' found, but no vertices assigned.")
#         else:
#             print(f"Warning: Bone '{bone_name}' has no corresponding vertex group (this should not happen now).")

#     # return to Object Mode
#     bpy.ops.object.mode_set(mode='OBJECT')
#     print("Rig aligned to mesh landmarks.")

# # ——— MAIN ———
# def main():
#     a=sys.argv; args=a[a.index("--")+1:]
#     if len(args)<3:
#         print("Usage: decimate_rig.py TARGET COLLAPSE/UNSUBDIV/PLANAR PARAM"); return
#     tf=int(args[0]); mode=args[1].upper(); p=float(args[2])

#     glbs=glob.glob(os.path.join(models_folder,"*.glb"))
#     if not glbs: print("No input .glb"); return
#     inp=max(glbs, key=os.path.getmtime)
#     out=os.path.join(output_folder,os.path.splitext(os.path.basename(inp))[0]+"_rigged.glb")

#     clear_scene()

#     mesh=import_glb(inp)
#     if not mesh: print("Import failed"); return
#     decimate_mesh(mesh,tf,mode,p)

#     arm=append_object(template_blend,arm_name)
#     if not arm or arm.type!="ARMATURE": print("Armature failed"); return

#     # Automatically create vertex groups and then fit the rig
#     align_rig_to_mesh(mesh, arm)

#     parent_to_armature(mesh,arm)

#     tmpl=import_template_mesh()
#     if tmpl:
#         transfer_weights(tmpl,mesh)
#     else:
#         print("⚠️ no template mesh—weights missing")

#     export_glb(out,mesh,arm)

# if __name__=="__main__":

# BEST ONE


# import bpy
# import os
# import sys
# import glob
# from mathutils import Vector

# # ——— USER SETTINGS ———
# models_folder       = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
# output_folder       = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"
# template_blend      = r"C:\Users\sarthak mohapatra\Downloads\mehses\royal1.blend"
# arm_name            = "metarig.001"       # Armature name in the template blend
# template_mesh_name  = "villager"           # Mesh name in the template blend

# # ——— SCENE UTILITIES ———
# def clear_scene():
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete(use_global=False)
#     if hasattr(bpy.ops.outliner, "orphans_purge"):
#         bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

# def import_glb(filepath):
#     bpy.ops.import_scene.gltf(filepath=filepath)
#     for o in bpy.context.selected_objects:
#         if o.type == "MESH":
#             bpy.context.view_layer.objects.active = o
#             print(f"[Import] GLB mesh: {o.name}")
#             return o
#     return None

# def append_object(blend, name):
#     if not os.path.exists(blend):
#         print(f"[Error] Blend not found: {blend}")
#         return None
#     with bpy.data.libraries.load(blend, link=False) as (src, dst):
#         if name in src.objects:
#             dst.objects = [name]
#     obj = bpy.data.objects.get(name)
#     if obj:
#         bpy.context.collection.objects.link(obj)
#         obj.select_set(True)
#         bpy.context.view_layer.objects.active = obj
#         print(f"[Append] {name} from {blend}")
#     return obj

# def import_template_mesh():
#     return append_object(template_blend, template_mesh_name)

# # ——— MESH PROCESSING ———
# def decimate_mesh(obj, tf, mode, p):
#     faces = len(obj.data.polygons)
#     if faces <= tf:
#         print("[Decimate] Skip; under threshold")
#         return
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     bpy.context.view_layer.objects.active = obj
#     mod = obj.modifiers.new("Decimate", "DECIMATE")
#     if mode == "COLLAPSE":
#         mod.decimate_type = "COLLAPSE"
#         mod.ratio = min(1.0, tf / faces)
#     elif mode == "UNSUBDIV":
#         mod.decimate_type = "UNSUBDIV"
#         mod.iterations = int(p)
#     else:
#         mod.decimate_type = "PLANAR"
#         mod.angle_limit = p
#     bpy.ops.object.modifier_apply(modifier=mod.name)
#     print(f"[Decimate] {faces} → {len(obj.data.polygons)} faces")

# def transfer_weights(src, dst):
#     dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
#     dt.object = src
#     dt.use_vert_data = True
#     dt.data_types_verts = {'VGROUP_WEIGHTS'}
#     dt.vert_mapping = 'NEAREST'
#     bpy.context.view_layer.objects.active = dst
#     bpy.ops.object.modifier_apply(modifier=dt.name)
#     print("[Weights] Transferred template → target")

# def parent_to_armature(mesh, arm):
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     arm.select_set(True)
#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.parent_set(type='ARMATURE_NAME')
#     mod = mesh.modifiers.get("Armature")
#     if not mod:
#         mod = mesh.modifiers.new("Armature", "ARMATURE")
#     mod.object = arm
#     print("[Parent] Mesh → Armature")

# def export_glb(path, mesh, arm):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     nm = os.path.splitext(os.path.basename(path))[0]
#     mesh.name = nm
#     mesh.data.name = nm
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     bpy.context.view_layer.objects.active = mesh
#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     arm.select_set(True)
#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.export_scene.gltf(
#         filepath=path,
#         export_format='GLB',
#         use_selection=True,
#         export_apply=True,
#         export_skins=True
#     )
#     print(f"[Export] {path}")

# # ——— ALIGNMENT HELPERS ———
# def get_bounding_box_corners(obj):
#     return [obj.matrix_world @ Vector(b) for b in obj.bound_box]

# def get_bounding_box_bottom_center(corners):
#     bottom_z_values = sorted(list(set(c.z for c in corners)))[:1]
#     bottom_corners = [c for c in corners if c.z == bottom_z_values[0]]
#     min_x = min(c.x for c in bottom_corners)
#     max_x = max(c.x for c in bottom_corners)
#     min_y = min(c.y for c in bottom_corners)
#     max_y = max(c.y for c in bottom_corners)
#     return Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, bottom_z_values[0]))

# def align_meshes(target, template):
#     target_corners = get_bounding_box_corners(target)
#     template_corners = get_bounding_box_corners(template)
#     translation = get_bounding_box_bottom_center(template_corners) - get_bounding_box_bottom_center(target_corners)
#     target.location += translation
#     print("[Align] Target mesh bottom-center to template bottom-center")

# def get_vgroup_centroid(mesh, group_name):
#     vg = mesh.vertex_groups.get(group_name)
#     if not vg:
#         return None
#     total_w = 0.0
#     sum_pos = Vector((0.0, 0.0, 0.0))
#     for v in mesh.data.vertices:
#         for g in v.groups:
#             if mesh.vertex_groups[g.group].name == group_name:
#                 w = g.weight
#                 sum_pos += (mesh.matrix_world @ v.co) * w
#                 total_w += w
#                 break
#     return (sum_pos / total_w) if total_w > 0 else None

# # ——— MAIN ———
# def main():
#     argv = sys.argv
#     if "--" not in argv:
#         print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM")
#         return
#     args = argv[argv.index("--") + 1:]
#     if len(args) < 3:
#         print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM")
#         return

#     tf = int(args[0])
#     mode = args[1].upper()
#     param = float(args[2])

#     clear_scene()

#     template = import_template_mesh()
#     arm = append_object(template_blend, arm_name)
#     if not template or not arm:
#         print("[Error] Template or Armature not found")
#         return

#     glbs = glob.glob(os.path.join(models_folder, "*.glb"))
#     if not glbs:
#         print("[Error] No GLB found")
#         return
#     latest = max(glbs, key=os.path.getmtime)
#     target = import_glb(latest)
#     if not target:
#         print("[Error] Failed to import GLB")
#         return

#     align_meshes(target, template)
#     decimate_mesh(target, tf, mode, param)

#     for vg in template.vertex_groups:
#         if vg.name not in target.vertex_groups:
#             target.vertex_groups.new(name=vg.name)

#     transfer_weights(template, target)

#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.mode_set(mode='EDIT')

#     vgroup_to_bone = {
#         "hand.L": "wrist.L",
#         "hand.R": "wrist.R",
#         "neck": ["neck.001", "neck.002"]
#     }

#     for bone in arm.data.edit_bones:
#         vgroup_name = bone.name
#         for vg_prefix, mapped_bones in vgroup_to_bone.items():
#             if isinstance(mapped_bones, list) and bone.name in mapped_bones:
#                 vgroup_name = vg_prefix
#                 break
#             elif bone.name.startswith(vg_prefix):
#                 vgroup_name = vg_prefix
#                 break

#         cen = get_vgroup_centroid(target, vgroup_name)
#         if cen:
#             local_centroid = arm.matrix_world.inverted() @ cen
#             delta = local_centroid - bone.head
#             bone.head = local_centroid.copy()
#             bone.tail += delta
#             print(f"[Align] Bone {bone.name} to centroid of {vgroup_name}")
#         else:
#             print(f"[Align] No centroid for {vgroup_name}")

#     bpy.ops.object.mode_set(mode='OBJECT')

#     parent_to_armature(target, arm)

#     # Apply Voxel Heat Diffuse Skinning
#     bpy.ops.object.select_all(action='DESELECT')
#     target.select_set(True)
#     arm.select_set(True)
#     bpy.context.view_layer.objects.active = arm
#     try:
#         bpy.ops.wm.voxel_heat_diffuse()
#         print("[VHDS] Voxel Heat Diffuse Skinning applied successfully.")
#     except Exception as e:
#         print(f"[VHDS] Failed to apply Voxel Heat Diffuse: {e}")

#     out_path = os.path.join(
#         output_folder,
#         f"{os.path.splitext(os.path.basename(latest))[0]}_rigged.glb"
#     )
#     export_glb(out_path, target, arm)

# if __name__ == "__main__":
#     main()

#MAYBE BE GOOD ENOUGH 

# import bpy
# import os
# import sys
# import glob
# from mathutils import Vector

# # ——— USER SETTINGS ———
# models_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
# output_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"
# template_blend = r"C:\Users\sarthak mohapatra\Downloads\mehses\royal1.blend"
# arm_name = "metarig.001"  # Armature name in the template blend
# template_mesh_name = "villager"  # Mesh name in the template blend

# # ——— SCENE UTILITIES ———
# def clear_scene():
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete(use_global=False)
#     if hasattr(bpy.ops.outliner, "orphans_purge"):
#         bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True,
#                          do_recursive=True)


# def import_glb(filepath):
#     bpy.ops.import_scene.gltf(filepath=filepath)
#     for o in bpy.context.selected_objects:
#         if o.type == "MESH":
#             bpy.context.view_layer.objects.active = o
#             print(f"[Import] GLB mesh: {o.name}")
#             return o
#     return None


# def append_object(blend, name):
#     if not os.path.exists(blend):
#         print(f"[Error] Blend not found: {blend}")
#         return None
#     with bpy.data.libraries.load(blend, link=False) as (src, dst):
#         if name in src.objects:
#             dst.objects = [name]
#     obj = bpy.data.objects.get(name)
#     if obj:
#         bpy.context.collection.objects.link(obj)
#         obj.select_set(True)
#         bpy.context.view_layer.objects.active = obj
#         print(f"[Append] {name} from {blend}")
#     return obj


# def import_template_mesh():
#     return append_object(template_blend, template_mesh_name)


# # ——— MESH PROCESSING ———
# def decimate_mesh(obj, tf, mode, p):
#     faces = len(obj.data.polygons)
#     if faces <= tf:
#         print("[Decimate] Skip; under threshold")
#         return
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     bpy.context.view_layer.objects.active = obj
#     mod = obj.modifiers.new("Decimate", "DECIMATE")
#     if mode == "COLLAPSE":
#         mod.decimate_type = "COLLAPSE"
#         mod.ratio = min(1.0, tf / faces)
#     elif mode == "UNSUBDIV":
#         mod.decimate_type = "UNSUBDIV"
#         mod.iterations = int(p)
#     else:
#         mod.decimate_type = "PLANAR"
#         mod.angle_limit = p
#     bpy.ops.object.modifier_apply(modifier=mod.name)
#     print(f"[Decimate] {faces} → {len(obj.data.polygons)} faces")


# def transfer_weights(src, dst):
#     dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
#     dt.object = src
#     dt.use_vert_data = True
#     dt.data_types_verts = {'VGROUP_WEIGHTS'}
#     dt.vert_mapping = 'NEAREST'
#     bpy.context.view_layer.objects.active = dst
#     bpy.ops.object.modifier_apply(modifier=dt.name)
#     print("[Weights] Transferred template → target")


# def parent_to_armature(mesh, arm):
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     arm.select_set(True)
#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.parent_set(type='ARMATURE_NAME')
#     mod = mesh.modifiers.get("Armature")
#     if not mod:
#         mod = mesh.modifiers.new("Armature", "ARMATURE")
#     mod.object = arm
#     print("[Parent] Mesh → Armature")


# def export_glb(path, mesh, arm):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     nm = os.path.splitext(os.path.basename(path))[0]
#     mesh.name = nm
#     mesh.data.name = nm
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     bpy.context.view_layer.objects.active = mesh
#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     arm.select_set(True)
#     bpy.ops.export_scene.gltf(
#         filepath=path,
#         export_format='GLB',
#         use_selection=True,
#         export_apply=True,
#         export_skins=True)
#     print(f"[Export] {path}")


# # ——— ALIGNMENT HELPERS ———
# def get_bounding_box_corners(obj):
#     return [obj.matrix_world @ Vector(b) for b in obj.bound_box]


# def get_bounding_box_bottom_center(corners):
#     bottom_z_values = sorted(list(set(c.z for c in corners)))[:1]
#     bottom_corners = [c for c in corners if c.z == bottom_z_values[0]]
#     min_x = min(c.x for c in bottom_corners)
#     max_x = max(c.x for c in corners)
#     min_y = min(c.y for c in corners)
#     max_y = max(c.y for c in corners)
#     return Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, bottom_z_values[0]))


# def align_meshes(target, template):
#     target_corners = get_bounding_box_corners(target)
#     template_corners = get_bounding_box_corners(template)
#     translation = get_bounding_box_bottom_center(
#         template_corners) - get_bounding_box_bottom_center(target_corners)
#     target.location += translation
#     print("[Align] Target mesh bottom-center to template bottom-center")


# def get_vgroup_centroid(mesh, group_name):
#     vg = mesh.vertex_groups.get(group_name)
#     if not vg:
#         return None
#     total_w = 0.0
#     sum_pos = Vector((0.0, 0.0, 0.0))
#     for v in mesh.data.vertices:
#         for g in v.groups:
#             if mesh.vertex_groups[g.group].name == group_name:
#                 w = g.weight
#                 sum_pos += (mesh.matrix_world @ v.co) * w
#                 total_w += w
#                 break
#     return (sum_pos / total_w) if total_w > 0 else None


# # ——— MAIN ———
# def main():
#     argv = sys.argv
#     if "--" not in argv:
#         print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM align_method vhds_res vhds_smooth")
#         print("  align_method: 0 = use centroid, 1 = use existing bone positions")
#         print("  vhds_res: Voxel size for VHDS (e.g., 0.1)")
#         print("  vhds_smooth: Smooth iterations for VHDS (e.g., 5)")
#         return
#     args = argv[argv.index("--") + 1:]
#     if len(args) < 3:
#         print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM align_method vhds_res vhds_smooth")
#         print("  align_method: 0 = use centroid, 1 = use existing bone positions")
#         print("  vhds_res: Voxel size for VHDS (e.g., 0.1)")
#         print("  vhds_smooth: Smooth iterations for VHDS (e.g., 5)")
#         return

#     tf = int(args[0])
#     mode = args[1].upper()
#     param = float(args[2])
#     align_method = int(args[3]) if len(args) > 3 else 0  # 0: centroid, 1: existing
#     vhds_res = float(args[4]) if len(args) > 4 else 0.1
#     vhds_smooth = int(args[5]) if len(args) > 5 else 5

#     clear_scene()

#     template = import_template_mesh()
#     arm = append_object(template_blend, arm_name)
#     if not template or not arm:
#         print("[Error] Template or Armature not found")
#         return

#     glbs = glob.glob(os.path.join(models_folder, "*.glb"))
#     if not glbs:
#         print("[Error] No GLB found")
#         return
#     latest = max(glbs, key=os.path.getmtime)
#     target = import_glb(latest)
#     if not target:
#         print("[Error] Failed to import GLB")
#         return

#     align_meshes(target, template)
#     decimate_mesh(target, tf, mode, param)

#     for vg in template.vertex_groups:
#         if vg.name not in target.vertex_groups:
#             target.vertex_groups.new(name=vg.name)

#     transfer_weights(template, target)

#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.mode_set(mode='EDIT')

#     vgroup_to_bone = {
#         "hand.L": "wrist.L",
#         "hand.R": "wrist.R",
#         "forearm.L": "elbow.L",
#         "forearm.R": "elbow.R",
#         "upperarm.L": "upper_arm.L",
#         "upperarm.R": "upper_arm.R",
#         "neck": ["neck.001", "neck.002"]
#     }

#     for bone in arm.data.edit_bones:
#         vgroup_name = bone.name
#         for vg_prefix, mapped_bones in vgroup_to_bone.items():
#             if isinstance(mapped_bones, list) and bone.name in mapped_bones:
#                 vgroup_name = vg_prefix
#                 break
#             elif bone.name.startswith(vg_prefix):
#                 vgroup_name = vg_prefix
#                 break

#         cen = get_vgroup_centroid(target, vgroup_name)
#         if cen:
#             local_centroid = arm.matrix_world.inverted() @ cen

#             if align_method == 0:  # Use centroid
#                 bone.head = local_centroid
#                 bone.tail = local_centroid + (bone.tail - bone.head)
#             elif align_method == 1:  # Use existing bone positions
#                 original_head = bone.head
#                 original_tail = bone.tail
#                 # Move towards centroid, but keep some of the original position
#                 bone.head = original_head + (local_centroid - original_head) * 0.5
#                 bone.tail = original_tail + (local_centroid - original_tail) * 0.5

#             print(f"[Align] Bone {bone.name} to centroid of {vgroup_name}")
#         else:
#             print(f"[Align] No centroid for {vgroup_name}")

#     bpy.ops.object.mode_set(mode='OBJECT')

#     # Change the armature display type
#     arm.data.display_type = 'STICK'  # Changed from arm.display_type

#     parent_to_armature(target, arm)

#     # Apply Voxel Heat Diffuse Skinning
#     bpy.ops.object.select_all(action='DESELECT')
#     target.select_set(True)
#     arm.select_set(True)
#     bpy.context.view_layer.objects.active = arm
#     try:
#         bpy.ops.object.modifier_set_active(modifier="Armature")
#         bpy.ops.object.voxel_remesh(mode='BOUNDED', resolution=vhds_res)
#         bpy.ops.object.modifier_add(type='SMOOTH')
#         bpy.context.object.modifiers["Smooth"].factor = 0.5
#         bpy.context.object.modifiers["Smooth"].iterations = vhds_smooth
#         bpy.ops.object.modifier_apply(modifier="Smooth")
#         print(
#             f"[VHDS] Voxel Heat Diffuse Skinning applied successfully. Res: {vhds_res}, Smooth: {vhds_smooth}")
#     except Exception as e:
#         print(f"[VHDS] Failed to apply Voxel Heat Diffuse: {e}")

#     out_path = os.path.join(output_folder,
#                             f"{os.path.splitext(os.path.basename(latest))[0]}_rigged.glb"
#                             )
#     export_glb(out_path, target, arm)


# if __name__ == "__main__":
#     main()
r"""
"C:\Program Files (x86)\Steam\steamapps\common\Blender\blender.exe" --background --python "C:\Users\sarthak mohapatra\Downloads\mehses\decimate_rig.py" -- 10000 COLLAPSE 0.5 1 0.03 10
"""

#TILL NOW GOOD
# import bpy
# import os
# import sys
# import glob
# from mathutils import Vector

# # ——— USER SETTINGS ———
# models_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
# output_folder = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"
# template_blend = r"C:\Users\sarthak mohapatra\Downloads\mehses\royal1.blend"
# arm_name = "metarig.001"  # Armature name in the template blend
# template_mesh_name = "villager"  # Mesh name in the template blend

# # ——— SCENE UTILITIES ———
# def clear_scene():
#     bpy.ops.object.select_all(action='SELECT')
#     bpy.ops.object.delete(use_global=False)
#     if hasattr(bpy.ops.outliner, "orphans_purge"):
#         bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True,
#                          do_recursive=True)


# def import_glb(filepath):
#     bpy.ops.import_scene.gltf(filepath=filepath)
#     for o in bpy.context.selected_objects:
#         if o.type == "MESH":
#             bpy.context.view_layer.objects.active = o
#             print(f"[Import] GLB mesh: {o.name}")
#             return o
#     return None


# def append_object(blend, name):
#     if not os.path.exists(blend):
#         print(f"[Error] Blend not found: {blend}")
#         return None
#     with bpy.data.libraries.load(blend, link=False) as (src, dst):
#         if name in src.objects:
#             dst.objects = [name]
#     obj = bpy.data.objects.get(name)
#     if obj:
#         bpy.context.collection.objects.link(obj)
#         obj.select_set(True)
#         bpy.context.view_layer.objects.active = obj
#         print(f"[Append] {name} from {blend}")
#     return obj


# def import_template_mesh():
#     return append_object(template_blend, template_mesh_name)


# # ——— MESH PROCESSING ———
# def decimate_mesh(obj, tf, mode, p):
#     faces = len(obj.data.polygons)
#     if faces <= tf:
#         print("[Decimate] Skip; under threshold")
#         return
#     bpy.ops.object.select_all(action='DESELECT')
#     obj.select_set(True)
#     bpy.context.view_layer.objects.active = obj
#     mod = obj.modifiers.new("Decimate", "DECIMATE")
#     if mode == "COLLAPSE":
#         mod.decimate_type = "COLLAPSE"
#         mod.ratio = min(1.0, tf / faces)
#     elif mode == "UNSUBDIV":
#         mod.decimate_type = "UNSUBDIV"
#         mod.iterations = int(p)
#     else:
#         mod.decimate_type = "PLANAR"
#         mod.angle_limit = p
#     bpy.ops.object.modifier_apply(modifier=mod.name)
#     print(f"[Decimate] {faces} → {len(obj.data.polygons)} faces")


# def transfer_weights(src, dst):
#     dt = dst.modifiers.new("WeightTransfer", "DATA_TRANSFER")
#     dt.object = src
#     dt.use_vert_data = True
#     dt.data_types_verts = {'VGROUP_WEIGHTS'}
#     dt.vert_mapping = 'NEAREST'
#     bpy.context.view_layer.objects.active = dst
#     bpy.ops.object.modifier_apply(modifier=dt.name)
#     print("[Weights] Transferred template → target")


# def parent_to_armature(mesh, arm):
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     arm.select_set(True)
#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.parent_set(type='ARMATURE_NAME')
#     mod = mesh.modifiers.get("Armature")
#     if not mod:
#         mod = mesh.modifiers.new("Armature", "ARMATURE")
#     mod.object = arm
#     print("[Parent] Mesh → Armature")


# def export_glb(path, mesh, arm):
#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     nm = os.path.splitext(os.path.basename(path))[0]
#     mesh.name = nm
#     mesh.data.name = nm
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     bpy.context.view_layer.objects.active = mesh
#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
#     bpy.ops.object.select_all(action='DESELECT')
#     mesh.select_set(True)
#     arm.select_set(True)
#     bpy.ops.export_scene.gltf(
#         filepath=path,
#         export_format='GLB',
#         use_selection=True,
#         export_apply=True,
#         export_skins=True)
#     print(f"[Export] {path}")


# # ——— ALIGNMENT HELPERS ———
# def get_bounding_box_corners(obj):
#     return [obj.matrix_world @ Vector(b) for b in obj.bound_box]


# def get_bounding_box_bottom_center(corners):
#     bottom_z_values = sorted(list(set(c.z for c in corners)))[:1]
#     bottom_corners = [c for c in corners if c.z == bottom_z_values[0]]
#     min_x = min(c.x for c in bottom_corners)
#     max_x = max(c.x for c in corners)
#     min_y = min(c.y for c in bottom_corners)
#     max_y = max(c.y for c in corners)
#     return Vector(((min_x + max_x) / 2, (min_y + max_y) / 2, bottom_z_values[0]))


# def align_meshes(target, template):
#     target_corners = get_bounding_box_corners(target)
#     template_corners = get_bounding_box_corners(template)
#     translation = get_bounding_box_bottom_center(
#         template_corners) - get_bounding_box_bottom_center(target_corners)
#     target.location += translation
#     print("[Align] Target mesh bottom-center to template bottom-center")


# def get_vgroup_centroid(mesh, group_name):
#     vg = mesh.vertex_groups.get(group_name)
#     if not vg:
#         return None
#     total_w = 0.0
#     sum_pos = Vector((0.0, 0.0, 0.0))
#     for v in mesh.data.vertices:
#         for g in v.groups:
#             if mesh.vertex_groups[g.group].name == group_name:
#                 w = g.weight
#                 sum_pos += (mesh.matrix_world @ v.co) * w
#                 total_w += w
#                 break
#     return (sum_pos / total_w) if total_w > 0 else None


# # ——— MAIN ———
# def main():
#     argv = sys.argv
#     if "--" not in argv:
#         print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM align_method vhds_res vhds_smooth")
#         print("  align_method: 0 = use centroid, 1 = use existing bone positions")
#         print("  vhds_res: Voxel size for VHDS (e.g., 0.1)")
#         print("  vhds_smooth: Smooth iterations for VHDS (e.g., 5)")
#         return
#     args = argv[argv.index("--") + 1:]
#     if len(args) < 3:
#         print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM align_method vhds_res vhds_smooth")
#         print("  align_method: 0 = use centroid, 1 = use existing bone positions")
#         print("  vhds_res: Voxel size for VHDS (e.g., 0.1)")
#         print("  vhds_smooth: Smooth iterations for VHDS (e.g., 5)")
#         return

#     tf = int(args[0])
#     mode = args[1].upper()
#     param = float(args[2])
#     align_method = int(args[3]) if len(args) > 3 else 0  # 0: centroid, 1: existing
#     vhds_res = float(args[4]) if len(args) > 4 else 0.1
#     vhds_smooth = int(args[5]) if len(args) > 5 else 5

#     clear_scene()

#     template = import_template_mesh()
#     arm = append_object(template_blend, arm_name)
#     if not template or not arm:
#         print("[Error] Template or Armature not found")
#         return

#     glbs = glob.glob(os.path.join(models_folder, "*.glb"))
#     if not glbs:
#         print("[Error] No GLB found")
#         return
#     latest = max(glbs, key=os.path.getmtime)
#     target = import_glb(latest)
#     if not target:
#         print("[Error] Failed to import GLB")
#         return

#     align_meshes(target, template)
#     decimate_mesh(target, tf, mode, param)

#     for vg in template.vertex_groups:
#         if vg.name not in target.vertex_groups:
#             target.vertex_groups.new(name=vg.name)

#     transfer_weights(template, target)

#     bpy.context.view_layer.objects.active = arm
#     bpy.ops.object.mode_set(mode='EDIT')

#     vgroup_to_bone = {
#         "hand.L": "wrist.L",
#         "hand.R": "wrist.R",
#         "forearm.L": "elbow.L",
#         "forearm.R": "elbow.R",
#         "upperarm.L": "upper_arm.L",
#         "upperarm.R": "upper_arm.R",
#         "neck": ["neck.001", "neck.002"]
#     }

#     for bone in arm.data.edit_bones:
#         vgroup_name = bone.name
#         for vg_prefix, mapped_bones in vgroup_to_bone.items():
#             if isinstance(mapped_bones, list) and bone.name in mapped_bones:
#                 vgroup_name = vg_prefix
#                 break
#             elif bone.name.startswith(vg_prefix):
#                 vgroup_name = vg_prefix
#                 break

#         cen = get_vgroup_centroid(target, vgroup_name)
#         if cen:
#             local_centroid = arm.matrix_world.inverted() @ cen

#             if align_method == 1:
#                 # Use existing bone head location (no change)
#                 pass
#             else:
#                 bone.head = local_centroid
#                 bone.tail = local_centroid + Vector((0, 0, 0.1))
#     bpy.ops.object.mode_set(mode='OBJECT')

#     parent_to_armature(target, arm)

#     # Apply transforms except scale, set scale=1, quaternion w=1, x,y,z unchanged
#     bpy.ops.object.select_all(action='DESELECT')
#     target.select_set(True)
#     bpy.context.view_layer.objects.active = target

#     bpy.ops.object.transform_apply(location=True, rotation=True, scale=False)
#     target.scale = (1.0, 1.0, 1.0)

#     if target.rotation_mode != 'QUATERNION':
#         target.rotation_mode = 'QUATERNION'
#     q = target.rotation_quaternion
#     target.rotation_quaternion = (1.0, q.x, q.y, q.z)

#     # Select armature and mesh for voxel heat diffuse skinning
#     bpy.ops.object.mode_set(mode='OBJECT')
#     bpy.ops.object.select_all(action='DESELECT')
#     arm.select_set(True)
#     target.select_set(True)
#     bpy.context.view_layer.objects.active = arm

#     bpy.ops.wm.voxel_heat_diffuse()

#     output_path = os.path.join(output_folder, os.path.basename(latest))
#     export_glb(output_path, target, arm)

# if __name__ == "__main__":
#     main()


import bpy
import os
import sys
import glob
from mathutils import Vector
import time

# --- USER SETTINGS ---
models_folder   = r"C:\Users\sarthak mohapatra\Downloads\mehses\models"
output_folder   = r"C:\Users\sarthak mohapatra\Downloads\mehses\output"
template_blend  = r"C:\Users\sarthak mohapatra\Downloads\mehses\royal1_t.blend"
arm_name        = "metarig.001"   # Armature name in the template blend
template_mesh   = "villager"      # Mesh name in the template blend

# --- SCENE UTILITIES ---
def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    if hasattr(bpy.ops.outliner, "orphans_purge"):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)

def import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)
    for o in bpy.context.selected_objects:
        if o.type == 'MESH':
            bpy.context.view_layer.objects.active = o
            print(f"[Import] {o.name}")
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
        bpy.context.view_layer.objects.active = obj
        print(f"[Append] {name}")
    return obj

def parent_to_armature(mesh, arm):
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = arm
    bpy.ops.object.parent_set(type='ARMATURE_NAME')
    # ensure the modifier is present
    mod = mesh.modifiers.get("Armature")
    if not mod:
        mod = mesh.modifiers.new("Armature", "ARMATURE")
    mod.object = arm
    print("[Parent] Mesh → Armature")

def export_glb(path, mesh, arm):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    name = os.path.splitext(os.path.basename(path))[0]
    mesh.name = name
    mesh.data.name = name
    bpy.ops.object.select_all(action='DESELECT')
    mesh.select_set(True)
    arm.select_set(True)
    bpy.context.view_layer.objects.active = mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_skins=True
    )
    print(f"[Export] {path}")

# --- MAIN PIPELINE ---
def main():
    # parse args
    argv = sys.argv
    if "--" not in argv:
        print("Usage: script.py TARGET_FACE_COUNT COLLAPSE/UNSUBDIV/PLANAR PARAM align_method")
        return
    args = argv[argv.index("--") + 1:]
    tf           = int(args[0])
    mode         = args[1].upper()
    param        = float(args[2])
    align_method = int(args[3]) if len(args) > 3 else 0

    clear_scene()

    # load template mesh + armature
    template = append_object(template_blend, template_mesh)
    arm      = append_object(template_blend, arm_name)
    if not template or not arm:
        print("[Error] Template or Armature not found")
        return

    # import latest GLB
    glbs = glob.glob(os.path.join(models_folder, "*.glb"))
    if not glbs:
        print("[Error] No GLB found")
        return
    latest = max(glbs, key=os.path.getmtime)
    target = import_glb(latest)
    if not target:
        print("[Error] Import failed")
        return

    # transfer skin weights from template → target
    dt = target.modifiers.new("WeightTransfer", "DATA_TRANSFER")
    dt.object = template
    dt.use_vert_data = True
    dt.data_types_verts = {'VGROUP_WEIGHTS'}
    dt.vert_mapping = 'NEAREST'
    bpy.context.view_layer.objects.active = target
    bpy.ops.object.modifier_apply(modifier=dt.name)
    print("[Weights] Template → Target")

    # Check vertex groups after transfer
    if len(target.vertex_groups) == 0:
        print("[Error] No vertex groups after transfer. Aborting.")
        return
    else:
        print(f"[Info] {len(target.vertex_groups)} vertex groups after transfer.")

    # call the VHDS operator as you already do:
    try:
        bpy.ops.wm.voxel_heat_diffuse()  
        print("[VHDS] Applied")
    except Exception as e:
        print(f"[VHDS] Failed: {e}")

    # Check weights post-VHDS
    weights_exist = any(any(vg.weight > 0 for vg in v.groups) for v in target.data.vertices)
    if not weights_exist:
        print("[Warning] All weights zero after VHDS. Attempting to fix.")

    # --- AUTOMATIC INFLUENCE CLEANUP ---
    print("[Cleanup] Cleaning vertex influences…")
    bone_positions = {b.name: arm.matrix_world @ b.head for b in arm.data.bones}

    def closest_bone(pt):
        return min(bone_positions.items(),
                   key=lambda bi: (pt - bi[1]).length)[0]

    # Ensure every vertex has at least one valid weight
    for v in target.data.vertices:
        world_pt = target.matrix_world @ v.co
        groups = [g.group for g in v.groups]
        if not groups:
            # Assign to closest bone if no group exists
            cb = closest_bone(world_pt)
            target.vertex_groups[cb].add([v.index], 1.0, 'REPLACE')
            continue
        cb = closest_bone(world_pt)
        has_valid = False
        # First pass: set weight to 1.0 for closest bone, 0.0 for others
        for gid in groups:
            name = target.vertex_groups[gid].name
            w = 1.0 if name == cb else 0.0
            target.vertex_groups[gid].add([v.index], w, 'REPLACE')
            if w == 1.0:
                has_valid = True
        # Fallback: assign to closest bone if none remain
        if not has_valid:
            target.vertex_groups[cb].add([v.index], 1.0, 'REPLACE')

    print("[Cleanup] Done.")

    # Final check for vertex groups and weights
    if len(target.vertex_groups) == 0:
        print("[Error] No vertex groups before export. Aborting.")
        return
    weights_exist = any(any(vg.weight > 0 for vg in v.groups) for v in target.data.vertices)
    if not weights_exist:
        print("[Error] No skin weights found before export. Aborting.")
        return
    else:
        print("[Info] Skin weights present before export.")

    # parent & export
    parent_to_armature(target, arm)
    out = os.path.join(output_folder,
                       f"{os.path.splitext(os.path.basename(latest))[0]}_rigged.glb")
    export_glb(out, target, arm)

if __name__ == "__main__":
    main()"""

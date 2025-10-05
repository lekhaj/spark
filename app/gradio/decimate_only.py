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
import time
def preprocess_mesh(obj):
    """
    Preprocess the mesh to reduce complexity and clean geometry.
    This step is modular and can be removed or commented out if not needed.
    """
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    # Merge by distance (remove doubles)
    try:
        bpy.ops.mesh.remove_doubles()
    except Exception as e:
        print(f"[WARN] remove_doubles failed: {e}")
    # Remove loose geometry
    try:
        bpy.ops.mesh.delete_loose()
    except Exception as e:
        print(f"[WARN] delete_loose failed: {e}")
    bpy.ops.object.mode_set(mode='OBJECT')
    
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

def remesh_mesh(obj, voxel_size=0.001):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    remesh = obj.modifiers.new("Remesh", 'REMESH')
    remesh.mode = 'VOXEL'
    remesh.voxel_size = voxel_size
    remesh.use_smooth_shade = True
    bpy.ops.object.modifier_apply(modifier=remesh.name)

def quadify_mesh(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.tris_convert_to_quads()
    bpy.ops.object.mode_set(mode='OBJECT')

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
    # Pack all external files (textures) into the .blend before export
    try:
        bpy.ops.file.pack_all()
    except Exception as e:
        print(f"[WARN] Could not pack all textures: {e}")
    bpy.ops.export_scene.fbx(
        filepath=filepath,
        use_selection=True,
        apply_unit_scale=True,
        apply_scale_options='FBX_SCALE_ALL',
        object_types={'MESH'},
        bake_space_transform=True,
        embed_textures=True
    )

# ---------------- MAIN PROCESSING ----------------

def main():
    # Ensure Cycles render engine is set for baking
    try:
        import bpy
        bpy.context.scene.render.engine = 'CYCLES'
    except Exception as e:
        print(f"[WARN] Could not set render engine to CYCLES: {e}")
    # Setup robust logging
    log_path = None
    def log(msg):
        print(msg)
        if log_path:
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            except Exception as e:
                print(f"[LOGGING ERROR] Could not write to log file: {e}")

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
        # Default to app/gradio/s3_downloads/decimated_output relative to this script
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        output_folder = os.path.join(workspace_root, 's3_downloads', 'decimated_output')
    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, "decimation.log")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"[START] Decimation run for {asset_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"[INFO] Logging to: {log_path}")
    except Exception as e:
        print(f"[LOGGING ERROR] Could not create log file: {e}")
        log_path = None
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
    voxel_size = 0.01  # Increased voxel size to reduce mesh complexity and resource usage
    try:
        for suffix, threshold, mode, param in DECIMATION_PROFILES:
            log(f"[INFO] Processing profile: {suffix}, threshold: {threshold}, mode: {mode}")
            try:
                clear_scene()
            except Exception as e:
                log(f"[ERROR] clear_scene failed: {e}")
                results[suffix] = {"error": f"clear_scene failed: {str(e)}"}
                continue
            try:
                obj = import_model(input_file)
            except Exception as e:
                log(f"[ERROR] import_model failed: {e}")
                results[suffix] = {"error": f"import_model failed: {str(e)}"}
                continue
            if not obj:
                log(f"[ERROR] Import failed for {input_file}")
                results[suffix] = {"error": f"Import failed for {input_file}"}
                continue
            poly_before = len(obj.data.polygons)
            # --- Preprocessing step (modular, can be removed if not needed) ---
            try:
                preprocess_mesh(obj)
            except Exception as e:
                log(f"[WARN] preprocess_mesh failed: {e}")

            # --- Texture baking setup ---
            # Duplicate the original mesh for baking reference
            original_obj = obj
            bpy.ops.object.select_all(action='DESELECT')
            original_obj.select_set(True)
            bpy.context.view_layer.objects.active = original_obj
            bpy.ops.object.duplicate()
            baked_obj = bpy.context.selected_objects[0]
            baked_obj.name = "RemeshedObj"

            # --- Remesh step on duplicate ---
            try:
                remesh_mesh(baked_obj, voxel_size=voxel_size)
            except Exception as e:
                log(f"[ERROR] Remesh failed: {e}")
                results[suffix] = {"error": f"Remesh failed: {str(e)}"}
                bpy.data.objects.remove(baked_obj, do_unlink=True)
                continue

            # --- Create new UVs for remeshed object ---
            try:
                bpy.context.view_layer.objects.active = baked_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.smart_project()
                bpy.ops.object.mode_set(mode='OBJECT')
                log("[INFO] UVs created for remeshed object.")
            except Exception as e:
                log(f"[WARN] Failed to create UVs for remeshed object: {e}")

            # --- Bake texture from original to remeshed object ---
            try:
                # Smart UV Project on source mesh to ensure full UV coverage
                bpy.context.view_layer.objects.active = original_obj
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                bpy.ops.uv.smart_project()
                bpy.ops.object.mode_set(mode='OBJECT')
                log("[INFO] Smart UV Project applied to source mesh.")
                # Assume first material and first image texture node
                orig_mat = original_obj.active_material
                if not orig_mat:
                    raise Exception("No material found on original object.")
                # Find image texture node on source
                img_node = None
                if orig_mat and orig_mat.use_nodes:
                    for n in orig_mat.node_tree.nodes:
                        if n.type == 'TEX_IMAGE':
                            img_node = n
                            break
                if not img_node or not img_node.image:
                    raise Exception("No image texture found on original material.")
                orig_img = img_node.image
                # Log source mesh UV layers
                uv_layers = list(original_obj.data.uv_layers)
                log(f"[DEBUG] Source mesh UV layers: {[uv.name for uv in uv_layers]}")
                if not uv_layers:
                    log("[WARN] Source mesh has no UV layers!")
                else:
                    log(f"[DEBUG] Source mesh active UV: {original_obj.data.uv_layers.active.name}")
                # Log source material node connections
                if orig_mat.use_nodes:
                    for node in orig_mat.node_tree.nodes:
                        if node.type == 'BSDF_PRINCIPLED':
                            for input in node.inputs:
                                if input.name == 'Base Color':
                                    for link in orig_mat.node_tree.links:
                                        if link.to_socket == input:
                                            log(f"[DEBUG] Base Color input is linked from: {link.from_node.name}")
                # Auto-connect image node to Base Color if not already
                principled = None
                for node in orig_mat.node_tree.nodes:
                    if node.type == 'BSDF_PRINCIPLED':
                        principled = node
                        break
                if principled:
                    base_color_input = principled.inputs['Base Color']
                    already_linked = any(
                        link.to_socket == base_color_input and link.from_node == img_node
                        for link in orig_mat.node_tree.links
                    )
                    if not already_linked:
                        orig_mat.node_tree.links.new(img_node.outputs['Color'], base_color_input)
                        log("[DEBUG] Connected image node to Base Color input.")
                # Set source image node as active and selected
                for n in orig_mat.node_tree.nodes:
                    n.select = False
                img_node.select = True
                orig_mat.node_tree.nodes.active = img_node
                log(f"[DEBUG] Source image node set active: {img_node.name}, image: {orig_img.name}")
                # Create new image for baking
                bake_img = bpy.data.images.new("BakedTexture", width=orig_img.size[0], height=orig_img.size[1])
                # Create new material for baked_obj
                baked_mat = bpy.data.materials.new(name="BakedMaterial")
                baked_mat.use_nodes = True
                nodes = baked_mat.node_tree.nodes
                links = baked_mat.node_tree.links
                nodes.clear()
                out_node = nodes.new(type='ShaderNodeOutputMaterial')
                diff_node = nodes.new(type='ShaderNodeBsdfPrincipled')
                tex_node = nodes.new(type='ShaderNodeTexImage')
                tex_node.image = bake_img
                tex_node.select = True  # Ensure node is selected
                nodes.active = tex_node  # Set as active node
                links.new(diff_node.outputs['BSDF'], out_node.inputs['Surface'])
                links.new(tex_node.outputs['Color'], diff_node.inputs['Base Color'])
                baked_obj.data.materials.clear()
                baked_obj.data.materials.append(baked_mat)
                # Ensure UV map exists and is active
                if not baked_obj.data.uv_layers:
                    bpy.context.view_layer.objects.active = baked_obj
                    bpy.ops.object.mode_set(mode='EDIT')
                    bpy.ops.mesh.select_all(action='SELECT')
                    bpy.ops.uv.smart_project()
                    bpy.ops.object.mode_set(mode='OBJECT')
                baked_obj.data.uv_layers.active_index = 0
                # Set up bake: select source, make target active
                bpy.ops.object.select_all(action='DESELECT')
                original_obj.select_set(True)
                baked_obj.select_set(True)
                bpy.context.view_layer.objects.active = baked_obj
                # Set bake image node as active and selected on target
                for n in baked_mat.node_tree.nodes:
                    n.select = False
                tex_node.select = True
                baked_mat.node_tree.nodes.active = tex_node
                log(f"[DEBUG] Target bake image node set active: {tex_node.name}, image: {bake_img.name}")
                # Bake using EMIT as a test
                try:
                    bpy.ops.object.bake(type='EMIT', use_selected_to_active=True, margin=2)
                    log("[INFO] Texture baked from original to remeshed object using EMIT.")
                except Exception as e:
                    log(f"[WARN] EMIT bake failed: {e}, trying DIFFUSE.")
                    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, use_selected_to_active=True, margin=2)
                    log("[INFO] Texture baked from original to remeshed object using DIFFUSE.")
                # Save the baked image to disk
                baked_img_path = os.path.join(output_folder, f"{asset_name}_{suffix}_baked.png")
                bake_img.filepath_raw = baked_img_path
                bake_img.file_format = 'PNG'
                bake_img.save()
                log(f"[INFO] Baked image saved to {baked_img_path}")
                # Optionally pack the image
                try:
                    bake_img.pack()
                    log("[INFO] Baked image packed into blend file.")
                except Exception as e:
                    log(f"[WARN] Could not pack baked image: {e}")
                # Ensure the baked material references the saved image and is active
                tex_node.image = bake_img
                # Set the image texture node as active for export
                try:
                    baked_mat.node_tree.nodes.active = tex_node
                    log("[INFO] Set baked image node as active for export.")
                except Exception as e:
                    log(f"[WARN] Could not set baked image node as active: {e}")
                # Double-check assignment
                if not tex_node.image or tex_node.image != bake_img:
                    log(f"[WARN] Baked image not assigned to texture node before export!")
            except Exception as e:
                log(f"[WARN] Texture baking failed: {e}")

            # --- Continue pipeline with baked_obj ---
            obj = baked_obj

            # --- Decimate step ---
            try:
                decimate_mesh(obj, threshold, mode, param)
            except Exception as e:
                log(f"[ERROR] Decimate failed: {e}")
                results[suffix] = {"error": f"Decimate failed: {str(e)}"}
                bpy.data.objects.remove(obj, do_unlink=True)
                continue
            # --- Quadify step ---
            try:
                quadify_mesh(obj)
            except Exception as e:
                log(f"[WARN] Quadify failed: {e}")
            try:
                poly_after = len(obj.data.polygons)
            except Exception as e:
                log(f"[WARN] Could not get poly_after: {e}")
                poly_after = None
            try:
                set_origin_to_bottom_face_cursor(obj)
            except Exception as e:
                log(f"[WARN] set_origin_to_bottom_face_cursor failed: {e}")
            out_name = f"{asset_name}_{suffix}_decimated.fbx"
            local_fbx = os.path.join(output_folder, out_name)
            try:
                export_fbx(local_fbx, obj)
            except Exception as e:
                log(f"[ERROR] export_fbx failed: {e}")
                results[suffix] = {"error": f"export_fbx failed: {str(e)}"}
                bpy.data.objects.remove(obj, do_unlink=True)
                continue
            results[suffix] = {
                "local_file": local_fbx,
                "poly_before": poly_before,
                "poly_after": poly_after,
                "reduction_ratio": round((poly_before - poly_after) / poly_before, 3) if poly_before and poly_after else None
            }
            # Clean up: remove the original object from the scene
            try:
                bpy.data.objects.remove(original_obj, do_unlink=True)
            except Exception as e:
                log(f"[WARN] Could not remove original object: {e}")
    except KeyboardInterrupt:
        log("[INTERRUPT] Script interrupted by user.")
        results['interrupted'] = {"error": "Script interrupted by user."}
    except Exception as e:
        log(f"[FATAL] Unhandled exception: {e}")
        import traceback
        log(traceback.format_exc())
        results['fatal'] = {"error": f"Unhandled exception: {str(e)}"}
    print(json.dumps(results))

if __name__ == "__main__":
    main()

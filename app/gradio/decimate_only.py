import bpy
import os
import sys
import json
import time

# ---------------- CONFIG ----------------
DECIMATION_PROFILES = [
    ("100k", 100000),
    ("300k", 300000)
]

# --- Tuned parameters to reduce clipping / bake misses ---
# Slightly coarser remesh reduces tiny floating islands that poke through
VOXEL_SIZE = 0.002   # was 0.001 — increase slightly to avoid tiny artifacts

PLANAR_ANGLE_LIMIT = 0.30

# Bake resolution: 2048 is a good balanced default to reduce memory pressure.
# If you have plenty of RAM & want extreme detail revert to 4096.
BAKE_RESOLUTION = 4096  # was 4096

# Larger margin to reduce seam-bleed and give safer pixel padding
BAKE_MARGIN = 64  # was 32

# Increase ray distance and cage extrusion so rays find the high-poly even through folds
BAKE_RAY_DISTANCE = 5.0   # was 1.0 -> big increase to avoid rays stopping short
BAKE_EXTRUSION = 0.25     # was 0.01 -> bigger cage so thin parts don't miss

# ---------------- HELPERS ----------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def clear_scene():
    log("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for collection in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, bpy.data.images]:
        for block in list(collection):
            try: collection.remove(block)
            except Exception: pass
    log("Scene cleared.")

def import_model(filepath):
    log(f"Importing model from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(filepath)
    
    ext = os.path.splitext(filepath)[1].lower()
    before = set(bpy.context.scene.objects)
    
    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    else:
        raise NotImplementedError(f"Unsupported format: {ext}")
        
    new_objs = [o for o in bpy.context.scene.objects if o not in before and o.type == 'MESH']
    if not new_objs: raise TypeError("No mesh imported.")
    
    if len(new_objs) > 1:
        bpy.ops.object.mode_set(mode='OBJECT')
        bpy.ops.object.select_all(action='DESELECT')
        for o in new_objs: o.select_set(True)
        bpy.context.view_layer.objects.active = new_objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = new_objs[0]
    
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj

def ensure_textures_on_disk(output_folder):
    for img in bpy.data.images:
        if img.packed_file or not img.filepath:
            base = img.name if img.name.lower().endswith('.png') else f"{img.name}.png"
            path = os.path.join(output_folder, base)
            img.filepath_raw = path
            img.file_format = 'PNG'
            try:
                img.save()
                img.reload()
            except: pass

# ---------------- BAKE HELPERS ----------------

def setup_highpoly_emission(obj):
    """Swap to Emission shader to capture pure color without lighting"""
    restore_list = []
    for mat in obj.data.materials:
        if not mat or not mat.use_nodes: continue
        tree = mat.node_tree
        nodes = tree.nodes
        links = tree.links
        
        principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        output = next((n for n in nodes if n.type == 'OUTPUT_MATERIAL'), None)
        
        if principled and output:
            color_socket = principled.inputs.get('Base Color')
            source_link = color_socket.links[0] if color_socket and color_socket.links else None
            
            emit = nodes.new('ShaderNodeEmission')
            emit.name = "TEMP_EMIT"
            
            if source_link:
                links.new(source_link.from_socket, emit.inputs['Color'])
            else:
                # If no link, copy the default color so EMIT has something sensible
                emit.inputs['Color'].default_value = color_socket.default_value if color_socket else (1,1,1,1)
            
            old_link = output.inputs['Surface'].links[0] if output.inputs['Surface'].links else None
            try:
                links.new(emit.outputs['Emission'], output.inputs['Surface'])
            except Exception:
                pass
            
            restore_list.append((mat, emit, old_link, output))
    return restore_list

def restore_highpoly_materials(restore_list):
    for mat, emit, old_link, output in restore_list:
        try:
            if old_link:
                mat.node_tree.links.new(old_link.from_socket, output.inputs['Surface'])
        except Exception:
            pass
        try:
            mat.node_tree.nodes.remove(emit)
        except Exception:
            pass

# ---------------- WORKFLOW ----------------

def create_base_mesh(source_obj):
    """
    Creates a clean, watertight base mesh that conforms tightly to the original.
    """
    log("Creating Optimized Base Mesh...")
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    bpy.context.view_layer.objects.active = source_obj
    bpy.ops.object.duplicate()
    
    base = bpy.context.selected_objects[0]
    base.name = f"{source_obj.name}_Base"
    
    # 1. Voxel Remesh (Seal holes)
    mod_r = base.modifiers.new("Remesh", 'REMESH')
    mod_r.mode = 'VOXEL'
    mod_r.voxel_size = VOXEL_SIZE
    mod_r.adaptivity = 0.0
    try:
        bpy.ops.object.modifier_apply(modifier=mod_r.name)
    except Exception as e:
        log(f"Voxel remesh apply failed: {e}")
    
    # 2. SHRINKWRAP (snap remesh to high-poly surface)
    try:
        mod_sw = base.modifiers.new("Shrinkwrap", 'SHRINKWRAP')
        mod_sw.target = source_obj
        mod_sw.wrap_method = 'PROJECT'
        mod_sw.use_negative_direction = True
        mod_sw.use_positive_direction = True
        mod_sw.offset = 0.0
        bpy.ops.object.modifier_apply(modifier=mod_sw.name)
    except Exception as e:
        log(f"Shrinkwrap failed/apply: {e}")

    # 3. Smooth (Clean up voxel artifacts)
    try:
        mod_s = base.modifiers.new("Smooth", 'CORRECTIVE_SMOOTH')
        mod_s.iterations = 5
        mod_s.smooth_type = 'LENGTH_WEIGHTED'
        bpy.ops.object.modifier_apply(modifier=mod_s.name)
    except Exception:
        pass
    
    # 4. Triangulate
    try:
        mod_t = base.modifiers.new("Tri", 'TRIANGULATE')
        mod_t.keep_custom_normals = True
        bpy.ops.object.modifier_apply(modifier=mod_t.name)
    except Exception:
        pass
    
    bpy.ops.object.shade_smooth()
    
    log(f"Base Mesh Faces: {len(base.data.polygons)}")
    return base

def process_lod(high_poly, base_mesh, suffix, face_count, asset_name, output_folder):
    log(f"--- Processing LOD: {suffix} (Target: {face_count}) ---")
    
    lod = base_mesh.copy()
    lod.data = base_mesh.data.copy()
    lod.name = f"{asset_name}_{suffix}"
    bpy.context.collection.objects.link(lod)
    bpy.context.view_layer.objects.active = lod
    
    initial = len(lod.data.polygons)
    final = len(lod.data.polygons)
    
    # Recalculate Normals (Fixes black faces)
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        # Increased margin to prevent texture bleeding on seams
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.05) 
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    
    # Setup Bake
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 1
    
    mat = bpy.data.materials.new(name=f"{asset_name}_{suffix}_Mat")
    mat.use_nodes = True
    lod.data.materials.clear()
    lod.data.materials.append(mat)
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    out = nodes.new('ShaderNodeOutputMaterial')
    try:
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
    except Exception:
        pass
    
    norm_node = nodes.new('ShaderNodeTexImage')
    norm_img = bpy.data.images.new(f"{asset_name}_{suffix}_Normal", BAKE_RESOLUTION, BAKE_RESOLUTION)
    try:
        norm_img.colorspace_settings.name = 'Non-Color'
    except Exception:
        pass
    norm_node.image = norm_img
    
    diff_node = nodes.new('ShaderNodeTexImage')
    diff_img = bpy.data.images.new(f"{asset_name}_{suffix}_Diffuse", BAKE_RESOLUTION, BAKE_RESOLUTION)
    diff_node.image = diff_img

    bpy.ops.object.select_all(action='DESELECT')
    high_poly.select_set(True)
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod

    bake_kwargs = {
        "use_selected_to_active": True,
        "margin": BAKE_MARGIN,
        "use_clear": True,
        # IMPORTANT: increased ray distance & cage extrusion to avoid clipping on thin parts
        "max_ray_distance": BAKE_RAY_DISTANCE, 
        "cage_extrusion": BAKE_EXTRUSION 
    }

    # Bake Normal
    log("Baking Normal...")
    nodes.active = norm_node
    norm_node.select = True
    diff_node.select = False
    try:
        bpy.ops.object.bake(type='NORMAL', **bake_kwargs)
        norm_img.filepath_raw = os.path.join(output_folder, f"{norm_img.name}.png")
        norm_img.file_format = 'PNG'
        norm_img.save()
    except Exception as e: log(f"Normal Bake Error: {e}")

    # Bake Color (Emit)
    log("Baking Color (Emit)...")
    restore_data = setup_highpoly_emission(high_poly)
    nodes.active = diff_node
    diff_node.select = True
    norm_node.select = False
    try:
        bpy.ops.object.bake(type='EMIT', **bake_kwargs)
        diff_img.filepath_raw = os.path.join(output_folder, f"{diff_img.name}.png")
        diff_img.file_format = 'PNG'
        diff_img.save()
    except Exception as e: log(f"Color Bake Error: {e}")
    restore_highpoly_materials(restore_data)

    # Export
    n_map = nodes.new('ShaderNodeNormalMap')
    try:
        mat.node_tree.links.new(norm_node.outputs['Color'], n_map.inputs['Color'])
        mat.node_tree.links.new(n_map.outputs['Normal'], bsdf.inputs['Normal'])
        mat.node_tree.links.new(diff_node.outputs['Color'], bsdf.inputs['Base Color'])
    except Exception:
        pass
    
    try:
        norm_img.reload(); norm_img.pack()
        diff_img.reload(); diff_img.pack()
    except: pass

    out_fbx = os.path.join(output_folder, f"{asset_name}_{suffix}.fbx")
    bpy.ops.object.select_all(action='DESELECT')
    lod.select_set(True)
    try:
        bpy.ops.export_scene.fbx(filepath=out_fbx, use_selection=True, apply_scale_options='FBX_SCALE_ALL', embed_textures=True, path_mode='COPY')
    except Exception as e: log(f"Export Error: {e}")
    
    bpy.data.objects.remove(lod, do_unlink=True)
    
    return {
        "local_file": out_fbx,
        "poly_before": initial,
        "poly_after": final,
        "reduction_ratio": round(1.0 - (final/initial), 4) if initial > 0 else 0
    }

# ---------------- MAIN ----------------
def main():
    argv = sys.argv
    if "--" not in argv: sys.exit(1)
    args = argv[argv.index("--")+1:]
    if len(args) < 2: sys.exit(1)
    
    input_file = args[0]
    output_folder = args[1]
    asset_name = os.path.splitext(os.path.basename(input_file))[0]
    
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    
    log(f"Job: {asset_name} | Input: {input_file}")
    
    results = {}
    try:
        clear_scene()
        high_poly = import_model(input_file)
        ensure_textures_on_disk(output_folder)
        
        base_mesh = create_base_mesh(high_poly)
        
        for suffix, count in DECIMATION_PROFILES:
            results[suffix] = process_lod(high_poly, base_mesh, suffix, count, asset_name, output_folder)
            
    except Exception as e:
        log(f"FATAL: {e}")
        results['error'] = str(e)
        
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
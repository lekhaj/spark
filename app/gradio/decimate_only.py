import bpy
import os
import sys
import json
import time

# ---------------- CONFIGURATION PARAMETERS ----------------

# --- Quality & Stability ---
# Controls the poly count of the master bake mesh. Smaller values are higher quality
# but use more memory. Aim for a base mesh higher than your highest LOD target.
VOXEL_SIZE = 0.015 

# Detail level for baking. 3 is stable, 4 is high quality.
MULTIRES_LEVELS = 4 

# --- Baking & Seam Fix ---
BAKE_RESOLUTION = 2048
BAKE_MARGIN = 16
CAGE_EXTRUSION = 0.1
MAX_RAY_DISTANCE = 0.1

# --- Decimation Profiles ---
# IMPORTANT: List from HIGHEST poly count to LOWEST for progressive decimation.
DECIMATION_PROFILES = [
    ("10k", 10000),
    ("8k", 8000),
    ("5k", 5000),
]

# ---------------- BLENDER OPERATIONS ----------------

def log(message):
    """Prints a message to the console (standard output) with a timestamp."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}")

def clear_scene():
    log("INFO: Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in bpy.data.meshes: bpy.data.meshes.remove(block)
    for block in bpy.data.materials: bpy.data.materials.remove(block)
    for block in bpy.data.textures: bpy.data.textures.remove(block)
    for block in bpy.data.images: bpy.data.images.remove(block)
    log("INFO: Scene cleared.")

def import_model(filepath):
    log(f"INFO: Importing model from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.glb', '.gltf']:
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.obj':
        bpy.ops.import_scene.obj(filepath=filepath)
    else:
        raise NotImplementedError(f"Unsupported file type: {ext}")
    
    for obj in bpy.context.selected_objects:
        if obj.type == 'MESH':
            bpy.context.view_layer.objects.active = obj
            log(f"INFO: Imported mesh object: {obj.name} with {len(obj.data.polygons)} polygons.")
            return obj
    raise TypeError("No mesh object found in the imported file.")

def preprocess_mesh(obj):
    log(f"INFO: Preprocessing mesh: {obj.name}")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=0.001)
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode='OBJECT')
    log("INFO: Preprocessing complete.")

def create_retopo_mesh(source_obj, voxel_size, multires_levels):
    log("INFO: Creating projected master LOD for baking...")
    
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    bpy.context.view_layer.objects.active = source_obj
    bpy.ops.object.duplicate()
    lowpoly_obj = bpy.context.selected_objects[0]
    lowpoly_obj.name = f"{source_obj.name}_Retopo"
    
    log(f"INFO: Applying Voxel Remesh with size: {voxel_size} to create a base shell...")
    remesh_mod = lowpoly_obj.modifiers.new(name="VoxelRemesh", type='REMESH')
    remesh_mod.mode = 'VOXEL'
    remesh_mod.voxel_size = voxel_size
    bpy.ops.object.modifier_apply(modifier=remesh_mod.name)
    bpy.ops.object.shade_smooth()
    log(f"INFO: Voxel Remesh complete. Base mesh has {len(lowpoly_obj.data.polygons)} polygons.")
    
    log("INFO: Setting up Multiresolution and Shrinkwrap modifiers for detail projection...")
    multires_mod = lowpoly_obj.modifiers.new(name="Multires", type='MULTIRES')
    shrinkwrap_mod = lowpoly_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    shrinkwrap_mod.target = source_obj
    shrinkwrap_mod.wrap_method = 'PROJECT'
    
    log(f"INFO: Subdividing Multires {multires_levels} times to project details...")
    for i in range(multires_levels):
        bpy.ops.object.multires_subdivide(modifier=multires_mod.name)
        log(f"  ...subdivision level {i+1}")
        
    log("INFO: Applying Shrinkwrap modifier to capture details...")
    bpy.ops.object.modifier_apply(modifier=shrinkwrap_mod.name)
    
    log("INFO: Master LOD projection complete.")
    return lowpoly_obj

def bake_maps(source_obj, target_obj, asset_name, output_folder, resolution, bake_margin, cage_extrusion, max_ray_distance):
    """Bakes Diffuse, Normal, Roughness, and AO maps with improved seam handling."""
    log("--- Starting Full PBR Texture Baking Process ---")
    
    log("INFO: Setting up renderer for CPU baking...")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 16
    
    log("INFO: Generating Smart UVs for the low-poly mesh...")
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode='OBJECT')
    
    log(f"INFO: Creating new material and {resolution}x{resolution} images...")
    baked_material = bpy.data.materials.new(name=f"{asset_name}_Baked_Material")
    baked_material.use_nodes = True
    nodes = baked_material.node_tree.nodes
    links = baked_material.node_tree.links
    nodes.clear()
    
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf_node = nodes.new(type='ShaderNodeBsdfPrincipled')
    
    diffuse_tex_node = nodes.new(type='ShaderNodeTexImage')
    normal_tex_node = nodes.new(type='ShaderNodeTexImage')
    roughness_tex_node = nodes.new(type='ShaderNodeTexImage')
    ao_tex_node = nodes.new(type='ShaderNodeTexImage')
    normal_map_node = nodes.new(type='ShaderNodeNormalMap')
    
    diffuse_image = bpy.data.images.new(f"{asset_name}_D", width=resolution, height=resolution)
    normal_image = bpy.data.images.new(f"{asset_name}_N", width=resolution, height=resolution, is_data=True)
    roughness_image = bpy.data.images.new(f"{asset_name}_R", width=resolution, height=resolution, is_data=True)
    ao_image = bpy.data.images.new(f"{asset_name}_AO", width=resolution, height=resolution, is_data=True)

    for img in [normal_image, roughness_image, ao_image]:
        img.colorspace_settings.name = 'Non-Color'

    diffuse_tex_node.image = diffuse_image
    normal_tex_node.image = normal_image
    roughness_tex_node.image = roughness_image
    ao_tex_node.image = ao_image

    links.new(bsdf_node.outputs['BSDF'], output_node.inputs['Surface'])
    links.new(diffuse_tex_node.outputs['Color'], bsdf_node.inputs['Base Color'])
    links.new(roughness_tex_node.outputs['Color'], bsdf_node.inputs['Roughness'])
    links.new(normal_tex_node.outputs['Color'], normal_map_node.inputs['Color'])
    links.new(normal_map_node.outputs['Normal'], bsdf_node.inputs['Normal'])
    
    target_obj.data.materials.clear()
    target_obj.data.materials.append(baked_material)
    
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    
    bake_kwargs = {
        "use_selected_to_active": True,
        "max_ray_distance": max_ray_distance,
        "cage_extrusion": cage_extrusion,
        "margin": bake_margin
    }

    log("INFO: Baking Normal map...")
    nodes.active = normal_tex_node
    bpy.ops.object.bake(type='NORMAL', **bake_kwargs)
    normal_image.filepath_raw = os.path.join(output_folder, f"{asset_name}_Normal.png")
    normal_image.save()
    log(f"  ...Normal map saved to {normal_image.filepath_raw}")
    
    log("INFO: Baking Roughness map...")
    nodes.active = roughness_tex_node
    bpy.ops.object.bake(type='ROUGHNESS', **bake_kwargs)
    roughness_image.filepath_raw = os.path.join(output_folder, f"{asset_name}_Roughness.png")
    roughness_image.save()
    log(f"  ...Roughness map saved to {roughness_image.filepath_raw}")

    log("INFO: Baking Ambient Occlusion map...")
    nodes.active = ao_tex_node
    bpy.ops.object.bake(type='AO', **bake_kwargs)
    ao_image.filepath_raw = os.path.join(output_folder, f"{asset_name}_AO.png")
    ao_image.save()
    log(f"  ...AO map saved to {ao_image.filepath_raw}")

    log("INFO: Baking Diffuse map...")
    nodes.active = diffuse_tex_node
    bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'}, **bake_kwargs)
    diffuse_image.filepath_raw = os.path.join(output_folder, f"{asset_name}_Diffuse.png")
    diffuse_image.save()
    log(f"  ...Diffuse map saved to {diffuse_image.filepath_raw}")
    
    log("--- Texture Baking Complete ---")
    return baked_material

# ---------------- MAIN PROCESSING ----------------

def main():
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--") + 2:
        print("Usage: blender --background --python your_script.py -- <asset_name> <input_file> [<output_folder>]")
        sys.exit(1)
        
    args = argv[argv.index("--") + 1:]

    input_file = args[1] if len(args) > 1 else None
    if not input_file:
        log("ERROR: Input file is required.")
        sys.exit(1)

    if len(args) > 0 and args[0]:
        asset_name = args[0]
    else:
        asset_name = os.path.splitext(os.path.basename(input_file))[0]

    if len(args) > 2 and args[2]:
        output_folder = args[2]
    else:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        output_folder = os.path.join(workspace_root, 's3_downloads', 'decimated_output')
    
    os.makedirs(output_folder, exist_ok=True)
    
    results = {}
    
    try:
        log(f"--- Starting Decimation Process for Asset: {asset_name} ---")
        log(f"  Input: {input_file}")
        log(f"  Output: {output_folder}")
        
        clear_scene()
        
        source_obj = import_model(input_file)
        
        preprocess_mesh(source_obj)
        
        master_lod = create_retopo_mesh(source_obj, VOXEL_SIZE, MULTIRES_LEVELS)
        
        baked_material = bake_maps(source_obj, master_lod, asset_name, output_folder, BAKE_RESOLUTION, BAKE_MARGIN, CAGE_EXTRUSION, MAX_RAY_DISTANCE)
        master_lod.data.materials.clear()
        master_lod.data.materials.append(baked_material)
        
        log("INFO: Cleaning up high-poly source object...")
        bpy.data.objects.remove(source_obj, do_unlink=True)
        
        log("INFO: Applying multires modifier at base level to create final master LOD...")
        bpy.context.view_layer.objects.active = master_lod
        if "Multires" in master_lod.modifiers:
            mod = master_lod.modifiers["Multires"]
            mod.levels = 0
            mod.sculpt_levels = 0
            mod.render_levels = 0
            bpy.ops.object.modifier_apply(modifier=mod.name)

        log("--- Starting Progressive Decimation ---")
        
        current_lod = master_lod.copy()
        current_lod.data = master_lod.data.copy()
        bpy.context.collection.objects.link(current_lod)

        for suffix, face_count in DECIMATION_PROFILES:
            log(f"\n--- Processing Profile: {suffix} (Target: {face_count} faces) ---")
            
            bpy.context.view_layer.objects.active = current_lod
            poly_before = len(current_lod.data.polygons)
            
            if poly_before > face_count:
                log(f"INFO: Decimating from {poly_before} to {face_count} faces...")
                dec_mod = current_lod.modifiers.new(name="Decimate", type='DECIMATE')
                dec_mod.decimate_type = 'COLLAPSE'
                dec_mod.ratio = face_count / poly_before
                bpy.ops.object.modifier_apply(modifier=dec_mod.name)
            else:
                log(f"WARN: Skipping decimation, face count ({poly_before}) is already below target ({face_count}).")
            
            poly_after = len(current_lod.data.polygons)
            log(f"INFO: Final polygon count for {suffix}: {poly_after}")
            
            filepath = os.path.join(output_folder, f"{asset_name}_{suffix}.fbx")
            log(f"INFO: Exporting FBX to: {filepath}")
            bpy.ops.object.select_all(action='DESELECT')
            current_lod.select_set(True)
            bpy.ops.export_scene.fbx(
                filepath=filepath,
                use_selection=True,
                apply_scale_options='FBX_SCALE_ALL',
                object_types={'MESH'},
                embed_textures=True,
                path_mode='COPY'
            )
            
            reduction = (poly_before - poly_after) / poly_before if poly_before > 0 and (poly_before - poly_after) > 0 else 0
            results[suffix] = {
                "output_file": filepath,
                "poly_before": poly_before,
                "poly_after": poly_after,
                "reduction_ratio": round(reduction, 3)
            }
        
        log("INFO: Cleaning up temporary LOD objects...")
        bpy.data.objects.remove(master_lod, do_unlink=True)
        bpy.data.objects.remove(current_lod, do_unlink=True)
            
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        import traceback
        # Also log the full traceback to the console
        log(traceback.format_exc())
        results['error'] = str(e)
            
    finally:
        # This block ensures the final JSON is printed, even on error.
        print("\n--- Final Results ---")
        print(json.dumps(results, indent=2))
        log("--- Script Finished ---")

if __name__ == "__main__":
    main()
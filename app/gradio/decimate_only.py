# blender_bake_decimate_final_with_finalize.py
# Final: unpack packed GLB images, EMIT bake for color, ensure baked images are file-backed and re-assigned
# Voxel size 0.02, decimation profiles: 5k,8k,6k,10k

import bpy, os, sys, json, time

# ---------------- CONFIG ----------------
VOXEL_SIZE = 0.02
MULTIRES_LEVELS = 4

BAKE_RESOLUTION = 2048
BAKE_MARGIN = 32
CAGE_EXTRUSION = 0.1
MAX_RAY_DISTANCE = 0.1

DECIMATION_PROFILES = [
    # ("5k", 5000),
    ("20k", 20000),
    ("40k", 40000),
    ("100k", 100000),
]

# ---------------- HELPERS ----------------
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

def clear_scene():
    log("Clearing scene...")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for blk in list(bpy.data.meshes):
        try: bpy.data.meshes.remove(blk)
        except Exception: pass
    for blk in list(bpy.data.materials):
        try: bpy.data.materials.remove(blk)
        except Exception: pass
    for blk in list(bpy.data.textures):
        try: bpy.data.textures.remove(blk)
        except Exception: pass
    for blk in list(bpy.data.images):
        try: bpy.data.images.remove(blk)
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
        raise NotImplementedError(ext)
    new_objs = [o for o in bpy.context.scene.objects if o not in before and o.type == 'MESH']
    if not new_objs:
        new_objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not new_objs:
        raise TypeError("No mesh imported.")
    if len(new_objs) > 1:
        log(f"Joining {len(new_objs)} imported meshes into one")
        bpy.ops.object.mode_set(mode='OBJECT')
        for o in new_objs: o.select_set(True)
        bpy.context.view_layer.objects.active = new_objs[0]
        bpy.ops.object.join()
        joined = bpy.context.view_layer.objects.active
        log(f"Joined mesh: {joined.name}")
        return joined
    obj = new_objs[0]
    bpy.context.view_layer.objects.active = obj
    log(f"Imported mesh object: {obj.name}")
    return obj

# Write any packed images (or images without a filepath) to disk so Cycles can access them
def ensure_textures_unpacked_to_disk(output_folder):
    log("Writing packed/unfiled images to disk...")
    written = []
    for img in list(bpy.data.images):
        try:
            is_packed = bool(getattr(img, "packed_file", None))
            has_fp = bool(img.filepath)
            if is_packed or not has_fp:
                # create an on-disk name
                base = img.name
                if not base.lower().endswith('.png'):
                    base = base + ".png"
                out = os.path.join(output_folder, base)
                img.filepath_raw = out
                img.file_format = 'PNG'
                try:
                    img.save()
                    img.reload()
                    written.append(out)
                    log(f"Wrote image to disk: {out}")
                except Exception as e:
                    log(f"Failed writing {img.name}: {e}")
        except Exception as e:
            log(f"Error handling image {getattr(img,'name',str(img))}: {e}")
    if not written:
        log("No packed/unfiled images needed writing.")
    return written

def preprocess_mesh(obj):
    log(f"Preprocessing mesh: {obj.name}")
    bpy.context.view_layer.objects.active = obj
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.remove_doubles(threshold=0.001)
        bpy.ops.mesh.delete_loose()
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    try:
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    except Exception:
        pass
    log("Preprocess complete.")

def create_retopo_mesh(source_obj, voxel_size, multires_levels):
    log("Creating retopo mesh via voxel remesh...")
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    bpy.context.view_layer.objects.active = source_obj
    bpy.ops.object.duplicate()
    low = bpy.context.selected_objects[0]
    low.name = f"{source_obj.name}_Retopo"
    rem = low.modifiers.new("VoxelRemesh","REMESH")
    rem.mode = 'VOXEL'
    rem.voxel_size = voxel_size
    bpy.ops.object.modifier_apply(modifier=rem.name)
    bpy.ops.object.shade_smooth()
    # Add multires & shrinkwrap
    m = low.modifiers.new("Multires","MULTIRES")
    s = low.modifiers.new("Shrinkwrap","SHRINKWRAP")
    s.target = source_obj
    s.wrap_method = 'PROJECT'
    for _ in range(multires_levels):
        try:
            bpy.ops.object.multires_subdivide(modifier=m.name)
        except Exception:
            break
    try:
        bpy.ops.object.modifier_apply(modifier=s.name)
    except Exception:
        pass
    # recalc normals
    bpy.context.view_layer.objects.active = low
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass
    log("Retopo complete.")
    return low

# Save image to disk, reload, pack and return outpath
def save_and_pack_image(img, filename, out_folder):
    outpath = os.path.join(out_folder, filename)
    img.filepath_raw = outpath
    img.file_format = 'PNG'
    try:
        img.save()
    except Exception as e:
        log(f"Warning saving {img.name}: {e}")
    try:
        img.reload()
    except Exception:
        pass
    try:
        img.pack()
    except Exception:
        pass
    log(f"Saved and packed image: {outpath}")
    return outpath

# Ensure the baked material's texture node points to the disk-backed image (avoid internal-only images)
def finalize_baked_image(baked_material, tex_node, outpath):
    try:
        # load image from disk (if not already present)
        if not os.path.exists(outpath):
            log(f"Warning: baked outpath does not exist: {outpath}")
            return
        # If the tex_node has an image, set its filepath to outpath and reload
        img = tex_node.image
        if img is None:
            # load new image data-block and assign
            newimg = bpy.data.images.load(outpath)
            tex_node.image = newimg
            img = newimg
        else:
            try:
                img.filepath = outpath
                img.filepath_raw = outpath
                img.reload()
            except Exception:
                # try forcing a fresh load
                try:
                    newimg = bpy.data.images.load(outpath)
                    tex_node.image = newimg
                    img = newimg
                except Exception:
                    pass
        # pack so FBX exporter can embed
        try:
            img.pack()
        except Exception:
            pass
        log(f"Finalized baked image assigned to node '{tex_node.name}' -> {outpath}")
    except Exception as e:
        log(f"finalize_baked_image error: {e}")

# Link image->Principled Base Color if missing (best-effort)
def ensure_textures_linked_to_basecolor(source_obj):
    log("Ensuring Image Texture nodes are linked to Principled Base Color if missing...")
    for mat in list(getattr(source_obj.data, "materials", [])):
        if not mat or not mat.use_nodes:
            continue
        nt = mat.node_tree
        nodes = nt.nodes
        links = nt.links
        principled = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if principled:
            # find any image node
            imgnode = next((n for n in nodes if n.type == 'TEX_IMAGE' and getattr(n,'image',None)), None)
            if imgnode:
                already = any(l.to_node==principled and l.to_socket.name=='Base Color' for l in links)
                if not already:
                    try:
                        links.new(imgnode.outputs['Color'], principled.inputs['Base Color'])
                        log(f"Linked {imgnode.name} -> {principled.name} Base Color")
                    except Exception:
                        pass

# Bake routine with finalize steps
def bake_maps(source_obj, target_obj, asset_name, output_folder, resolution, bake_margin, cage_extrusion, max_ray_distance):
    log("Starting baking process...")
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 16

    # apply transforms
    for o in (source_obj, target_obj):
        try:
            bpy.ops.object.select_all(action='DESELECT')
            o.select_set(True)
            bpy.context.view_layer.objects.active = o
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        except Exception:
            pass

    # UVs
    bpy.context.view_layer.objects.active = target_obj
    try:
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='SELECT')
        bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
        bpy.ops.object.mode_set(mode='OBJECT')
    except Exception:
        pass

    # create bake material
    baked_mat = bpy.data.materials.new(f"{asset_name}_Baked_Material")
    baked_mat.use_nodes = True
    nodes = baked_mat.node_tree.nodes
    links = baked_mat.node_tree.links
    nodes.clear()
    out_node = nodes.new(type='ShaderNodeOutputMaterial')
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')

    diffuse_node = nodes.new(type='ShaderNodeTexImage')
    normal_node = nodes.new(type='ShaderNodeTexImage')
    rough_node = nodes.new(type='ShaderNodeTexImage')
    ao_node = nodes.new(type='ShaderNodeTexImage')
    normal_map_node = nodes.new(type='ShaderNodeNormalMap')
    normal_map_node.space = 'TANGENT'

    diffuse_img = bpy.data.images.new(f"{asset_name}_D", BAKE_RESOLUTION, BAKE_RESOLUTION)
    normal_img  = bpy.data.images.new(f"{asset_name}_N", BAKE_RESOLUTION, BAKE_RESOLUTION, is_data=True)
    rough_img   = bpy.data.images.new(f"{asset_name}_R", BAKE_RESOLUTION, BAKE_RESOLUTION, is_data=True)
    ao_img      = bpy.data.images.new(f"{asset_name}_AO", BAKE_RESOLUTION, BAKE_RESOLUTION, is_data=True)

    for img in (diffuse_img, normal_img, rough_img, ao_img):
        img.file_format = 'PNG'
    for img in (normal_img, rough_img, ao_img):
        try:
            img.colorspace_settings.name = 'Non-Color'
        except Exception:
            pass

    diffuse_node.image = diffuse_img
    normal_node.image = normal_img
    rough_node.image = rough_img
    ao_node.image = ao_img

    links.new(bsdf.outputs['BSDF'], out_node.inputs['Surface'])
    links.new(diffuse_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(rough_node.outputs['Color'], bsdf.inputs['Roughness'])
    links.new(normal_node.outputs['Color'], normal_map_node.inputs['Color'])
    links.new(normal_map_node.outputs['Normal'], bsdf.inputs['Normal'])

    target_obj.data.materials.clear()
    target_obj.data.materials.append(baked_mat)

    # ensure source uses nodes
    for m in source_obj.data.materials:
        if m and not m.use_nodes:
            m.use_nodes = True

    # select source+target
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    # create explicit cage by duplicating target and scaling slightly
    bpy.ops.object.select_all(action='DESELECT')
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.duplicate()
    cage = bpy.context.selected_objects[0]
    cage.name = f"{target_obj.name}_BakeCage"
    scale_val = 1.0 + max(0.01, cage_extrusion)
    cage.scale = (scale_val, scale_val, scale_val)
    try:
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    except Exception:
        pass

    # reselect source & target (duplicate changed selection)
    bpy.ops.object.select_all(action='DESELECT')
    source_obj.select_set(True)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    bake_kwargs = {
        "use_selected_to_active": True,
        "max_ray_distance": max_ray_distance,
        "cage_extrusion": cage_extrusion,
        "margin": bake_margin,
        "use_cage": True,
        "cage_object": cage.name,
        "use_clear": True,
    }

    def bake_with_fallback(btype, extra=None):
        kw = dict(bake_kwargs)
        if extra: kw.update(extra)
        try:
            bpy.ops.object.bake(type=btype, **kw)
        except TypeError:
            safe = {k: v for k, v in kw.items() if k in ('use_selected_to_active','max_ray_distance','cage_extrusion','margin','use_cage','cage_object','use_clear')}
            bpy.ops.object.bake(type=btype, **safe)

    log(f"Active before bake: {bpy.context.view_layer.objects.active.name if bpy.context.view_layer.objects.active else 'None'}, Selected: {[o.name for o in bpy.context.selected_objects]}")

    # NORMAL
    log("Baking NORMAL...")
    nodes.active = normal_node
    bpy.context.view_layer.update()
    try:
        bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', **bake_kwargs)
    except TypeError:
        bake_with_fallback('NORMAL')
    out_normal = save_and_pack_image(normal_img, f"{asset_name}_Normal.png", output_folder)
    finalize_baked_image(baked_mat, normal_node, out_normal)

    # ROUGHNESS
    log("Baking ROUGHNESS...")
    nodes.active = rough_node
    bpy.context.view_layer.update()
    bake_with_fallback('ROUGHNESS')
    out_rough = save_and_pack_image(rough_img, f"{asset_name}_Roughness.png", output_folder)
    finalize_baked_image(baked_mat, rough_node, out_rough)

    # AO
    log("Baking AO...")
    nodes.active = ao_node
    bpy.context.view_layer.update()
    bake_with_fallback('AO')
    out_ao = save_and_pack_image(ao_img, f"{asset_name}_AO.png", output_folder)
    finalize_baked_image(baked_mat, ao_node, out_ao)

    # EMIT (color) bake using temporary emission nodes if needed
    # Setup emission fallback so EMIT picks up the Base Color texture
    temp_nodes = []
    log("Preparing emission fallbacks for EMIT bake...")
    for mat in list(getattr(source_obj.data, "materials", [])):
        if not mat or not mat.use_nodes:
            continue
        nt = mat.node_tree
        nodes_local = nt.nodes
        links_local = nt.links
        principled = next((n for n in nodes_local if n.type=='BSDF_PRINCIPLED'), None)
        img_node = None
        if principled:
            base_links = [l for l in links_local if l.to_node==principled and l.to_socket.name=='Base Color']
            if base_links:
                img_node = base_links[0].from_node
        if not img_node:
            img_node = next((n for n in nodes_local if n.type=='TEX_IMAGE' and getattr(n,'image',None)), None)
        if not img_node:
            continue
        out_node = next((n for n in nodes_local if n.type=='OUTPUT_MATERIAL'), None)
        original = []
        if out_node:
            for l in list(links_local):
                if l.to_node==out_node and l.to_socket.name=='Surface':
                    original.append((l.from_node.name, l.from_socket.name))
                    try: links_local.remove(l)
                    except Exception: pass
        emit = nodes_local.new(type='ShaderNodeEmission')
        emit.name = f"TempEmit_{mat.name}"
        try:
            links_local.new(img_node.outputs['Color'], emit.inputs['Color'])
        except Exception:
            pass
        if out_node:
            try:
                links_local.new(emit.outputs['Emission'], out_node.inputs['Surface'])
            except Exception:
                pass
        temp_nodes.append((mat, emit, original))

    # EMIT bake
    log("Baking EMIT (color)...")
    nodes.active = diffuse_node
    bpy.context.view_layer.update()
    bake_with_fallback('EMIT')
    out_diffuse = save_and_pack_image(diffuse_img, f"{asset_name}_Diffuse.png", output_folder)
    finalize_baked_image(baked_mat, diffuse_node, out_diffuse)

    # restore materials (remove temp emission nodes and restore original links)
    log("Restoring original materials...")
    for mat, emit_node, original in temp_nodes:
        try:
            nt = mat.node_tree
            # remove any links involving emit_node
            for l in list(nt.links):
                if l.from_node == emit_node or l.to_node == emit_node:
                    try: nt.links.remove(l)
                    except Exception: pass
            out_node = next((n for n in nt.nodes if n.type=='OUTPUT_MATERIAL'), None)
            if out_node and original:
                for src_name, src_socket in original:
                    src_node = next((n for n in nt.nodes if n.name==src_name), None)
                    if src_node:
                        try:
                            nt.links.new(src_node.outputs[src_socket], out_node.inputs['Surface'])
                        except Exception:
                            pass
            try:
                nt.nodes.remove(emit_node)
            except Exception:
                pass
        except Exception as e:
            log(f"Restore material error: {e}")

    # cleanup cage
    try:
        bpy.data.objects.remove(cage, do_unlink=True)
    except Exception:
        pass

    log("Baking finished.")
    return baked_mat

def decimate_and_export(base_obj, suffix, face_count, asset_name, output_folder):
    log(f"Processing LOD {suffix} target {face_count} faces")
    lod = base_obj.copy()
    lod.data = base_obj.data.copy()
    lod.name = f"{asset_name}_{suffix}"
    bpy.context.collection.objects.link(lod)
    bpy.context.view_layer.objects.active = lod

    # apply multires if present
    for mod in list(lod.modifiers):
        if mod.type == 'MULTIRES':
            mod.levels = 0
            try:
                bpy.ops.object.modifier_apply(modifier=mod.name)
            except Exception:
                pass

    initial = len(lod.data.polygons)
    if initial > face_count:
        dec = lod.modifiers.new("Decimate","DECIMATE")
        dec.decimate_type = 'COLLAPSE'
        dec.ratio = float(face_count)/float(initial)
        try:
            bpy.ops.object.modifier_apply(modifier=dec.name)
        except Exception:
            pass
    else:
        log("Skipping decimation (already below target)")

    final = len(lod.data.polygons)
    if not lod.data.uv_layers:
        log("Warning: LOD has no UVs")

    out_fbx = os.path.join(output_folder, f"{asset_name}_{suffix}.fbx")
    log(f"Exporting FBX: {out_fbx}")
    bpy.ops.object.select_all(action='DESELECT')
    lod.select_set(True)
    try:
        bpy.ops.export_scene.fbx(filepath=out_fbx, use_selection=True, apply_scale_options='FBX_SCALE_ALL', object_types={'MESH'}, embed_textures=True, path_mode='COPY')
    except Exception as e:
        log(f"FBX export failed: {e}")
        raise
    try: bpy.data.objects.remove(lod, do_unlink=True)
    except Exception: pass
    # Normalize output to the format expected by the decimation_page consumer
    # Provide local_file, poly_before, poly_after and reduction_ratio
    # reduction_ratio: fraction of polygons removed (e.g., 0.60 means 60% removed)
    reduction_ratio = None
    try:
        if initial and final is not None and initial > 0:
            reduction_ratio = (float(initial) - float(final)) / float(initial)
            # keep a compact representation
            reduction_ratio = round(reduction_ratio, 4)
        else:
            reduction_ratio = None
    except Exception:
        reduction_ratio = None
    return {
        "local_file": out_fbx,
        "poly_before": initial,
        "poly_after": final,
        "reduction_ratio": reduction_ratio
    }

# ---------------- MAIN ----------------
def main():
    argv = sys.argv
    if "--" not in argv or len(argv) < argv.index("--")+2:
        print("Usage: blender --background --python script.py -- <asset_name | input_file> <input_file | output_folder> [<output_folder>]")
        sys.exit(1)
    args = argv[argv.index("--")+1:]
    if len(args) == 0:
        print("No args")
        sys.exit(1)

    possible_first = args[0]
    if os.path.exists(possible_first):
        asset_name = os.path.splitext(os.path.basename(possible_first))[0]
        input_file = possible_first
        output_folder = args[1] if len(args) > 1 else None
    else:
        asset_name = possible_first
        input_file = args[1] if len(args) > 1 else None
        output_folder = args[2] if len(args) > 2 else None

    if not input_file:
        print("Input file is required")
        sys.exit(1)

    if not output_folder:
        workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        output_folder = os.path.join(workspace_root, "s3_downloads", "decimated_output")

    os.makedirs(output_folder, exist_ok=True)

    log(f"Input: {input_file} | Output: {output_folder} | Asset: {asset_name}")

    results = {}
    try:
        clear_scene()
        source = import_model(input_file)

        # write packed images to disk so cycles can read them
        ensure_textures_unpacked_to_disk(output_folder)

        preprocess_mesh(source)
        ensure_textures_linked_to_basecolor(source)

        lowpoly = create_retopo_mesh(source, VOXEL_SIZE, MULTIRES_LEVELS)

        baked_mat = bake_maps(source, lowpoly, asset_name, output_folder, BAKE_RESOLUTION, BAKE_MARGIN, CAGE_EXTRUSION, MAX_RAY_DISTANCE)

        # assign baked material to lowpoly for export
        lowpoly.data.materials.clear()
        lowpoly.data.materials.append(baked_mat)

        # remove source
        try: bpy.data.objects.remove(source, do_unlink=True)
        except Exception: pass

        # cleanup
        try: bpy.ops.outliner.orphans_purge()
        except Exception: pass

        for suffix, faces in DECIMATION_PROFILES:
            results[suffix] = decimate_and_export(lowpoly, suffix, faces, asset_name, output_folder)

    except Exception as e:
        log(f"[FATAL] {e}")
        import traceback; traceback.print_exc()
        results['error'] = str(e)

    print("\n--- Final Results ---")
    print(json.dumps(results, indent=2))
    log("Script finished.")

if __name__ == "__main__":
    main()
    
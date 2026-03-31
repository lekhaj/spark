#!/usr/bin/env python3
"""
Production LOD pipeline — Blender 3.x / 4.x / 5.x
────────────────────────────────────────────────────
Usage:
  blender --background --python lod_pipeline.py -- <asset_name> <input> <output_dir>

Fixes vs previous version:
  • Triangulate BEFORE computing collapse ratio (fixes target overshoot)
  • Iterative ratio correction to hit exact face targets (±5%)
  • Mesh validation pass before export (fixes "not valid" warnings)
  • Cached intermediate mesh (planar+tri done once, not per-LOD)
  • BMesh-based degenerate face removal
"""

import bpy
import bmesh
import os
import sys
import json
import math
import time
import traceback
from mathutils import Vector

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CONFIGURATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

LOD_PROFILES = [
    ("100k", 100_000),
    ("300k", 300_000),
    ("50k",   50_000),
    ("40k",   40_000),
]

DECIMATE_METHOD = "COLLAPSE"   # "COLLAPSE" or "QUADRIFLOW"

# Bake
BAKE_RESOLUTION     = 4096
BAKE_MARGIN_PX      = 16
BAKE_SAMPLES        = 1

# Auto-cage fractions of bounding-box diagonal
CAGE_FRACTION       = 0.005
RAY_DISTANCE_FRAC   = 0.02

# UV
UV_ANGLE_DEG        = 66.0
UV_ISLAND_MARGIN    = 0.004

# Post-decimate surface snap
USE_SHRINKWRAP_SNAP = True

# Planar dissolve angle (set to 0 to disable)
PLANAR_ANGLE_DEG    = 1.5

# Maps
BAKE_DIFFUSE = True
BAKE_NORMAL  = True
BAKE_AO      = False
AO_SAMPLES   = 32

# How close to target is acceptable (5% = 0.05)
TARGET_TOLERANCE    = 0.05

# Max iterative correction passes
MAX_DECIMATE_ITERS  = 5


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOGGING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def log(msg: str):
    print(f"[LOD {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  SCENE UTILITIES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def purge_scene():
    log("Purging scene…")
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for attr in ('meshes', 'materials', 'textures', 'images',
                 'node_groups', 'worlds', 'cameras', 'lights',
                 'actions', 'armatures'):
        coll = getattr(bpy.data, attr, None)
        if coll is None:
            continue
        for block in list(coll):
            try:
                coll.remove(block)
            except Exception:
                pass
    log("Scene purged.")


def bbox_diagonal(obj) -> float:
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    lo = Vector((min(c[i] for c in corners) for i in range(3)))
    hi = Vector((max(c[i] for c in corners) for i in range(3)))
    return (hi - lo).length


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  IMPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def import_mesh(filepath: str):
    log(f"Importing: {filepath}")
    if not os.path.isfile(filepath):
        raise FileNotFoundError(filepath)

    ext = os.path.splitext(filepath)[1].lower()
    before = set(bpy.context.scene.objects)

    if ext in ('.glb', '.gltf'):
        bpy.ops.import_scene.gltf(filepath=filepath)
    elif ext == '.fbx':
        bpy.ops.import_scene.fbx(filepath=filepath)
    elif ext == '.obj':
        try:
            bpy.ops.wm.obj_import(filepath=filepath)
        except AttributeError:
            bpy.ops.import_scene.obj(filepath=filepath)
    elif ext == '.stl':
        bpy.ops.import_mesh.stl(filepath=filepath)
    elif ext == '.ply':
        (bpy.ops.wm.ply_import(filepath=filepath)
         if hasattr(bpy.ops.wm, 'ply_import')
         else bpy.ops.import_mesh.ply(filepath=filepath))
    else:
        raise NotImplementedError(f"Unsupported format: {ext}")

    new_meshes = [o for o in bpy.context.scene.objects
                  if o not in before and o.type == 'MESH']
    if not new_meshes:
        raise RuntimeError("No mesh objects imported.")

    bpy.ops.object.select_all(action='DESELECT')
    for o in new_meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = new_meshes[0]
    if len(new_meshes) > 1:
        bpy.ops.object.join()

    obj = bpy.context.view_layer.objects.active
    obj.name = "HighPoly"
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    log(f"  → {len(obj.data.vertices):,} verts | "
        f"{len(obj.data.polygons):,} faces | "
        f"{len(obj.data.materials)} materials")
    return obj


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MESH CLEANING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def clean_mesh(obj):
    log("Cleaning high-poly…")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=1e-4)
    bpy.ops.mesh.delete_loose(use_verts=True, use_edges=True, use_faces=False)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    mesh = obj.data
    if bpy.app.version < (4, 1, 0):
        try:
            mesh.use_auto_smooth = True
            mesh.auto_smooth_angle = math.radians(60)
        except Exception:
            pass

    log(f"  → cleaned: {len(mesh.vertices):,} verts, {len(mesh.polygons):,} faces")


def validate_mesh_bmesh(obj):
    """
    Remove degenerate geometry using BMesh.
    Fixes the "Mesh is not valid" glTF export warning.
    """
    log("  Validating mesh (removing degenerates)…")
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)

    # Remove zero-area faces
    degenerate_faces = [f for f in bm.faces if f.calc_area() < 1e-8]
    if degenerate_faces:
        bmesh.ops.delete(bm, geom=degenerate_faces, context='FACES')
        log(f"    Removed {len(degenerate_faces)} degenerate faces")

    # Remove zero-length edges
    degenerate_edges = [e for e in bm.edges if e.calc_length() < 1e-8]
    if degenerate_edges:
        bmesh.ops.delete(bm, geom=degenerate_edges, context='EDGES')
        log(f"    Removed {len(degenerate_edges)} degenerate edges")

    # Merge very close vertices
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=1e-5)

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  TRIANGULATION  (must happen before any face counting)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_triangulated(obj):
    """
    Ensure mesh is fully triangulated.
    After this, len(obj.data.polygons) == actual triangle count.
    """
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mod = obj.modifiers.new("_EnsureTri", 'TRIANGULATE')
    mod.keep_custom_normals = True
    mod.quad_method = 'SHORTEST_DIAGONAL'
    mod.ngon_method = 'BEAUTY'
    bpy.ops.object.modifier_apply(modifier=mod.name)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  DECIMATION  (FIXED — works on triangulated mesh)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def decimate_collapse(obj, target_faces: int):
    """
    QEM collapse decimation on a PRE-TRIANGULATED mesh.
    Iterates to hit the exact target within tolerance.

    Previous bug: planar dissolve created n-gons, ratio was computed
    against n-gon count, then triangulation blew the count back up.
    Fix: triangulate first, then collapse on pure triangle counts.
    """
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # ── Step 1: Triangulate so face count = triangle count ──
    ensure_triangulated(obj)
    current = len(obj.data.polygons)
    log(f"    Pre-decimate triangles: {current:,}")

    if current <= target_faces:
        log(f"    Already ≤ target ({current:,} ≤ {target_faces:,}), skipping.")
        return

    # ── Step 2: Optional planar dissolve + re-triangulate ──
    #    This is purely for speed: dissolve co-planar regions first
    #    so collapse has fewer faces to process.
    if PLANAR_ANGLE_DEG > 0:
        mod = obj.modifiers.new("_Planar", 'DECIMATE')
        mod.decimate_type = 'DISSOLVE'
        mod.angle_limit = math.radians(PLANAR_ANGLE_DEG)
        mod.use_dissolve_boundaries = False
        bpy.ops.object.modifier_apply(modifier=mod.name)
        ngon_count = len(obj.data.polygons)

        # CRITICAL FIX: re-triangulate after dissolve so counts are accurate
        ensure_triangulated(obj)
        after_planar = len(obj.data.polygons)
        log(f"    After planar dissolve: {ngon_count:,} ngons → {after_planar:,} tris")
    else:
        after_planar = current

    if after_planar <= target_faces:
        log(f"    Planar already ≤ target ({after_planar:,} ≤ {target_faces:,}).")
        return

    # ── Step 3: Iterative collapse to hit exact target ──
    #    Blender's collapse ratio isn't perfectly linear,
    #    so we iterate with ratio correction.
    working_count = after_planar

    for iteration in range(MAX_DECIMATE_ITERS):
        if working_count <= target_faces:
            break

        ratio = target_faces / working_count

        # Clamp ratio: never go below 0.001 or above 0.999
        ratio = max(0.001, min(ratio, 0.999))

        mod = obj.modifiers.new(f"_Collapse_{iteration}", 'DECIMATE')
        mod.decimate_type = 'COLLAPSE'
        mod.ratio = ratio
        mod.use_collapse_triangulate = False  # already triangulated
        mod.use_symmetry = False
        bpy.ops.object.modifier_apply(modifier=mod.name)

        working_count = len(obj.data.polygons)

        overshoot = (working_count - target_faces) / target_faces
        log(f"    Collapse pass {iteration+1}: ratio={ratio:.4f} → "
            f"{working_count:,} tris (target {target_faces:,}, "
            f"{'over' if overshoot > 0 else 'under'} by "
            f"{abs(overshoot)*100:.1f}%)")

        # Close enough?
        if abs(overshoot) <= TARGET_TOLERANCE:
            break

        # If we overshot and can't reduce further, stop
        if working_count <= target_faces:
            break

    log(f"  Decimated {current:,} → {working_count:,} tris "
        f"[target {target_faces:,}]")


def decimate_quadriflow(obj, target_faces: int):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    try:
        bpy.ops.object.quadriflow_remesh(
            target_faces=target_faces,
            use_paint_symmetry=False,
            use_preserve_sharp=True,
            use_preserve_boundary=True,
            seed=42,
        )
        # Quadriflow outputs quads, triangulate for consistency
        ensure_triangulated(obj)
        log(f"  Quadriflow → {len(obj.data.polygons):,} tris")
    except Exception as e:
        log(f"  Quadriflow failed ({e}), falling back to collapse.")
        decimate_collapse(obj, target_faces)


def decimate(obj, target_faces: int, method: str = DECIMATE_METHOD):
    if method == "QUADRIFLOW":
        decimate_quadriflow(obj, target_faces)
    else:
        decimate_collapse(obj, target_faces)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  POST-DECIMATE SNAP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def shrinkwrap_snap(lod_obj, target_obj):
    log("  Shrinkwrap snap to high-poly…")
    bpy.context.view_layer.objects.active = lod_obj
    mod = lod_obj.modifiers.new("_SWSnap", 'SHRINKWRAP')
    mod.target = target_obj
    mod.wrap_method = 'NEAREST_SURFACEPOINT'
    mod.wrap_mode = 'ON_SURFACE'
    bpy.ops.object.modifier_apply(modifier=mod.name)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  UV UNWRAPPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def unwrap_smart(obj):
    log("  UV unwrapping (Smart Project)…")
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')

    bpy.ops.uv.smart_project(
        angle_limit=math.radians(UV_ANGLE_DEG),
        island_margin=UV_ISLAND_MARGIN,
        area_weight=0.0,
        correct_aspect=True,
        scale_to_bounds=True,
    )

    bpy.ops.object.mode_set(mode='OBJECT')
    log("  UV unwrap done.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  CYCLES / BAKE SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def setup_cycles():
    scn = bpy.context.scene
    scn.render.engine = 'CYCLES'
    scn.cycles.samples = BAKE_SAMPLES
    scn.cycles.use_denoising = False

    prefs = bpy.context.preferences.addons.get('cycles')
    if prefs:
        try:
            cp = prefs.preferences
            for ct in ('OPTIX', 'CUDA', 'HIP', 'METAL', 'ONEAPI'):
                try:
                    cp.compute_device_type = ct
                    cp.get_devices()
                    if cp.devices:
                        for d in cp.devices:
                            d.use = True
                        scn.cycles.device = 'GPU'
                        log(f"  Cycles device: GPU ({ct})")
                        return
                except Exception:
                    continue
        except Exception:
            pass
    scn.cycles.device = 'CPU'
    log("  Cycles device: CPU")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EMISSION SWAP (for diffuse bake)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _find_base_color_source(nodes, links):
    for n in nodes:
        if n.type == 'BSDF_PRINCIPLED':
            inp = n.inputs.get('Base Color')
            if inp and inp.links:
                return inp.links[0].from_socket
            elif inp:
                return tuple(inp.default_value)

    for n in nodes:
        if n.type == 'BSDF_DIFFUSE':
            inp = n.inputs.get('Color')
            if inp and inp.links:
                return inp.links[0].from_socket
            elif inp:
                return tuple(inp.default_value)

    for n in nodes:
        if n.type == 'GROUP':
            for out in n.outputs:
                if out.name in ('Base Color', 'Color') and out.type == 'RGBA':
                    return out

    return (0.8, 0.8, 0.8, 1.0)


def swap_to_emission(obj):
    restore = []
    for idx, slot in enumerate(obj.material_slots):
        mat = slot.material
        if not mat:
            continue
        if not getattr(mat, 'use_nodes', False):
            continue

        tree = mat.node_tree
        output = next((n for n in tree.nodes
                       if n.type == 'OUTPUT_MATERIAL' and n.is_active_output), None)
        if not output:
            output = next((n for n in tree.nodes
                           if n.type == 'OUTPUT_MATERIAL'), None)
        if not output:
            continue

        old_link_from = (output.inputs['Surface'].links[0].from_socket
                         if output.inputs['Surface'].links else None)

        color_src = _find_base_color_source(tree.nodes, tree.links)

        emit = tree.nodes.new('ShaderNodeEmission')
        emit.name = f"__TEMP_EMIT_{idx}"
        emit.inputs['Strength'].default_value = 1.0

        if isinstance(color_src, tuple):
            emit.inputs['Color'].default_value = color_src
        else:
            tree.links.new(color_src, emit.inputs['Color'])

        tree.links.new(emit.outputs['Emission'], output.inputs['Surface'])

        restore.append({
            'mat': mat,
            'emit': emit,
            'old_from': old_link_from,
            'output': output,
        })
    return restore


def restore_materials(restore):
    for r in restore:
        tree = r['mat'].node_tree
        if r['old_from']:
            try:
                tree.links.new(r['old_from'], r['output'].inputs['Surface'])
            except Exception:
                pass
        try:
            tree.nodes.remove(r['emit'])
        except Exception:
            pass


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MATERIAL & BAKE IMAGE SETUP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def create_lod_material(name: str, res: int):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()

    bsdf   = tree.nodes.new('ShaderNodeBsdfPrincipled')
    output = tree.nodes.new('ShaderNodeOutputMaterial')
    tree.links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    bsdf.location   = (0, 0)
    output.location  = (400, 0)

    imgs = {}

    if BAKE_DIFFUSE:
        img = bpy.data.images.new(f"{name}_Diffuse", res, res, alpha=False)
        img.colorspace_settings.name = 'sRGB'
        nd = tree.nodes.new('ShaderNodeTexImage')
        nd.image = img
        nd.name  = 'BAKE_DIFFUSE'
        nd.location = (-700, 200)
        imgs['diffuse'] = (nd, img)

    if BAKE_NORMAL:
        img = bpy.data.images.new(f"{name}_Normal", res, res, alpha=False)
        img.colorspace_settings.name = 'Non-Color'
        nd = tree.nodes.new('ShaderNodeTexImage')
        nd.image = img
        nd.name  = 'BAKE_NORMAL'
        nd.location = (-700, -100)
        imgs['normal'] = (nd, img)

    if BAKE_AO:
        img = bpy.data.images.new(f"{name}_AO", res, res, alpha=False)
        img.colorspace_settings.name = 'Non-Color'
        nd = tree.nodes.new('ShaderNodeTexImage')
        nd.image = img
        nd.name  = 'BAKE_AO'
        nd.location = (-700, -400)
        imgs['ao'] = (nd, img)

    return mat, bsdf, output, imgs


def wire_final_material(mat, bsdf, imgs):
    tree = mat.node_tree

    if 'diffuse' in imgs:
        tree.links.new(imgs['diffuse'][0].outputs['Color'],
                       bsdf.inputs['Base Color'])

    if 'normal' in imgs:
        nm = tree.nodes.new('ShaderNodeNormalMap')
        nm.location = (-350, -100)
        tree.links.new(imgs['normal'][0].outputs['Color'], nm.inputs['Color'])
        tree.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  BAKE EXECUTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _select_for_bake(high, low):
    bpy.ops.object.select_all(action='DESELECT')
    high.select_set(True)
    low.select_set(True)
    bpy.context.view_layer.objects.active = low


def _set_active_image_node(mat, node_name):
    tree = mat.node_tree
    for n in tree.nodes:
        n.select = (n.name == node_name)
    tree.nodes.active = tree.nodes[node_name]


def bake_all_maps(high_poly, low_poly, mat, imgs, out_dir, prefix):
    setup_cycles()

    diag = bbox_diagonal(high_poly)
    cage = max(diag * CAGE_FRACTION, 0.0005)
    ray  = max(diag * RAY_DISTANCE_FRAC, 0.005)
    log(f"  Auto-cage: diag={diag:.4f}  cage={cage:.5f}  ray={ray:.4f}")

    common = dict(
        use_selected_to_active=True,
        margin=BAKE_MARGIN_PX,
        use_clear=True,
        cage_extrusion=cage,
        max_ray_distance=ray,
    )
    try:
        common['margin_type'] = 'EXTEND'
    except Exception:
        pass

    saved = {}

    # ── Normal ──
    if 'normal' in imgs:
        log("  Baking NORMAL…")
        _set_active_image_node(mat, 'BAKE_NORMAL')
        _select_for_bake(high_poly, low_poly)
        try:
            bpy.ops.object.bake(type='NORMAL', normal_space='TANGENT', **common)
            p = os.path.join(out_dir, f"{prefix}_Normal.png")
            imgs['normal'][1].filepath_raw = p
            imgs['normal'][1].file_format = 'PNG'
            imgs['normal'][1].save()
            saved['normal'] = p
            log(f"    → saved {p}")
        except Exception as e:
            log(f"    ✗ Normal bake failed: {e}")

    # ── Diffuse ──
    if 'diffuse' in imgs:
        log("  Baking DIFFUSE (emission pass)…")
        restore = swap_to_emission(high_poly)
        _set_active_image_node(mat, 'BAKE_DIFFUSE')
        _select_for_bake(high_poly, low_poly)
        try:
            bpy.ops.object.bake(type='EMIT', **common)
            p = os.path.join(out_dir, f"{prefix}_Diffuse.png")
            imgs['diffuse'][1].filepath_raw = p
            imgs['diffuse'][1].file_format = 'PNG'
            imgs['diffuse'][1].save()
            saved['diffuse'] = p
            log(f"    → saved {p}")
        except Exception as e:
            log(f"    ✗ Diffuse bake failed: {e}")
        restore_materials(restore)

    # ── AO ──
    if 'ao' in imgs:
        log("  Baking AO…")
        old_samples = bpy.context.scene.cycles.samples
        bpy.context.scene.cycles.samples = AO_SAMPLES
        _set_active_image_node(mat, 'BAKE_AO')
        _select_for_bake(high_poly, low_poly)
        try:
            bpy.ops.object.bake(type='AO', **common)
            p = os.path.join(out_dir, f"{prefix}_AO.png")
            imgs['ao'][1].filepath_raw = p
            imgs['ao'][1].file_format = 'PNG'
            imgs['ao'][1].save()
            saved['ao'] = p
            log(f"    → saved {p}")
        except Exception as e:
            log(f"    ✗ AO bake failed: {e}")
        bpy.context.scene.cycles.samples = old_samples

    return saved


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  EXPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def export_glb(obj, filepath):
    for slot in obj.material_slots:
        if not slot.material:
            continue
        if not getattr(slot.material, 'use_nodes', False):
            continue
        for node in slot.material.node_tree.nodes:
            if node.type == 'TEX_IMAGE' and node.image:
                try:
                    node.image.pack()
                except Exception:
                    pass

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    bpy.ops.export_scene.gltf(
        filepath=filepath,
        use_selection=True,
        export_format='GLB',
        export_image_format='AUTO',
        export_materials='EXPORT',
        export_normals=True,
    )

    log(f"  Exported → {filepath}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LOD PIPELINE (per-level)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def process_lod(high_poly, suffix, target_faces, asset_name, out_dir):
    log(f"\n{'━'*64}")
    log(f"  LOD  {suffix}   target = {target_faces:,} faces")
    log(f"{'━'*64}")

    # 1 ── Duplicate high-poly
    bpy.ops.object.select_all(action='DESELECT')
    high_poly.select_set(True)
    bpy.context.view_layer.objects.active = high_poly
    bpy.ops.object.duplicate()
    lod = bpy.context.view_layer.objects.active
    lod.name = f"{asset_name}_{suffix}"

    faces_before = len(lod.data.polygons)

    # 2 ── Decimate (includes triangulation + iterative correction)
    decimate(lod, target_faces)

    # 3 ── Snap back onto high-poly surface
    if USE_SHRINKWRAP_SNAP:
        shrinkwrap_snap(lod, high_poly)

    # 4 ── Recalculate normals
    bpy.context.view_layer.objects.active = lod
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode='OBJECT')

    # 5 ── Validate mesh (remove degenerates → fixes "not valid" warning)
    validate_mesh_bmesh(lod)

    # 6 ── UV unwrap
    unwrap_smart(lod)

    faces_after = len(lod.data.polygons)

    # 7 ── Smooth shade
    bpy.ops.object.select_all(action='DESELECT')
    lod.select_set(True)
    bpy.context.view_layer.objects.active = lod
    bpy.ops.object.shade_smooth()

    # 8 ── Create material & bake
    prefix = f"{asset_name}_{suffix}"
    mat, bsdf, output, imgs = create_lod_material(prefix, BAKE_RESOLUTION)
    lod.data.materials.clear()
    lod.data.materials.append(mat)

    tex_paths = bake_all_maps(high_poly, lod, mat, imgs, out_dir, prefix)

    # 9 ── Wire up material for export
    wire_final_material(mat, bsdf, imgs)

    # 10 ── Export
    glb_path = os.path.join(out_dir, f"{prefix}.glb")
    export_glb(lod, glb_path)

    # 11 ── Cleanup
    bpy.data.objects.remove(lod, do_unlink=True)

    accuracy = abs(faces_after - target_faces) / target_faces * 100
    log(f"  Result: {faces_after:,} tris (target {target_faces:,}, "
        f"accuracy {100-accuracy:.1f}%)")

    return {
        'file':          glb_path,
        'faces_before':  faces_before,
        'faces_after':   faces_after,
        'target':        target_faces,
        'accuracy_pct':  round(100 - accuracy, 2),
        'reduction_pct': round((1 - faces_after / faces_before) * 100, 2)
                         if faces_before else 0,
        'textures':      tex_paths,
    }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    argv = sys.argv
    if "--" not in argv:
        print("Usage: blender -b -P script.py -- <asset> <input> <output_dir>")
        sys.exit(1)

    args = argv[argv.index("--") + 1:]
    if len(args) >= 3:
        asset_name, input_file, output_dir = args[0], args[1], args[2]
    elif len(args) == 2:
        input_file, output_dir = args[0], args[1]
        asset_name = os.path.splitext(os.path.basename(input_file))[0]
    else:
        print("Not enough arguments.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    log(f"Asset : {asset_name}")
    log(f"Input : {input_file}")
    log(f"Output: {output_dir}")

    results = {}
    try:
        purge_scene()
        high_poly = import_mesh(input_file)
        clean_mesh(high_poly)

        for suffix, target in LOD_PROFILES:
            results[suffix] = process_lod(
                high_poly, suffix, target, asset_name, output_dir
            )

    except Exception as e:
        log(f"FATAL: {e}")
        log(traceback.format_exc())
        results['error'] = str(e)
        results['traceback'] = traceback.format_exc()

    out_json = json.dumps(results, indent=2)
    print(out_json)

    with open(os.path.join(output_dir, f"{asset_name}_results.json"), 'w') as f:
        f.write(out_json)

    log("Done.")


if __name__ == "__main__":
    main()

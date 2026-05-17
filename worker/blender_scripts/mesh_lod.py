"""
mesh_lod.py — Blender headless mesh-LOD generator.

Invoked as:
    blender --background --python mesh_lod.py -- <profile> <in_glb> <out_glb> [json_args]

Profiles:
    a  Merge-by-Distance + Delete-Loose only (truly lossless).
    b  + Decimate COLLAPSE (ratio configurable, default 0.4, preserve UV).
    c  + heavier Decimate COLLAPSE + Decimate Planar.
    d  + Voxel Remesh + Smart UV + bake old texture → new UV. (optional QuadriFlow)

Args JSON (positional 4) supports keys:
    ratio_b         (float, default 0.4)
    target_faces_c  (int,   default 20000)  — PyMeshLab QECD target face count
    ratio_c         (float, default 0.17)   — fallback Blender ratio if pymeshlab absent
    voxel_size_d    (float, default 0.030)
    quadriflow      (bool,  default False)
    quadriflow_target (int, default 5000)
    bake_resolution (int,   default 1024)

Profile C requires PyMeshLab in the system Python (not Blender's embedded Python).
Install with:  pip install pymeshlab
Override interpreter: PYMESHLAB_PYTHON=/path/to/python3

Exits 0 on success. Stats line on stdout:
    MESH_LOD_RESULT: profile=X tris=N seconds=S
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import math
import traceback

import bpy


# ─── arg parsing ──────────────────────────────────────────────────────────────

def _argv():
    if "--" in sys.argv:
        return sys.argv[sys.argv.index("--") + 1:]
    return []


def _parse_extras(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _import_glb(path):
    bpy.ops.import_scene.gltf(filepath=path)


def _gather_meshes():
    return [o for o in bpy.data.objects if o.type == 'MESH']


def _join_meshes(meshes):
    if len(meshes) <= 1:
        return meshes[0] if meshes else None
    bpy.ops.object.select_all(action='DESELECT')
    for m in meshes:
        m.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    bpy.ops.object.join()
    return bpy.context.active_object


def _activate(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def _merge_by_distance(obj, threshold=0.0001):
    _activate(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.remove_doubles(threshold=threshold)
    bpy.ops.object.mode_set(mode='OBJECT')


def _delete_loose(obj):
    _activate(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.delete_loose()
    bpy.ops.object.mode_set(mode='OBJECT')


def _fill_holes(obj, sides=16):
    """
    Fill only *small* open boundary loops — artifact holes from the generation
    pipeline or decimation, NOT intentional character openings.

    sides=16 means: only fill holes whose boundary loop has ≤ 16 edges.
    Eyes (~40-150 edges), mouth (~80-200 edges), nostrils stay open.
    Tiny degenerate holes from mesh generation (usually 3-10 edges) get capped.
    Increase `sides` if you're still seeing small artifacts.
    """
    _activate(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.fill_holes(sides=sides)   # sides > 0 caps only loops ≤ N edges
    bpy.ops.object.mode_set(mode='OBJECT')


def _add_decimate_collapse(obj, ratio, name="Decimate"):
    _activate(obj)
    mod = obj.modifiers.new(name=name, type='DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = max(0.001, min(1.0, ratio))
    mod.use_collapse_triangulate = True
    mod.use_symmetry = False
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _add_decimate_planar(obj, angle_deg=15.0, name="DecimatePlanar"):
    _activate(obj)
    mod = obj.modifiers.new(name=name, type='DECIMATE')
    mod.decimate_type = 'DISSOLVE'
    mod.angle_limit = math.radians(angle_deg)
    mod.use_dissolve_boundaries = False
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _voxel_remesh(obj, voxel_size=0.03):
    _activate(obj)
    obj.data.remesh_voxel_size = voxel_size
    # Newer Blenders dropped/renamed the preserve_paint_mask flag — set only
    # the attributes that exist on this build.
    for attr_name in ("use_remesh_preserve_volume",):
        if hasattr(obj.data, attr_name):
            setattr(obj.data, attr_name, True)
    bpy.ops.object.voxel_remesh()


def _shrinkwrap_snap(dst_obj, src_obj):
    """Snap dst_obj surface back onto src_obj to undo voxel/decimate inflation."""
    _activate(dst_obj)
    mod = dst_obj.modifiers.new(name="Shrinkwrap", type='SHRINKWRAP')
    mod.target = src_obj
    mod.wrap_method = 'PROJECT'
    mod.use_negative_direction = True
    mod.use_positive_direction = True
    mod.offset = 0.0
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as e:
        print(f"[mesh_lod] shrinkwrap apply failed: {e}")


def _corrective_smooth(obj, iterations=5):
    """Corrective smooth to remove voxel/decimate stairstepping artifacts."""
    _activate(obj)
    mod = obj.modifiers.new(name="CorrectiveSmooth", type='CORRECTIVE_SMOOTH')
    mod.iterations = iterations
    mod.smooth_type = 'LENGTH_WEIGHTED'
    try:
        bpy.ops.object.modifier_apply(modifier=mod.name)
    except Exception as e:
        print(f"[mesh_lod] corrective smooth apply failed: {e}")


def _pymeshlab_qecd(in_obj: str, out_obj: str, target_faces: int) -> bool:
    """
    Quadric Edge Collapse Decimation via PyMeshLab (system Python subprocess).
    Blender's embedded Python does not ship PyMeshLab — we call out to the
    system/conda Python that does.

    Set PYMESHLAB_PYTHON env var to override the interpreter path.
    Returns True on success, False on any error.
    """
    python = os.getenv("PYMESHLAB_PYTHON", "python3")
    code = "\n".join([
        "import pymeshlab",
        "ms = pymeshlab.MeshSet()",
        f"ms.load_new_mesh({repr(in_obj)})",
        "m = ms.current_mesh()",
        "print(f'[pymeshlab] input  faces={m.face_number()} verts={m.vertex_number()}')",
        # Close only tiny artifact holes before decimation — NOT intentional
        # character openings (eyes/mouth are typically 40-200 boundary edges).
        # maxholesize=16 matches the Blender _fill_holes(sides=16) threshold.
        "try:",
        "    ms.meshing_close_holes(maxholesize=16, newfaceselected=False, selfintersection=True)",
        "    m_h = ms.current_mesh()",
        "    print(f'[pymeshlab] after hole-close faces={m_h.face_number()}')",
        "except Exception as _he:",
        "    print(f'[pymeshlab] close_holes skipped: {_he}')",
        "ms.meshing_decimation_quadric_edge_collapse_decimation(",
        f"    targetfacenum={target_faces},",
        "    qualitythr=0.5,",
        "    preserveboundary=True,",
        "    boundaryweight=2.0,",
        "    preservenormal=True,",
        "    preservetopology=True,",
        "    optimalplacement=True,",
        "    planarquadric=True,",
        "    autoremeshing=False,",
        ")",
        "m2 = ms.current_mesh()",
        "print(f'[pymeshlab] output faces={m2.face_number()} verts={m2.vertex_number()}')",
        f"ms.save_current_mesh({repr(out_obj)})",
    ])
    try:
        result = subprocess.run(
            [python, "-c", code],
            capture_output=True, text=True, timeout=120,
        )
        for line in result.stdout.splitlines():
            print(line)
        if result.returncode != 0:
            print(f"[mesh_lod] pymeshlab stderr: {result.stderr[-400:]}")
            return False
        return True
    except FileNotFoundError:
        print(f"[mesh_lod] Python not found: {python!r} — set PYMESHLAB_PYTHON env var")
        return False
    except subprocess.TimeoutExpired:
        print("[mesh_lod] pymeshlab timed out after 120s")
        return False
    except Exception as e:
        print(f"[mesh_lod] pymeshlab subprocess error: {e}")
        return False


def _smart_uv_project(obj, angle_limit_deg=66.0, island_margin=0.005):
    _activate(obj)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(
        angle_limit=math.radians(angle_limit_deg),
        island_margin=island_margin,
    )
    bpy.ops.object.mode_set(mode='OBJECT')


def _triangulate(obj):
    _activate(obj)
    mod = obj.modifiers.new(name="Tri", type='TRIANGULATE')
    mod.quad_method = 'FIXED'
    mod.ngon_method = 'BEAUTY'
    bpy.ops.object.modifier_apply(modifier=mod.name)


def _bake_diffuse(src_obj, dst_obj, resolution=1024):
    """Bake src_obj's material → dst_obj's new UV. Result image attached to dst_obj's new material."""
    # New image to receive the bake
    img = bpy.data.images.new(name="lod_bake", width=resolution, height=resolution, alpha=True)

    # Create a fresh material on dst with image texture node
    mat = bpy.data.materials.new(name="lod_baked")
    mat.use_nodes = True
    nodes  = mat.node_tree.nodes
    links  = mat.node_tree.links
    # Find principled BSDF, hook image as Base Color
    bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if bsdf is None:
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
    tex  = nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.select = True
    nodes.active = tex   # bake target node must be active selected image
    links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])

    # Assign material to dst
    if dst_obj.data.materials:
        dst_obj.data.materials[0] = mat
    else:
        dst_obj.data.materials.append(mat)

    # Cycles is required for bake — switch render engine
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'CPU'
    bpy.context.scene.cycles.samples = 8
    bpy.context.scene.cycles.bake_type = 'DIFFUSE'
    bpy.context.scene.render.bake.use_pass_direct   = False
    bpy.context.scene.render.bake.use_pass_indirect = False
    bpy.context.scene.render.bake.use_pass_color    = True
    bpy.context.scene.render.bake.use_selected_to_active = True
    bpy.context.scene.render.bake.cage_extrusion = 0.02

    # Select source + active=dst
    bpy.ops.object.select_all(action='DESELECT')
    src_obj.select_set(True)
    dst_obj.select_set(True)
    bpy.context.view_layer.objects.active = dst_obj

    try:
        bpy.ops.object.bake(type='DIFFUSE')
    except Exception as e:
        print(f"[mesh_lod] bake failed: {e} — continuing without baked texture")
        return None

    # Pack the image so it gets exported in the GLB
    img.pack()
    return img


def _export_glb(obj, path):
    _activate(obj)
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format='GLB',
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_normals=True,
        export_texcoords=True,
        export_materials='EXPORT',
        export_image_format='AUTO',
    )


def _tri_count(obj):
    """Count triangles after final triangulation."""
    return sum(1 for _ in obj.data.polygons)


# ─── profile pipelines ────────────────────────────────────────────────────────

def run_profile_a(obj, args):
    _merge_by_distance(obj)
    _delete_loose(obj)
    _fill_holes(obj)          # close open boundary loops from generation pipeline


def run_profile_b(obj, args):
    _merge_by_distance(obj)
    _delete_loose(obj)
    _fill_holes(obj)          # close holes before decimation
    _add_decimate_collapse(obj, ratio=float(args.get("ratio_b", 0.4)))


def run_profile_c(obj, args):
    """
    PyMeshLab QECD — ~6:1 reduction (120k → 20k) preserving silhouette,
    UV seams and normals via Quadric Error Metric.

    Falls back to Blender Decimate COLLAPSE if PyMeshLab is unavailable.
    Returns the decimated object (may be a new object if PyMeshLab path taken).
    """
    _merge_by_distance(obj)
    _delete_loose(obj)
    _fill_holes(obj)          # seal input holes before exporting to PyMeshLab

    target_faces = int(args.get("target_faces_c", 20000))
    tmp_dir = tempfile.mkdtemp(prefix="mesh_lod_c_")
    tmp_in  = os.path.join(tmp_dir, "in.obj")
    tmp_out = os.path.join(tmp_dir, "out.obj")

    # ── Export current mesh to OBJ for PyMeshLab ──────────────────────────
    _activate(obj)
    try:
        bpy.ops.wm.obj_export(           # Blender 3.6+ exporter
            filepath=tmp_in,
            export_selected_objects=True,
            export_uv=True,
            export_normals=True,
            export_materials=True,
            export_triangulated_mesh=True,
        )
    except AttributeError:
        bpy.ops.export_scene.obj(        # legacy exporter (Blender < 3.6)
            filepath=tmp_in,
            use_selection=True,
            use_normals=True,
            use_uvs=True,
            use_materials=True,
            use_triangles=True,
        )

    # ── PyMeshLab QECD ────────────────────────────────────────────────────
    ok = _pymeshlab_qecd(tmp_in, tmp_out, target_faces)

    if not ok or not os.path.exists(tmp_out):
        print("[mesh_lod] profile C: pymeshlab unavailable — falling back to Blender COLLAPSE")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        _add_decimate_collapse(obj, ratio=float(args.get("ratio_c", 0.17)))
        _add_decimate_planar(obj, angle_deg=15.0)
        return None   # obj modified in-place; caller keeps original reference

    # ── Import decimated OBJ back into Blender ────────────────────────────
    bpy.data.objects.remove(obj, do_unlink=True)
    try:
        bpy.ops.wm.obj_import(filepath=tmp_out)   # Blender 3.6+
    except AttributeError:
        bpy.ops.import_scene.obj(filepath=tmp_out) # legacy

    shutil.rmtree(tmp_dir, ignore_errors=True)

    new_obj = next(
        (o for o in bpy.context.selected_objects if o.type == 'MESH'),
        None,
    )
    if new_obj is None:
        print("[mesh_lod] profile C: OBJ re-import yielded no mesh")
        return new_obj

    # Post-QECD hole fill — QECD at high reduction can leave small boundary
    # gaps at UV seams and mesh boundaries; close them before export.
    _fill_holes(new_obj)
    return new_obj


def run_profile_d(obj, args):
    _merge_by_distance(obj)
    _delete_loose(obj)
    _fill_holes(obj)          # watertight input gives cleaner voxel remesh

    # Source mesh is `obj`. Duplicate it to receive the voxel remesh.
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.duplicate()
    dst = bpy.context.active_object
    dst.name = obj.name + "_lod_d"

    # Voxel remesh on dst
    voxel = float(args.get("voxel_size_d", 0.030))
    _voxel_remesh(dst, voxel_size=voxel)

    # Snap remeshed surface back to original to prevent voxel inflation
    _shrinkwrap_snap(dst, obj)
    _corrective_smooth(dst, iterations=int(args.get("smooth_iterations_d", 5)))

    # Optional QuadriFlow
    if bool(args.get("quadriflow", False)):
        _activate(dst)
        try:
            bpy.ops.object.quadriflow_remesh(
                target_faces=int(args.get("quadriflow_target", 5000)),
                use_preserve_sharp=True,
                use_preserve_boundary=True,
                use_mesh_symmetry=False,
            )
        except Exception as e:
            print(f"[mesh_lod] quadriflow failed: {e} — continuing")

    # Smart UV on dst + bake old material → new texture
    _smart_uv_project(dst, angle_limit_deg=66.0, island_margin=0.005)
    _bake_diffuse(obj, dst, resolution=int(args.get("bake_resolution", 1024)))

    # Remove the source mesh so export only ships dst
    bpy.data.objects.remove(obj, do_unlink=True)
    return dst


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    argv = _argv()
    if len(argv) < 3:
        print("usage: mesh_lod.py -- <profile> <in.glb> <out.glb> [json]")
        sys.exit(2)

    profile, in_glb, out_glb = argv[0], argv[1], argv[2]
    args = _parse_extras(argv[3] if len(argv) > 3 else "")

    t0 = time.time()
    _reset_scene()
    _import_glb(in_glb)
    meshes = _gather_meshes()
    if not meshes:
        print("MESH_LOD_RESULT: ERROR no meshes in input")
        sys.exit(3)
    obj = _join_meshes(meshes)

    profile = profile.lower()
    if profile == "a":
        run_profile_a(obj, args)
    elif profile == "b":
        run_profile_b(obj, args)
    elif profile == "c":
        new_obj = run_profile_c(obj, args)
        obj = new_obj if new_obj else obj
    elif profile == "d":
        new_obj = run_profile_d(obj, args)
        obj = new_obj if new_obj else obj
    else:
        print(f"MESH_LOD_RESULT: ERROR unknown profile '{profile}'")
        sys.exit(4)

    _triangulate(obj)
    _export_glb(obj, out_glb)

    tris = _tri_count(obj)
    elapsed = time.time() - t0
    print(f"MESH_LOD_RESULT: profile={profile} tris={tris} seconds={elapsed:.1f}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(99)

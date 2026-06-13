"""
Auto-Rig Pro Smart Rigging — Blender desktop script
====================================================
Called by rig_model.py via blender_desktop_run.sh:

    bash blender_desktop_run.sh blender --python auto_rig_smart.py -- \\
         --input        /tmp/char.glb \\
         --output       /tmp/char_rigged.glb \\
         --output_fbx   /tmp/char_rigged.fbx \\
         --character_type humanoid \\
         --morphology   B1_humanoid

Produces TWO exports from one Blender session:
  * a clean **deform-only GLB** (web/three.js/Babylon) — control/reference bones
    stripped via deform isolation + ``export_def_bones=True``;
  * an engine-ready **ARP FBX** via ``arp.arp_export_fbx_panel`` (rig-type
    HUMANOID for B1/B2/B3, UNIVERSAL for quadruped/other).

Rigging is **non-fatal**. If ARP auto-detect can't produce an armature, we fall
back to exporting the *unrigged* mesh in both formats and print
``[RIG_RESULT] rig_status=manual`` so an artist can finish it. A successful
auto-rig prints ``[RIG_RESULT] rig_status=auto``.

Folds in the two artist-shared scripts (arp_smart_automation_v2.py +
arp_export_automation.py), with interactive dialogs removed for headless and the
mesh pre-flight relaxed (generated meshes are frequently non-manifold — we clean
and warn rather than hard-stop).

Architecture (virtual desktop, timer, quit machinery) is unchanged from the
proven humanoid pipeline — see the timer/quit notes below.
"""

from __future__ import annotations

import sys
import os
import argparse
import atexit

import bpy
import bmesh
import addon_utils
import mathutils

# ── Parse CLI args (must happen at module level before timer) ─────────────────
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_parser = argparse.ArgumentParser()
_parser.add_argument("--input",          required=True)
_parser.add_argument("--output",         required=True)
_parser.add_argument("--output_fbx",     default="")
_parser.add_argument("--character_type", default="humanoid",
                     choices=["humanoid", "quadruped", "bird", "fish", "other"])
_parser.add_argument("--morphology",     default="B1_humanoid")
_parser.add_argument("--scale",          type=float, default=1.0)
_args = _parser.parse_args(_argv)

INPUT_GLB      = _args.input
OUTPUT_GLB     = _args.output
OUTPUT_FBX     = _args.output_fbx
CHARACTER_TYPE = _args.character_type
MORPHOLOGY     = _args.morphology
CHAR_SCALE     = _args.scale

ARP_ADDON_NAME = os.getenv("ARP_ADDON_NAME", "auto_rig_pro-master")

# ARP "Rig" FBX export type: HUMANOID only for humanoid-torso morphologies.
_HUMANOID_MORPHS = {"B1_humanoid", "B2_centaur", "B3_naga"}
FBX_RIG_TYPE = "HUMANOID" if MORPHOLOGY in _HUMANOID_MORPHS else "UNIVERSAL"

DEFORM_COLLECTION_NAMES = ["deform", "def", "deform bones", "deform_bones"]

# ARP FBX export settings (source-verified from auto_rig_ge.py — see the artist's
# arp_export_automation.py). Rig-type is morphology-dependent.
EXPORT_SETTINGS = {
    "arp_engine_type":        "OTHERS",
    "arp_export_rig_type":    FBX_RIG_TYPE,
    "arp_export_show_panels": "RIG",
    "arp_ge_sel_only":        True,
    "arp_ge_sel_bones_only":  False,
    "arp_export_twist":       False,
    "arp_show_ge_advanced":   False,
    "arp_full_facial":        False,
    "arp_ge_export_metacarp": False,
    "arp_export_noparent":    False,
    "arp_export_rig_name":    "root",
    "arp_units_x100":         True,
    "arp_ue_root_motion":     False,
}

print(f"[ARP] Input:      {INPUT_GLB}")
print(f"[ARP] Output GLB: {OUTPUT_GLB}")
print(f"[ARP] Output FBX: {OUTPUT_FBX or '(none)'}")
print(f"[ARP] Type:       {CHARACTER_TYPE}  morphology={MORPHOLOGY}  fbx_rig_type={FBX_RIG_TYPE}")


# ── Quit helper ───────────────────────────────────────────────────────────────
_exit_code = [0]

def _quit(code: int = 0) -> None:
    """Exit Blender cleanly. Without --background Blender would hang otherwise."""
    _exit_code[0] = code
    atexit.unregister(_atexit_quit)
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass
    sys.exit(code)


def _atexit_quit() -> None:
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass


atexit.register(_atexit_quit)


# ── Step 0: Disable ARP startup handlers (runs at import time, before UI) ────
print("[ARP] Disabling ARP startup handlers ...")
for _key in list(bpy.context.preferences.addons.keys()):
    if "auto_rig_pro" in _key:
        try:
            addon_utils.disable(_key, default_set=False)
            print(f"[ARP]   disabled: {_key}")
        except Exception as _e:
            print(f"[ARP]   disable warning: {_e}")


# ── VIEW_3D context helper ────────────────────────────────────────────────────
def _view3d_ctx() -> dict:
    wm     = bpy.context.window_manager
    window = wm.windows[0]
    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    return {"window": window, "area": area, "region": region}
    area = window.screen.areas[0]
    old_type  = area.type
    area.type = "VIEW_3D"
    region    = next((r for r in area.regions if r.type == "WINDOW"), area.regions[0])
    print(f"[ARP] WARNING: no VIEW_3D found — converted area[0] from {old_type}")
    return {"window": window, "area": area, "region": region}


# ── Mesh pre-flight (relaxed for generated meshes) ────────────────────────────
def _clean_mesh(char_mesh) -> None:
    """Auto-clean the mesh before rigging. Unlike the artist's interactive script
    (which hard-stops), generated meshes are routinely non-manifold and never
    have prior vertex groups, so we CLEAR groups and WARN on non-manifold +
    merge-by-distance rather than aborting (no-fail contract)."""
    if len(char_mesh.vertex_groups) > 0:
        n = len(char_mesh.vertex_groups)
        char_mesh.vertex_groups.clear()
        print(f"[ARP] Cleared {n} pre-existing vertex group(s) before rigging.")

    bpy.ops.object.select_all(action="DESELECT")
    char_mesh.select_set(True)
    bpy.context.view_layer.objects.active = char_mesh

    # Merge-by-distance to close tiny gaps that break voxel binding.
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    bm = bmesh.new()
    bm.from_mesh(char_mesh.data)
    bm.edges.ensure_lookup_table()
    non_manifold = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    if non_manifold:
        print(f"[ARP] WARNING: {non_manifold} non-manifold edge(s) on "
              f"'{char_mesh.name}' — continuing (voxel bind tolerates this).")
    else:
        print("[ARP] Mesh is manifold ✓")


# ── Smart rig (humanoid detector; attempted for all morphologies) ─────────────
def _smart_rig(char_mesh, ctx):
    """Run the ARP Smart pipeline. Returns the armature on success, raises on
    failure. ARP Smart is humanoid-trained; for B4..B7 this is a best-effort
    attempt (the caller treats a raise as the manual fallback)."""
    scn = bpy.context.scene
    scn.arp_smart_type            = "BODY"
    scn.arp_smart_sym             = True
    scn.arp_smart_fingers_engine  = "AI"
    scn.arp_fingers_to_detect     = 0

    bpy.ops.object.select_all(action="DESELECT")
    char_mesh.select_set(True)
    bpy.context.view_layer.objects.active = char_mesh

    print("[ARP] id.get_selected_objects ...")
    with bpy.context.temp_override(**ctx):
        bpy.ops.id.get_selected_objects("EXEC_DEFAULT")
    bpy.context.view_layer.update()

    body_temp = bpy.data.objects.get("body_temp")
    if body_temp is None:
        raise RuntimeError("body_temp not created by get_selected_objects")

    bpy.ops.object.select_all(action="DESELECT")
    body_temp.select_set(True)
    bpy.context.view_layer.objects.active = body_temp

    print("[ARP] arp.guess_markers (AI inference) ...")
    with bpy.context.temp_override(**ctx):
        bpy.ops.arp.guess_markers()
    bpy.context.view_layer.update()

    # Patch the results popup to a no-op before go_detect.
    for mod_key in (f"{ARP_ADDON_NAME}.src.auto_rig",
                    f"{ARP_ADDON_NAME}.src.auto_rig_smart"):
        mod = sys.modules.get(mod_key)
        if mod and hasattr(mod, "display_popup_message"):
            mod.display_popup_message = lambda *a, **kw: None

    body_temp = bpy.data.objects.get("body_temp") or char_mesh
    bpy.ops.object.select_all(action="DESELECT")
    body_temp.select_set(True)
    bpy.context.view_layer.objects.active = body_temp
    bpy.context.view_layer.update()

    print("[ARP] id.go_detect ...")
    with bpy.context.temp_override(**ctx):
        bpy.ops.id.go_detect("EXEC_DEFAULT")
    bpy.context.view_layer.update()

    armature = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)
    if armature is None:
        raise RuntimeError("go_detect produced no armature")
    print(f"[ARP] Armature created: {armature.name}")

    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception:
        pass

    # ── Bind (voxel → heat map → envelope fallbacks) ──────────────────────────
    char_mesh = bpy.data.objects.get("Character")
    bpy.ops.object.select_all(action="DESELECT")
    char_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    scn.arp_bind_engine = "PSEUDO_VOXELS"
    try:
        bpy.ops.arp.bind_to_rig("EXEC_DEFAULT")
        print("[ARP] bound via PSEUDO_VOXELS")
    except Exception as e:
        print(f"[ARP] PSEUDO_VOXELS failed ({e}); HEAT_MAP ...")
        try:
            scn.arp_bind_engine = "HEAT_MAP"
            bpy.ops.arp.bind_to_rig("EXEC_DEFAULT")
            print("[ARP] bound via HEAT_MAP")
        except Exception as e2:
            print(f"[ARP] HEAT_MAP failed ({e2}); ARMATURE_AUTO weights ...")
            bpy.ops.object.parent_set(type="ARMATURE_AUTO")
            print("[ARP] bound via Automatic Weights (fallback)")
    return armature


# ── Deform isolation (ported from arp_export_automation.py) ───────────────────
def _isolate_deform(armature) -> None:
    """Make only the Deform bone collection / layer visible so exports carry the
    deform skeleton only (no ARP control/reference/mechanism bones)."""
    arm = armature.data
    if hasattr(arm, "collections"):
        deform = [c for c in arm.collections
                  if any(k in c.name.lower() for k in DEFORM_COLLECTION_NAMES)]
        if deform:
            names = {c.name for c in deform}
            for c in arm.collections:
                c.is_visible = c.name in names
            print(f"[ARP] Deform isolation: visible={sorted(names)}")
        else:
            print("[ARP] No named Deform collection — relying on export_def_bones.")
    elif hasattr(arm, "layers"):
        arm.layers = [i == 29 for i in range(32)]  # ARP deform = layer 29 (3.x)
        print("[ARP] Deform isolation: layer 29 only (Blender 3.x)")


# ── Exports ───────────────────────────────────────────────────────────────────
def _export_glb(armature) -> None:
    out_dir = os.path.dirname(OUTPUT_GLB)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if armature is not None:
        _isolate_deform(armature)
    print(f"[ARP] Exporting GLB: {OUTPUT_GLB}")
    bpy.ops.export_scene.gltf(
        filepath          = OUTPUT_GLB,
        export_format     = "GLB",
        use_selection     = False,
        export_apply      = True,
        export_animations = True,
        export_skins      = True,
        export_morph      = True,
        export_lights     = False,
        export_cameras    = False,
        export_yup        = True,
        export_def_bones  = True,
    )
    if not os.path.exists(OUTPUT_GLB):
        raise RuntimeError(f"GLB export ran but file not found: {OUTPUT_GLB}")
    print(f"[ARP] GLB OK — {os.path.getsize(OUTPUT_GLB)/1e6:.2f} MB")


def _export_fbx(armature, char_mesh) -> None:
    """Engine FBX. With a rig → ARP-native arp_export_fbx_panel (clean deform
    skeleton, root bone, x100). Without a rig → plain mesh FBX."""
    if not OUTPUT_FBX:
        return
    out_dir = os.path.dirname(OUTPUT_FBX)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if armature is None or not hasattr(bpy.ops.arp, "arp_export_fbx_panel"):
        # Manual fallback (or ARP GE export unavailable): plain mesh FBX.
        bpy.ops.object.select_all(action="DESELECT")
        char_mesh.select_set(True)
        bpy.context.view_layer.objects.active = char_mesh
        print(f"[ARP] Exporting plain FBX (no rig): {OUTPUT_FBX}")
        bpy.ops.export_scene.fbx(filepath=OUTPUT_FBX, use_selection=True,
                                 apply_unit_scale=True, global_scale=1.0)
    else:
        scn = bpy.context.scene
        for k, v in EXPORT_SETTINGS.items():
            if hasattr(scn, k):
                try:
                    setattr(scn, k, v)
                except Exception as e:
                    print(f"[ARP] export setting {k} failed: {e}")
        bpy.ops.object.select_all(action="DESELECT")
        char_mesh.select_set(True)
        armature.select_set(True)
        bpy.context.view_layer.objects.active = armature
        if hasattr(scn, "arp_ge_fp"):
            scn.arp_ge_fp = OUTPUT_FBX
        print(f"[ARP] Exporting ARP FBX (rig-type={FBX_RIG_TYPE}): {OUTPUT_FBX}")
        try:
            res = bpy.ops.arp.arp_export_fbx_panel(
                "EXEC_DEFAULT", filepath=OUTPUT_FBX, quick_export=True)
            print(f"[ARP] arp_export_fbx_panel → {res}")
        except Exception as e:
            print(f"[ARP] ARP FBX export failed ({e}) — plain FBX fallback")
            bpy.ops.export_scene.fbx(filepath=OUTPUT_FBX, use_selection=True)

    if os.path.exists(OUTPUT_FBX):
        print(f"[ARP] FBX OK — {os.path.getsize(OUTPUT_FBX)/1e6:.2f} MB")
    else:
        print(f"[ARP] WARNING: FBX not found after export: {OUTPUT_FBX}")


def _strip_armatures() -> None:
    """Manual fallback: delete any partial armature so we export the bare mesh."""
    for obj in [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]:
        bpy.data.objects.remove(obj, do_unlink=True)
    mesh = bpy.data.objects.get("Character")
    if mesh:
        for m in [md for md in mesh.modifiers if md.type == "ARMATURE"]:
            mesh.modifiers.remove(m)


# ── Main pipeline ─────────────────────────────────────────────────────────────
def _run_pipeline() -> None:
    print("[ARP] Timer fired — running pipeline ...")
    try:
        _pipeline()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ARP] PIPELINE ERROR: {exc}")
        _quit(1)
    return None


def _pipeline() -> None:
    # ── 1. Clear scene ────────────────────────────────────────────────────────
    print("[ARP] Clearing scene ...")
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=True)
    for block in list(bpy.data.meshes):
        bpy.data.meshes.remove(block)

    # ── 2. Import GLB ─────────────────────────────────────────────────────────
    print(f"[ARP] Importing: {INPUT_GLB}")
    bpy.ops.import_scene.gltf(filepath=INPUT_GLB)
    mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not mesh_objs:
        raise RuntimeError(f"No mesh objects after importing {INPUT_GLB}")
    print(f"[ARP] Imported {len(mesh_objs)} mesh(es)")

    # ── 3. Merge into single mesh ─────────────────────────────────────────────
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    char_mesh      = bpy.context.active_object
    char_mesh.name = "Character"

    # ── 4. Normalise scale + apply transforms ─────────────────────────────────
    bpy.ops.object.select_all(action="DESELECT")
    char_mesh.select_set(True)
    bpy.context.view_layer.objects.active = char_mesh
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    bbox   = [char_mesh.matrix_world @ mathutils.Vector(v) for v in char_mesh.bound_box]
    height = max(v.z for v in bbox) - min(v.z for v in bbox)
    if height > 0.01 and CHAR_SCALE == 1.0:
        sf = 2.0 / height
        char_mesh.scale = (sf, sf, sf)
        bpy.ops.object.transform_apply(scale=True)
        print(f"[ARP] Auto-scaled: {height:.3f}m → 2.0m")
    elif CHAR_SCALE != 1.0:
        char_mesh.scale = (CHAR_SCALE,) * 3
        bpy.ops.object.transform_apply(scale=True)

    bbox  = [char_mesh.matrix_world @ mathutils.Vector(v) for v in char_mesh.bound_box]
    char_mesh.location.z -= min(v.z for v in bbox)
    bpy.ops.object.transform_apply(location=True)

    # ── 5. Mesh pre-flight ────────────────────────────────────────────────────
    _clean_mesh(char_mesh)

    # ── 6. Enable ARP + patch AI path ─────────────────────────────────────────
    print(f"[ARP] Enabling addon: {ARP_ADDON_NAME} ...")
    addon_utils.enable(ARP_ADDON_NAME, default_set=False, persistent=False)
    bpy.context.view_layer.update()
    if not hasattr(bpy.ops, "arp"):
        raise RuntimeError(f"ARP operators unavailable after enabling {ARP_ADDON_NAME!r}.")

    _ai_path = os.getenv("ARP_AI_PATH", os.path.expanduser("~/Documents/AI"))
    _smart_mod = sys.modules.get(f"{ARP_ADDON_NAME}.src.auto_rig_smart")
    if _smart_mod and hasattr(_smart_mod, "get_AI_path"):
        def _patched_get_ai_path(root_dir: bool = False) -> str:
            return (_ai_path + "/") if root_dir else os.path.join(_ai_path, "inference")
        _smart_mod.get_AI_path = _patched_get_ai_path
        print(f"[ARP] Patched get_AI_path → {os.path.join(_ai_path, 'inference')}")

    ctx = _view3d_ctx()

    # ── 7. Attempt auto-rig; manual fallback on any failure ───────────────────
    rig_status = "manual"
    armature   = None
    try:
        armature   = _smart_rig(char_mesh, ctx)
        rig_status = "auto"
        print(f"[ARP] Auto-rig succeeded (morphology={MORPHOLOGY}).")
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ARP] Auto-rig failed ({exc}) — MANUAL fallback, exporting "
              f"unrigged mesh in both formats.")
        _strip_armatures()
        armature = None

    char_mesh = bpy.data.objects.get("Character")

    # ── 8. Dual export ────────────────────────────────────────────────────────
    _export_glb(armature)
    _export_fbx(armature, char_mesh)

    # Machine-readable result marker parsed by rig_model.py.
    print(f"[RIG_RESULT] rig_status={rig_status} glb={OUTPUT_GLB} fbx={OUTPUT_FBX or ''}")
    print(f"[ARP] RIGGING_COMPLETE: {OUTPUT_GLB}")
    _quit(0)


# ── Register timer — fires 1 second after Blender startup is complete ─────────
print("[ARP] Registering pipeline timer (1.0s) ...")
bpy.app.timers.register(_run_pipeline, first_interval=1.0)

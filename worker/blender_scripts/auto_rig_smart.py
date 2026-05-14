"""
Auto-Rig Pro Smart Rigging — Blender desktop script
====================================================
Called by rig_model.py via blender_desktop_run.sh:

    bash blender_desktop_run.sh blender --python auto_rig_smart.py -- \\
         --input  /tmp/char.glb \\
         --output /tmp/char_rigged.glb \\
         --character_type humanoid

Architecture
------------
Blender is launched inside a virtual desktop (Xvfb + fluxbox WM) by the
wrapper script.  This means Blender's default workspace fully initialises
before our code runs — a real VIEW_3D area exists and is managed by the WM.

Our pipeline code is registered as a bpy.app.timers callback (1 second delay).
By the time the timer fires:
  - Blender's startup is complete
  - The default Layout workspace is loaded
  - VIEW_3D area is the active area with all regions ready
  - ARP's view3d.* operators poll correctly

Blender calls bpy.ops.wm.quit_blender() at the end (or on error), which lets
the wrapper script tear down Xvfb + fluxbox cleanly.

Pipeline steps
--------------
  0. (at import time) Disable ARP startup handlers → prevents depsgraph SIGSEGV
  1. Register 1-second timer → return, let Blender finish startup
  --- timer fires ---
  2. Clear scene
  3. Import GLB
  4. Merge meshes → single "Character" object
  5. Normalise scale (auto-scale to 2m height) + apply transforms
  6. Enable ARP addon
  7. Set ARP scene params (BODY type, symmetry, AI fingers)
  8. id.get_selected_objects  → creates body_temp (ARP's working copy)
  9. arp.guess_markers         → AI screenshots + inference → place markers
  10. id.go_detect             → build armature from markers
  11. arp.bind_to_rig          → voxel skinning (best quality)
  12. Export GLB
  13. bpy.ops.wm.quit_blender()
"""

from __future__ import annotations

import sys
import os
import argparse
import atexit

import bpy
import addon_utils
import mathutils

# ── Parse CLI args (must happen at module level before timer) ─────────────────
_argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
_parser = argparse.ArgumentParser()
_parser.add_argument("--input",          required=True)
_parser.add_argument("--output",         required=True)
_parser.add_argument("--character_type", default="humanoid",
                     choices=["humanoid", "quadruped", "bird", "fish", "other"])
_parser.add_argument("--scale",          type=float, default=1.0)
_args = _parser.parse_args(_argv)

INPUT_GLB      = _args.input
OUTPUT_GLB     = _args.output
CHARACTER_TYPE = _args.character_type
CHAR_SCALE     = _args.scale

ARP_ADDON_NAME = os.getenv("ARP_ADDON_NAME", "auto_rig_pro-master")

print(f"[ARP] Input:  {INPUT_GLB}")
print(f"[ARP] Output: {OUTPUT_GLB}")
print(f"[ARP] Type:   {CHARACTER_TYPE}")


# ── Quit helper ───────────────────────────────────────────────────────────────
_exit_code = [0]   # mutable box so atexit can read the final code

def _quit(code: int = 0) -> None:
    """Exit Blender cleanly. Without --background Blender would hang otherwise."""
    _exit_code[0] = code
    # Unregister the atexit handler FIRST so it doesn't fire a second
    # quit_blender() call during Python finalization — that double-call
    # causes a SIGSEGV in PyObject_GC_UnTrack (Blender already shutting down).
    atexit.unregister(_atexit_quit)
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass
    sys.exit(code)   # fallback if quit_blender() hasn't fired yet


def _atexit_quit() -> None:
    """Guarantee Blender quits if the pipeline crashes before _quit() is called."""
    try:
        bpy.ops.wm.quit_blender()
    except Exception:
        pass


atexit.register(_atexit_quit)


# ── Step 0: Disable ARP startup handlers (runs at import time, before UI) ────
# ARP saves itself to user prefs (default_set=True) so Blender auto-enables it.
# Its depsgraph_update_post handlers fire during GLB import and try to access
# bpy.context.space_data — which is None while the UI is still starting up —
# causing a SIGSEGV.  Disable the addon NOW (before we touch the scene) so the
# handlers are removed.  We re-enable it after the timer fires.
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
    """
    Return a dict suitable for bpy.context.temp_override(…) that points at the
    VIEW_3D area + WINDOW region.  Belt-and-suspenders alongside the fluxbox WM:
    even though the desktop session means ARP's operators usually have correct
    context, we keep this wrapper to handle any edge cases in Blender 4.4 where
    timer callbacks still have a null area.
    """
    wm     = bpy.context.window_manager
    window = wm.windows[0]

    for area in window.screen.areas:
        if area.type == "VIEW_3D":
            for region in area.regions:
                if region.type == "WINDOW":
                    return {"window": window, "area": area, "region": region}

    # Fallback: convert the first available area to VIEW_3D temporarily.
    # This should never happen when fluxbox is running, but is a safety net.
    area = window.screen.areas[0]
    old_type  = area.type
    area.type = "VIEW_3D"
    region    = next((r for r in area.regions if r.type == "WINDOW"), area.regions[0])
    print(f"[ARP] WARNING: no VIEW_3D found — converted area[0] from {old_type}")
    return {"window": window, "area": area, "region": region}


# ── Main pipeline (runs inside the timer, after full Blender startup) ─────────
def _run_pipeline() -> None:
    """Full ARP rigging pipeline.  Called by the timer 1 second after startup."""

    print("[ARP] Timer fired — Blender startup complete, running pipeline ...")

    try:
        _pipeline()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[ARP] PIPELINE ERROR: {exc}")
        _quit(1)

    # Return None from the timer callback so it does NOT repeat
    return None


def _pipeline() -> None:

    # ── 1. Clear default scene ────────────────────────────────────────────────
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
    print(f"[ARP] Imported {len(mesh_objs)} mesh(es): {[o.name for o in mesh_objs]}")

    # ── 3. Merge into single mesh ─────────────────────────────────────────────
    bpy.ops.object.select_all(action="DESELECT")
    for obj in mesh_objs:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = mesh_objs[0]
    if len(mesh_objs) > 1:
        bpy.ops.object.join()
    char_mesh      = bpy.context.active_object
    char_mesh.name = "Character"
    print(f"[ARP] Character mesh: {char_mesh.name}  verts={len(char_mesh.data.vertices)}")

    # ── 4. Normalise scale + apply all transforms ─────────────────────────────
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
        print(f"[ARP] Auto-scaled: {height:.3f}m → 2.0m (×{sf:.3f})")
    elif CHAR_SCALE != 1.0:
        char_mesh.scale = (CHAR_SCALE,) * 3
        bpy.ops.object.transform_apply(scale=True)

    # Place origin at ground (min Z = 0)
    bbox  = [char_mesh.matrix_world @ mathutils.Vector(v) for v in char_mesh.bound_box]
    min_z = min(v.z for v in bbox)
    char_mesh.location.z -= min_z
    bpy.ops.object.transform_apply(location=True)

    # ── 5. Enable ARP ─────────────────────────────────────────────────────────
    print(f"[ARP] Enabling addon: {ARP_ADDON_NAME} ...")
    addon_utils.enable(ARP_ADDON_NAME, default_set=False, persistent=False)
    bpy.context.view_layer.update()

    if not hasattr(bpy.ops, "arp"):
        raise RuntimeError(f"ARP operators unavailable after enabling {ARP_ADDON_NAME!r}.")
    print("[ARP] ARP operators ready.")

    # ── 5b. Patch get_AI_path in ARP's smart module ───────────────────────────
    # addon_utils.enable(..., default_set=False) does NOT add an entry to
    # bpy.context.preferences.addons, so ARP's get_prefs() always falls back to
    # _HeadlessFallback which returns '' for ai_presets_path.  get_AI_path()
    # then returns '/inference' (root fs) and screenshot saves fail.
    #
    # Fix: replace get_AI_path entirely in the ARP smart module so it returns
    # the correct path regardless of preferences state.
    _ai_path = os.getenv(
        "ARP_AI_PATH",
        os.path.expanduser("~/Documents/AI")
    )
    _smart_mod = sys.modules.get(f"{ARP_ADDON_NAME}.src.auto_rig_smart")
    if _smart_mod and hasattr(_smart_mod, "get_AI_path"):
        def _patched_get_ai_path(root_dir: bool = False) -> str:
            if root_dir:
                return _ai_path + "/"
            return os.path.join(_ai_path, "inference")
        _smart_mod.get_AI_path = _patched_get_ai_path
        print(f"[ARP] Patched get_AI_path → {os.path.join(_ai_path, 'inference')}")
    else:
        print(f"[ARP] WARNING: could not patch get_AI_path (module not found) — "
              f"check ARP_ADDON_NAME={ARP_ADDON_NAME!r}")

    # ── 6. Configure ARP scene settings ───────────────────────────────────────
    scn = bpy.context.scene
    scn.arp_smart_type            = "BODY"
    scn.arp_smart_sym             = True
    scn.arp_smart_fingers_engine  = "AI"
    scn.arp_fingers_to_detect     = 0      # auto-detect
    print(f"[ARP] Scene: type=BODY  sym=True  fingers=AI")

    # Select Character, make active
    bpy.ops.object.select_all(action="DESELECT")
    char_mesh.select_set(True)
    bpy.context.view_layer.objects.active = char_mesh

    # Get VIEW_3D context (belt-and-suspenders alongside fluxbox WM)
    ctx = _view3d_ctx()
    print(f"[ARP] VIEW_3D context: area={ctx['area'].type}  region={ctx['region'].type}")

    # ── 7. id.get_selected_objects ────────────────────────────────────────────
    # Creates body_temp (a duplicate of Character) which ARP uses as its working
    # copy.  Must run before guess_markers.  Internally calls view3d.view_axis
    # and view3d.view_selected — hence the temp_override.
    print("[ARP] Running id.get_selected_objects ...")
    with bpy.context.temp_override(**ctx):
        result = bpy.ops.id.get_selected_objects("EXEC_DEFAULT")
    print(f"[ARP] get_selected_objects → {result}")
    bpy.context.view_layer.update()

    body_temp = bpy.data.objects.get("body_temp")
    if body_temp is None:
        raise RuntimeError("body_temp not created by get_selected_objects — ARP failed")

    bpy.ops.object.select_all(action="DESELECT")
    body_temp.select_set(True)
    bpy.context.view_layer.objects.active = body_temp

    # ── 8. arp.guess_markers (AI inference) ───────────────────────────────────
    # Renders the mesh from front/side/top, runs ELF inference binaries, places
    # landmark markers.  Needs VIEW_3D for render.opengl and camera view setup.
    print("[ARP] Running arp.guess_markers (AI inference) ...")
    with bpy.context.temp_override(**ctx):
        result = bpy.ops.arp.guess_markers()
    print(f"[ARP] guess_markers → {result}")
    bpy.context.view_layer.update()

    # ── 9. Patch display_popup_message before go_detect ───────────────────────
    # go_detect calls display_popup_message to show a results dialog.
    # Even with a real window this can cause issues in automated runs; patch it
    # to a no-op (the armature is already built before the popup fires).
    for mod_key in [f"{ARP_ADDON_NAME}.src.auto_rig",
                    f"{ARP_ADDON_NAME}.src.auto_rig_smart"]:
        mod = sys.modules.get(mod_key)
        if mod and hasattr(mod, "display_popup_message"):
            mod.display_popup_message = lambda *a, **kw: None
            print(f"[ARP] Patched display_popup_message → no-op in {mod_key}")

    # ── 10. id.go_detect ──────────────────────────────────────────────────────
    # Reads the placed markers, generates the ARP armature.
    body_temp = bpy.data.objects.get("body_temp") or char_mesh
    bpy.ops.object.select_all(action="DESELECT")
    body_temp.select_set(True)
    bpy.context.view_layer.objects.active = body_temp
    bpy.context.view_layer.update()

    print("[ARP] Running id.go_detect ...")
    with bpy.context.temp_override(**ctx):
        result = bpy.ops.id.go_detect("EXEC_DEFAULT")
    print(f"[ARP] go_detect → {result}")
    bpy.context.view_layer.update()

    # ── 11. Verify armature ───────────────────────────────────────────────────
    armature_objs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if not armature_objs:
        raise RuntimeError(
            "go_detect did not create an armature — detection failed. "
            "Check Blender log for ARP warnings above."
        )
    armature = armature_objs[0]
    print(f"[ARP] Armature created: {armature.name}")

    # Return to OBJECT mode (go_detect can leave us in POSE mode)
    try:
        bpy.ops.object.mode_set(mode="OBJECT")
    except Exception as e:
        print(f"[ARP] mode_set warning (ignored): {e}")

    # ── 12. Bind mesh to armature (voxel skinning) ────────────────────────────
    # ARP's voxel binding is the highest-quality option and is stable headless.
    # Falls back to ARMATURE_ENVELOPE if arp.bind_to_rig is unavailable.
    bpy.ops.object.select_all(action="DESELECT")
    char_mesh = bpy.data.objects.get("Character")
    armature  = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)

    if char_mesh is None:
        raise RuntimeError("Character mesh not found after go_detect")
    if armature is None:
        raise RuntimeError("No armature found after go_detect")

    char_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature

    # Use ARP's built-in PSEUDO_VOXELS engine: volume-based skinning that needs
    # no external addons and is more consistent than Heat Maps on complex meshes.
    # Call with EXEC_DEFAULT to skip the interactive dialog/warnings popup.
    scn.arp_bind_engine = "PSEUDO_VOXELS"
    try:
        result = bpy.ops.arp.bind_to_rig("EXEC_DEFAULT")
        print(f"[ARP] arp.bind_to_rig PSEUDO_VOXELS → {result}")
    except Exception as e:
        print(f"[ARP] arp.bind_to_rig PSEUDO_VOXELS failed ({e}), trying HEAT_MAP ...")
        try:
            scn.arp_bind_engine = "HEAT_MAP"
            result = bpy.ops.arp.bind_to_rig("EXEC_DEFAULT")
            print(f"[ARP] arp.bind_to_rig HEAT_MAP → {result}")
        except Exception as e2:
            print(f"[ARP] arp.bind_to_rig HEAT_MAP also failed ({e2}), "
                  f"falling back to basic ARMATURE_ENVELOPE")
            bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
            print("[ARP] Bound via ARMATURE_ENVELOPE (last-resort fallback)")

    print("[ARP] Mesh bound to armature OK.")

    # ── 13. Export GLB ────────────────────────────────────────────────────────
    out_dir = os.path.dirname(OUTPUT_GLB)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[ARP] Exporting: {OUTPUT_GLB}")
    bpy.ops.export_scene.gltf(
        filepath        = OUTPUT_GLB,
        export_format   = "GLB",
        use_selection   = False,
        export_apply    = True,
        export_animations = True,
        export_skins    = True,
        export_morph    = True,
        export_lights   = False,
        export_cameras  = False,
        export_yup      = True,
    )

    if not os.path.exists(OUTPUT_GLB):
        raise RuntimeError(f"Export operator ran but GLB not found: {OUTPUT_GLB}")

    size_mb = os.path.getsize(OUTPUT_GLB) / 1e6
    print(f"[ARP] Export OK — {size_mb:.2f} MB")
    print(f"[ARP] RIGGING_COMPLETE: {OUTPUT_GLB}")

    _quit(0)


# ── Register timer — fires 1 second after Blender startup is complete ─────────
# At this point Blender's default workspace is loaded, VIEW_3D area exists,
# fluxbox has focused the window, and all regions are initialised.
print("[ARP] Registering pipeline timer (1.0s) ...")
bpy.app.timers.register(_run_pipeline, first_interval=1.0)

"""
Auto-Rig Pro Smart Rigging — Blender headless script
=====================================================
Called by rig_cpu_worker.py via subprocess:
  blender --background --python worker/blender_scripts/auto_rig_smart.py -- \
          --input /tmp/char.glb \
          --output /tmp/char_rigged.glb \
          --character_type humanoid

Uses Auto-Rig Pro 3.76+ Smart detection with programmatically placed markers.
Markers are placed at mesh extremities (outside the mesh surface) so ARP's
BVHTree raycasting can find the mesh boundaries from each marker position.

Note: ARP addon files must be patched for headless mode (bpy.context.space_data
and bpy.context.area guard-patched). See deploy_and_run.sh for patch commands.
"""

import sys
import os
import argparse

import bpy
import addon_utils
import mathutils
import bmesh

# ── Parse args ────────────────────────────────────────────────────────────────
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--input",          required=True)
parser.add_argument("--output",         required=True)
parser.add_argument("--character_type", default="humanoid",
                    choices=["humanoid", "quadruped", "bird", "fish", "other"])
parser.add_argument("--scale",          type=float, default=1.0)
args = parser.parse_args(argv)

INPUT_GLB      = args.input
OUTPUT_GLB     = args.output
CHARACTER_TYPE = args.character_type
CHAR_SCALE     = args.scale

print(f"[ARP] Input:  {INPUT_GLB}")
print(f"[ARP] Output: {OUTPUT_GLB}")
print(f"[ARP] Type:   {CHARACTER_TYPE}")


# ── 0. Disable ARP if already active (from saved user prefs) ─────────────────
# ARP was previously saved to user prefs (default_set=True) so Blender
# auto-enables it at startup.  Its depsgraph_update_post handlers then fire
# during GLB import, try bpy.context.space_data (None headless) → SIGSEGV.
# Disable it now so its handlers are gone before we import; the prefs entry
# it created at startup still exists so re-enabling later will succeed.
print("[ARP] Disabling ARP handlers (if active from startup prefs)...")
for _key in list(bpy.context.preferences.addons.keys()):
    if "auto_rig_pro" in _key:
        try:
            addon_utils.disable(_key, default_set=False)
            print(f"[ARP]   disabled addon: {_key}")
        except Exception as _e:
            print(f"[ARP]   disable warning: {_e}")


# ── 1. Clean scene ────────────────────────────────────────────────────────────
print("[ARP] Clearing scene...")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=True)
for block in bpy.data.meshes:
    bpy.data.meshes.remove(block)


# ── 2. Import GLB ─────────────────────────────────────────────────────────────
print(f"[ARP] Importing: {INPUT_GLB}")
bpy.ops.import_scene.gltf(filepath=INPUT_GLB)

mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError(f"No mesh objects after importing {INPUT_GLB}")
print(f"[ARP] Imported {len(mesh_objs)} mesh object(s): {[o.name for o in mesh_objs]}")


# ── 3. Merge meshes ───────────────────────────────────────────────────────────
bpy.ops.object.select_all(action="DESELECT")
for obj in mesh_objs:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objs[0]
if len(mesh_objs) > 1:
    bpy.ops.object.join()
char_mesh = bpy.context.active_object
char_mesh.name = "Character"


# ── 4. Normalize scale + origin ───────────────────────────────────────────────
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
    print(f"[ARP] Auto-scaled: {height:.3f} → 2.0 (factor {sf:.3f})")
elif CHAR_SCALE != 1.0:
    char_mesh.scale = (CHAR_SCALE, CHAR_SCALE, CHAR_SCALE)
    bpy.ops.object.transform_apply(scale=True)

bbox  = [char_mesh.matrix_world @ mathutils.Vector(v) for v in char_mesh.bound_box]
min_z = min(v.z for v in bbox)
char_mesh.location.z -= min_z
bpy.ops.object.transform_apply(location=True)


# ── 5. Analyse mesh vertices to find extremity positions ─────────────────────
print("[ARP] Analysing mesh to find body landmarks...")

verts = [char_mesh.matrix_world @ v.co for v in char_mesh.data.vertices]

H      = max(v.z for v in verts)
min_z  = min(v.z for v in verts)
max_x  = max(v.x for v in verts)
min_x  = min(v.x for v in verts)
max_y  = max(v.y for v in verts)
min_y  = min(v.y for v in verts)
cx     = (max_x + min_x) / 2
cy     = (max_y + min_y) / 2
W      = max_x - min_x
print(f"[ARP] Bounds: H={H:.3f}  W={W:.3f}  X=[{min_x:.3f},{max_x:.3f}]  cy={cy:.3f}")

# ── Find arm tip positions ────────────────────────────────────────────────
# ARP fires Y-axis rays from the marker's (X, Z) position to find mesh depth.
# So the marker must be at an (X, Z) where mesh geometry actually exists.
# Strategy: find verts near the X extreme (within 15% of W from the tip),
# excluding the head zone (Z > 82% of H) to avoid sphere head skewing Z.

arm_zone_max_z = H * 0.82
left_tip_verts  = [v for v in verts if v.x > max_x - W * 0.15 and v.z < arm_zone_max_z]
right_tip_verts = [v for v in verts if v.x < min_x + W * 0.15 and v.z < arm_zone_max_z]

if left_tip_verts:
    # Use median Z of the cluster to avoid outliers
    zs = sorted(v.z for v in left_tip_verts)
    hand_l_z = zs[len(zs) // 2]
    hand_l_x = max_x - W * 0.03   # slightly inside the surface
else:
    hand_l_z = H * 0.72
    hand_l_x = max_x - W * 0.03

if right_tip_verts:
    zs = sorted(v.z for v in right_tip_verts)
    hand_r_z = zs[len(zs) // 2]
    hand_r_x = min_x + W * 0.03
else:
    hand_r_z = H * 0.72
    hand_r_x = min_x + W * 0.03

hand_z = (hand_l_z + hand_r_z) / 2
print(f"[ARP] Arm tip: hand_z={hand_z:.3f}  left_x={hand_l_x:.3f}  right_x={hand_r_x:.3f}")

# ── Find shoulder ──────────────────────────────────────────────────────────
# Shoulder = upper body Z, at moderate X (where torso meets arm)
upper_verts = [v for v in verts if v.z > H * 0.65 and v.z < arm_zone_max_z]
if upper_verts:
    # Find the widest X extent in upper body, placed slightly inward from max
    shoulder_max_x = max(v.x for v in upper_verts)
    shoulder_l_x   = cx + (shoulder_max_x - cx) * 0.70
    # Z: centroid of upper-body wide verts
    wide_up = [v for v in upper_verts if v.x > cx + (shoulder_max_x - cx) * 0.40]
    shoulder_z = sum(v.z for v in wide_up) / len(wide_up) if wide_up else H * 0.78
else:
    shoulder_l_x = cx + W * 0.30
    shoulder_z   = H * 0.78

# ── Find feet ─────────────────────────────────────────────────────────────
foot_verts = [v for v in verts if v.z < H * 0.18]
if foot_verts:
    foot_l_x = max((v.x for v in foot_verts if v.x > cx), default=cx + W * 0.08)
    foot_r_x = min((v.x for v in foot_verts if v.x < cx), default=cx - W * 0.08)
else:
    foot_l_x =  cx + W * 0.08
    foot_r_x =  cx - W * 0.08

foot_z = H * 0.04

# ── Root, neck, chin heights ──────────────────────────────────────────────
root_z  = H * 0.52
neck_z  = H * 0.84
chin_z  = H * 0.88

print(f"[ARP] Landmarks: shoulder=({shoulder_l_x:.3f},{shoulder_z:.3f}) "
      f"hand=({hand_l_x:.3f},{hand_z:.3f}) foot_l={foot_l_x:.3f}")


# ── 6. Enable ARP  (after mesh processed; prefs entry already exists from step 0) ──
print("[ARP] Enabling Auto-Rig Pro...")
ARP_ADDON_NAME = os.getenv("ARP_ADDON_NAME", "auto_rig_pro-master")
addon_utils.enable(ARP_ADDON_NAME, default_set=False, persistent=False)
bpy.context.view_layer.update()

if not hasattr(bpy.ops, "arp"):
    raise RuntimeError(f"ARP operators not found after enabling {ARP_ADDON_NAME!r}.")
print("[ARP] ARP operators available.")


# ── 7. Place ARP Smart body markers (OUTSIDE mesh surface) ────────────────
# Markers must be outside the mesh so ARP's raycast can hit the surface.
# We place them just beyond the extremity points found above.
print("[ARP] Placing Smart markers...")

scn = bpy.context.scene
scn.arp_body_name      = "Character"
scn.arp_smart_type     = "BODY"
scn.arp_smart_sym      = True    # mirror: only need left-side markers
scn.arp_fingers_enable = False   # skip AI finger detection (needs viewport)
scn.arp_fingers_to_detect = 0

def make_marker(name, x, y, z):
    bpy.ops.object.empty_add(type="PLAIN_AXES", radius=0.04, location=(x, y, z))
    obj = bpy.context.active_object
    obj.name = name
    obj.show_in_front = True
    container = bpy.data.objects.get("arp_markers")
    if container:
        obj.parent = container
    return obj

# Create arp_markers parent container
bpy.ops.object.empty_add(type="PLAIN_AXES", radius=0.01, location=(0, 0, 0))
arp_markers = bpy.context.active_object
arp_markers.name = "arp_markers"

# Markers at mesh-internal positions (ARP fires Y-rays from marker X,Z to find depth)
make_marker("root_loc",      cx,           cy,  root_z)
make_marker("neck_loc",      cx,           cy,  neck_z)
make_marker("chin_loc",      cx,           cy,  chin_z)
make_marker("shoulder_loc",  shoulder_l_x, cy,  shoulder_z)
make_marker("hand_loc",      hand_l_x,     cy,  hand_z)
make_marker("foot_loc",      foot_l_x,     cy,  foot_z)

print("[ARP] Markers placed.")
for obj in bpy.data.objects:
    if obj.name.endswith("_loc") and not obj.name.endswith("_sym"):
        print(f"  {obj.name}: {obj.location.x:.3f}, {obj.location.z:.3f}")


# ── 8. Run ARP Smart detection ────────────────────────────────────────────────

# HEADLESS PATCH: ARP calls display_popup_message() after go_detect to show a
# success/warning dialog.  That function calls bpy.types.UILayout.popup_menu()
# which needs bpy.context.window to be non-None (it's None in --background).
# This causes SIGSEGV at the C level even though rigging itself succeeded.
# Patching display_popup_message to a no-op before calling go_detect is safe:
# the armature is already created by the time the popup fires.
try:
    import auto_rig_pro.src.auto_rig as _arp_auto_rig  # type: ignore
    _orig_popup = getattr(_arp_auto_rig, "display_popup_message", None)
    _arp_auto_rig.display_popup_message = lambda *a, **kw: None
    print("[ARP] Patched display_popup_message → no-op (headless-safe)")
except Exception as _pe:
    print(f"[ARP] Could not patch display_popup_message ({_pe}) — continuing anyway")

# Also patch via the smart module path
try:
    import auto_rig_pro.src.auto_rig_smart as _arp_smart_mod  # type: ignore
    if hasattr(_arp_smart_mod, "display_popup_message"):
        _arp_smart_mod.display_popup_message = lambda *a, **kw: None
        print("[ARP] Patched auto_rig_smart.display_popup_message → no-op")
except Exception as _pe2:
    pass  # not present in all ARP versions

print("[ARP] Running ARP Smart detection (EXEC_DEFAULT)...")
bpy.context.view_layer.objects.active = char_mesh
char_mesh.select_set(True)
bpy.context.view_layer.update()

result = bpy.ops.id.go_detect("EXEC_DEFAULT")
print(f"[ARP] go_detect result: {result}")
bpy.context.view_layer.update()


# ── 9. Verify armature ────────────────────────────────────────────────────────

armature_objs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if not armature_objs:
    raise RuntimeError("ARP Smart did not create an armature — detection failed. "
                       "Check marker positions vs mesh geometry.")
armature = armature_objs[0]
print(f"[ARP] Armature: {armature.name}")


# ── 10. Bind mesh to armature ─────────────────────────────────────────────────

# ARMATURE_AUTO (bone-heat) segfaults on headless/cloud instances with complex
# meshes.  Use ARMATURE_ENVELOPE (distance-based) which is stable everywhere.
# ARP's own arp.bind operator is tried first when available.

# go_detect leaves Blender in POSE mode — return to OBJECT mode
try:
    bpy.ops.object.mode_set(mode="OBJECT")
except Exception as e:
    print(f"[ARP] mode_set warning: {e}")

bpy.ops.object.select_all(action="DESELECT")
# Re-fetch objects in case references changed during ARP processing
char_mesh = bpy.data.objects.get("Character")
armature  = next((o for o in bpy.context.scene.objects if o.type == "ARMATURE"), None)

if char_mesh is None:
    raise RuntimeError("Character mesh not found after ARP detection")
if armature is None:
    raise RuntimeError("No armature found after ARP detection")

char_mesh.select_set(True)
armature.select_set(True)
bpy.context.view_layer.objects.active = armature

# Try ARP's native bind first (preserves IK/FK setup); fall back to envelope
bound = False
if hasattr(bpy.ops, "arp") and hasattr(bpy.ops.arp, "bind"):
    try:
        result = bpy.ops.arp.bind("EXEC_DEFAULT")
        print(f"[ARP] arp.bind result: {result}")
        bound = True
    except Exception as e:
        print(f"[ARP] arp.bind failed ({e}), falling back to envelope")

if not bound:
    # ARMATURE_ENVELOPE: stable on headless, no bone-heat graph traversal
    bpy.ops.object.parent_set(type="ARMATURE_ENVELOPE")
    print("[ARP] Bound with ARMATURE_ENVELOPE (headless-safe fallback)")

print("[ARP] Mesh bound OK.")


# ── 11. Export GLB ────────────────────────────────────────────────────────────

print(f"[ARP] Exporting to: {OUTPUT_GLB}")
os.makedirs(os.path.dirname(OUTPUT_GLB) or ".", exist_ok=True)

bpy.ops.export_scene.gltf(
    filepath=OUTPUT_GLB,
    export_format="GLB",
    use_selection=False,
    export_apply=True,
    export_animations=True,
    export_skins=True,
    export_morph=True,
    export_lights=False,
    export_cameras=False,
    export_yup=True,
)

if not os.path.exists(OUTPUT_GLB):
    raise RuntimeError(f"Export failed — output not found: {OUTPUT_GLB}")

size_mb = os.path.getsize(OUTPUT_GLB) / 1e6
print(f"[ARP] Export OK: {OUTPUT_GLB} ({size_mb:.2f} MB)")
print(f"[ARP] RIGGING_COMPLETE: {OUTPUT_GLB}")

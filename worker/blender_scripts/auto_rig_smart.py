"""
Auto-Rig Pro Smart Rigging — Blender headless script
=====================================================
Called by rig_worker.py via subprocess:
  blender --background --python worker/blender_scripts/auto_rig_smart.py -- \
          --input /tmp/char.glb \
          --output /tmp/char_rigged.glb \
          --character_type humanoid   (or: quadruped, bird, fish)

Supports all character types via Auto-Rig Pro Smart detection.
The Smart feature auto-detects body landmarks (head, hands, feet, spine)
and builds a full deformation rig automatically.

ARP Smart flow:
  1. Import GLB mesh
  2. Normalize scale + orientation
  3. Run ARP Smart detection (body part landmarks)
  4. Generate full ARP armature
  5. Bind mesh with Automatic Weights
  6. Export rigged GLB
"""

import sys
import os
import argparse

import bpy
import addon_utils
import mathutils


# ── Parse args passed after "--" ─────────────────────────────────────────────
argv = sys.argv
if "--" in argv:
    argv = argv[argv.index("--") + 1:]
else:
    argv = []

parser = argparse.ArgumentParser()
parser.add_argument("--input",          required=True,  help="Input GLB path")
parser.add_argument("--output",         required=True,  help="Output GLB path")
parser.add_argument("--character_type", default="humanoid",
                    choices=["humanoid", "quadruped", "bird", "fish", "other"],
                    help="Character type for ARP Smart")
parser.add_argument("--scale",          type=float, default=1.0,
                    help="Character scale override (default: auto)")
args = parser.parse_args(argv)

INPUT_GLB       = args.input
OUTPUT_GLB      = args.output
CHARACTER_TYPE  = args.character_type
CHAR_SCALE      = args.scale

print(f"[ARP] Input:  {INPUT_GLB}")
print(f"[ARP] Output: {OUTPUT_GLB}")
print(f"[ARP] Type:   {CHARACTER_TYPE}")


# ── 1. Clean scene ────────────────────────────────────────────────────────────
print("[ARP] Clearing scene...")
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=True)
for block in bpy.data.meshes:
    bpy.data.meshes.remove(block)


# ── 2. Enable Auto-Rig Pro addon ─────────────────────────────────────────────
print("[ARP] Enabling Auto-Rig Pro addon...")
ARP_ADDON_NAME = os.getenv("ARP_ADDON_NAME", "auto_rig_pro-master")
addon_utils.enable(ARP_ADDON_NAME, default_set=True, persistent=True)
bpy.context.view_layer.update()

# Verify ARP loaded
if not hasattr(bpy.ops, "arp"):
    raise RuntimeError(
        f"Auto-Rig Pro (arp.*) operators not found after enabling {ARP_ADDON_NAME!r}. "
        "Check addon is installed in Blender's addons directory."
    )
print("[ARP] Auto-Rig Pro operators available.")


# ── 3. Import GLB ─────────────────────────────────────────────────────────────
print(f"[ARP] Importing: {INPUT_GLB}")
bpy.ops.import_scene.gltf(filepath=INPUT_GLB)

# Collect mesh objects
mesh_objs = [o for o in bpy.context.scene.objects if o.type == "MESH"]
if not mesh_objs:
    raise RuntimeError(f"No mesh objects found after importing {INPUT_GLB}")

print(f"[ARP] Imported {len(mesh_objs)} mesh object(s): {[o.name for o in mesh_objs]}")


# ── 4. Merge all meshes into one (ARP Smart works best with single mesh) ──────
print("[ARP] Joining meshes into one...")
bpy.ops.object.select_all(action="DESELECT")
for obj in mesh_objs:
    obj.select_set(True)
bpy.context.view_layer.objects.active = mesh_objs[0]
bpy.ops.object.join()
char_mesh = bpy.context.active_object
char_mesh.name = "Character"


# ── 5. Normalize scale and orientation ───────────────────────────────────────
print("[ARP] Normalizing transforms...")
bpy.ops.object.select_all(action="DESELECT")
char_mesh.select_set(True)
bpy.context.view_layer.objects.active = char_mesh

# Apply all transforms
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Auto-scale: normalize character height to ~2 Blender units (standard ARP scale)
bbox = [char_mesh.matrix_world @ mathutils.Vector(v) for v in char_mesh.bound_box]
height = max(v.z for v in bbox) - min(v.z for v in bbox)
if height > 0.01 and CHAR_SCALE == 1.0:
    scale_factor = 2.0 / height   # normalize to 2m
    char_mesh.scale = (scale_factor, scale_factor, scale_factor)
    bpy.ops.object.transform_apply(scale=True)
    print(f"[ARP] Auto-scaled: height {height:.3f} → 2.0 (factor {scale_factor:.3f})")
elif CHAR_SCALE != 1.0:
    char_mesh.scale = (CHAR_SCALE, CHAR_SCALE, CHAR_SCALE)
    bpy.ops.object.transform_apply(scale=True)

# Move to origin (feet at Z=0)
bbox = [char_mesh.matrix_world @ mathutils.Vector(v) for v in char_mesh.bound_box]
min_z = min(v.z for v in bbox)
char_mesh.location.z -= min_z
bpy.ops.object.transform_apply(location=True)


# ── 6. Run ARP Smart Rigging ──────────────────────────────────────────────────
print(f"[ARP] Running Smart Rigging (type={CHARACTER_TYPE})...")

bpy.context.view_layer.objects.active = char_mesh
char_mesh.select_set(True)

# Set character type in ARP scene settings
scene = bpy.context.scene
if hasattr(scene, "arp_smart_type"):
    if CHARACTER_TYPE == "quadruped":
        scene.arp_smart_type = "quadruped"
    elif CHARACTER_TYPE in ("bird",):
        scene.arp_smart_type = "bird"
    else:
        scene.arp_smart_type = "humanoid"

# Run ARP Smart — this is the one-click auto-rig operator
try:
    result = bpy.ops.arp.search_and_rig()
    print(f"[ARP] search_and_rig result: {result}")
except Exception as e:
    # Fallback: try the older smart_bones_object operator
    print(f"[ARP] search_and_rig failed ({e}), trying smart_bones_object...")
    try:
        result = bpy.ops.arp.smart_bones_object()
        print(f"[ARP] smart_bones_object result: {result}")
    except Exception as e2:
        raise RuntimeError(f"ARP Smart operators failed: {e} / {e2}")

bpy.context.view_layer.update()


# ── 7. Bind mesh with Automatic Weights (if not already bound) ───────────────
print("[ARP] Checking skin binding...")
armature_objs = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
if not armature_objs:
    raise RuntimeError("ARP Smart did not create an armature — rigging failed.")

armature = armature_objs[0]
print(f"[ARP] Armature: {armature.name}")

# Check if mesh is already parented with armature modifier
has_armature_mod = any(
    m.type == "ARMATURE" for m in char_mesh.modifiers
)
if not has_armature_mod:
    print("[ARP] Binding skin with Automatic Weights...")
    bpy.ops.object.select_all(action="DESELECT")
    char_mesh.select_set(True)
    armature.select_set(True)
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type="ARMATURE_AUTO")
    print("[ARP] Skin bound OK.")
else:
    print("[ARP] Skin already bound.")


# ── 8. Export rigged GLB ──────────────────────────────────────────────────────
print(f"[ARP] Exporting to: {OUTPUT_GLB}")
os.makedirs(os.path.dirname(OUTPUT_GLB) or ".", exist_ok=True)

bpy.ops.export_scene.gltf(
    filepath=OUTPUT_GLB,
    export_format="GLB",
    export_apply=True,          # apply modifiers
    export_animations=True,
    export_skins=True,
    export_morph=True,
    export_lights=False,
    export_cameras=False,
    export_yup=True,            # Y-up for game engines
)

# Verify output exists
if not os.path.exists(OUTPUT_GLB):
    raise RuntimeError(f"Export failed — output not found: {OUTPUT_GLB}")

size_mb = os.path.getsize(OUTPUT_GLB) / 1e6
print(f"[ARP] Export OK: {OUTPUT_GLB} ({size_mb:.2f} MB)")
print(f"[ARP] RIGGING_COMPLETE: {OUTPUT_GLB}")

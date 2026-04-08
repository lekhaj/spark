#!/usr/bin/env bash
# ════════════════════════════════════════════════════════════════════════════
# install_blender_arp.sh — Install Blender + Auto-Rig Pro on the GPU server
# ════════════════════════════════════════════════════════════════════════════
#
# Downloads:
#   • Blender 4.2 LTS (headless, Linux x64)
#   • Auto-Rig Pro 3.76.22 from S3 (sparkassets-us)
#
# Usage:
#   bash worker/gpu_setup/install_blender_arp.sh 2>&1 | tee /tmp/blender_install.log
# ════════════════════════════════════════════════════════════════════════════
set -uo pipefail

BLENDER_VERSION="4.2.9"
BLENDER_DIR="/home/ec2-user/blender"
BLENDER_BIN="$BLENDER_DIR/blender"
ARP_S3_URL="https://sparkassets-us.s3.us-east-1.amazonaws.com/auto_rig/Auto-Rig+Pro+3.76.22.zip"
ARP_ZIP="/tmp/auto_rig_pro.zip"

log()  { echo ""; echo "════ $* ════"; echo ""; }
ok()   { echo "  [OK] $*"; }
warn() { echo "  [WARN] $*"; }

# ── 1. Install Blender ────────────────────────────────────────────────────────
log "STEP 1: Install Blender $BLENDER_VERSION"
if [ -f "$BLENDER_BIN" ]; then
  echo "Blender already installed at $BLENDER_BIN"
  "$BLENDER_BIN" --version | head -1
else
  BLENDER_TGZ="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
  BLENDER_URL="https://download.blender.org/release/Blender4.2/$BLENDER_TGZ"

  echo "Downloading Blender $BLENDER_VERSION..."
  wget -q --show-progress "$BLENDER_URL" -O /tmp/blender.tar.xz

  mkdir -p "$BLENDER_DIR"
  echo "Extracting..."
  tar -xf /tmp/blender.tar.xz -C /tmp/
  # The extracted folder name varies slightly — find it
  EXTRACTED=$(ls /tmp/ | grep "^blender-${BLENDER_VERSION}" | head -1)
  mv "/tmp/$EXTRACTED"/* "$BLENDER_DIR/"
  rm /tmp/blender.tar.xz

  ok "Blender installed at $BLENDER_BIN"
  "$BLENDER_BIN" --version | head -1
fi

# Add blender to PATH
grep -q 'blender' /home/ec2-user/.bashrc || \
  echo "export PATH=$BLENDER_DIR:\$PATH" >> /home/ec2-user/.bashrc
export PATH="$BLENDER_DIR:$PATH"

# ── 2. Install system deps for headless Blender ───────────────────────────────
log "STEP 2: System libraries for headless rendering"
sudo dnf install -y libXi libXrender libXext libXfixes libSM libICE libGLU \
  libXxf86vm libXrandr mesa-libGL mesa-libEGL 2>/dev/null || \
  warn "Some X11/GL libs may be missing (usually ok for headless)"

# ── 3. Download Auto-Rig Pro from S3 ─────────────────────────────────────────
log "STEP 3: Download Auto-Rig Pro 3.76.22 from S3"
if [ -f "$ARP_ZIP" ]; then
  echo "ARP zip already downloaded at $ARP_ZIP"
else
  echo "Downloading from S3..."
  wget -q --show-progress "$ARP_S3_URL" -O "$ARP_ZIP"
  ok "Downloaded: $ARP_ZIP ($(du -sh $ARP_ZIP | cut -f1))"
fi

# ── 4. Install Auto-Rig Pro addon into Blender ───────────────────────────────
log "STEP 4: Install Auto-Rig Pro addon into Blender"

# Blender 4.x user addons live in ~/.config/blender/4.2/scripts/addons
BLENDER_ADDON_DIR="$HOME/.config/blender/4.2/scripts/addons"
mkdir -p "$BLENDER_ADDON_DIR"

if [ -f "$BLENDER_ADDON_DIR/auto_rig_pro/__init__.py" ]; then
  echo "Auto-Rig Pro already installed at $BLENDER_ADDON_DIR/auto_rig_pro"
else
  # The outer zip is a package containing auto_rig_pro_3.76.22.zip inside
  # Extract the inner ARP addon zip first
  echo "Extracting ARP addon zip from package..."
  unzip -p "$ARP_ZIP" \
    'Auto-Rig Pro 3.76.22/Auto-Rig Pro 3.76.22/auto_rig_pro_3.76.22.zip' \
    > /tmp/arp_addon.zip

  # Extract ARP addon to user addons dir (extracts as auto_rig_pro-master)
  cd "$BLENDER_ADDON_DIR"
  unzip -q /tmp/arp_addon.zip

  # Rename master suffix if present
  [ -d "auto_rig_pro-master" ] && mv auto_rig_pro-master auto_rig_pro

  # Also extract rig_tools (required companion)
  unzip -p "$ARP_ZIP" \
    'Auto-Rig Pro 3.76.22/Auto-Rig Pro 3.76.22/rig_tools_3.76.18.zip' \
    > /tmp/rig_tools.zip
  unzip -q /tmp/rig_tools.zip 2>/dev/null || true

  ok "Auto-Rig Pro installed at $BLENDER_ADDON_DIR/auto_rig_pro"
fi

# ── 5. Enable Auto-Rig Pro in Blender's startup ──────────────────────────────
log "STEP 5: Enable Auto-Rig Pro addon in Blender"
"$BLENDER_BIN" --background --python-expr "
import bpy, addon_utils
for mod in ['auto_rig_pro', 'rig_tools']:
    mods = [m.__name__ for m in addon_utils.modules()]
    if mod in mods:
        addon_utils.enable(mod, default_set=True, persistent=True)
        print('Enabled:', mod)
    else:
        print('WARNING: module not found:', mod)
bpy.ops.wm.save_userpref()
print('ARP ops available:', hasattr(bpy.ops, 'arp'))
print('AUTO_RIG_PRO ENABLED OK')
" 2>&1 | grep -E 'AUTO_RIG|Enabled|arp|ERROR|WARNING' | head -10

# ── 6. Set BLENDER_PATH in .env ──────────────────────────────────────────────
log "STEP 6: Update .env with Blender path"
ENV_FILE=/home/ec2-user/spark/.env
grep -q 'BLENDER_PATH' "$ENV_FILE" 2>/dev/null || cat >> "$ENV_FILE" << ENVEOF

# Blender + Auto-Rig Pro rigging
BLENDER_PATH=$BLENDER_BIN
ARP_CHARACTER_SCALE=1.0
ENVEOF
ok ".env updated"

# ── 7. Verify ─────────────────────────────────────────────────────────────────
log "STEP 7: Verify Blender + ARP"
"$BLENDER_BIN" --background --python-expr "
import bpy
import addon_utils
enabled = [m.bl_info.get('name','') for m in addon_utils.modules() if addon_utils.check(m.__name__)[1]]
arp_found = any('Auto-Rig' in n for n in enabled)
print('ARP enabled:', arp_found)
print('Blender OK')
" 2>&1 | grep -E 'ARP|Blender|WARN'

echo ""
echo "════════════════════════════════════════════════════════"
echo "  Blender + Auto-Rig Pro install complete!"
echo "  Blender: $BLENDER_BIN"
echo "  ARP:     $ARP_ZIP"
echo ""
echo "  Test rigging:"
echo "    python3 worker/gpu_main.py --workers rig --no-shutdown"
echo "════════════════════════════════════════════════════════"

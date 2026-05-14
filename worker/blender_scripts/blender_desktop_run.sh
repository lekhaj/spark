#!/usr/bin/env bash
# blender_desktop_run.sh
# ──────────────────────────────────────────────────────────────────────────────
# Starts a virtual X desktop (Xvfb + fluxbox window manager), runs Blender
# inside it, then tears down cleanly regardless of success or failure.
#
# Why fluxbox:
#   Blender with --python scripts needs a VIEW_3D area to exist AND be managed
#   by a window manager so that area.regions are fully initialised.
#   xvfb-run alone provides a display but no WM → operators that poll
#   context.space_data / context.area.regions fail.  A WM fixes this.
#
# Usage:
#   bash blender_desktop_run.sh <blender_binary> [blender_args...]
#
# Called by rig_model.py — no manual steps needed.
# ──────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── 1. Find a free display number ─────────────────────────────────────────────
DISP=99
while [ -e "/tmp/.X${DISP}-lock" ] || [ -S "/tmp/.X11-unix/X${DISP}" ]; do
    DISP=$((DISP + 1))
    if [ "$DISP" -gt 199 ]; then
        echo "[DESKTOP] ERROR: Could not find free display (tried :99 – :199)"
        exit 1
    fi
done
export DISPLAY=":${DISP}"
echo "[DESKTOP] Using display :${DISP}"

XVFB_PID=""
FLUXBOX_PID=""

# ── 2. Cleanup on exit (crash or success) ─────────────────────────────────────
cleanup() {
    local code=$?
    echo "[DESKTOP] Cleaning up display :${DISP} ..."
    [ -n "$FLUXBOX_PID" ] && kill "$FLUXBOX_PID" 2>/dev/null || true
    [ -n "$XVFB_PID"    ] && kill "$XVFB_PID"    2>/dev/null || true
    # Wait briefly so sockets are released before next job can grab same display
    sleep 0.2
    exit $code
}
trap cleanup EXIT INT TERM

# ── 3. Start Xvfb ─────────────────────────────────────────────────────────────
Xvfb ":${DISP}" \
    -screen 0 1920x1080x24 \
    -ac \
    +extension GLX \
    +extension RANDR \
    +render \
    -noreset &
XVFB_PID=$!
echo "[DESKTOP] Xvfb PID=${XVFB_PID}"

# Wait up to 4 seconds for the socket
for i in $(seq 1 40); do
    [ -S "/tmp/.X11-unix/X${DISP}" ] && break
    sleep 0.1
done
if [ ! -S "/tmp/.X11-unix/X${DISP}" ]; then
    echo "[DESKTOP] ERROR: Xvfb socket /tmp/.X11-unix/X${DISP} did not appear"
    exit 1
fi
echo "[DESKTOP] Xvfb socket ready"

# ── 4. Start fluxbox window manager ───────────────────────────────────────────
# fluxbox is needed so Blender's window is "managed" — areas get focus events,
# regions are fully initialised, and view3d operators poll correctly.
DISPLAY=":${DISP}" fluxbox -display ":${DISP}" -log /dev/null 2>/dev/null &
FLUXBOX_PID=$!
echo "[DESKTOP] fluxbox PID=${FLUXBOX_PID}"

# Give fluxbox 400 ms to start up and claim the display
sleep 0.4

# ── 5. Run Blender with all forwarded arguments ────────────────────────────────
echo "[DESKTOP] Starting: $*"
DISPLAY=":${DISP}" "$@"
# Exit code propagated via trap cleanup

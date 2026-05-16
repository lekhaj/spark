#!/bin/bash
# install_gltfpack.sh — download gltfpack binary on Linux x86_64.
# Idempotent: skips if already installed and up to date.
set -euo pipefail

VERSION="${GLTFPACK_VERSION:-0.21}"
DEST="${GLTFPACK_BIN:-/usr/local/bin/gltfpack}"

if [ -x "$DEST" ]; then
    cur="$("$DEST" -v 2>&1 | head -1 || true)"
    echo "gltfpack already present: $cur"
    exit 0
fi

URL="https://github.com/zeux/meshoptimizer/releases/download/v${VERSION}/gltfpack-ubuntu.zip"
TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

echo "downloading gltfpack v${VERSION} → $URL"
curl -sSL "$URL" -o "$TMPDIR/gltfpack.zip"
unzip -q "$TMPDIR/gltfpack.zip" -d "$TMPDIR"

# Binary inside zip is named "gltfpack" (on linux)
if [ ! -f "$TMPDIR/gltfpack" ]; then
    # Some releases have it directly without zip wrapping — try as binary
    if file "$TMPDIR/gltfpack.zip" | grep -qi executable; then
        cp "$TMPDIR/gltfpack.zip" "$TMPDIR/gltfpack"
    else
        echo "ERROR: gltfpack binary not found in archive"
        ls -la "$TMPDIR"
        exit 2
    fi
fi

sudo install -m 0755 -o root -g root "$TMPDIR/gltfpack" "$DEST"
echo "installed: $("$DEST" -v 2>&1 | head -1)"

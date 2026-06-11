#!/bin/bash
# spark-prewarm.sh — read critical model weights into the Linux page cache
# so the first inference on a fresh spot doesn't pay the EBS snapshot
# lazy-load penalty (which throttles reads to ~10 MB/s for the first touch
# of each block).
#
# Runs as ec2-user on boot via spark-prewarm.service. Touches
# /var/run/spark-prewarm.done when complete — AutoShutdown skips idle
# counting until that sentinel exists, so the instance never auto-stops
# itself mid-warmup.

set -euo pipefail

LOG=/var/log/spark-prewarm.log
SENTINEL=/home/ec2-user/spark-prewarm.done
HF_HUB="$HOME/.cache/huggingface/hub"

# Idempotency: don't redo if already done THIS BOOT. The sentinel lives on
# EBS so it survives reboots — but the page cache doesn't. A sentinel older
# than the current boot is stale and must not skip the warmup (a stale skip
# made the first post-reboot flux load take ~90 min on 2026-06-11).
if [ -f "$SENTINEL" ]; then
    boot_ts=$(date -d "$(uptime -s)" +%s)
    sent_ts=$(stat -c %Y "$SENTINEL")
    if [ "$sent_ts" -ge "$boot_ts" ]; then
        echo "[prewarm] sentinel from this boot exists, skipping"
        exit 0
    fi
    echo "[prewarm] stale sentinel (pre-boot) — re-running warmup"
    rm -f "$SENTINEL"
fi

exec > >(tee -a "$LOG") 2>&1
echo "=== spark-prewarm START $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "Reading model weights into page cache to defeat EBS snapshot lazy-load."

# ── 1. Critical HF model snapshots ────────────────────────────────────────
# Only the directories we actually use. Skipping TRELLIS.2 (4B) unless we
# need it — saves ~30 GB of cache pressure on the 61 GB host.
HF_MODELS=(
    "models--black-forest-labs--FLUX.1-schnell"
    "models--TencentARC--Pixal3D"
    "models--camenduru--dinov3-vitl16-pretrain-lvd1689m"
    "models--facebook--dinov3-vitl16-pretrain-lvd1689m"
    "models--Ruicheng--moge-2-vitl"
    "models--ZhengPeng7--BiRefNet"
    "models--microsoft--TRELLIS-image-large"
)

total_bytes=0
for m in "${HF_MODELS[@]}"; do
    dir="$HF_HUB/$m"
    if [ ! -d "$dir" ]; then
        echo "[prewarm] skip (missing): $m"
        continue
    fi
    echo "[prewarm] warming $m"
    # Parallel-cat all weights — 4 streams to saturate EBS lazy-load.
    # || true: with pipefail, one unreadable file (broken HF-cache symlink)
    # makes xargs exit 123 and killed the whole unit (seen 2026-06-11).
    bytes=$(find "$dir" \( -name "*.safetensors" -o -name "*.bin" -o -name "*.pt" \
            -o -name "*.json" -o -name "*.txt" \) -size +0c -print0 \
            | xargs -0 -n4 -P 4 cat 2>/dev/null | wc -c || true)
    echo "[prewarm]   $m: ${bytes:-0} bytes"
    total_bytes=$((total_bytes + ${bytes:-0}))
done

# ── 2. Conda env shared libraries (avoids slow first import) ──────────────
echo "[prewarm] warming conda env .so files"
find "$HOME/miniconda3/envs/spark/lib" \
     "$HOME/miniconda3/envs/pixal3d/lib" \
     -name "*.so*" -size +1M -print0 2>/dev/null \
    | xargs -0 -n4 -P 4 cat 2>/dev/null | wc -c > /dev/null || true

# ── 3. Pixal3D code + autotune cache + Pixal3D ckpt JSON ──────────────────
if [ -d "$HOME/Pixal3D" ]; then
    find "$HOME/Pixal3D" -type f -print0 2>/dev/null \
        | xargs -0 -n4 -P 4 cat 2>/dev/null | wc -c > /dev/null || true
fi

echo "=== spark-prewarm complete — total HF bytes: $total_bytes ==="
echo "=== spark-prewarm DONE $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# Sentinel is on EBS (/home/ec2-user) — writable by ec2-user directly.
touch "$SENTINEL"
chmod 644 "$SENTINEL"
echo "[prewarm] sentinel created at $SENTINEL"

#!/bin/bash
# g7e_blackwell_patches.sh — sm_120 / RTX PRO 6000 Blackwell build patches
# ==========================================================================
# The g7e.2xlarge GPU is a Blackwell-class card (compute capability 12.0).
# Several third-party deps and on-disk model assets need patches that cannot
# live in the spark git repo (they touch the HF cache and cloned upstream
# repos). This script reapplies them all, idempotently, so the golden AMI can
# be rebuilt from scratch. Run once on a fresh g7e after the base bootstrap
# has created the conda envs + cloned Pixal3D / trellis (trellis2) / Hunyuan3D.
#
# Envs assumed (created by bootstrap):
#   spark      — in-process worker (FLUX, TRELLIS.2)        torch 2.11+cu128
#   pixal3d    — Pixal3D subprocess                          torch 2.8 +cu128
#   hunyuan3d  — Hunyuan3D-2 subprocess                      torch *  +cu128
#
# Safe to re-run.
set -uo pipefail

PIXAL3D_PY=/home/ec2-user/miniconda3/envs/pixal3d/bin/python
PIXAL3D_PIP=/home/ec2-user/miniconda3/envs/pixal3d/bin/pip
HUNYUAN_PIP=/home/ec2-user/miniconda3/envs/hunyuan3d/bin/pip
TRELLIS_REPO=/home/ec2-user/trellis
PIXAL3D_REPO=/home/ec2-user/Pixal3D
DINOV3_DIR=/home/ec2-user/models/dinov3-vitl16
HF_HUB=/home/ec2-user/.cache/huggingface/hub

log(){ echo "[g7e-patch] $*"; }

# ── 1. NATTEN for Pixal3D (NAF upsampler is *mandatory* — the checkpoint's
#       cross-attention proj_linear expects embed_dim*2 = 2048, which only
#       exists when use_naf_upsample=True). natten has no PyPI cp310 wheel for
#       Blackwell; pull the sm_120 build from whl.natten.org. ─────────────────
log "installing natten (torch280cu128, sm_120) into pixal3d env"
$PIXAL3D_PIP install "natten==0.21.1+torch280cu128" -f https://whl.natten.org/ --only-binary=:all: \
  || log "WARN natten install failed — pin/index may have moved; check whl.natten.org"

# ── 2. Pixal3D: ensure NAF upsampling is enabled (a debug pass once disabled
#       it to dodge the natten dep — that breaks the feature dimension). ───────
if [ -f "$PIXAL3D_REPO/inference.py" ]; then
  log "enabling use_naf_upsample in Pixal3D/inference.py"
  sed -i 's/"use_naf_upsample": False/"use_naf_upsample": True/g' "$PIXAL3D_REPO/inference.py"
fi

# ── 3. Hunyuan3D paint pipeline import fix: diffusers 0.30 imports
#       FLAX_WEIGHTS_NAME, removed in transformers >=4.46. Pin transformers to
#       the last release that still exports it. ────────────────────────────────
log "pinning transformers==4.45.2 in hunyuan3d env (FLAX_WEIGHTS_NAME)"
$HUNYUAN_PIP install "transformers==4.45.2" \
  || log "WARN transformers pin failed"

# ── 4. DINOv3 weights — facebook/dinov3 is gated; use the public camenduru
#       mirror, downloaded to a local dir TRELLIS.2 + Pixal3D point at. ─────────
if [ ! -f "$DINOV3_DIR/model.safetensors" ]; then
  log "downloading camenduru/dinov3-vitl16 mirror -> $DINOV3_DIR"
  mkdir -p "$DINOV3_DIR"
  /home/ec2-user/miniconda3/envs/spark/bin/hf download \
    camenduru/dinov3-vitl16-pretrain-lvd1689m --local-dir "$DINOV3_DIR" \
    || log "WARN dinov3 download failed"
else
  log "dinov3 mirror already present"
fi

# ── 5. Repoint TRELLIS.2 pipeline configs at the local DINOv3 dir (the gated
#       facebook id 404s/Access-denies). ────────────────────────────────────────
for snap in "$HF_HUB"/models--microsoft--TRELLIS.2-4B/snapshots/*/; do
  for j in pipeline.json texturing_pipeline.json; do
    f="$snap$j"
    [ -f "$f" ] || continue
    if grep -q "facebook/dinov3-vitl16-pretrain-lvd1689m" "$f"; then
      log "repointing dinov3 in $f"
      sed -i "s#facebook/dinov3-vitl16-pretrain-lvd1689m#$DINOV3_DIR#g" "$f"
    fi
  done
done

# ── 6. trellis2 BiRefNet dtype: cast input to model param dtype (mirror
#       weights are fp16 -> "Input type (float) and bias type (Half)"). ─────────
BIREF="$TRELLIS_REPO/trellis2/pipelines/rembg/BiRefNet.py"
if [ -f "$BIREF" ] && ! grep -q "next(self.model.parameters()).dtype" "$BIREF"; then
  log "patching BiRefNet.py dtype cast"
  $PIXAL3D_PY - "$BIREF" <<'PY'
import sys, re
p = sys.argv[1]
s = open(p).read()
anchor = 'input_images = self.transform_image(image).unsqueeze(0).to("cuda")'
patch  = ('\n        # match model weight dtype (mirror weights may be fp16) to avoid\n'
          '        # "Input type (float) and bias type (Half)" on the first conv.\n'
          '        input_images = input_images.to(next(self.model.parameters()).dtype)')
if anchor in s and 'next(self.model.parameters()).dtype' not in s:
    s = s.replace(anchor, anchor + patch, 1)
    open(p, "w").write(s)
    print("patched")
else:
    print("skip (anchor missing or already patched)")
PY
else
  log "BiRefNet.py already patched / missing"
fi

# ── 7. trellis2 image_feature_extractor: transformers 5.x nests DINOv3 layers
#       differently; use the native forward with output_hidden_states. ──────────
IMGF="$TRELLIS_REPO/trellis2/modules/image_feature_extractor.py"
if [ -f "$IMGF" ] && ! grep -q "output_hidden_states=True" "$IMGF"; then
  log "WARN image_feature_extractor.py is NOT patched — extract_features must use"
  log "     self.model(pixel_values=image, output_hidden_states=True) then"
  log "     F.layer_norm(hidden_states[-1]). Apply manually (structure varies)."
else
  log "image_feature_extractor.py already patched / missing"
fi

log "done. Verify:"
log "  $PIXAL3D_PY -c 'import natten; from natten.functional import na2d; print(natten.__version__)'"
log "  /home/ec2-user/miniconda3/envs/hunyuan3d/bin/python -c 'from transformers.utils import FLAX_WEIGHTS_NAME; print(\"ok\")'"

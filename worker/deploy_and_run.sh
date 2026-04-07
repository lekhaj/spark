#!/usr/bin/env bash
# ════════════════════════════════════════════════════════
# deploy_and_run.sh
# Assumes both CPU (18.207.13.85) and L4 (3.215.211.192)
# are already running. Run from your Mac:
#   bash worker/deploy_and_run.sh
# ════════════════════════════════════════════════════════
set -e

KEY="/Users/lekhaj/Documents/us_cpu_key.pem"
A10="ec2-user@3.215.211.192"
CPU="ubuntu@18.207.13.85"
REMOTE_DIR="/home/ec2-user/worker/sd15"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30"
SCP="scp -i $KEY -o StrictHostKeyChecking=no"

# ════════════════════════════════════════════════════════
echo "════════════════════════════════════════"
echo " STEP 1 — Deploy worker files to L4"
echo "════════════════════════════════════════"
$SSH $A10 "mkdir -p $REMOTE_DIR"
$SCP "$REPO_DIR/worker/sd15_image_worker.py"       $A10:$REMOTE_DIR/
$SCP "$REPO_DIR/worker/prepare_controlnet_refs.py" $A10:$REMOTE_DIR/
echo "Files deployed."

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 2 — Install / verify Python deps on L4"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
set -e
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv

if [ ! -d "$VENV" ]; then
    python3 -m venv $VENV
fi
PIP="$VENV/bin/pip"

# torch already on A10 — only install what's missing
$PIP install -q --upgrade \
    "diffusers>=0.27" \
    transformers \
    accelerate \
    xformers \
    controlnet_aux \
    boto3 \
    redis \
    pymongo \
    python-dotenv \
    Pillow

echo "Deps OK."
ENDSSH

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 3 — Generate ControlNet reference images on L4"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv
mkdir -p /home/ubuntu/controlnet_refs
$VENV/bin/python /home/ubuntu/worker/sd15/prepare_controlnet_refs.py
ls -lh /home/ubuntu/controlnet_refs/
ENDSSH

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 4 — Insert biome schema into MongoDB (CPU)"
echo "════════════════════════════════════════"
$SCP "$REPO_DIR/worker/mongo_schema_update.py" $CPU:/tmp/
$SSH $CPU "python3 /tmp/mongo_schema_update.py"

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 5 — Start SD15 worker on L4 (screen)"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
set -e
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv
REMOTE_DIR=/home/ubuntu/worker/sd15

EXISTING_ENV=/home/ubuntu/worker/Hunyuan3D-2/.env
if [ -f "$EXISTING_ENV" ]; then
    grep -E 'REDIS|MONGO|AWS|S3|BUCKET|REGION' "$EXISTING_ENV" > $REMOTE_DIR/.env
    echo "Config copied from $EXISTING_ENV"
else
    echo "ERROR: $EXISTING_ENV not found. Create $REMOTE_DIR/.env manually."
    exit 1
fi

screen -S sd15-worker -X quit 2>/dev/null || true
sleep 1

screen -S sd15-worker -d -m bash -c "
    cd $REMOTE_DIR
    set -a; source .env; set +a
    $VENV/bin/python sd15_image_worker.py 2>&1 | tee /tmp/sd15_worker.log
"
sleep 3
screen -ls | grep sd15 && echo "Worker screen running." || echo "WARNING: screen not found"
ENDSSH

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 6 — Queue cultivation_master task (via CPU)"
echo "════════════════════════════════════════"
$SCP "$REPO_DIR/worker/queue_sd15_test.py" $CPU:/tmp/
$SSH $CPU "python3 /tmp/queue_sd15_test.py"

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " ALL DONE — Tailing worker log on A10"
echo " (Ctrl+C to stop watching — worker keeps running)"
echo " S3 key: images/claude-sd15-test-001/cultivation_master_refined.png"
echo "════════════════════════════════════════"
$SSH $A10 "tail -f /tmp/sd15_worker.log"

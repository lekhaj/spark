#!/usr/bin/env bash
# ════════════════════════════════════════════════════════
# deploy_and_run.sh
# Run from your Mac:  bash worker/deploy_and_run.sh
# ════════════════════════════════════════════════════════
set -e

KEY="/Users/lekhaj/Downloads/s_spu_key.pem"
A10="ubuntu@13.203.114.77"
CPU="ubuntu@15.206.99.66"
REMOTE_DIR="/home/ubuntu/worker/sd15"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH="ssh -i $KEY -o StrictHostKeyChecking=no"
SCP="scp -i $KEY -o StrictHostKeyChecking=no"

echo "════════════════════════════════════════"
echo " STEP 1 — Deploy worker files to A10"
echo "════════════════════════════════════════"
$SSH $A10 "mkdir -p $REMOTE_DIR"
$SCP "$REPO_DIR/worker/sd15_image_worker.py"       $A10:$REMOTE_DIR/
$SCP "$REPO_DIR/worker/prepare_controlnet_refs.py" $A10:$REMOTE_DIR/
echo "Files deployed."

echo ""
echo "════════════════════════════════════════"
echo " STEP 2 — Check / install Python deps on A10"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
set -e
cd /home/ubuntu/worker/sd15

# Use existing myenv if it has diffusers, else install into it
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv
if [ -d "$VENV" ]; then
    PIP="$VENV/bin/pip"
else
    python3 -m venv $VENV
    PIP="$VENV/bin/pip"
fi

# Install/upgrade needed packages (fast if already present)
$PIP install -q --upgrade \
    diffusers>=0.27 \
    transformers \
    accelerate \
    xformers \
    controlnet_aux \
    boto3 \
    redis \
    pymongo \
    python-dotenv \
    Pillow \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Deps OK."
ENDSSH

echo ""
echo "════════════════════════════════════════"
echo " STEP 3 — Generate ControlNet reference images on A10"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv
$VENV/bin/python /home/ubuntu/worker/sd15/prepare_controlnet_refs.py
ENDSSH

echo ""
echo "════════════════════════════════════════"
echo " STEP 4 — Insert biome document into MongoDB (via CPU)"
echo "════════════════════════════════════════"
$SCP "$REPO_DIR/worker/mongo_schema_update.py" $CPU:/tmp/
$SSH $CPU bash <<'ENDSSH'
python3 /tmp/mongo_schema_update.py
ENDSSH

echo ""
echo "════════════════════════════════════════"
echo " STEP 5 — Start SD15 worker on A10 (screen session)"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
set -e
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv

# Copy AWS creds + infra config from the existing worker .env on this machine
# (avoids hardcoding secrets in this script)
EXISTING_ENV=/home/ubuntu/worker/Hunyuan3D-2/.env
if [ -f "$EXISTING_ENV" ]; then
    grep -E 'REDIS|MONGO|AWS|S3|BUCKET|REGION' "$EXISTING_ENV" \
        > /home/ubuntu/worker/sd15/.env
    echo "Copied config from $EXISTING_ENV"
else
    echo "ERROR: $EXISTING_ENV not found — create /home/ubuntu/worker/sd15/.env manually"
    exit 1
fi

# Kill any existing session
screen -S sd15-worker -X quit 2>/dev/null || true
sleep 1

# Start new screen session
screen -S sd15-worker -d -m bash -c "
    cd /home/ubuntu/worker/sd15
    source .env
    /home/ubuntu/worker/Hunyuan3D-2/myenv/bin/python sd15_image_worker.py \
        2>&1 | tee /tmp/sd15_worker.log
"
sleep 3
screen -ls | grep sd15 && echo "Worker screen session running." || echo "WARNING: screen session not found"
ENDSSH

echo ""
echo "════════════════════════════════════════"
echo " STEP 6 — Queue cultivation_master test task"
echo "════════════════════════════════════════"
# Run queue script locally (has direct Redis access)
python3 "$REPO_DIR/worker/queue_sd15_test.py"

echo ""
echo "════════════════════════════════════════"
echo " DONE — Monitoring worker log"
echo "════════════════════════════════════════"
echo "Tailing log (Ctrl+C to stop watching, worker continues running)..."
echo ""
$SSH $A10 "tail -f /tmp/sd15_worker.log"

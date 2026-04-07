#!/usr/bin/env bash
# ════════════════════════════════════════════════════════
# deploy_and_run.sh
# Run from your Mac:  bash worker/deploy_and_run.sh
# ════════════════════════════════════════════════════════
set -e

KEY="/Users/lekhaj/Downloads/s_spu_key.pem"
A10_IP="43.205.175.32"
CPU_IP="15.206.99.66"
A10="ubuntu@${A10_IP}"
CPU="ubuntu@${CPU_IP}"
A10_INSTANCE_ID="i-09d9e7be52c7c8560"
CPU_INSTANCE_ID="i-0f53b275935e3ea6b"
AWS_REGION="ap-south-1"

REMOTE_DIR="/home/ubuntu/worker/sd15"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30"
SCP="scp -i $KEY -o StrictHostKeyChecking=no"

# ════════════════════════════════════════════════════════
echo "════════════════════════════════════════"
echo " STEP 0 — Start EC2 instances"
echo "════════════════════════════════════════"
aws ec2 start-instances --instance-ids $CPU_INSTANCE_ID $A10_INSTANCE_ID \
    --region $AWS_REGION --output text --query 'StartingInstances[*].[InstanceId,CurrentState.Name]'

echo "Waiting for both instances to be running..."
aws ec2 wait instance-running \
    --instance-ids $CPU_INSTANCE_ID $A10_INSTANCE_ID \
    --region $AWS_REGION
echo "Instances running."

# Extra 30s for SSH daemon to be ready
echo "Waiting 30s for SSH to be ready..."
sleep 30

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 1 — Deploy worker files to A10"
echo "════════════════════════════════════════"
$SSH $A10 "mkdir -p $REMOTE_DIR"
$SCP "$REPO_DIR/worker/sd15_image_worker.py"       $A10:$REMOTE_DIR/
$SCP "$REPO_DIR/worker/prepare_controlnet_refs.py" $A10:$REMOTE_DIR/
echo "Files deployed."

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " STEP 2 — Install / verify Python deps on A10"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
set -e
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv

# Create venv if missing (first-time setup)
if [ ! -d "$VENV" ]; then
    python3 -m venv $VENV
fi
PIP="$VENV/bin/pip"

# Install/upgrade only what's needed (torch already present on A10)
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
echo " STEP 3 — Generate ControlNet reference images on A10"
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
echo " STEP 5 — Start SD15 worker on A10 (screen)"
echo "════════════════════════════════════════"
$SSH $A10 bash <<'ENDSSH'
set -e
VENV=/home/ubuntu/worker/Hunyuan3D-2/myenv
REMOTE_DIR=/home/ubuntu/worker/sd15

# Pull .env from the existing worker config on this machine
EXISTING_ENV=/home/ubuntu/worker/Hunyuan3D-2/.env
if [ -f "$EXISTING_ENV" ]; then
    grep -E 'REDIS|MONGO|AWS|S3|BUCKET|REGION' "$EXISTING_ENV" > $REMOTE_DIR/.env
    echo "Config copied from $EXISTING_ENV"
else
    echo "ERROR: $EXISTING_ENV not found. Create $REMOTE_DIR/.env manually."
    exit 1
fi

# Kill any stale session
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
# Run via CPU so Redis is reachable (security group allows CPU→Redis)
$SCP "$REPO_DIR/worker/queue_sd15_test.py" $CPU:/tmp/
$SSH $CPU "python3 /tmp/queue_sd15_test.py"

# ════════════════════════════════════════════════════════
echo ""
echo "════════════════════════════════════════"
echo " ALL DONE — Tailing worker log on A10"
echo " (Ctrl+C to stop watching — worker keeps running)"
echo " S3 key will appear as: images/claude-sd15-test-001/cultivation_master_refined.png"
echo "════════════════════════════════════════"
$SSH $A10 "tail -f /tmp/sd15_worker.log"

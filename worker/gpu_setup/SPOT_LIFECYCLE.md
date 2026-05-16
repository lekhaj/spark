# Persistent-Spot GPU Lifecycle

How the CPU launches, warms, and reclaims a persistent-spot GPU instance.

## Components

| Where | What | Purpose |
|---|---|---|
| **CPU** | `worker/lib/gpu_launcher.py` | `ensure_gpu_ready()` — describe → start/launch → wait |
| **CPU** | `app/gradio/pages/generation_studio_page.py` | UI dropdown `🖥️ Target GPU queue` |
| **GPU** | `/usr/local/bin/spark-prewarm.sh` | Reads model weights into page cache at boot |
| **GPU** | `spark-prewarm.service` | systemd one-shot wrapper for the above |
| **GPU** | `worker/workers/auto_shutdown.py` | Sentinel-aware — defers idle counting during prewarm |
| **GPU AMI** | `ami-0d689e40322983537` | Baseline image with conda envs + ~/.cache/huggingface |

## Boot flow on a fresh spot

```
spot boot
  ├─ network-online.target
  ├─ spark-prewarm.service (oneshot, ~20 min on cold EBS)
  │     └─ touches /var/run/spark-prewarm.done
  ├─ manual_gen_worker.service starts (in parallel, can accept tasks)
  │     └─ AutoShutdown thread defers idle counting until sentinel exists
  └─ trellis_worker.service (parallel)
```

While prewarm is running, the worker still picks up tasks. The first
inference is slow (model weights still loading from EBS lazy-load), but
once they're in page cache subsequent calls run at full speed. The
AutoShutdown sentinel prevents the spot from stopping itself mid-warmup
or during a slow first inference.

## CPU env vars (add to `/home/ubuntu/spark/.env`)

```bash
# Activate launcher
GPU_AUTO_LAUNCH=1

# Existing instance lookup (env first, then tag-based fallback)
AWS_GPU_INSTANCE_ID=i-076db2d5aee6e16bb        # current L40S spot (changes on relaunch)
AWS_REGION=us-east-1

# Spot-from-AMI provisioning (used when the existing instance is missing/terminated)
AWS_GPU_AMI_ID=ami-0d689e40322983537
AWS_GPU_INSTANCE_TYPE=g6e.2xlarge
AWS_GPU_SUBNET_ID=subnet-0c5b465f9ede9e6ce
AWS_GPU_SG_ID=sg-0a4a561065082e3c9
AWS_GPU_KEY_NAME=us_cpu_key
AWS_GPU_INSTANCE_PROFILE=ec2_s3
AWS_GPU_EIP_ALLOC_ID=eipalloc-09678f3d9f0162d2d
AWS_GPU_PROJECT_TAG=spark-gpu

# Where the gradio app pushes tasks
MANUAL_GEN_QUEUE=manual_gen_tasks_spot   # routes to L40S spot
# MANUAL_GEN_QUEUE=manual_gen_tasks      # routes to old on-demand L4

# Boot timeout (waits for state=running)
GPU_BOOT_TIMEOUT_S=180
```

The UI dropdown can override `MANUAL_GEN_QUEUE` per session without a
restart — the env var is just the default.

## GPU env vars (already in `~/spark/worker/.env` on the spot)

```bash
REDIS_HOST=172.31.26.6              # CPU private IP (public IP path is firewalled)
REDIS_PORT=6379
MANUAL_GEN_QUEUE=manual_gen_tasks_spot

# L40S-tuned (do NOT set on L4 instances)
PIXAL3D_LOW_VRAM=0                  # 48 GB VRAM, full GPU residency
FLUX_OFFLOAD=none                   # FLUX.1-schnell fully on GPU

# AutoShutdown
IDLE_SHUTDOWN_MINUTES=15
PREWARM_SENTINEL=/var/run/spark-prewarm.done
```

## Rebake checklist

When source code or AMI contents need to change, rebake the AMI:

1. SSH to the running spot.
2. `cd ~/spark && git pull && sudo systemctl restart manual_gen_worker`
3. Install any new files at root-owned paths (e.g. `/usr/local/bin/`,
   `/etc/systemd/system/`) and `systemctl enable` them.
4. Clear caches to slim the snapshot:
   ```
   ~/miniconda3/envs/spark/bin/pip cache purge
   ~/miniconda3/envs/pixal3d/bin/pip cache purge
   sudo dnf clean all
   sudo rm -rf /tmp/pixal3d_* /tmp/pix_*.log /tmp/test.glb
   sudo journalctl --vacuum-time=7d
   sudo rm -f /var/run/spark-prewarm.done     # reset sentinel for next boot
   sync
   ```
5. From CPU:
   ```
   aws ec2 create-image \
     --instance-id <spot-instance-id> \
     --name "spark-gpu-pixal3d-<YYYY-MM-DD>" \
     --description "ASCII description, no unicode" \
     --no-reboot \
     --tag-specifications \
       "ResourceType=image,Tags=[{Key=Project,Value=spark-gpu},{Key=BakeDate,Value=<YYYY-MM-DD>}]" \
       "ResourceType=snapshot,Tags=[{Key=Project,Value=spark-gpu}]" \
     --region us-east-1
   ```
6. Wait for the AMI to be `available` (10-15 min).
7. Update `AWS_GPU_AMI_ID` in CPU `.env` so the next spot launch uses the new image.
8. (Optional) `aws ec2 deregister-image --image-id <old-ami>` to clean up the previous AMI; snapshots persist unless explicitly deleted.

## Spot interruption handling

`InstanceInterruptionBehavior=stop` (set in `_launch_spot_from_ami()`)
means AWS will **stop** the instance — not terminate — when capacity is
reclaimed. The EBS volume persists. When AWS capacity returns, the
instance auto-starts. The EIP stays attached across stop/start, so the
CPU sees no IP change.

If AWS does fully terminate (rare, e.g. zonal capacity exhaustion),
`ensure_gpu_ready()` notices the missing/terminated instance on the next
queue push and calls `_launch_spot_from_ami()` to create a fresh one. The
new instance picks up the EIP, runs prewarm, and joins the queue. No
manual intervention required.

## Monitoring

```bash
# On GPU
sudo journalctl -u spark-prewarm -f
sudo journalctl -u manual_gen_worker -f
tail -f /var/log/spark-prewarm.log

# Sentinel check
ls -la /var/run/spark-prewarm.done    # exists when prewarm complete
```

# GPU failover & AZ relocation

The GPU compute is a **single roaming 256 GB root volume** (OS + conda + all
models) that we attach to whichever g6e.2xlarge can actually run. We never keep
two big EBS volumes. A stable **Elastic IP** rides with whatever instance is
active, so the CPU never has to chase a changing address.

## Why this exists

- Spot g6e in **us-east-1d** kept getting denied (capacity), and on-demand g6e
  is unavailable in 1d too — the AZ is dry for that family.
- The models volume was stranded as the **root** of the 1d spot.
- EBS is AZ-locked, so reaching a live AZ (**us-east-1b**) requires a
  snapshot → new-volume-in-target-AZ hop (`relocate_models_az.sh`).
- Once both candidate instances live in the **same** AZ, switching between them
  is a direct detach/attach of the one roaming root (`switch_gpu.sh`).

## Canonical resources (update if they change)

| Thing | ID |
|---|---|
| Models volume (roaming root, 256 GB, us-east-1b) | `vol-03e4c94a61881e7b2` |
| On-demand g6e (us-east-1b) | `i-089872baa8e109ca3` |
| Spot g6e (us-east-1d, dead AZ) | `i-0f7766b2b07e8b372` |
| Elastic IP (52.91.128.47) | `eipalloc-0db12aa4d8be94e92` |
| DR AMI (hot, 256 GB, models baked 2026-05-16) | `ami-03fbbf973df14672f` |
| Region | `us-east-1` |

## Operating model

- **Primary = on-demand in the live AZ + aggressive autoshutdown.** Reliable;
  billed only while a job runs. This is the day-to-day runtime.
- **Spot = opportunistic, same-AZ only.** If g6e spot capacity returns *in the
  live AZ*, `switch_gpu.sh spot` flips the roaming root onto a spot for the
  discount, and rolls back to on-demand automatically if the spot won't start.
- **AZ relocation = rare.** Only when the whole AZ goes dry. `relocate_models_az.sh`.
- **DR = the AMI.** `ami-03fbbf973df14672f` already contains the models; a cold
  start (bootstrap re-pull) is the last resort if the roaming volume is ever lost.

## DeleteOnTermination

The roaming volume must always be `DeleteOnTermination=false` so a terminate
(spot reclaim or manual) never destroys the models. Both scripts enforce this
after every attach.

## Files

| File | Use |
|---|---|
| `relocate_models_az.sh` | Move the models volume to another AZ via snapshot. Rare. |
| `switch_gpu.sh` | Same-AZ: move the roaming root between two instances + EIP, with rollback. |

Both scripts are **idempotent-ish and gated**: they print the plan and require
`--yes` to mutate. Destructive deletes are never automatic.

## Post-switch checklist (learned 2026-06-11, the 1d→1b relocation)

A new/changed instance is NOT done when it boots. Verify each of these — every
one bit us during the first relocation:

1. **Security group**: the CPU SG must allow Redis 6379 + Mongo 27017 from the
   *new instance's* SG (rules are SG-to-SG, they don't follow the volume).
2. **CPU private IP pinning**: `.env.gpu` (REDIS_*/MONGO*) and `app/infra.py`
   pin the CPU's private IP — verify it matches `172.31.26.6` (it changes if
   the CPU instance is ever replaced).
3. **`app/infra.py`**: GPU_INSTANCE_ID / GPU_PUBLIC_IP / GPU_PRIVATE_IP and
   GPU_QUEUES must reflect the active instance + queue, or the orchestrator
   manages a ghost.
4. **CPU `.env.cpu`**: AWS_GPU_INSTANCE_ID, AWS_GPU_IS_SPOT_INSTANCE,
   GPU_INSTANCE_MAP → restart `fastapi_app.service`.
5. **Retry capacity errors**: `start-instances` can throw
   InsufficientInstanceCapacity transiently — retry a few times before
   declaring an AZ dry (1b succeeded on attempt 2).
6. **spark-prewarm ran?** `systemctl status spark-prewarm` + sentinel
   `/var/run/spark-prewarm.done`. On a snapshot-restored volume the first
   model load crawls (~4 MB/s mmap) until prewarm initializes the blocks.
7. **Exactly one worker service**: `manual_gen_worker.service` only.
   Legacy units (`trellis_worker`, `spark-gpu-worker`) each carry their own
   AutoShutdown clock → racing stop_instances. Keep them disabled.
8. **IAM instance profile**: a NEW instance launches with no profile — the
   worker then fails S3 uploads with "Unable to locate credentials" *after*
   a full inference. Attach `ec2_s3` (`aws ec2 associate-iam-instance-profile
   --instance-id <id> --iam-instance-profile Name=ec2_s3`, works while
   running) and restart `manual_gen_worker` so boto3 re-resolves credentials.
9. **End-to-end test from the CPU**: queue a flux job via
   `/manual-gen/queue/flux` and confirm image_url lands in the run doc.
   If a stale run blocks queueing ("already queued"), mark the orphan
   `status=error` in `manual_gen_stage_runs` first.

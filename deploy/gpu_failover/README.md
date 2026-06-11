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
| Models volume (roaming root, 256 GB) | `vol-0c7bb27c340f01b05` *(replaced after relocation)* |
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

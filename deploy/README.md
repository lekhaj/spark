# Spark GPU bootstrap

Auto-installs missing stages, models, and pip deps on the L40S GPU instance
at every boot. **The git repo is the only source of truth** — never edit code
on the live instance.

> The GPU (`spark_gpu` `i-089872baa8e109ca3`, 52.91.128.47) is now an
> **on-demand** instance, normally **stopped** — start it before a run, stop it
> after. It was a spot box historically; this bootstrap workflow is identical
> either way (it runs at every boot regardless of lifecycle).

## What it does

On every boot of the GPU instance, `spark-bootstrap.service` runs **before**
`manual_gen_worker.service` and:

1. `git fetch && git reset --hard origin/main` in `/home/ec2-user/spark`
   (unless `~/.spark/pin_branch` exists — pin escape hatch)
2. Parses [`spec.yaml`](./spec.yaml) to learn what stages this role needs.
3. For each stage:
   - If `~/.spark/installed/<stage>.v<N>` sentinel exists → skip.
   - Else runs `install/<stage>.sh` (pip + HF model pulls + patches).
   - On success, writes the sentinel.
4. Syncs systemd units from `deploy/systemd/` → `/etc/systemd/system/`.
5. Publishes `bootstrap:status:<instance_id>` to Redis (`ready` / `installing:<stage>` / `failed:<reason>`).

Idempotent — completes in ~5s when nothing has changed.

---

## Adding a new stage (the only workflow you should ever use)

1. Write the model file (e.g. `worker/models/<new>_model.py`).
2. Wire the handler in `worker/workers/manual_gen_worker.py`.
3. Add the stage to `worker/lib/manual_gen_schema.py:STAGE_NAMES`.
4. Add UI to `app/gradio/pages/generation_studio_page.py`.
5. Write `deploy/install/<new>.sh` — handles its pip deps, HF model pulls,
   any patches. Mark executable: `chmod +x deploy/install/<new>.sh`.
6. Add a `stage_defs` entry in `deploy/spec.yaml` and append the name to
   `stages:`. Set `version: 1`.
7. Commit + push to `main`.
8. (Spot will auto-pick this up at next boot, or restart `spark-bootstrap`
   manually for an immediate apply: `sudo systemctl restart spark-bootstrap`.)

**To force a stage to re-install** (e.g. after changing its installer or
adding a model): bump `version:` in `spec.yaml` and commit. Next bootstrap
sees a missing sentinel for the new version and re-runs.

---

## One-time AMI setup (already done, document for future)

When baking the lean spot AMI:

```bash
# As ec2-user
git clone https://github.com/lekhaj/spark.git /home/ec2-user/spark
chmod +x /home/ec2-user/spark/deploy/bootstrap.sh /home/ec2-user/spark/deploy/install/*.sh

# Install systemd units (this is the ONLY direct file write — after this,
# bootstrap.sh manages systemd from the repo).
sudo cp /home/ec2-user/spark/deploy/systemd/spark-bootstrap.service /etc/systemd/system/
sudo mkdir -p /etc/systemd/system/manual_gen_worker.service.d
sudo cp /home/ec2-user/spark/deploy/systemd/dropins/manual_gen_worker.service.d/bootstrap-dep.conf \
        /etc/systemd/system/manual_gen_worker.service.d/
sudo systemctl daemon-reload
sudo systemctl enable spark-bootstrap.service

# Allow ec2-user passwordless sudo for the few things bootstrap needs to
# touch (cp into /etc/systemd, daemon-reload, systemctl). Add to sudoers:
#   ec2-user ALL=(ALL) NOPASSWD: /bin/install -m 644 *, /bin/cp *, \
#                                /bin/mkdir *, /bin/systemctl daemon-reload, \
#                                /bin/cmp *
```

After that, bake the AMI. All future code/model/dep changes happen via git
commits to `main` — never via SSH edits.

---

## Manual commands

| Command | What |
|---|---|
| `sudo systemctl restart spark-bootstrap` | Re-run install (pulls latest git + delta-installs) |
| `journalctl -u spark-bootstrap -f` | Watch bootstrap progress live |
| `ls ~/.spark/installed/` | See which stages are installed at which versions |
| `rm ~/.spark/installed/<stage>.v<N>` | Force re-install of a single stage on next bootstrap |
| `touch ~/.spark/pin_branch` | Pin current commit, skip git update on next bootstrap |
| `touch ~/.spark/skip_bootstrap` | Disable bootstrap entirely (emergency) |
| `redis-cli GET bootstrap:status:<iid>` | Check status from CPU |

---

## File map

| File | Purpose |
|---|---|
| `deploy/spec.yaml` | What each stage needs (pip, HF models, env, version) |
| `deploy/bootstrap.sh` | The orchestrator (oneshot at boot) |
| `deploy/lib/common.sh` | Helpers: sentinel check, hf_pull, pip_install, conda activate |
| `deploy/install/<stage>.sh` | Per-stage installer — one file per stage |
| `deploy/systemd/spark-bootstrap.service` | The systemd unit |
| `deploy/systemd/dropins/manual_gen_worker.service.d/bootstrap-dep.conf` | Gates worker on bootstrap success |

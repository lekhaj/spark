#!/bin/bash
# ensure_access.sh — unify SSH access across BOTH g7e GPU boxes.
#
# WHY: the on-demand box was created with KeyName=us_cpu_key (cloud-init injected
# us_cpu_key for ec2-user); the spot box was created with KeyName=None (only a
# hand-added key, likely s_spu_key). That mismatch caused repeated "which key /
# which user" confusion. Both boxes run Amazon Linux 2023, so the user is ALWAYS
# ec2-user. This script makes BOTH local keys work on whichever box it runs on.
#
# RUN once on each GPU box while it is up (re-run safe — idempotent):
#   scp -i us_cpu_key.pem deploy/gpu/ensure_access.sh ec2-user@<box-ip>:/tmp/
#   ssh -i us_cpu_key.pem ec2-user@<box-ip> 'bash /tmp/ensure_access.sh "<us_cpu_pub>" "<s_spu_pub>"'
# (pass the two PUBLIC keys as args; get them locally with `ssh-keygen -y -f <key>.pem`)
#
# Better long-term: create every new GPU instance with `--key-name us_cpu_key` so
# cloud-init injects it automatically and this script becomes unnecessary.
set -euo pipefail

AUTH="/home/ec2-user/.ssh/authorized_keys"
mkdir -p /home/ec2-user/.ssh
touch "$AUTH"
chmod 700 /home/ec2-user/.ssh
chmod 600 "$AUTH"

added=0
for pub in "$@"; do
  [ -z "$pub" ] && continue
  if ! grep -qF "$pub" "$AUTH"; then
    echo "$pub" >> "$AUTH"
    added=$((added+1))
  fi
done
chown -R ec2-user:ec2-user /home/ec2-user/.ssh
echo "ensure_access: added $added key(s); $(wc -l < "$AUTH") total authorized."

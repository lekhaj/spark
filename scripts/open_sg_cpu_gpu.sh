#!/usr/bin/env bash
# Open Security Group rules for CPU ↔ GPU communication.
#
# WHY: The GPU worker uploads images to S3 (works), then must update MongoDB
# on the CPU instance (FAILS — port 27017 is blocked by AWS SG). This script
# adds the missing ingress rules so the GPU SG can reach CPU's MongoDB and
# Redis ports over the private network.
#
# Run from a machine that has fresh AWS credentials (e.g. your laptop after
# `aws sso login` or with valid keys in ~/.aws/credentials).
#
# Re-running is safe — AuthorizeSecurityGroupIngress errors on duplicates
# are caught and ignored.

set -u
REGION="us-east-1"
CPU_SG="sg-068df98169b9850cc"   # spark_cpu (i-0663f7700a2da31c7)
GPU_SG="sg-0a4a561065082e3c9"   # spark_gpu (i-0d6b9d6d34ccc053d)

authorize() {
    local group=$1 port=$2 src=$3 desc=$4
    echo "  → ${desc} (port ${port})"
    aws ec2 authorize-security-group-ingress \
        --region "${REGION}" \
        --group-id "${group}" \
        --protocol tcp --port "${port}" \
        --source-group "${src}" \
        --output text 2>&1 | grep -v "InvalidPermission.Duplicate" || true
}

echo "Opening CPU SG (${CPU_SG}) for GPU SG (${GPU_SG}):"
authorize "${CPU_SG}" 27017 "${GPU_SG}" "MongoDB"
authorize "${CPU_SG}" 6379  "${GPU_SG}" "Redis (default)"
authorize "${CPU_SG}" 6380  "${GPU_SG}" "Redis (alt)"

echo
echo "Opening GPU SG (${GPU_SG}) for CPU SG (${CPU_SG}):"
authorize "${GPU_SG}" 22 "${CPU_SG}" "SSH (orchestrator)"

echo
echo "✅ Done. Verify from GPU:"
echo "  ssh -i us_cpu_key.pem ec2-user@3.215.211.192 \\"
echo "    \"timeout 5 bash -c '</dev/tcp/172.31.92.14/27017' && echo OK\""

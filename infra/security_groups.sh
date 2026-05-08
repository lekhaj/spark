#!/usr/bin/env bash
# infra/security_groups.sh — Spark Security Group reconciliation (idempotent).
#
# WHY: Single source of truth for which ports/sources are allowed between
# the spark CPU and GPU instances. Run this after any infra change to
# ensure the SGs match the declared spec.
#
# IMPORTANT: This script ONLY manages spark-owned rules. The CPU instance
# also hosts Helmio (frontend, backend, postgres on ports 80/3000/7000/8001/8081)
# — those rules are explicitly left alone.
#
# Run with fresh AWS credentials:
#   aws sso login   # or export AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
#   ./infra/security_groups.sh           # show diff
#   ./infra/security_groups.sh --apply   # apply changes
#
# Re-running is safe: existing rules trigger InvalidPermission.Duplicate,
# which is suppressed.

set -u

# ── Topology ──────────────────────────────────────────────────────────────────
REGION="us-east-1"
CPU_SG="sg-068df98169b9850cc"   # spark_cpu (i-0663f7700a2da31c7)
GPU_SG="sg-0a4a561065082e3c9"   # spark_gpu (i-0d6b9d6d34ccc053d)

# ── Desired state ─────────────────────────────────────────────────────────────
# Format: "PORT:DESCRIPTION"
# Public = exposed to 0.0.0.0/0 (user/internet facing)
# CPU↔GPU = restricted to the peer SG only (private VPC traffic)

# CPU public-facing (Spark only — Helmio's 80/3000/7000/8001/8081 are NOT managed here)
CPU_PUBLIC=(
    "22:SSH"
    "7860:Gradio UI"
    "8000:FastAPI"
)

# CPU ports reachable only from GPU SG (worker → CPU services)
CPU_FROM_GPU=(
    "6379:Redis (broker)"
    "6380:Redis (alt)"
    "27017:MongoDB"
)

# GPU public-facing
GPU_PUBLIC=(
    "22:SSH"
)

# GPU ports reachable only from CPU SG (orchestrator → GPU)
GPU_FROM_CPU=(
    "22:Orchestrator SSH"
)

# ── AWS helpers ───────────────────────────────────────────────────────────────
APPLY=0
[[ "${1:-}" == "--apply" ]] && APPLY=1

aws_authorize_cidr() {
    local sg=$1 port=$2 cidr=$3 desc=$4
    if (( APPLY == 0 )); then
        echo "  [DRY-RUN] would authorize ${sg}: tcp/${port} from ${cidr}  (${desc})"
        return
    fi
    aws ec2 authorize-security-group-ingress \
        --region "${REGION}" --group-id "${sg}" \
        --ip-permissions "IpProtocol=tcp,FromPort=${port},ToPort=${port},IpRanges=[{CidrIp=${cidr},Description=${desc// /-}}]" \
        --output text 2>&1 | grep -v "InvalidPermission.Duplicate" || true
}

aws_authorize_sg() {
    local sg=$1 port=$2 src_sg=$3 desc=$4
    if (( APPLY == 0 )); then
        echo "  [DRY-RUN] would authorize ${sg}: tcp/${port} from ${src_sg}  (${desc})"
        return
    fi
    aws ec2 authorize-security-group-ingress \
        --region "${REGION}" --group-id "${sg}" \
        --ip-permissions "IpProtocol=tcp,FromPort=${port},ToPort=${port},UserIdGroupPairs=[{GroupId=${src_sg},Description=${desc// /-}}]" \
        --output text 2>&1 | grep -v "InvalidPermission.Duplicate" || true
}

# ── Reconcile ────────────────────────────────────────────────────────────────
echo "════════════════════════════════════════════════════════════════════"
echo "  Spark Security Group reconciliation  (mode: $((APPLY)) ? apply : dry-run)"
echo "  Region: ${REGION}"
echo "  CPU SG: ${CPU_SG}"
echo "  GPU SG: ${GPU_SG}"
echo "════════════════════════════════════════════════════════════════════"
[[ ${APPLY} -eq 0 ]] && echo "  (Pass --apply to actually make changes)"
echo

echo "── CPU public-facing (0.0.0.0/0) ─────────────────────────────────────"
for entry in "${CPU_PUBLIC[@]}"; do
    port=${entry%%:*}; desc=${entry#*:}
    aws_authorize_cidr "${CPU_SG}" "${port}" "0.0.0.0/0" "${desc}"
done

echo
echo "── CPU ← GPU SG only (private) ──────────────────────────────────────"
for entry in "${CPU_FROM_GPU[@]}"; do
    port=${entry%%:*}; desc=${entry#*:}
    aws_authorize_sg "${CPU_SG}" "${port}" "${GPU_SG}" "${desc}"
done

echo
echo "── GPU public-facing (0.0.0.0/0) ─────────────────────────────────────"
for entry in "${GPU_PUBLIC[@]}"; do
    port=${entry%%:*}; desc=${entry#*:}
    aws_authorize_cidr "${GPU_SG}" "${port}" "0.0.0.0/0" "${desc}"
done

echo
echo "── GPU ← CPU SG only (private) ──────────────────────────────────────"
for entry in "${GPU_FROM_CPU[@]}"; do
    port=${entry%%:*}; desc=${entry#*:}
    aws_authorize_sg "${GPU_SG}" "${port}" "${CPU_SG}" "${desc}"
done

echo
echo "── Audit: rules currently in CPU SG NOT in this spec ─────────────────"
echo "    (Helmio uses 80, 3000, 7000, 8001, 8081 — left untouched)"
echo "    Review with:"
echo "      aws ec2 describe-security-groups --group-ids ${CPU_SG} --region ${REGION}"
echo
echo "✅ Done. To verify CPU ↔ GPU connectivity:"
echo "  ssh -i us_cpu_key.pem ec2-user@3.215.211.192 \\"
echo "    \"timeout 5 bash -c '</dev/tcp/172.31.92.14/27017' && echo MONGO_OK\""

#!/usr/bin/env bash
# Smoke test for the new /manual-gen REST surface.
#
# Run on the CPU instance after deploying:
#   bash scripts/smoke_manual_gen_api.sh
#
# Or from anywhere with the BASE env var:
#   BASE=https://api.example.com bash scripts/smoke_manual_gen_api.sh
#
# This is READ-ONLY by default. Pass QUEUE=1 to also POST a queue/flux job
# (spends GPU credits when the worker picks it up).

set -uo pipefail

BASE="${BASE:-http://localhost:8000}"
DEMO_CHAR="${DEMO_CHAR:-VIC_warden}"

ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$1"; exit 1; }

echo "smoke: $BASE"

echo "→ GET /manual-gen/health"
curl -fsS "$BASE/manual-gen/health" | python3 -m json.tool || fail "health"
ok "health"

echo "→ POST /manual-gen/characters (register $DEMO_CHAR)"
curl -fsS -X POST "$BASE/manual-gen/characters" \
  -H "Content-Type: application/json" \
  -d "{\"label\": \"$DEMO_CHAR\"}" \
  | python3 -m json.tool || fail "create_character"
ok "create $DEMO_CHAR"

echo "→ GET /manual-gen/characters"
curl -fsS "$BASE/manual-gen/characters" | python3 -m json.tool >/dev/null || fail "list_chars"
ok "list_characters"

echo "→ GET /manual-gen/characters/$DEMO_CHAR/stages"
curl -fsS "$BASE/manual-gen/characters/$DEMO_CHAR/stages" | python3 -m json.tool >/dev/null || fail "list_stages"
ok "list_stages_for_char"

echo "→ GET /manual-gen/characters/$DEMO_CHAR/stages/flux/versions"
curl -fsS "$BASE/manual-gen/characters/$DEMO_CHAR/stages/flux/versions" | python3 -m json.tool >/dev/null || fail "list_versions"
ok "list_stage_versions"

if [[ "${QUEUE:-0}" == "1" ]]; then
  echo "→ POST /manual-gen/queue/flux (will burn GPU)"
  curl -fsS -X POST "$BASE/manual-gen/queue/flux" \
    -H "Content-Type: application/json" \
    -d "$(cat <<EOF
{
  "char": "$DEMO_CHAR",
  "major": 1,
  "minor": 0,
  "prompt": "isometric stylised portrait of a weathered hooded traveler in a long coat, soft amber rim light, painterly, three-quarter view, no background, character sheet style",
  "negative": "blurry, low quality, signature, watermark",
  "width": 1024,
  "height": 1024,
  "steps": 30,
  "guidance": 4.0
}
EOF
)" | python3 -m json.tool || fail "queue_flux"
  ok "queue_flux (check status via GET /manual-gen/runs/<run_id>)"
else
  echo "  ⏸  Skipping queue test (set QUEUE=1 to enable)"
fi

echo
echo "All reads OK against $BASE"

# Pending architecture proposals (not yet started)

System works fine right now without these — on-demand g6e in us-east-1b,
roaming root vol-03e4c94a61881e7b2, EIP 52.91.128.47, autoshutdown validated
2026-06-11 (delegated-stop path via Redis `autoshutdown:stop_requested:<iid>`).

These are optional follow-ups, suggested order C → A → B.

## C. Rotate Mongo creds out of tracked .env files
Move MONGO_URI/MONGODB_URL (and any other secrets currently in `.env.cpu` /
`.env.gpu`) into `.env.secrets` (gitignored). Tracked env files keep only
non-secret config (instance IDs, queue maps, hosts). Smallest, lowest-risk
change — do this first.

## A. Stateless GPU via baked AMI
Bake a fresh AMI with all models pre-loaded (refresh `ami-03fbbf973df14672f`
or build new). New GPU instances launch from this AMI with no roaming EBS
volume — disposable, no AZ-lock problem, no cross-AZ snapshot dance ever
again. Tradeoff: AMI rebake needed whenever models/deps change; storage cost
moves from EBS to AMI snapshot (similar $).

## B. Modular async refactor
- StageRegistry pattern for the 9 manual-gen stages (currently dispatch logic
  in manual_gen_worker.py)
- Non-blocking aws_service calls (orchestrator currently does sync SSH calls
  in its poll loop)
- Split orchestrator's single loop into independent async tasks (queue watch,
  idle/stop logic, worker health) so one slow check doesn't stall others

---
Pick one and pick up the conversation referencing this file.

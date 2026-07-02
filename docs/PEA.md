# Player-Experience Analytics (PEA) — integrated into Spark

Mixpanel → deterministic moods/FELT/personality → grounded narration → Postgres → Studio.
Lives inside this backend (`app/pea/` + `app/routes/pea_routes.py`, mounted at `/pea`) and the
studio frontend (`spark_studio` `/studio/analytics` + the "Player XP" dock view). Reuses the
CycleZero Postgres (tables prefixed `pea_`). Keyed by `game_id` to generalize to other games.

> ⚠️ Pre-launch: ~14 users, no `env` flag → every output stamped "internal/QA, not player behavior".

## Backend layout
```
app/pea/
  config.py       # ALL thresholds + Mixpanel/DB wiring (secrets from app.config.settings)
  extract.py      # incremental Mixpanel raw export -> pea_raw_events (dedupe $insert_id)
  reconcile.py    # 3 level encodings -> one int level_id (buildIndex-1 == LevelNumber)
  moods.py        # deterministic entry/exit mood + feeling + persona (NO LLM)
  felt.py         # FELT labels: tension / mastery / autonomy (aligns w/ FELT Scoring Plan)
  personality.py  # spectrum archetypes: puzzle-solver/casual-gamer/analytical-serious/...
  aggregate.py    # session->player rollup, daily digest, level friction, funnel/retention
  narrate.py      # LLM narration, STRICTLY grounded (falls back to templates w/o key)
  bringback.py    # who/when/what bring-back list (manual CSV; no push channel yet)
  store.py        # psycopg2 upserts + pandas loader over the CycleZero Postgres
  run_batch.py    # nightly orchestrator: python -m app.pea.run_batch
  schema.sql      # pea_* derived tables
app/routes/pea_routes.py   # /pea/* read-API (SQLAlchemy session; reads pea_* only)
```

## One-time setup on the CPU box (18.207.13.85)
1. Add to `.env.secrets` (gitignored) — **use the ROTATED Mixpanel secret, not the leaked one**:
   ```
   MIXPANEL_SA_USER=spark.9682f0.mp-service-account
   MIXPANEL_SA_SECRET=<rotated secret>
   MIXPANEL_PROJECT_ID=3631004
   ```
   (Deps already in requirements.txt: pandas, requests, psycopg2-binary, anthropic, SQLAlchemy.)
2. Bootstrap tables + 30-day backfill:
   ```
   cd /home/ubuntu/spark && python -m app.pea.run_batch --backfill 30
   ```

## Nightly schedule (systemd timer, alongside the existing service)
`/etc/systemd/system/pea-batch.service` (Type=oneshot, `ExecStart=/usr/bin/python -m app.pea.run_batch`,
`WorkingDirectory=/home/ubuntu/spark`) + `pea-batch.timer` (`OnCalendar=*-*-* 02:30 Asia/Kolkata`).
`sudo systemctl enable --now pea-batch.timer`.

## Deploy (standard Spark flow)
Backend: `git push origin main` → on 18.207.13.85 `git pull --ff-only && sudo systemctl restart fastapi_app.service`.
Frontend: `cd spark_studio && nvm use 22 && npm install && npm run build && npm run deploy` (wrangler → Cloudflare Pages).
The `/_api` Pages Function already proxies `/pea/*` to the HTTP backend (no new proxy config needed).

## Endpoints (mounted at /pea)
`/pea/health · /digest · /mood-trends · /personality · /player/{distinct_id} · /friction · /watch-list · /bringback · /funnel`

## Closing the data gaps
`env` flag, booster/decision events (unlock the AUTONOMY / `independence` personality axis),
real `session_id`, unified `level_id`, `shot_fired.deliberation_ms`. Ready-to-paste Unity asks:
[PEA_UNITY_EVENT_PROMPT.md](PEA_UNITY_EVENT_PROMPT.md).

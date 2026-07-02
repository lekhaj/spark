# Unity event-instrumentation prompt (paste into the AuraBeam thread)

> This closes the gaps the analytics service found in the live Mixpanel data (project 3631004).
> Order matters: **envelope + hygiene first (P0)**, then the **booster/decision events (P1)** that
> unlock the AUTONOMY / independence personality axis, then the **richer gameplay signals (P2)**.
> All go through the existing `GameEvents` → `GameAnalyticsListener` seam + `MixpanelManager`
> (`Mixpanel.Register` for super properties). Keep numerics numeric. Do not rename existing events
> without telling the analytics side — we key off the exact names below.

---
## P0 — Common envelope (super properties on EVERY event, via `Mixpanel.Register`)
Set once at startup / first open:
- `game_id` (string, e.g. "aurabeam") — required so multi-game reuse works.
- `env` (string: `dev` in editor/debug builds, `prod` in release) — **highest priority**; without it we
  cannot separate testers from players and every metric stays labelled "internal/QA".
- `build_version` (already have `$app_version_string` — fine, just keep it stable).
- `session_id` (GUID minted at each app-foreground; stamped on every event until app background/close).
- `platform` / `device_class` (have `$os`/`$model`; add a coarse `device_class` low/mid/high if easy).
- `days_since_install` (from first-open date) and `ab_variant` (null for now).
- `$insert_id` on every event (dedupe) — we already rely on this; keep it.

## P0 — Fix these existing inconsistencies
1. **Unify level id.** `Game Start.Level` currently sends a numeric **string**; `Level Completed`
   sends `LevelNumber` (int). Emit a single **`level_id` (int)** on BOTH (and on all `Journey:*`
   events alongside `buildIndex`). Convention we use: `level_id = buildIndex − 1` (MainMenu = buildIndex 1).
2. **Split `App Opened`.** It fires once-ever today. Emit **`first_open`** (once = install) and
   **`app_open`** (every cold start) so DAU/retention work.
3. **One client timestamp.** Add `ts_client` (unix seconds) on every event; drop ad-hoc
   `Timestamp`/`CompletedAt`/`WatchedAt`.
4. **Keep numerics numeric** — retries, hearts_remaining, time_s, reflections, shards must be numbers, not strings.
5. **Resolve the two undocumented events** `Journey: StartGame` and `Journey: LevelNavigation.StartGame`
   — tell us which is canonical vs `Game Start`, or fold them together.

---
## P1 — Booster & decision events (unlocks AUTONOMY / independence — currently a hard DATA GAP)
The personality/FELT AUTONOMY axis (confident-independent / strategic-user / dependent / stuck-without-tools)
is unresolvable without these. Boosters that exist: **Pulse Beam, Mirror Reveal, Employee Remover**
(+ WIP Aura Overload, Boss Shield).

- **`booster_offered`** { level_id, booster_type, offered_free (bool), shards_balance }
- **`booster_used`** { level_id, booster_type, cost_type (`free`|`shards`|`ad`), cost_shards, uses_left }
- **`booster_declined`** — *derive it if you can't fire it*: at attempt end emit
  **`booster_available_not_used`** { level_id, booster_type } (strong autonomy signal).
- **`booster_upgraded`** { booster_type, param, new_tier, shards_spent }
- **`decision_point`** (generic "meaningful choice") { level_id, kind, options_count, chosen } — for
  AuraBeam the richest is the booster-choice moment and the mirror-rotation-before-fire moment.

## P1 — Retry/fail clarity (we infer frustration from these; make them explicit)
- Keep the `Journey: LEVEL FAILED (*)` / `Life lost` / `TryAgain` events, but add
  **`retry`** { level_id, attempt_index, hearts_remaining, reason } and
  **`level_abandon`** { level_id, last_action } for a clean "interrupted vs frustrated" split.
- **`app_background`** { last_event, level_id } — the abandon-point detector; lets us stop guessing
  "interrupted" from a missing `$session_end`.

---
## P2 — Richer gameplay signals (sharpen method/mastery/exploration dimensions)
These upgrade personality accuracy from proxies to real signals:
- **`shot_fired`** { level_id, aim_angle, mirrors_rotated_before_fire, deliberation_ms (time from
  level load / last action to fire), sweet_spot_hit (bool) } — `deliberation_ms` + `mirrors_rotated`
  turn the "method/analytical vs impulsive" axis from a fail-rate proxy into a real measure.
- **`target_hit` / `target_missed`** { level_id, npc_type (Worker|Boss|God), polarity_correct (bool) }
  — hitting Boss = instant fail; wrong-polarity kill is a distinct mistake worth its own signal.
- **`near_miss`** { level_id, workers_left, hearts_remaining } — fired when a run ends 1 target short;
  converts a loss into TENSION (earned-relief vs defeated).
- **`hint_shown` / `hint_used`** { level_id, hint_type } — strategic-user vs stuck signal.
- **`resource_spend` / `resource_grant`** { currency (aura|shards), amount, reason } and
  **`iap_view` / `iap_purchase`** { product_id, price } — for monetization-vs-frustration tension.

---
## Attribution (separate track, not blocking)
Add the **Google Play Install Referrer API** → push referrer into a super property on first open, so
Play Store UAC installs get attributed (current `utm_*` only fire on deep-link opens).

---
### What each unblocks on the analytics side
| You add | We can finally compute |
|---|---|
| `env` | separate real players from testers → drop the QA banner |
| booster_* / decision_point | AUTONOMY label + `independence` personality dimension (today = "unknown") |
| session_id / app_background | reliable sessions + true "interrupted" (stop stitching) |
| level_id everywhere | clean per-level friction + start→complete join |
| first_open/app_open | correct DAU, D1/D7 retention |
| shot_fired.deliberation_ms | real "analytical/methodical" personality (not a fail-rate proxy) |
| near_miss / polarity_correct | FELT TENSION earned-relief vs defeated; boss-mistake analytics |

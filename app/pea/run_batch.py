"""
PEA nightly batch. Run on the CPU box:
    cd /home/ubuntu/spark && python -m app.pea.run_batch --backfill 30   # bootstrap
    cd /home/ubuntu/spark && python -m app.pea.run_batch                 # incremental nightly

Order: extract -> load cache -> sessions+moods+FELT -> player rollup(+personality)
-> digest/friction/funnel -> narrate -> bring-back -> upsert pea_* tables.
Compute reads the Postgres cache, never live Mixpanel.
"""
from __future__ import annotations

import argparse
import datetime as dt
from collections import defaultdict

from . import config as C, extract, store, aggregate, narrate, bringback
from .moods import build_session_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backfill", type=int, default=0)
    ap.add_argument("--no-pull", action="store_true")
    args = ap.parse_args()

    store.init_schema()

    if not args.no_pull:
        if args.backfill:
            today = dt.datetime.now(C.TIMEZONE).date()
            extract.pull_range(today - dt.timedelta(days=args.backfill), today - dt.timedelta(days=1))
            store.set_watermark(C.GAME_ID, today - dt.timedelta(days=1))
        else:
            extract.incremental()

    df = store.load_cached_events()
    if df.empty:
        print("[pea.batch] no cached events; nothing to compute.")
        return

    if C.ENV_PROPERTY in df.columns and df["env"].notna().any():
        df = df[~df["env"].isin(C.ENV_EXCLUDE_VALUES)]
    else:
        print(f"[pea.batch] NOTE: no '{C.ENV_PROPERTY}' flag -> {C.PRELAUNCH_BANNER}")

    sessions = build_session_state(df)
    _resolve_churn_risk(sessions)
    players = aggregate.rollup_players(sessions)
    for p in players:
        p["narrative"] = narrate.narrate_player(p)

    sessions_by_player = defaultdict(list)
    for s in sessions:
        sessions_by_player[s["distinct_id"]].append(s)
    dates = sorted({p["date"] for p in players})
    digests, friction_rows, funnel_rows = [], [], []
    for date in dates:
        dg = aggregate.build_digest(players, date)
        dg["insights"] = narrate.narrate_digest_insights(dg)
        digests.append(dg)
        friction_rows += aggregate.build_level_friction(sessions, date)
        funnel_rows.append(aggregate.build_funnel_retention(sessions, df, date))

    latest = dates[-1]
    bb = bringback.build_bringback(players, sessions_by_player, latest)

    store.upsert_sessions(sessions)
    store.upsert_players(players)
    store.upsert_digest(digests)
    store.upsert_friction(friction_rows)
    store.upsert_bringback(bb)
    store.upsert_funnel(funnel_rows)
    print(f"[pea.batch] done: {len(sessions)} sessions, {len(players)} player-days, "
          f"{len(digests)} digests, {len(friction_rows)} friction, {len(funnel_rows)} funnel, {len(bb)} bring-back.")


def _resolve_churn_risk(sessions: list[dict]):
    from .config import MOODS
    by_player = defaultdict(list)
    for s in sessions:
        by_player[s["distinct_id"]].append(s)
    for did, ss in by_player.items():
        ss.sort(key=lambda s: s["started_at"] or dt.datetime.min)
        gaps = []
        for i in range(1, len(ss)):
            if ss[i]["session_date"] and ss[i - 1]["session_date"]:
                gaps.append((ss[i]["session_date"] - ss[i - 1]["session_date"]).days)
        for i, s in enumerate(ss):
            if s["exit_mood"] != "frustrated":
                continue
            if i == len(ss) - 1:  # no later session -> churn-risk
                s["exit_mood"] = "churn-risk"
                s["evidence"].append({"exit": ["frustrated + no subsequent session within cadence"]})


if __name__ == "__main__":
    main()

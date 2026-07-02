"""
Incremental Mixpanel Raw Export -> pea_raw_events (Postgres cache).
HTTP Basic auth = base64(SA_USER:SA_SECRET) from settings. Dedupe on $insert_id.
Pull by date, resume from watermark; never re-pull full history.
"""
from __future__ import annotations

import base64
import datetime as dt
import json
from typing import Iterator

import requests

from . import config as C
from . import store
from .reconcile import reconcile_level_id, normalize_event


def _auth_header() -> dict:
    if not C.MIXPANEL_SA_USER or not C.MIXPANEL_SA_SECRET:
        raise RuntimeError(
            "MIXPANEL_SA_USER / MIXPANEL_SA_SECRET not set in .env.secrets. "
            "Rotate the secret that was pasted in chat; never hardcode it.")
    token = base64.b64encode(f"{C.MIXPANEL_SA_USER}:{C.MIXPANEL_SA_SECRET}".encode()).decode()
    return {"Authorization": f"Basic {token}", "Accept": "text/plain"}


def export_day(day: dt.date, timeout: int = 300) -> Iterator[dict]:
    params = {"project_id": C.MIXPANEL_PROJECT_ID, "from_date": day.isoformat(), "to_date": day.isoformat()}
    resp = requests.get(C.RAW_EXPORT_URL, params=params, headers=_auth_header(), stream=True, timeout=timeout)
    if resp.status_code == 400:
        raise RuntimeError(f"Export 400 for {day}: {resp.text[:300]} — verify params vs docs.mixpanel.com")
    resp.raise_for_status()
    for line in resp.iter_lines():
        if line:
            yield json.loads(line)


def _to_row(raw: dict) -> dict:
    props = raw.get("properties", {})
    ts = props.get("time")
    ts_server = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc) if ts else None
    ts_client = props.get("$client_time") or props.get("mp_client_time")
    return {
        "game_id": C.GAME_ID,
        "insert_id": props.get("$insert_id"),
        "distinct_id": props.get("$distinct_id") or props.get("distinct_id"),
        "event_name": normalize_event(raw.get("event")),
        "ts_server": ts_server,
        "ts_client": dt.datetime.fromtimestamp(ts_client, tz=dt.timezone.utc) if ts_client else ts_server,
        "build_version": props.get(C.P_VERSION),
        "platform": props.get(C.P_OS) or props.get(C.P_DEVICE),
        "env": props.get(C.ENV_PROPERTY),
        "level_id": reconcile_level_id(raw.get("event"), props),
        "properties": props,
    }


def pull_range(start: dt.date, end: dt.date) -> int:
    total, seen = 0, set()
    day = start
    while day <= end:
        rows = []
        for raw in export_day(day):
            r = _to_row(raw)
            key = (r["game_id"], r["insert_id"])
            if r["insert_id"] and key in seen:
                continue
            seen.add(key)
            rows.append(r)
        if rows:
            store.upsert_raw(rows)  # ON CONFLICT dedupes across runs too
        total += len(rows)
        print(f"[pea.extract] {day}: {len(rows)} rows -> pea_raw_events")
        day += dt.timedelta(days=1)
    return total


def incremental(days_back_if_empty: int = 30) -> int:
    today = dt.datetime.now(C.TIMEZONE).date()
    last = store.get_watermark(C.GAME_ID)
    start = (last + dt.timedelta(days=1)) if last else today - dt.timedelta(days=days_back_if_empty)
    end = today - dt.timedelta(days=1)
    if start > end:
        print("[pea.extract] up to date")
        return 0
    n = pull_range(start, end)
    store.set_watermark(C.GAME_ID, end)
    return n

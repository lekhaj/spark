"""
PEA storage — reuses the shared CycleZero Postgres (CYCLEZERO_DATABASE_URL).

Batch writes use a psycopg2 connection (Json adapter + execute_values upserts);
the FastAPI read routes use the SQLAlchemy session (app.cyclezero.db) per convention.
Raw events are cached in `pea_raw_events` (no Parquet dependency on the shared box).
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import psycopg2
import psycopg2.extras

from app.config import settings
from . import config as C


def _dsn() -> str:
    url = getattr(settings, "CYCLEZERO_DATABASE_URL", None)
    if not url:
        raise RuntimeError("CYCLEZERO_DATABASE_URL not set (needed for PEA tables)")
    # The shared URL is SQLAlchemy-style (postgresql+psycopg2://...); raw psycopg2
    # wants a bare postgresql:// DSN — strip the driver suffix.
    return url.replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgres+psycopg2://", "postgresql://")


def connect():
    return psycopg2.connect(_dsn())


def init_schema():
    ddl = (Path(__file__).parent / "schema.sql").read_text()
    with connect() as cx, cx.cursor() as cur:
        cur.execute(ddl)
        cx.commit()


# ------------------------------------------------------------------ watermark
def get_watermark(game_id: str):
    with connect() as cx, cx.cursor() as cur:
        cur.execute("SELECT last_date_pulled FROM pea_ingest_watermark "
                    "WHERE game_id=%s AND source='mixpanel_export'", (game_id,))
        row = cur.fetchone()
        return row[0] if row else None


def set_watermark(game_id: str, day: dt.date):
    with connect() as cx, cx.cursor() as cur:
        cur.execute(
            "INSERT INTO pea_ingest_watermark (game_id, source, last_date_pulled, last_run_at) "
            "VALUES (%s,'mixpanel_export',%s, now()) "
            "ON CONFLICT (game_id, source) DO UPDATE SET "
            "last_date_pulled=EXCLUDED.last_date_pulled, last_run_at=now()",
            (game_id, day))
        cx.commit()


# ------------------------------------------------------------------ raw events
def upsert_raw(rows: list[dict]):
    _write("pea_raw_events", rows, ("game_id", "insert_id"))


def load_cached_events(start: dt.date | None = None, end: dt.date | None = None):
    """Load cached raw events into a DataFrame (compute reads cache, not live Mixpanel)."""
    import pandas as pd
    sql = ("SELECT game_id, insert_id, distinct_id, event_name, ts_server, ts_client, "
           "build_version, platform, env, level_id, properties FROM pea_raw_events "
           "WHERE game_id=%s")
    params = [C.GAME_ID]
    if start:
        sql += " AND ts_server >= %s"; params.append(dt.datetime.combine(start, dt.time.min))
    if end:
        sql += " AND ts_server < %s"; params.append(dt.datetime.combine(end + dt.timedelta(days=1), dt.time.min))
    with connect() as cx:
        df = pd.read_sql(sql, cx, params=params)
    return df


# ------------------------------------------------------------------ upserts
def _upsert(cur, table: str, rows: list[dict], pk: tuple[str, ...]):
    if not rows:
        return
    cols = list(rows[0].keys())
    updates = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in pk)
    tmpl = "(" + ",".join(["%s"] * len(cols)) + ")"
    vals = [[psycopg2.extras.Json(r[c]) if isinstance(r[c], (dict, list)) else r[c] for c in cols]
            for r in rows]
    sql = (f"INSERT INTO {table} ({','.join(cols)}) VALUES %s "
           f"ON CONFLICT ({','.join(pk)}) DO UPDATE SET {updates}")
    psycopg2.extras.execute_values(cur, sql, vals, template=tmpl)


def _write(table, rows, pk):
    with connect() as cx, cx.cursor() as cur:
        _upsert(cur, table, rows, pk)
        cx.commit()


def upsert_sessions(rows): _write("pea_session_state", rows, ("game_id", "session_id"))
def upsert_players(rows): _write("pea_player_state", rows, ("game_id", "distinct_id", "date"))
def upsert_digest(rows): _write("pea_daily_digest", rows, ("game_id", "date"))
def upsert_friction(rows): _write("pea_level_friction", rows, ("game_id", "date", "level_id"))
def upsert_bringback(rows): _write("pea_bringback_list", rows, ("game_id", "date", "distinct_id"))
def upsert_funnel(rows): _write("pea_funnel_retention", rows, ("game_id", "date"))

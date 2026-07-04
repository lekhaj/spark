-- PEA derived tables, living in the shared CycleZero Postgres. All prefixed `pea_`
-- to avoid clashing with the design-graph tables. Dashboard reads ONLY these.

CREATE TABLE IF NOT EXISTS pea_raw_events (
    game_id        TEXT        NOT NULL,
    insert_id      TEXT        NOT NULL,
    distinct_id    TEXT        NOT NULL,
    event_name     TEXT        NOT NULL,
    ts_server      TIMESTAMPTZ NOT NULL,
    ts_client      TIMESTAMPTZ,
    build_version  TEXT,
    platform       TEXT,
    env            TEXT,
    level_id       INT,
    properties     JSONB       NOT NULL DEFAULT '{}',
    PRIMARY KEY (game_id, insert_id)
);
CREATE INDEX IF NOT EXISTS idx_pea_raw_did_ts ON pea_raw_events (game_id, distinct_id, ts_server);
CREATE INDEX IF NOT EXISTS idx_pea_raw_evt ON pea_raw_events (game_id, event_name, ts_server);

CREATE TABLE IF NOT EXISTS pea_session_state (
    game_id TEXT NOT NULL, distinct_id TEXT NOT NULL, session_id TEXT NOT NULL,
    session_date DATE, started_at TIMESTAMPTZ, ended_at TIMESTAMPTZ, duration_s INT,
    build_version TEXT, platform TEXT, is_new BOOLEAN, level_reached INT, levels_played JSONB DEFAULT '[]',
    retries INT DEFAULT 0, fails INT DEFAULT 0, wins INT DEFAULT 0,
    entry_mood TEXT, entry_mood_runner_up TEXT, exit_mood TEXT,
    overall_feeling TEXT, feeling_score INT,
    felt_tension TEXT, felt_mastery TEXT, felt_autonomy TEXT,
    confidence TEXT, stitched BOOLEAN DEFAULT FALSE, evidence JSONB DEFAULT '[]',
    PRIMARY KEY (game_id, session_id)
);
CREATE INDEX IF NOT EXISTS idx_pea_sess_did ON pea_session_state (game_id, distinct_id, session_date);

CREATE TABLE IF NOT EXISTS pea_player_state (
    game_id TEXT NOT NULL, distinct_id TEXT NOT NULL, date DATE NOT NULL,
    sessions_today INT DEFAULT 0, level_reached INT,
    retries INT DEFAULT 0, fails INT DEFAULT 0, wins INT DEFAULT 0,
    build_version TEXT, platform TEXT,
    entry_mood TEXT, exit_mood TEXT, overall_feeling TEXT, feeling_score INT,
    persona TEXT, persona_axes JSONB DEFAULT '{}', prev_persona TEXT,
    personality TEXT, personality_runner_up TEXT, personality_spectrum JSONB DEFAULT '{}',
    felt_tension TEXT, felt_mastery TEXT, felt_autonomy TEXT,
    confidence TEXT, is_new BOOLEAN, flipped_to_risk BOOLEAN DEFAULT FALSE,
    evidence JSONB DEFAULT '[]', narrative TEXT,
    PRIMARY KEY (game_id, distinct_id, date)
);
CREATE INDEX IF NOT EXISTS idx_pea_player_date ON pea_player_state (game_id, date);

CREATE TABLE IF NOT EXISTS pea_daily_digest (
    game_id TEXT NOT NULL, date DATE NOT NULL, dau INT NOT NULL,
    new_users INT NOT NULL, returning_users INT NOT NULL,
    by_build JSONB DEFAULT '{}', by_platform JSONB DEFAULT '{}',
    entry_mood_dist JSONB DEFAULT '{}', during_tension_dist JSONB DEFAULT '{}',
    exit_mood_dist JSONB DEFAULT '{}', personality_dist JSONB DEFAULT '{}',
    entry_mood_dod JSONB DEFAULT '{}', exit_mood_dod JSONB DEFAULT '{}',
    top_friction_levels JSONB DEFAULT '[]', watch_list JSONB DEFAULT '[]',
    insights JSONB DEFAULT '[]', confidence TEXT, banner TEXT,
    PRIMARY KEY (game_id, date)
);

CREATE TABLE IF NOT EXISTS pea_level_friction (
    game_id TEXT NOT NULL, date DATE NOT NULL, level_id INT NOT NULL,
    attempts INT DEFAULT 0, retries INT DEFAULT 0, fails INT DEFAULT 0, wins INT DEFAULT 0,
    frustrated_sessions INT DEFAULT 0, churn_risk_sessions INT DEFAULT 0,
    unique_players INT DEFAULT 0, confidence TEXT,
    PRIMARY KEY (game_id, date, level_id)
);

CREATE TABLE IF NOT EXISTS pea_bringback_list (
    game_id TEXT NOT NULL, date DATE NOT NULL, distinct_id TEXT NOT NULL,
    persona TEXT, mood_history JSONB DEFAULT '[]', lapse_risk NUMERIC,
    recommended_send_hour_local INT, recommended_message TEXT, recommended_incentive TEXT,
    included BOOLEAN DEFAULT TRUE, overridden BOOLEAN DEFAULT FALSE,
    PRIMARY KEY (game_id, date, distinct_id)
);

CREATE TABLE IF NOT EXISTS pea_funnel_retention (
    game_id TEXT NOT NULL, date DATE NOT NULL,
    funnel_steps JSONB DEFAULT '[]', d1_retention NUMERIC, d7_retention NUMERIC,
    cohort_size INT DEFAULT 0, confidence TEXT,
    PRIMARY KEY (game_id, date)
);

CREATE TABLE IF NOT EXISTS pea_ingest_watermark (
    game_id TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'mixpanel_export',
    last_date_pulled DATE, last_run_at TIMESTAMPTZ,
    PRIMARY KEY (game_id, source)
);

-- GROW: a log of share packages the creator built from detected moments. clicks/plays
-- are populated later when the /play landing + Mixpanel UTM join lands (data gap today).
CREATE TABLE IF NOT EXISTS pea_share_log (
    game_id TEXT NOT NULL, share_id TEXT NOT NULL, created_at TIMESTAMPTZ DEFAULT now(),
    moment_key TEXT, distinct_id TEXT, level_id INT, felt_tension TEXT, exit_mood TEXT,
    channel TEXT, caption TEXT, play_url TEXT,
    clicks INT DEFAULT 0, plays INT DEFAULT 0,
    PRIMARY KEY (game_id, share_id)
);
CREATE INDEX IF NOT EXISTS idx_pea_share_created ON pea_share_log (game_id, created_at DESC);

"""
PEA configuration — integrated into the Spark backend.

Connection/secrets come from Spark's pydantic settings (app.config.settings), which
load from .env / .env.secrets on the EC2 box. Add these to .env.secrets:
    MIXPANEL_SA_USER, MIXPANEL_SA_SECRET, MIXPANEL_PROJECT_ID (=3631004)
DB reuses the existing CYCLEZERO_DATABASE_URL Postgres (see app.pea.store).

EVERYTHING tunable (mood/persona/felt/personality/bring-back thresholds) lives here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

try:  # use Spark settings when running inside the app; fall back to env for CLI/tests
    from app.config import settings as _s
except Exception:  # pragma: no cover
    _s = None


def _cfg(name: str, default: str = "") -> str:
    if _s is not None and getattr(_s, name, None):
        return str(getattr(_s, name))
    return os.getenv(name, default)


# -------------------------------------------------------------------- connection
GAME_ID = _cfg("PEA_GAME_ID", "aurabeam")
MIXPANEL_PROJECT_ID = int(_cfg("MIXPANEL_PROJECT_ID", "3631004"))
MIXPANEL_REGION = _cfg("MIXPANEL_REGION", "US")
MIXPANEL_SA_USER = _cfg("MIXPANEL_SA_USER", "")
MIXPANEL_SA_SECRET = _cfg("MIXPANEL_SA_SECRET", "")  # NEVER hardcode; rotate the leaked one.

RAW_EXPORT_URL = "https://data.mixpanel.com/api/2.0/export"
QUERY_BASE_URL = "https://mixpanel.com/api/query"
ENGAGE_URL = "https://mixpanel.com/api/query/engage"

# LLM narration (optional). Spark uses Bedrock; if no ANTHROPIC_API_KEY, narrate.py
# falls back to deterministic templates so nothing breaks.
ANTHROPIC_API_KEY = _cfg("ANTHROPIC_API_KEY", "")
NARRATION_MODEL = _cfg("PEA_NARRATION_MODEL", "claude-opus-4-8")

# -------------------------------------------------------------------- hygiene
TIMEZONE = ZoneInfo(_cfg("PEA_TIMEZONE", "Asia/Kolkata"))
SESSION_INACTIVITY_MINUTES = 30
LOW_CONFIDENCE_USER_THRESHOLD = 20
DEDUPE_KEY = "$insert_id"
ENV_PROPERTY = "env"
ENV_EXCLUDE_VALUES = {"dev", "editor", "internal", "qa"}
PRELAUNCH_BANNER = "internal/QA, not player behavior (no env flag yet)"

# -------------------------------------------------------------------- event names (verified 2026-07-02)
E_SESSION_START = "$session_start"
E_SESSION_END = "$session_end"
E_APP_OPENED = "App Opened"
E_LOGIN, E_LOGOUT = "Login", "Logout"
E_GAME_START = "Game Start"
E_LEVEL_COMPLETED = "Level Completed"
E_GAME_LOST = "Game Lost"
E_AD_WATCHED = "Rewarded Ad Watched"
E_CRASH = "Crash / Error"
E_FIRST_PLAY = "Journey: First play (no save)"
E_CONTINUE = "Journey: Continue saved progress"
E_LEVEL_WON = "Journey: LEVEL WON"
E_FAIL_HEARTS = "Journey: LEVEL FAILED (out of hearts)"
E_FAIL_BEAM = "Journey: LEVEL FAILED (beam missed targets)"
E_LIFE_LOST = "Journey: Life lost -> restarting level"
E_TRY_AGAIN = "Journey: TryAgain (reload level)"
E_TARGET_HIT = "Journey: Target hit"
E_CHECK_WIN = "Journey: CheckWinCondition"
E_LOAD_NEXT = "Journey: LoadNextLevel"
E_LOAD_SAVED = "Journey: LoadCurrentSavedLevel"
E_QUIT_MENU = "Journey: Quit to MainMenu"
E_INTERSTITIAL = "Journey: Next -> milestone, requesting interstitial ad"
E_JOURNEY_STARTGAME = "Journey: StartGame"
E_JOURNEY_NAV_STARTGAME = "Journey: LevelNavigation.StartGame"

RETRY_EVENTS = {E_LIFE_LOST, E_TRY_AGAIN}
FAIL_EVENTS = {E_FAIL_HEARTS, E_FAIL_BEAM, E_GAME_LOST}
WIN_EVENTS = {E_LEVEL_WON}
COMPLETE_EVENTS = {E_LEVEL_COMPLETED, E_LEVEL_WON}
ABANDON_EVENTS = {E_QUIT_MENU}
NEW_PLAYER_EVENTS = {E_FIRST_PLAY}
RETURN_PLAYER_EVENTS = {E_CONTINUE, E_LOAD_SAVED}

P_DISTINCT_ID = "distinct_id"
P_INSERT_ID = "$insert_id"
P_VERSION = "$app_version_string"
P_OS = "$os"
P_DEVICE = "$device"
P_BUILD_INDEX = "buildIndex"
P_SCENE = "scene"
P_LEVEL_STR = "Level"
P_LEVEL_NUM = "LevelNumber"
P_REASON = "Reason"
P_REWARD = "RewardAmount"


# -------------------------------------------------------------------- mood thresholds
@dataclass(frozen=True)
class MoodConfig:
    new_max_days_since_install: int = 1
    new_max_session_count: int = 2
    returning_engaged_min_sessions: int = 5
    cadence_multiplier_at_risk: float = 2.0
    puzzle_seeker_max_retries_per_level: float = 0.5
    explorer_max_session_minutes: float = 4.0
    explorer_min_distinct_levels: int = 3
    frustrated_min_attempts_one_level: int = 3
    interrupted_max_session_minutes: float = 2.0
    comeback_min_trailing_wins: int = 1
    feeling_scores: dict = field(default_factory=lambda: {
        "happy": 2, "comeback-tomorrow": 2, "ok-satisfied": 1,
        "interrupted": 0, "frustrated": -1, "churn-risk": -2,
    })
    feeling_labels: dict = field(default_factory=lambda: {
        2: "great", 1: "good", 0: "neutral", -1: "rough", -2: "rough",
    })
    persona_window_days: int = 14
    climbing_min_new_levels: int = 2
    struggling_min_retries_per_level: float = 3.0
    binge_min_sessions_per_active_day: float = 3.0
    grazer_max_sessions_per_active_day: float = 1.5
    cautious_max_fail_rate: float = 0.2


MOODS = MoodConfig()


@dataclass(frozen=True)
class BringBackConfig:
    quiet_hours = (23, 7)
    default_send_hour = 19
    max_sends_per_user_per_week = 2
    suppress_if_returned_today = True
    lapse_risk_days = 3
    templates: dict = field(default_factory=lambda: {
        "puzzle-seeker": ("A fresh hard puzzle is waiting for you.", "new_hard_level"),
        "frustrated": ("Stuck? Here's a free booster + 30 min of infinite lives.", "booster+infinite_lives_30m"),
        "churn-risk": ("We saved your progress — jump back in with a free boost.", "booster"),
        "comeback-tomorrow": ("You were on a roll — keep your streak alive!", "streak_nudge"),
        "at-risk-returner": ("Your beams missed you. A quick level to warm up?", "easy_level"),
        "default": ("New levels just dropped in AuraBeam.", "none"),
    })
    send_decision_fields = {
        "distinct_id": "available", "exit_mood": "available", "persona": "available",
        "lapse_risk": "available", "active_hour_local": "available (from event ts)",
        "push_token": "MISSING — no push channel", "consent/opt_in": "MISSING",
        "timezone_per_user": "MISSING — approximated from mp geo",
    }


BRINGBACK = BringBackConfig()


@dataclass(frozen=True)
class FeltConfig:
    tension_frustrated_min_attempts: int = 3
    tension_relief_max_attempts: int = 2
    autonomy_default: str = "unknown"


FELT = FeltConfig()


@dataclass(frozen=True)
class PersonalityConfig:
    window_days: int = 14
    min_sessions_for_confidence: int = 4
    high: int = 66
    mid: int = 40
    archetypes: tuple = (
        "puzzle-solver", "analytical-serious", "creative-experimenter", "determined-grinder",
        "casual-gamer", "booster-reliant", "at-risk-drifter", "steady-improver",
    )


PERSONALITY = PersonalityConfig()

"""Studio access control — email allowlist + waitlist.

Spark Studio is invite-only for now: only allowlisted emails get past sign-in.
Everyone else is recorded as a waitlist request ("your request is registered,
we'll email next steps") and politely held back.

The allowlist is the union of:
  - the owner email (always allowed),
  - the comma-separated ``STUDIO_ALLOWED_EMAILS`` env var,
  - any email with ``allowed: true`` in the Mongo ``studio_users`` collection
    (so you can grant access by flipping a flag without a redeploy).

Every sign-in is upserted into ``studio_users`` so we have a record of who
signed in (or tried to). No passwords here — Firebase owns authentication; this
is pure authorization + bookkeeping.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter
from pydantic import BaseModel

from app.services.mongo_service import get_db

logger = logging.getLogger(__name__)
router = APIRouter()

OWNER_EMAIL = "lekhaj123btmn@gmail.com"
_COLLECTION = "studio_users"


def _env_allowlist() -> set[str]:
    raw = os.getenv("STUDIO_ALLOWED_EMAILS", "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _is_allowed(email: str, db) -> bool:
    email = email.lower()
    if email == OWNER_EMAIL.lower():
        return True
    if email in _env_allowlist():
        return True
    if db is not None:
        doc = db[_COLLECTION].find_one({"email": email, "allowed": True})
        if doc:
            return True
    return False


class LoginIn(BaseModel):
    email: str
    uid: str | None = None
    name: str | None = None


class LoginOut(BaseModel):
    allowed: bool
    status: str  # "allowed" | "waitlist"


@router.post("/login", response_model=LoginOut)
def access_login(body: LoginIn) -> LoginOut:
    """Called right after a successful Firebase sign-in. Records the user and
    returns whether they're allowed in. Best-effort: never blocks sign-in
    bookkeeping on a DB hiccup (falls back to env/owner allowlist)."""
    email = str(body.email).lower()
    db = get_db()
    allowed = _is_allowed(email, db)
    status = "allowed" if allowed else "waitlist"

    if db is not None:
        try:
            now = datetime.now(timezone.utc)
            db[_COLLECTION].update_one(
                {"email": email},
                {
                    "$set": {
                        "uid": body.uid,
                        "name": body.name,
                        "last_seen": now,
                        "status": status,
                    },
                    "$setOnInsert": {
                        "email": email,
                        "first_seen": now,
                        # Don't grant access on insert — allowlist drives `allowed`.
                        "allowed": allowed,
                    },
                },
                upsert=True,
            )
        except Exception as exc:  # noqa: BLE001 — bookkeeping must never 500 sign-in
            logger.warning("studio access_login bookkeeping failed: %s", exc)

    return LoginOut(allowed=allowed, status=status)

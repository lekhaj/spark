"""
identity.py — lightweight studio-user attribution for FastAPI routes
====================================================================

Firebase already authenticates the user client-side and the access gate
(``/access/login``) authorizes them. The studio frontend forwards the signed-in
user's identity on every request via two headers so the backend can *attribute*
events and LLM usage to a person:

    X-Studio-Uid    Firebase uid (opaque)
    X-Studio-Email  the signed-in email

This is attribution, NOT authentication — never trust it for access control.
Use ``require_owner`` to gate owner-only admin reads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException

OWNER_EMAIL = "lekhaj123btmn@gmail.com"


@dataclass
class StudioUser:
    uid: Optional[str]
    email: Optional[str]

    @property
    def is_owner(self) -> bool:
        return bool(self.email) and self.email.lower() == OWNER_EMAIL.lower()


def current_user(
    x_studio_uid: Optional[str] = Header(default=None),
    x_studio_email: Optional[str] = Header(default=None),
) -> StudioUser:
    """FastAPI dependency — resolves the caller from the studio headers."""
    email = (x_studio_email or "").strip().lower() or None
    return StudioUser(uid=(x_studio_uid or None), email=email)


def require_owner(user: StudioUser) -> StudioUser:
    if not user.is_owner:
        raise HTTPException(403, "owner only")
    return user

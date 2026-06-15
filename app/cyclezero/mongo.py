"""Accessor for the CycleZero Mongo DB (freeform / generated content + job
artifacts). Reuses the same MongoDB server as spark, but a separate database
(``CYCLEZERO_MONGO_DB``, default ``cyclezero``) so game content stays isolated
from spark's ``World_builder``."""
from __future__ import annotations

from typing import Optional

from pymongo import MongoClient
from pymongo.database import Database

from app.config import settings

_client: Optional[MongoClient] = None


def get_mongo() -> Database:
    global _client
    if _client is None:
        _client = MongoClient(settings.MONGODB_URL)
    db_name = getattr(settings, "CYCLEZERO_MONGO_DB", "cyclezero")
    return _client[db_name]

#!/usr/bin/env python3
"""
Base GPU Worker
===============
Shared foundation for all GPU workers (SD1.5, TRELLIS, Rig).
Provides: queue polling, S3 upload/download, MongoDB updates, logging, task TTL guard.
"""

import io
import json
import logging
import os
import time
import uuid
from abc import ABC, abstractmethod
from typing import Optional

import boto3
import redis
from PIL import Image
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)


# ── Shared Config (reads from .env) ───────────────────────────────────────────
class WorkerConfig:
    REDIS_HOST     = os.getenv("REDIS_HOST",   "localhost")
    REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
    MONGO_URI      = os.getenv("MONGO_URI",    "mongodb://kartik:Kartikg421@localhost:27017")
    MONGO_DB       = os.getenv("MONGO_DB",     "World_builder")
    S3_BUCKET      = os.getenv("AWS_S3_BUCKET", os.getenv("S3_BUCKET", "sparkassets-us"))
    S3_REGION      = os.getenv("AWS_REGION",   "ap-south-1")
    TASK_TTL       = int(os.getenv("TASK_TTL", 14400))   # 4 hours
    INSTANCE_ID    = (
        os.getenv("AWS_GPU_INSTANCE_ID") or
        os.getenv("GPU") or
        "i-0e029990527fa2b73"
    )


# ── Base Worker ───────────────────────────────────────────────────────────────
class BaseWorker(ABC):
    """
    Abstract base for all GPU pipeline workers.

    Subclasses must implement:
        worker_name  : str  (e.g. "SD15Worker")
        input_queue  : str  (Redis key to pop from)
        process_task(task, redis_client, mongo_db) -> None
    """

    worker_name: str = "BaseWorker"
    input_queue: str = ""

    def __init__(self):
        self.logger = logging.getLogger(self.worker_name)
        self.cfg    = WorkerConfig()
        self._s3    = None
        self._redis = None
        self._mongo = None

    # ── Connections ───────────────────────────────────────────────────────────

    def get_redis(self) -> redis.Redis:
        if self._redis is None or not self._redis.ping():
            self._redis = redis.Redis(
                host=self.cfg.REDIS_HOST,
                port=self.cfg.REDIS_PORT,
                db=0,
                decode_responses=True,
                socket_connect_timeout=10,
                socket_timeout=35,  # must be > BLPOP timeout (30s) to avoid spurious timeouts
            )
        return self._redis

    def get_mongo(self):
        if self._mongo is None:
            client = MongoClient(self.cfg.MONGO_URI, serverSelectionTimeoutMS=10000)
            self._mongo = client[self.cfg.MONGO_DB]
        return self._mongo

    def get_s3(self):
        if self._s3 is None:
            # Support both standard boto3 names and the project's shorter names
            key    = os.getenv("AWS_ACCESS_KEY_ID")    or os.getenv("AWS_ACCESS_KEY")
            secret = os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_KEY")
            self._s3 = boto3.client(
                "s3",
                region_name=self.cfg.S3_REGION,
                aws_access_key_id=key,
                aws_secret_access_key=secret,
            )
        return self._s3

    # ── S3 Helpers ────────────────────────────────────────────────────────────

    def upload_image(self, img: Image.Image, s3_key: str) -> str:
        """Upload a PIL Image to S3; returns the s3_key."""
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        self.get_s3().put_object(
            Bucket=self.cfg.S3_BUCKET,
            Key=s3_key,
            Body=buf,
            ContentType="image/png",
        )
        self.logger.info(f"Uploaded image → s3://{self.cfg.S3_BUCKET}/{s3_key}")
        return s3_key

    def upload_file(self, local_path: str, s3_key: str, content_type: str = "application/octet-stream") -> str:
        """Upload any local file to S3; returns the s3_key."""
        self.get_s3().upload_file(
            Filename=local_path,
            Bucket=self.cfg.S3_BUCKET,
            Key=s3_key,
            ExtraArgs={"ContentType": content_type},
        )
        self.logger.info(f"Uploaded file  → s3://{self.cfg.S3_BUCKET}/{s3_key}")
        return s3_key

    def download_file(self, s3_key: str, local_path: str) -> str:
        """Download S3 object to local_path; returns local_path."""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        self.get_s3().download_file(self.cfg.S3_BUCKET, s3_key, local_path)
        return local_path

    def s3_public_url(self, s3_key: str) -> str:
        return f"https://{self.cfg.S3_BUCKET}.s3.{self.cfg.S3_REGION}.amazonaws.com/{s3_key}"

    # ── MongoDB Helpers ───────────────────────────────────────────────────────

    def mongo_update(self, biome_id: str, char_name: str, fields: dict):
        """Dot-notation upsert for a character in biomes collection."""
        self.get_mongo().biomes.update_one(
            {"_id": biome_id},
            {"$set": {
                f"possible_structures.characters.{char_name}.{k}": v
                for k, v in fields.items()
            }},
        )

    # ── Queue Push ────────────────────────────────────────────────────────────

    def push_task(self, queue: str, payload: dict):
        payload.setdefault("task_id",   str(uuid.uuid4()))
        payload.setdefault("timestamp", time.time())
        self.get_redis().rpush(queue, json.dumps(payload))
        self.logger.info(f"Pushed task → {queue}  ({payload.get('character_name','?')})")

    # ── TTL Guard ─────────────────────────────────────────────────────────────

    def is_expired(self, task: dict) -> bool:
        age = time.time() - task.get("timestamp", 0)
        if age > self.cfg.TASK_TTL:
            self.logger.warning(
                f"Task expired ({age:.0f}s old), skipping: {task.get('character_name')}"
            )
            return True
        return False

    # ── Main Loop ─────────────────────────────────────────────────────────────

    @abstractmethod
    def load_models(self):
        """Load ML models into GPU memory. Called once at startup."""
        ...

    @abstractmethod
    def process_task(self, task: dict, r: redis.Redis, db) -> None:
        """Process one task popped from input_queue."""
        ...

    def run(self, idle_notify_callback=None):
        """
        Main blocking loop.
        idle_notify_callback(seconds_idle) is called each poll with no task,
        used by AutoShutdown to track idle time.
        """
        self.logger.info(f"{self.worker_name} starting — queue: {self.input_queue}")
        self.load_models()

        r  = self.get_redis()
        db = self.get_mongo()
        idle_since = time.time()

        while True:
            try:
                raw = r.blpop(self.input_queue, timeout=30)
            except Exception as e:
                self.logger.error(f"Redis error: {e}; reconnecting in 5s")
                time.sleep(5)
                self._redis = None
                continue

            if raw is None:
                if idle_notify_callback:
                    idle_notify_callback(time.time() - idle_since)
                continue

            idle_since = time.time()
            _, payload = raw

            try:
                task = json.loads(payload)
            except json.JSONDecodeError:
                self.logger.error(f"Bad JSON in queue: {payload[:120]}")
                continue

            if self.is_expired(task):
                continue

            self.logger.info(
                f"Task → biome={task.get('biome_id')}  "
                f"char={task.get('character_name')}  type={task.get('character_type')}"
            )
            try:
                self.process_task(task, r, db)
            except Exception as exc:
                self.logger.exception(f"Task failed for {task.get('character_name')}: {exc}")
                try:
                    self.mongo_update(
                        task["biome_id"], task["character_name"],
                        {"status": "error", "error_msg": str(exc)},
                    )
                except Exception:
                    pass

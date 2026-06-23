"""Unit tests for worker/lib/gpu_heartbeat.py — the GPU busy signal.

Pure logic with a tiny in-memory fake Redis (no server, no torch). Proves the
freshness semantics both stop-paths rely on, including the fail-safe behavior.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from lib import gpu_heartbeat  # noqa: E402


class FakeRedis:
    """Minimal Redis: set(ex ignored) / get. Optionally raise to simulate down."""

    def __init__(self, raise_on=()):
        self.store = {}
        self.raise_on = set(raise_on)

    def set(self, key, val, ex=None):
        if "set" in self.raise_on:
            raise ConnectionError("redis down")
        self.store[key] = val

    def get(self, key):
        if "get" in self.raise_on:
            raise ConnectionError("redis down")
        return self.store.get(key)


def test_touch_then_fresh():
    r = FakeRedis()
    gpu_heartbeat.touch(r, "i-abc")
    s = gpu_heartbeat.seconds_since(r, "i-abc")
    assert s is not None and s < 2.0
    assert gpu_heartbeat.is_busy(r, "i-abc", fresh_seconds=120) is True


def test_absent_key_reads_as_long_idle():
    r = FakeRedis()
    assert gpu_heartbeat.seconds_since(r, "i-none") == float("inf")
    assert gpu_heartbeat.is_busy(r, "i-none", fresh_seconds=120) is False


def test_stale_heartbeat_not_busy():
    r = FakeRedis()
    old = str(time.time() - 600)  # 10 min ago
    r.store[gpu_heartbeat.KEY_GLOBAL] = old
    r.store[gpu_heartbeat.KEY_IID.format(instance_id="i-abc")] = old
    s = gpu_heartbeat.seconds_since(r, "i-abc")
    assert s is not None and s > 300
    assert gpu_heartbeat.is_busy(r, "i-abc", fresh_seconds=120) is False


def test_per_instance_beats_global_when_newer():
    r = FakeRedis()
    r.store[gpu_heartbeat.KEY_GLOBAL] = str(time.time() - 600)
    r.store[gpu_heartbeat.KEY_IID.format(instance_id="i-abc")] = str(time.time())
    # freshest of the two wins → busy
    assert gpu_heartbeat.is_busy(r, "i-abc", fresh_seconds=120) is True


def test_redis_error_fails_safe_busy():
    r = FakeRedis(raise_on={"get"})
    assert gpu_heartbeat.seconds_since(r, "i-abc") is None
    # unknown → assume busy so nothing ever stops a box it can't read
    assert gpu_heartbeat.is_busy(r, "i-abc", fresh_seconds=120) is True


def test_touch_never_raises_on_redis_error():
    r = FakeRedis(raise_on={"set"})
    gpu_heartbeat.touch(r, "i-abc")  # must swallow — heartbeat can't crash worker

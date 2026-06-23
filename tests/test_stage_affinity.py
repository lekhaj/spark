"""Unit tests for the stage-affinity scheduler (worker/lib/stage_affinity.py).

Pure logic — uses a tiny in-memory fake Redis (no server, no torch), so it proves
the reordering behavior that keeps a stage's model warm without any GPU.
"""
import json
import sys
from pathlib import Path

# worker/ is importable as a top-level package root (matches the worker runtime,
# which runs with cwd=/home/ec2-user/spark and worker/ on the path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

from lib.stage_affinity import pop_next_task  # noqa: E402


class FakeRedis:
    """Minimal Redis list supporting the ops pop_next_task uses."""

    def __init__(self, items):
        self.q = list(items)
        self.blpop_calls = 0

    def lrange(self, _key, start, end):
        # mimic redis inclusive end / -1 semantics
        if end == -1:
            return list(self.q[start:])
        return list(self.q[start:end + 1])

    def lrem(self, _key, count, value):
        removed = 0
        i = 0
        while i < len(self.q) and (count == 0 or removed < count):
            if self.q[i] == value:
                self.q.pop(i)
                removed += 1
            else:
                i += 1
        return removed

    def blpop(self, _key, timeout=0):
        self.blpop_calls += 1
        if not self.q:
            return None
        return ("queue", self.q.pop(0))


def _t(stage, char):
    return json.dumps({"stage": stage, "char_label": char, "task_id": f"{stage}_{char}"})


def _drain(items):
    """Process the whole queue with stage-affinity and return the stage order."""
    r = FakeRedis(items)
    order = []
    last_stage = None
    while True:
        payload = pop_next_task(r, "queue", last_stage, timeout=0)
        if payload is None:
            break
        task = json.loads(payload)
        order.append((task["stage"], task["char_label"]))
        last_stage = task["stage"]
    return order


def test_groups_same_stage_consecutively():
    # Interleaved per-character fan-out (the wasteful order).
    items = [
        _t("trellis", "A"), _t("pixal3d", "A"), _t("hunyuan3d", "A"),
        _t("trellis", "B"), _t("pixal3d", "B"), _t("hunyuan3d", "B"),
    ]
    order = _drain(items)
    stages = [s for s, _ in order]
    # Each stage runs consecutively → one model load per stage, not per character.
    assert stages == ["trellis", "trellis", "pixal3d", "pixal3d", "hunyuan3d", "hunyuan3d"]
    # All tasks still processed exactly once (no content dropped).
    assert sorted(order) == sorted([
        ("trellis", "A"), ("trellis", "B"),
        ("pixal3d", "A"), ("pixal3d", "B"),
        ("hunyuan3d", "A"), ("hunyuan3d", "B"),
    ])


def test_first_pull_is_fifo_head():
    r = FakeRedis([_t("trellis", "A"), _t("pixal3d", "B")])
    # last_stage=None → FIFO head (BLPOP)
    first = json.loads(pop_next_task(r, "queue", None, timeout=0))
    assert (first["stage"], first["char_label"]) == ("trellis", "A")
    assert r.blpop_calls == 1


def test_falls_back_to_fifo_when_no_same_stage():
    r = FakeRedis([_t("pixal3d", "B"), _t("hunyuan3d", "C")])
    # Ask for 'trellis' affinity but none present → FIFO head.
    nxt = json.loads(pop_next_task(r, "queue", "trellis", timeout=0))
    assert nxt["stage"] == "pixal3d"  # head, via BLPOP fallback
    assert r.blpop_calls == 1


def test_empty_queue_returns_none():
    r = FakeRedis([])
    assert pop_next_task(r, "queue", "trellis", timeout=0) is None


def test_three_characters_each_model_loads_once():
    items = []
    for c in ("A", "B", "C"):
        items += [_t("trellis", c), _t("pixal3d", c), _t("hunyuan3d", c)]
    stages = [s for s, _ in _drain(items)]
    # 3 chars × 3 stages, grouped: exactly 3 stage-transitions (one per model).
    transitions = sum(1 for i in range(1, len(stages)) if stages[i] != stages[i - 1])
    assert transitions == 2  # trellis→pixal3d→hunyuan3d (2 switches = 3 loads total)
    assert stages == ["trellis"] * 3 + ["pixal3d"] * 3 + ["hunyuan3d"] * 3

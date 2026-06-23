"""Unit tests for the orchestrator's pipeline-aware work detection.

`_pipeline_has_work` is the single gate for idle-stop. These tests prove its
truth table (queued tasks / fresh heartbeat / in-flight run / fail-safe) without
any AWS, Redis server, or Mongo — every external is monkeypatched.
"""
import os

os.environ.setdefault("CELERY_BROKER_URL", "redis://localhost:6379/0")
os.environ.setdefault("MONGODB_URL", "mongodb://localhost:27017")

import pytest

from app.services import orchestrator_service as osvc


@pytest.fixture()
def orch(monkeypatch):
    o = osvc.GPUOrchestrator()
    monkeypatch.setattr(o, "_active_iid", lambda: "i-test")
    return o


def _set_hb(monkeypatch, busy):
    monkeypatch.setattr(osvc.gpu_heartbeat, "is_busy", lambda *a, **k: busy)


def test_queued_tasks_is_work(orch, monkeypatch):
    _set_hb(monkeypatch, False)
    monkeypatch.setattr(orch, "_has_inflight_asset_run", lambda: False)
    assert orch._pipeline_has_work(total=3) is True


def test_fresh_heartbeat_is_work(orch, monkeypatch):
    _set_hb(monkeypatch, True)
    monkeypatch.setattr(orch, "_has_inflight_asset_run", lambda: False)
    assert orch._pipeline_has_work(total=0) is True


def test_inflight_run_is_work(orch, monkeypatch):
    _set_hb(monkeypatch, False)
    monkeypatch.setattr(orch, "_has_inflight_asset_run", lambda: True)
    assert orch._pipeline_has_work(total=0) is True


def test_mongo_unknown_fails_safe_to_work(orch, monkeypatch):
    _set_hb(monkeypatch, False)
    monkeypatch.setattr(orch, "_has_inflight_asset_run", lambda: None)  # unknown
    assert orch._pipeline_has_work(total=0) is True


def test_truly_idle_is_no_work(orch, monkeypatch):
    _set_hb(monkeypatch, False)
    monkeypatch.setattr(orch, "_has_inflight_asset_run", lambda: False)
    assert orch._pipeline_has_work(total=0) is False

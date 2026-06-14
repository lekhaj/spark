"""
gpu_launcher ladder tests — spot-first / on-demand-fallback, EIP, parallel spot.

boto3 + redis are faked in-process (no AWS, no Redis server). Covers:
  - disabled flag
  - stick to an already-running active box (no EIP flip mid-job)
  - spot running / spot stopped→started
  - spot can't start (capacity) → on-demand fallback
  - spot deleted → parallel new-spot launch + on-demand fallback now
"""

import pytest

from worker.lib import gpu_launcher as gl


# ── Fakes ──────────────────────────────────────────────────────────────────────

class FakeClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code}}


class FakeEc2:
    def __init__(self, instances):
        # instances: {iid: {"state": str, "lifecycle": "spot"|None}}
        self.instances = instances
        self.associated = []      # (iid, alloc)
        self.launched = []        # run_instances kwargs
        self.start_calls = []     # iids started
        self.start_error = {}     # iid -> exception to raise on start

    def _inst(self, iid):
        meta = self.instances[iid]
        d = {"InstanceId": iid, "State": {"Name": meta["state"]}}
        if meta.get("lifecycle"):
            d["InstanceLifecycle"] = meta["lifecycle"]
        return d

    def describe_instances(self, InstanceIds=None, Filters=None):
        if InstanceIds:
            reservations = [{"Instances": [self._inst(iid)]}
                            for iid in InstanceIds if iid in self.instances]
            return {"Reservations": reservations}
        return {"Reservations": []}   # tag lookup → nothing

    def start_instances(self, InstanceIds):
        for iid in InstanceIds:
            if iid in self.start_error:
                raise self.start_error[iid]
            self.start_calls.append(iid)
            self.instances[iid]["state"] = "running"

    def run_instances(self, **kw):
        iid = "i-newspot01"
        self.launched.append(kw)
        self.instances[iid] = {"state": "pending", "lifecycle": "spot"}
        return {"Instances": [{"InstanceId": iid}]}

    def associate_address(self, InstanceId, AllocationId):
        self.associated.append((InstanceId, AllocationId))


class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v):
        self.store[k] = v

    def setex(self, k, ttl, v):
        self.store[k] = v

    def ping(self):
        return True


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("GPU_AUTO_LAUNCH", "1")
    monkeypatch.setenv("AWS_GPU_SPOT_INSTANCE_ID", "i-spot")
    monkeypatch.setenv("AWS_GPU_INSTANCE_ID", "i-ondemand")
    monkeypatch.setenv("AWS_GPU_EIP_ALLOC_ID", "eip-1")
    monkeypatch.setenv("GPU_BOOT_TIMEOUT_S", "5")
    # make waits instant
    monkeypatch.setattr(gl.time, "sleep", lambda *_: None)


def _wire(monkeypatch, ec2, redis):
    monkeypatch.setattr(gl, "_ec2_client", lambda: ec2)
    monkeypatch.setattr(gl, "_redis", lambda: redis)


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_disabled(monkeypatch):
    monkeypatch.setenv("GPU_AUTO_LAUNCH", "0")
    assert gl.ensure_gpu_ready() == (True, "disabled")


def test_stick_to_running_active(env, monkeypatch):
    ec2 = FakeEc2({"i-spot": {"state": "running", "lifecycle": "spot"}})
    rds = FakeRedis()
    rds.set(gl.REDIS_ACTIVE_KEY, "i-spot")   # already serving
    _wire(monkeypatch, ec2, rds)
    ok, reason = gl.ensure_gpu_ready()
    assert ok and reason == "active-running"
    assert ec2.start_calls == []             # never touched
    assert ec2.associated == []              # no EIP flip mid-job


def test_spot_running_finalizes(env, monkeypatch):
    ec2 = FakeEc2({
        "i-spot": {"state": "running", "lifecycle": "spot"},
        "i-ondemand": {"state": "stopped"},
    })
    rds = FakeRedis()
    _wire(monkeypatch, ec2, rds)
    ok, reason = gl.ensure_gpu_ready()
    assert ok and reason == "spot-running"
    assert ("i-spot", "eip-1") in ec2.associated
    assert rds.get(gl.REDIS_ACTIVE_KEY) == "i-spot"


def test_spot_stopped_starts(env, monkeypatch):
    ec2 = FakeEc2({
        "i-spot": {"state": "stopped", "lifecycle": "spot"},
        "i-ondemand": {"state": "stopped"},
    })
    rds = FakeRedis()
    _wire(monkeypatch, ec2, rds)
    ok, reason = gl.ensure_gpu_ready()
    assert ok and reason == "spot-started"
    assert ec2.start_calls == ["i-spot"]
    assert ("i-spot", "eip-1") in ec2.associated


def test_spot_capacity_falls_back_to_ondemand(env, monkeypatch):
    ec2 = FakeEc2({
        "i-spot": {"state": "stopped", "lifecycle": "spot"},
        "i-ondemand": {"state": "stopped"},
    })
    ec2.start_error["i-spot"] = FakeClientError("InsufficientInstanceCapacity")
    rds = FakeRedis()
    _wire(monkeypatch, ec2, rds)
    ok, reason = gl.ensure_gpu_ready()
    assert ok and reason == "ondemand-started"
    assert ec2.start_calls == ["i-ondemand"]
    assert ("i-ondemand", "eip-1") in ec2.associated
    assert rds.get(gl.REDIS_ACTIVE_KEY) == "i-ondemand"
    assert ec2.launched == []                # no relaunch — spot still exists


def test_spot_deleted_launches_parallel_and_uses_ondemand(env, monkeypatch):
    monkeypatch.setenv("AWS_GPU_AMI_ID", "ami-1")
    ec2 = FakeEc2({
        "i-spot": {"state": "terminated", "lifecycle": "spot"},
        "i-ondemand": {"state": "stopped"},
    })
    rds = FakeRedis()
    _wire(monkeypatch, ec2, rds)
    ok, reason = gl.ensure_gpu_ready()
    # current work served by on-demand…
    assert ok and reason == "ondemand-started"
    assert ec2.start_calls == ["i-ondemand"]
    assert ("i-ondemand", "eip-1") in ec2.associated
    # …and a fresh spot was launched in the background, its new id persisted.
    assert len(ec2.launched) == 1
    assert rds.get(gl.REDIS_SPOT_KEY) == "i-newspot01"


def test_spot_deleted_no_ami_still_uses_ondemand(env, monkeypatch):
    monkeypatch.delenv("AWS_GPU_AMI_ID", raising=False)
    ec2 = FakeEc2({
        "i-spot": {"state": "terminated", "lifecycle": "spot"},
        "i-ondemand": {"state": "stopped"},
    })
    rds = FakeRedis()
    _wire(monkeypatch, ec2, rds)
    ok, reason = gl.ensure_gpu_ready()
    assert ok and reason == "ondemand-started"
    assert ec2.launched == []                # couldn't relaunch (no AMI), that's fine

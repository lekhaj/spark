"""
Spot GPU Instance Manager
=========================
Manages a single on-demand spot GPU instance for the pipeline.

Flow:
  1. Tasks arrive in Redis queues
  2. Orchestrator calls ensure_gpu_available()
  3. If no spot instance → request one (with fallback instance types)
  4. Wait for running + SSH ready
  5. Deploy workers via SSH (systemd services)
  6. When idle → terminate spot instance (cost = $0)

The active instance is tracked in memory + tagged in AWS for recovery
after orchestrator restart.
"""

import time
import logging
import boto3
from botocore.exceptions import ClientError
from app.config import settings

logger = logging.getLogger("spot_gpu_service")

# ── Fallback instance types (try in order) ──────────────────────────────────
SPOT_INSTANCE_TYPES = [
    "g6.2xlarge",   # L4 24GB  (preferred)
    "g5.2xlarge",   # A10G 24GB (fallback)
]

# Tag used to find our spot instance after orchestrator restart
PROJECT_TAG = "spark-gpu-worker"

# How long to wait for instance to become reachable via SSH
SSH_READY_WAIT = 40  # seconds after "running" state


class SpotGPUManager:
    """Manages a single spot GPU instance lifecycle."""

    def __init__(self):
        self.ec2 = boto3.client(
            "ec2",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION,
        )
        # Active instance state (populated on startup or after launch)
        self.instance_id: str | None = None
        self.public_ip: str | None = None
        self.instance_type: str | None = None
        self.state: str = "none"  # none | pending | running | shutting-down | terminated

        # Config from settings (with sensible defaults)
        self.ami_id = getattr(settings, "SPOT_AMI_ID", None)
        self.key_name = getattr(settings, "SPOT_KEY_NAME", None)
        self.security_group_ids = self._parse_list(
            getattr(settings, "SPOT_SECURITY_GROUP_IDS", "")
        )
        self.subnet_id = getattr(settings, "SPOT_SUBNET_ID", None)
        self.iam_instance_profile = getattr(settings, "SPOT_IAM_INSTANCE_PROFILE", None)
        self.ssh_user = getattr(settings, "SPOT_SSH_USER", "ubuntu")
        self.ssh_key_path = getattr(settings, "GPU_SSH_KEY_PATH", "/home/ubuntu/.ssh/s_spu_key.pem")

        # Try to recover an existing spot instance on init
        self._recover_existing_instance()

    # ── Public API ───────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Check if we have a running spot instance."""
        if not self.instance_id:
            return False
        self._refresh_state()
        return self.state == "running"

    def ensure_gpu_available(self) -> bool:
        """Ensure a spot GPU instance is running. Returns True if available."""
        if self.is_running():
            logger.info(f"[SPOT] Instance already running: {self.instance_id} ({self.public_ip})")
            return True

        # No running instance — try to launch one
        logger.info("[SPOT] No running GPU instance — requesting spot instance...")
        return self._request_spot_instance()

    def terminate(self) -> bool:
        """Terminate the active spot instance."""
        if not self.instance_id:
            logger.info("[SPOT] No instance to terminate")
            return True

        try:
            logger.info(f"[SPOT] Terminating instance {self.instance_id}...")
            self.ec2.terminate_instances(InstanceIds=[self.instance_id])
            self._clear_state()
            logger.info("[SPOT] Instance terminated — cost is now $0")
            return True
        except ClientError as e:
            logger.error(f"[SPOT] Failed to terminate: {e}")
            return False

    def get_status(self) -> dict:
        """Return current spot instance status."""
        if self.instance_id:
            self._refresh_state()
        return {
            "instance_id": self.instance_id,
            "public_ip": self.public_ip,
            "instance_type": self.instance_type,
            "state": self.state,
            "ssh_user": self.ssh_user,
        }

    def ssh_command(self, command: str, timeout: int = 60) -> tuple[bool, str]:
        """Execute SSH command on the spot instance."""
        import subprocess

        if not self.public_ip:
            logger.error("[SPOT-SSH] No public IP — instance not running")
            return False, "no public IP"

        full_cmd = [
            "ssh", "-i", self.ssh_key_path,
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "ConnectTimeout=30",
            "-o", "BatchMode=yes",
            f"{self.ssh_user}@{self.public_ip}",
            command,
        ]

        try:
            logger.info(f"[SPOT-SSH] {command}")
            result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=timeout)
            if result.returncode == 0:
                return True, result.stdout.strip()
            else:
                logger.warning(f"[SPOT-SSH] Failed (rc={result.returncode}): {result.stderr.strip()}")
                return False, result.stderr.strip()
        except subprocess.TimeoutExpired:
            logger.error("[SPOT-SSH] Command timed out")
            return False, "timeout"
        except Exception as e:
            logger.error(f"[SPOT-SSH] Error: {e}")
            return False, str(e)

    def is_ssh_ready(self) -> bool:
        """Quick check if instance accepts SSH."""
        ok, _ = self.ssh_command("echo ready", timeout=15)
        return ok

    def is_worker_running(self, service_name: str) -> bool:
        """Check if a systemd service is active on the spot instance."""
        ok, output = self.ssh_command(f"systemctl is-active {service_name}", timeout=15)
        return ok and output.strip() == "active"

    def start_worker(self, service_name: str) -> bool:
        """Start a systemd service on the spot instance."""
        logger.info(f"[SPOT] Starting service: {service_name}")
        ok, _ = self.ssh_command(f"sudo systemctl start {service_name}")
        return ok

    def stop_worker(self, service_name: str) -> bool:
        """Stop a systemd service on the spot instance."""
        logger.info(f"[SPOT] Stopping service: {service_name}")
        ok, _ = self.ssh_command(f"sudo systemctl stop {service_name}")
        return ok

    def is_gpu_busy(self) -> bool:
        """Check if GPU has active VRAM usage (any worker processing)."""
        ok, output = self.ssh_command(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo 0",
            timeout=15,
        )
        if ok:
            try:
                vram_mb = int(output.strip().splitlines()[0])
                return vram_mb > 1000
            except (ValueError, IndexError):
                pass
        return False

    # ── Private methods ──────────────────────────────────────────────────

    def _request_spot_instance(self) -> bool:
        """Try to launch a spot instance with fallback instance types."""
        if not self.ami_id:
            logger.error("[SPOT] SPOT_AMI_ID not configured — cannot launch")
            return False
        if not self.key_name:
            logger.error("[SPOT] SPOT_KEY_NAME not configured — cannot launch")
            return False

        for instance_type in SPOT_INSTANCE_TYPES:
            logger.info(f"[SPOT] Trying instance type: {instance_type}...")
            try:
                launched = self._launch_spot(instance_type)
                if launched:
                    logger.info(
                        f"[SPOT] Launched {instance_type} → {self.instance_id}"
                    )
                    # Wait for running state
                    if self._wait_for_running():
                        logger.info(
                            f"[SPOT] Instance running — IP: {self.public_ip}"
                        )
                        return True
                    else:
                        logger.error("[SPOT] Instance didn't reach running state — terminating")
                        self.terminate()
            except ClientError as e:
                error_code = e.response["Error"]["Code"]
                if error_code in (
                    "InsufficientInstanceCapacity",
                    "SpotMaxPriceTooLow",
                    "MaxSpotInstanceCountExceeded",
                ):
                    logger.warning(
                        f"[SPOT] {instance_type} unavailable ({error_code}) — trying next"
                    )
                    continue
                else:
                    logger.error(f"[SPOT] Unexpected error: {e}")
                    return False

        logger.error("[SPOT] All instance types exhausted — no GPU available")
        return False

    def _launch_spot(self, instance_type: str) -> bool:
        """Launch a single spot instance."""
        launch_params = {
            "ImageId": self.ami_id,
            "InstanceType": instance_type,
            "KeyName": self.key_name,
            "MinCount": 1,
            "MaxCount": 1,
            "InstanceMarketOptions": {
                "MarketType": "spot",
                "SpotOptions": {
                    "SpotInstanceType": "one-time",
                    "InstanceInterruptionBehavior": "terminate",
                },
            },
            "TagSpecifications": [
                {
                    "ResourceType": "instance",
                    "Tags": [
                        {"Key": "Name", "Value": PROJECT_TAG},
                        {"Key": "Project", "Value": "spark"},
                        {"Key": "ManagedBy", "Value": "spot-gpu-service"},
                    ],
                }
            ],
        }

        if self.security_group_ids:
            launch_params["SecurityGroupIds"] = self.security_group_ids
        if self.subnet_id:
            launch_params["SubnetId"] = self.subnet_id
        if self.iam_instance_profile:
            launch_params["IamInstanceProfile"] = {"Name": self.iam_instance_profile}

        response = self.ec2.run_instances(**launch_params)
        instance = response["Instances"][0]
        self.instance_id = instance["InstanceId"]
        self.instance_type = instance_type
        self.state = instance["State"]["Name"]
        logger.info(f"[SPOT] Instance requested: {self.instance_id} (state={self.state})")
        return True

    def _wait_for_running(self, timeout: int = 300) -> bool:
        """Wait for instance to reach 'running' state and get public IP."""
        start = time.time()
        while time.time() - start < timeout:
            self._refresh_state()
            if self.state == "running" and self.public_ip:
                # Extra wait for SSH daemon
                logger.info(f"[SPOT] Waiting {SSH_READY_WAIT}s for SSH readiness...")
                time.sleep(SSH_READY_WAIT)
                return True
            elif self.state in ("shutting-down", "terminated"):
                logger.error(f"[SPOT] Instance went to {self.state} unexpectedly")
                return False
            time.sleep(10)

        logger.error(f"[SPOT] Timeout waiting for running state ({timeout}s)")
        return False

    def _refresh_state(self):
        """Refresh instance state and public IP from AWS."""
        if not self.instance_id:
            self.state = "none"
            return

        try:
            resp = self.ec2.describe_instances(InstanceIds=[self.instance_id])
            reservations = resp.get("Reservations", [])
            if not reservations or not reservations[0].get("Instances"):
                self._clear_state()
                return

            inst = reservations[0]["Instances"][0]
            self.state = inst["State"]["Name"]
            self.public_ip = inst.get("PublicIpAddress")
            self.instance_type = inst.get("InstanceType")

            if self.state in ("terminated", "shutting-down"):
                logger.info(f"[SPOT] Instance {self.instance_id} is {self.state} — clearing")
                self._clear_state()

        except ClientError as e:
            if "InvalidInstanceID.NotFound" in str(e):
                self._clear_state()
            else:
                logger.error(f"[SPOT] Error refreshing state: {e}")

    def _recover_existing_instance(self):
        """On startup, look for an existing tagged spot instance we own."""
        try:
            resp = self.ec2.describe_instances(
                Filters=[
                    {"Name": "tag:Name", "Values": [PROJECT_TAG]},
                    {"Name": "tag:ManagedBy", "Values": ["spot-gpu-service"]},
                    {"Name": "instance-state-name", "Values": ["pending", "running"]},
                ]
            )
            for reservation in resp.get("Reservations", []):
                for inst in reservation.get("Instances", []):
                    self.instance_id = inst["InstanceId"]
                    self.public_ip = inst.get("PublicIpAddress")
                    self.instance_type = inst.get("InstanceType")
                    self.state = inst["State"]["Name"]
                    logger.info(
                        f"[SPOT] Recovered existing instance: {self.instance_id} "
                        f"({self.instance_type}, {self.state}, IP={self.public_ip})"
                    )
                    return
        except Exception as e:
            logger.warning(f"[SPOT] Could not scan for existing instances: {e}")

        logger.info("[SPOT] No existing spot instance found")

    def _clear_state(self):
        """Reset tracked instance state."""
        self.instance_id = None
        self.public_ip = None
        self.instance_type = None
        self.state = "none"

    @staticmethod
    def _parse_list(val: str) -> list[str]:
        """Parse comma-separated string into list."""
        if not val:
            return []
        return [v.strip() for v in val.split(",") if v.strip()]


# ── Singleton ────────────────────────────────────────────────────────────────
spot_gpu = SpotGPUManager()

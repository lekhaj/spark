"""
AWS EC2 Manager for GPU Instance Management
This module provides functionality to manage EC2 instances for Hunyuan3D processing:
- Start/stop GPU instances
- Check instance status and health
- Auto-scaling logic
- Cost optimization features
"""

import boto3
import time
import logging
import os
from typing import Optional, Dict, Any, List
from botocore.exceptions import ClientError, NoCredentialsError
from dataclasses import dataclass
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class InstanceInfo:
    """Data class to hold EC2 instance information"""
    instance_id: str
    state: str
    instance_type: str
    public_ip: Optional[str] = None
    private_ip: Optional[str] = None
    launch_time: Optional[datetime] = None
    uptime_hours: Optional[float] = None

class AWSManager:
    """
    Manages AWS EC2 instances for GPU processing tasks.
    Provides methods for starting, stopping, and monitoring instances.
    """
    
    def __init__(self, region: str = 'us-east-1', instance_id: Optional[str] = None):
        """
        Initialize AWS Manager.
        
        Args:
            region: AWS region where instances are located
            instance_id: Default instance ID to manage
        """
        self.region = region
        self.instance_id = instance_id
        self.ec2_client = None
        self.ec2_resource = None
        
        try:
            self._initialize_aws_clients()
            logger.info(f"AWS Manager initialized for region: {region}")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise
    
    def _initialize_aws_clients(self):
        """Initialize boto3 clients and resources"""
        try:
            # Try to create clients - this will use default credentials or IAM roles
            self.ec2_client = boto3.client('ec2', region_name=self.region)
            self.ec2_resource = boto3.resource('ec2', region_name=self.region)
            
            # Test the connection by describing instances (this will raise an error if credentials are invalid)
            self.ec2_client.describe_instances(MaxResults=5)
            
        except NoCredentialsError:
            logger.error("AWS credentials not found. Please configure AWS credentials.")
            raise
        except ClientError as e:
            logger.error(f"AWS client error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing AWS clients: {e}")
            raise
    
    def get_instance_info(self, instance_id: Optional[str] = None) -> Optional[InstanceInfo]:
        """
        Get detailed information about an EC2 instance.
        
        Args:
            instance_id: Instance ID to check (uses default if not provided)
            
        Returns:
            InstanceInfo object with instance details, or None if instance not found
        """
        target_id = instance_id or self.instance_id
        if not target_id:
            logger.error("No instance ID provided")
            return None
        
        try:
            response = self.ec2_client.describe_instances(InstanceIds=[target_id])
            
            if not response['Reservations']:
                logger.warning(f"Instance {target_id} not found")
                return None
            
            instance = response['Reservations'][0]['Instances'][0]
            
            # Calculate uptime if instance is running
            uptime_hours = None
            launch_time = instance.get('LaunchTime')
            if launch_time and instance['State']['Name'] == 'running':
                uptime_hours = (datetime.now(launch_time.tzinfo) - launch_time).total_seconds() / 3600
            
            return InstanceInfo(
                instance_id=target_id,
                state=instance['State']['Name'],
                instance_type=instance['InstanceType'],
                public_ip=instance.get('PublicIpAddress'),
                private_ip=instance.get('PrivateIpAddress'),
                launch_time=launch_time,
                uptime_hours=uptime_hours
            )
            
        except ClientError as e:
            logger.error(f"Error getting instance info for {target_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting instance info: {e}")
            return None
    
    def start_instance(self, instance_id: Optional[str] = None) -> bool:
        """
        Start an EC2 instance.
        
        Args:
            instance_id: Instance ID to start (uses default if not provided)
            
        Returns:
            True if start command was successful, False otherwise
        """
        target_id = instance_id or self.instance_id
        if not target_id:
            logger.error("No instance ID provided for start operation")
            return False
        
        try:
            info = self.get_instance_info(target_id)
            if not info:
                logger.error(f"Cannot start instance {target_id}: instance not found")
                return False
            
            if info.state == 'running':
                logger.info(f"Instance {target_id} is already running")
                return True
            
            if info.state not in ['stopped', 'stopping']:
                logger.warning(f"Instance {target_id} is in state '{info.state}', cannot start")
                return False
            
            logger.info(f"Starting instance {target_id}...")
            self.ec2_client.start_instances(InstanceIds=[target_id])
            logger.info(f"Start command sent for instance {target_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error starting instance {target_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error starting instance: {e}")
            return False
    
    def stop_instance(self, instance_id: Optional[str] = None) -> bool:
        """
        Stop an EC2 instance.
        
        Args:
            instance_id: Instance ID to stop (uses default if not provided)
            
        Returns:
            True if stop command was successful, False otherwise
        """
        target_id = instance_id or self.instance_id
        if not target_id:
            logger.error("No instance ID provided for stop operation")
            return False
        
        try:
            info = self.get_instance_info(target_id)
            if not info:
                logger.error(f"Cannot stop instance {target_id}: instance not found")
                return False
            
            if info.state == 'stopped':
                logger.info(f"Instance {target_id} is already stopped")
                return True
            
            if info.state not in ['running', 'pending']:
                logger.warning(f"Instance {target_id} is in state '{info.state}', cannot stop")
                return False
            
            logger.info(f"Stopping instance {target_id}...")
            self.ec2_client.stop_instances(InstanceIds=[target_id])
            logger.info(f"Stop command sent for instance {target_id}")
            return True
            
        except ClientError as e:
            logger.error(f"Error stopping instance {target_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error stopping instance: {e}")
            return False
    
    def wait_for_state(self, target_state: str, instance_id: Optional[str] = None, 
                      max_wait_time: int = 300, check_interval: int = 10) -> bool:
        """
        Wait for an instance to reach a specific state.
        
        Args:
            target_state: State to wait for ('running', 'stopped', etc.)
            instance_id: Instance ID to monitor (uses default if not provided)
            max_wait_time: Maximum time to wait in seconds
            check_interval: How often to check state in seconds
            
        Returns:
            True if instance reached target state, False if timeout
        """
        target_id = instance_id or self.instance_id
        if not target_id:
            logger.error("No instance ID provided for state monitoring")
            return False
        
        start_time = time.time()
        logger.info(f"Waiting for instance {target_id} to reach state '{target_state}'...")
        
        while (time.time() - start_time) < max_wait_time:
            info = self.get_instance_info(target_id)
            if not info:
                logger.error(f"Failed to get instance info while waiting for state")
                return False
            
            current_state = info.state
            logger.debug(f"Instance {target_id} current state: {current_state}")
            
            if current_state == target_state:
                logger.info(f"Instance {target_id} reached target state '{target_state}'")
                return True
            
            # Check for error states
            if current_state in ['terminated', 'terminating']:
                logger.error(f"Instance {target_id} is terminated or terminating")
                return False
            
            time.sleep(check_interval)
        
        logger.error(f"Timeout waiting for instance {target_id} to reach state '{target_state}' "
                    f"after {max_wait_time} seconds")
        return False
    
    def ensure_instance_running(self, instance_id: Optional[str] = None,
                               max_wait_time: int = 300, check_interval: int = 10) -> bool:
        """
        Ensure an instance is running, starting it if necessary.
        
        Args:
            instance_id: Instance ID to ensure is running
            max_wait_time: Maximum time to wait for startup
            check_interval: How often to check status
            
        Returns:
            True if instance is running, False otherwise
        """
        target_id = instance_id or self.instance_id
        if not target_id:
            logger.error("No instance ID provided")
            return False
        
        info = self.get_instance_info(target_id)
        if not info:
            logger.error(f"Instance {target_id} not found")
            return False
        
        if info.state == 'running':
            logger.info(f"Instance {target_id} is already running")
            return True
        
        # Start the instance if it's stopped
        if info.state == 'stopped':
            if not self.start_instance(target_id):
                return False
        elif info.state == 'stopping':
            # Wait for it to stop, then start
            if not self.wait_for_state('stopped', target_id, max_wait_time//2, check_interval):
                return False
            if not self.start_instance(target_id):
                return False
        
        # Wait for it to be running
        return self.wait_for_state('running', target_id, max_wait_time, check_interval)
    
    def get_instance_cost_estimate(self, instance_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get cost estimate for an instance based on its uptime.
        
        Args:
            instance_id: Instance ID to analyze
            
        Returns:
            Dictionary with cost information
        """
        target_id = instance_id or self.instance_id
        if not target_id:
            return {"error": "No instance ID provided"}
        
        info = self.get_instance_info(target_id)
        if not info:
            return {"error": "Instance not found"}
        
        # Rough pricing for common GPU instances (as of 2024)
        # These should be updated with current AWS pricing
        hourly_rates = {
            'g4dn.xlarge': 0.526,
            'g4dn.2xlarge': 0.752,
            'g4dn.4xlarge': 1.204,
            'g4dn.8xlarge': 2.176,
            'g4dn.12xlarge': 3.912,
            'g4dn.16xlarge': 4.352,
            'p3.2xlarge': 3.06,
            'p3.8xlarge': 12.24,
            'p3.16xlarge': 24.48,
        }
        
        hourly_rate = hourly_rates.get(info.instance_type, 0.5)  # Default fallback rate
        
        cost_info = {
            "instance_type": info.instance_type,
            "hourly_rate_usd": hourly_rate,
            "current_state": info.state,
            "uptime_hours": info.uptime_hours or 0,
            "estimated_cost_usd": (info.uptime_hours or 0) * hourly_rate
        }
        
        if info.launch_time:
            cost_info["launch_time"] = info.launch_time.isoformat()
        
        return cost_info
    
    def list_gpu_instances(self) -> List[InstanceInfo]:
        """
        List all GPU instances in the region.
        
        Returns:
            List of InstanceInfo objects for GPU instances
        """
        try:
            # Filter for common GPU instance types
            gpu_instance_types = [
                'g4dn.*', 'g4ad.*', 'g5.*', 'g5g.*',
                'p3.*', 'p4.*', 'p5.*',
                'inf1.*', 'inf2.*'
            ]
            
            response = self.ec2_client.describe_instances()
            gpu_instances = []
            
            for reservation in response['Reservations']:
                for instance in reservation['Instances']:
                    instance_type = instance['InstanceType']
                    
                    # Check if it's a GPU instance
                    is_gpu = any(
                        instance_type.startswith(gpu_type.replace('.*', ''))
                        for gpu_type in gpu_instance_types
                    )
                    
                    if is_gpu and instance['State']['Name'] != 'terminated':
                        # Calculate uptime if running
                        uptime_hours = None
                        launch_time = instance.get('LaunchTime')
                        if launch_time and instance['State']['Name'] == 'running':
                            uptime_hours = (datetime.now(launch_time.tzinfo) - launch_time).total_seconds() / 3600
                        
                        gpu_instances.append(InstanceInfo(
                            instance_id=instance['InstanceId'],
                            state=instance['State']['Name'],
                            instance_type=instance_type,
                            public_ip=instance.get('PublicIpAddress'),
                            private_ip=instance.get('PrivateIpAddress'),
                            launch_time=launch_time,
                            uptime_hours=uptime_hours
                        ))
            
            return gpu_instances
            
        except ClientError as e:
            logger.error(f"Error listing GPU instances: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing instances: {e}")
            return []


def get_aws_manager(instance_id: Optional[str] = None, region: Optional[str] = None) -> AWSManager:
    """
    Factory function to create an AWSManager instance.
    
    Args:
        instance_id: Default instance ID to manage
        region: AWS region
        
    Returns:
        Configured AWSManager instance
    """
    from config import AWS_REGION, AWS_GPU_INSTANCE_ID
    
    # Use provided values or fall back to config
    final_region = region or AWS_REGION
    final_instance_id = instance_id or AWS_GPU_INSTANCE_ID
    
    return AWSManager(region=final_region, instance_id=final_instance_id)


# Convenience functions for common operations
def ensure_gpu_running(instance_id: Optional[str] = None, region: Optional[str] = None) -> bool:
    """Convenience function to ensure a GPU instance is running"""
    manager = get_aws_manager(instance_id, region)
    return manager.ensure_instance_running()

def stop_gpu_instance(instance_id: Optional[str] = None, region: Optional[str] = None) -> bool:
    """Convenience function to stop a GPU instance"""
    manager = get_aws_manager(instance_id, region)
    return manager.stop_instance()

def get_gpu_status(instance_id: Optional[str] = None, region: Optional[str] = None) -> Optional[InstanceInfo]:
    """Convenience function to get GPU instance status"""
    manager = get_aws_manager(instance_id, region)
    return manager.get_instance_info()

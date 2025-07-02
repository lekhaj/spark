#!/usr/bin/env python3
"""
Test suite for AWS Manager functionality.
Tests EC2 instance management, cost estimation, and AWS integration.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestAWSManager(unittest.TestCase):
    """Test cases for AWS Manager."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_instance_id = 'i-1234567890abcdef0'
        self.test_region = 'us-west-2'
        
        # Mock environment variables
        self.env_patcher = patch.dict(os.environ, {
            'AWS_REGION': self.test_region,
            'AWS_GPU_INSTANCE_ID': self.test_instance_id,
            'AWS_MAX_STARTUP_WAIT_TIME': '300',
            'AWS_EC2_CHECK_INTERVAL': '10'
        })
        self.env_patcher.start()
        
    def tearDown(self):
        """Clean up after each test method."""
        self.env_patcher.stop()
    
    @patch('boto3.Session')
    def test_aws_manager_initialization(self, mock_session):
        """Test AWS Manager initialization."""
        from aws_manager import AWSManager
        
        # Mock boto3 session and client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        
        self.assertEqual(manager.instance_id, self.test_instance_id)
        self.assertEqual(manager.region, self.test_region)
        mock_session.assert_called_once()
        mock_session.return_value.client.assert_called_with('ec2', region_name=self.test_region)
    
    @patch('boto3.Session')
    def test_get_instance_info_running(self, mock_session):
        """Test getting instance info for a running instance."""
        from aws_manager import AWSManager
        
        # Mock EC2 client and response
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        mock_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'running'},
                    'InstanceType': 'g4dn.xlarge',
                    'PublicIpAddress': '1.2.3.4',
                    'LaunchTime': datetime.now() - timedelta(hours=2)
                }]
            }]
        }
        mock_ec2.describe_instances.return_value = mock_response
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        info = manager.get_instance_info()
        
        self.assertIsNotNone(info)
        self.assertEqual(info.state, 'running')
        self.assertEqual(info.instance_type, 'g4dn.xlarge')
        self.assertEqual(info.public_ip, '1.2.3.4')
        self.assertGreater(info.uptime_hours, 1.5)
    
    @patch('boto3.Session')
    def test_get_instance_info_stopped(self, mock_session):
        """Test getting instance info for a stopped instance."""
        from aws_manager import AWSManager
        
        # Mock EC2 client and response
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        mock_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'stopped'},
                    'InstanceType': 'g4dn.xlarge',
                    'LaunchTime': datetime.now() - timedelta(hours=24)
                }]
            }]
        }
        mock_ec2.describe_instances.return_value = mock_response
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        info = manager.get_instance_info()
        
        self.assertIsNotNone(info)
        self.assertEqual(info.state, 'stopped')
        self.assertIsNone(info.public_ip)
        self.assertEqual(info.uptime_hours, 0)
    
    @patch('boto3.Session')
    def test_start_instance_success(self, mock_session):
        """Test successful instance start."""
        from aws_manager import AWSManager
        
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        # Mock start_instances response
        mock_ec2.start_instances.return_value = {
            'StartingInstances': [{
                'InstanceId': self.test_instance_id,
                'CurrentState': {'Name': 'pending'},
                'PreviousState': {'Name': 'stopped'}
            }]
        }
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        result = manager.start_instance()
        
        self.assertTrue(result)
        mock_ec2.start_instances.assert_called_once_with(InstanceIds=[self.test_instance_id])
    
    @patch('boto3.Session')
    def test_stop_instance_success(self, mock_session):
        """Test successful instance stop."""
        from aws_manager import AWSManager
        
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        # Mock stop_instances response
        mock_ec2.stop_instances.return_value = {
            'StoppingInstances': [{
                'InstanceId': self.test_instance_id,
                'CurrentState': {'Name': 'stopping'},
                'PreviousState': {'Name': 'running'}
            }]
        }
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        result = manager.stop_instance()
        
        self.assertTrue(result)
        mock_ec2.stop_instances.assert_called_once_with(InstanceIds=[self.test_instance_id])
    
    @patch('boto3.Session')
    @patch('time.sleep')
    def test_ensure_instance_running_already_running(self, mock_sleep, mock_session):
        """Test ensure_instance_running when instance is already running."""
        from aws_manager import AWSManager
        
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        # Mock instance already running
        mock_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'running'},
                    'InstanceType': 'g4dn.xlarge',
                    'PublicIpAddress': '1.2.3.4',
                    'LaunchTime': datetime.now() - timedelta(hours=1)
                }]
            }]
        }
        mock_ec2.describe_instances.return_value = mock_response
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        result = manager.ensure_instance_running(max_wait_time=60)
        
        self.assertTrue(result)
        mock_ec2.start_instances.assert_not_called()
        mock_sleep.assert_not_called()
    
    @patch('boto3.Session')
    @patch('time.sleep')
    def test_ensure_instance_running_needs_start(self, mock_sleep, mock_session):
        """Test ensure_instance_running when instance needs to be started."""
        from aws_manager import AWSManager
        
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        # Mock instance states: stopped -> pending -> running
        stopped_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'stopped'},
                    'InstanceType': 'g4dn.xlarge'
                }]
            }]
        }
        
        pending_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'pending'},
                    'InstanceType': 'g4dn.xlarge'
                }]
            }]
        }
        
        running_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'running'},
                    'InstanceType': 'g4dn.xlarge',
                    'PublicIpAddress': '1.2.3.4',
                    'LaunchTime': datetime.now()
                }]
            }]
        }
        
        # Setup mock responses in sequence
        mock_ec2.describe_instances.side_effect = [
            stopped_response,  # Initial check
            pending_response,  # After start
            running_response   # Final state
        ]
        
        mock_ec2.start_instances.return_value = {
            'StartingInstances': [{
                'InstanceId': self.test_instance_id,
                'CurrentState': {'Name': 'pending'}
            }]
        }
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        result = manager.ensure_instance_running(max_wait_time=60, check_interval=1)
        
        self.assertTrue(result)
        mock_ec2.start_instances.assert_called_once_with(InstanceIds=[self.test_instance_id])
        self.assertTrue(mock_sleep.called)
    
    @patch('boto3.Session')
    def test_get_instance_cost_estimate(self, mock_session):
        """Test instance cost estimation."""
        from aws_manager import AWSManager
        
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        # Mock instance running for 2 hours
        mock_response = {
            'Reservations': [{
                'Instances': [{
                    'InstanceId': self.test_instance_id,
                    'State': {'Name': 'running'},
                    'InstanceType': 'g4dn.xlarge',
                    'LaunchTime': datetime.now() - timedelta(hours=2)
                }]
            }]
        }
        mock_ec2.describe_instances.return_value = mock_response
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        cost_info = manager.get_instance_cost_estimate()
        
        self.assertIsNotNone(cost_info)
        self.assertIn('hourly_rate', cost_info)
        self.assertIn('current_session_cost', cost_info)
        self.assertIn('uptime_hours', cost_info)
        self.assertGreater(cost_info['hourly_rate'], 0)
        self.assertGreater(cost_info['current_session_cost'], 0)
    
    @patch('boto3.Session')
    def test_list_gpu_instances(self, mock_session):
        """Test listing GPU instances."""
        from aws_manager import AWSManager
        
        # Mock EC2 client
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        
        # Mock multiple GPU instances
        mock_response = {
            'Reservations': [
                {
                    'Instances': [{
                        'InstanceId': 'i-1234567890abcdef0',
                        'State': {'Name': 'running'},
                        'InstanceType': 'g4dn.xlarge',
                        'Tags': [{'Key': 'Name', 'Value': 'GPU-Worker-1'}]
                    }]
                },
                {
                    'Instances': [{
                        'InstanceId': 'i-0987654321fedcba0',
                        'State': {'Name': 'stopped'},
                        'InstanceType': 'p3.2xlarge',
                        'Tags': [{'Key': 'Name', 'Value': 'GPU-Worker-2'}]
                    }]
                }
            ]
        }
        mock_ec2.describe_instances.return_value = mock_response
        
        manager = AWSManager(region=self.test_region)
        instances = manager.list_gpu_instances()
        
        self.assertEqual(len(instances), 2)
        self.assertEqual(instances[0]['InstanceId'], 'i-1234567890abcdef0')
        self.assertEqual(instances[0]['State'], 'running')
        self.assertEqual(instances[1]['InstanceId'], 'i-0987654321fedcba0')
        self.assertEqual(instances[1]['State'], 'stopped')
    
    @patch('boto3.Session')
    def test_aws_error_handling(self, mock_session):
        """Test AWS error handling."""
        from aws_manager import AWSManager
        import botocore.exceptions
        
        # Mock EC2 client that raises an exception
        mock_ec2 = Mock()
        mock_session.return_value.client.return_value = mock_ec2
        mock_ec2.describe_instances.side_effect = botocore.exceptions.ClientError(
            {'Error': {'Code': 'InvalidInstanceID.NotFound', 'Message': 'Instance not found'}},
            'DescribeInstances'
        )
        
        manager = AWSManager(instance_id=self.test_instance_id, region=self.test_region)
        info = manager.get_instance_info()
        
        self.assertIsNone(info)
    
    def test_get_aws_manager_function(self):
        """Test the get_aws_manager convenience function."""
        from aws_manager import get_aws_manager
        
        manager = get_aws_manager(instance_id=self.test_instance_id, region=self.test_region)
        
        self.assertEqual(manager.instance_id, self.test_instance_id)
        self.assertEqual(manager.region, self.test_region)


class TestAWSManagerConfiguration(unittest.TestCase):
    """Test AWS Manager configuration loading."""
    
    def test_configuration_from_env(self):
        """Test loading configuration from environment variables."""
        with patch.dict(os.environ, {
            'AWS_REGION': 'eu-west-1',
            'AWS_GPU_INSTANCE_ID': 'i-test123',
            'AWS_MAX_STARTUP_WAIT_TIME': '600'
        }):
            from aws_manager import get_aws_manager
            
            manager = get_aws_manager()
            
            # Manager should use environment variables
            self.assertEqual(manager.region, 'eu-west-1')
    
    def test_configuration_missing_credentials(self):
        """Test behavior when AWS credentials are missing."""
        with patch.dict(os.environ, {}, clear=True):
            from aws_manager import AWSManager
            
            # Should not raise exception during initialization
            manager = AWSManager(region='us-east-1')
            
            # But should fail when trying to use AWS services
            with patch('boto3.Session') as mock_session:
                mock_session.side_effect = Exception("No credentials")
                
                info = manager.get_instance_info()
                self.assertIsNone(info)


if __name__ == '__main__':
    # Create a test suite
    unittest.main(verbosity=2)

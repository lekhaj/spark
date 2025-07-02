import boto3
import logging
import os

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- AWS Configuration ---
# boto3 will automatically pick up credentials from env vars or ~/.aws/credentials
# Ensure your AWS_DEFAULT_REGION environment variable is set, or configure it via `aws configure`.
# Example: export AWS_DEFAULT_REGION="ap-south-1"
ec2_client = boto3.client('ec2')

def start_ec2_instance(instance_id: str):
    """
    Directly attempts to start a specified EC2 instance by its Instance ID.
    Does NOT check current state or resolve from IP.
    """
    try:
        logger.info(f"Attempting to start instance directly: {instance_id}")
        response = ec2_client.start_instances(InstanceIds=[instance_id])
        
        starting_instances = [i['InstanceId'] for i in response['StartingInstances']]
        logger.info(f"Successfully sent start command for instances: {starting_instances}")
        
        # Optional: Wait until the instance is running (still requires DescribeInstances)
        # If you *also* want to avoid DescribeInstances for the waiter, remove this block.
        # However, it's generally good for script reliability to wait for the state.
        # If the user has permission to Start/Stop but NOT Describe, this waiter will fail.
        # Let's keep it optional but make a note.
        # If you continue to get UnauthorizedOperation, you may need to remove this waiter.
        logger.info(f"Waiting for instance '{instance_id}' to enter 'running' state... (Requires ec2:DescribeInstances permission)")
        waiter = ec2_client.get_waiter('instance_running')
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={'Delay': 15, 'MaxAttempts': 40})
        logger.info(f"Instance '{instance_id}' is now running.")

    except Exception as e:
        logger.error(f"Failed to start instance '{instance_id}': {e}", exc_info=True)
        if "UnauthorizedOperation" in str(e):
            logger.error("HINT: The IAM user needs 'ec2:DescribeInstances' permission for the waiter to function.")


def stop_ec2_instance(instance_id: str):
    """
    Directly attempts to stop a specified EC2 instance by its Instance ID.
    Does NOT check current state or resolve from IP.
    """
    try:
        logger.info(f"Attempting to stop instance directly: {instance_id}")
        response = ec2_client.stop_instances(InstanceIds=[instance_id])
        
        stopping_instances = [i['InstanceId'] for i in response['StoppingInstances']]
        logger.info(f"Successfully sent stop command for instances: {stopping_instances}")

        # Optional: Wait until the instance is stopped (still requires DescribeInstances)
        logger.info(f"Waiting for instance '{instance_id}' to enter 'stopped' state... (Requires ec2:DescribeInstances permission)")
        waiter = ec2_client.get_waiter('instance_stopped')
        waiter.wait(InstanceIds=[instance_id], WaiterConfig={'Delay': 15, 'MaxAttempts': 40})
        logger.info(f"Instance '{instance_id}' is now stopped.")

    except Exception as e:
        logger.error(f"Failed to stop instance '{instance_id}': {e}", exc_info=True)
        if "UnauthorizedOperation" in str(e):
            logger.error("HINT: The IAM user needs 'ec2:DescribeInstances' permission for the waiter to function.")

# --- Main Execution ---
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Start or stop an EC2 instance directly by Instance ID.")
    parser.add_argument('action', choices=['start', 'stop'], help="Action to perform: 'start' or 'stop'.")
    parser.add_argument('instance_id', help="The ID of the EC2 instance (e.g., i-0e029990527fa2b73).")
    args = parser.parse_args()

    if args.action == 'start':
        start_ec2_instance(args.instance_id)
    elif args.action == 'stop':
        stop_ec2_instance(args.instance_id)


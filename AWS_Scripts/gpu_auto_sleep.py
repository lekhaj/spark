import os
import time
import datetime
import subprocess
import logging
import boto3
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/var/log/gpu_auto_sleep.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Configuration
UTILIZATION_THRESHOLD = 1  # GPU utilization threshold in percent (1% instead of 0% to avoid false positives)
IDLE_TIME_THRESHOLD = 15 * 60  # 15 minutes in seconds
CHECK_INTERVAL = 60  # Check every minute
INSTANCE_ID = None  # Will be retrieved automatically

def get_gpu_utilization():
    """Get current GPU utilization percentage using nvidia-smi."""
    try:
        # Run nvidia-smi to get GPU utilization
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse the output
        utilization = float(result.stdout.strip())
        logger.info(f"Current GPU utilization: {utilization}%")
        return utilization
    except (subprocess.SubprocessError, ValueError) as e:
        logger.error(f"Error getting GPU utilization: {e}")
        return 100  # Return a high value to prevent shutdown in case of errors

def get_instance_id():
    """Get the EC2 instance ID using the instance metadata service."""
    try:
        response = requests.get('http://169.254.169.254/latest/meta-data/instance-id', timeout=2)
        instance_id = response.text
        logger.info(f"Running on EC2 instance: {instance_id}")
        return instance_id
    except requests.RequestException as e:
        logger.error(f"Error getting instance ID: {e}")
        logger.error("This might not be an EC2 instance or metadata service is not accessible")
        return None

def shutdown_instance(instance_id):
    """Shut down the EC2 instance."""
    if not instance_id:
        logger.error("Instance ID not available, can't shut down")
        return False
    
    try:
        # Create EC2 client
        ec2 = boto3.client('ec2')
        
        # Stop the instance
        logger.info(f"Stopping instance {instance_id}")
        ec2.stop_instances(InstanceIds=[instance_id])
        
        logger.info(f"Stop request sent for instance {instance_id}")
        return True
    except Exception as e:
        logger.error(f"Error stopping instance: {e}")
        return False

def main():
    """Main function to monitor GPU utilization and shut down if idle."""
    logger.info("GPU auto-sleep monitoring started")
    
    # Get instance ID once at startup
    global INSTANCE_ID
    INSTANCE_ID = get_instance_id()
    if not INSTANCE_ID:
        logger.warning("Not running on EC2 or unable to determine instance ID. Will log but not shut down.")
    
    idle_start_time = None
    
    while True:
        utilization = get_gpu_utilization()
        
        # Check if GPU utilization is below threshold
        if utilization <= UTILIZATION_THRESHOLD:
            current_time = time.time()
            
            # Start counting idle time if this is the first detection
            if idle_start_time is None:
                idle_start_time = current_time
                logger.info(f"GPU utilization below threshold ({UTILIZATION_THRESHOLD}%). Starting idle timer.")
            else:
                # Calculate how long the GPU has been idle
                idle_duration = current_time - idle_start_time
                logger.info(f"GPU idle for {idle_duration:.1f} seconds")
                
                # If idle for longer than threshold, shut down
                if idle_duration >= IDLE_TIME_THRESHOLD:
                    logger.warning(f"GPU has been idle for {idle_duration:.1f} seconds (threshold: {IDLE_TIME_THRESHOLD})")
                    logger.warning("Initiating instance shutdown")
                    
                    if shutdown_instance(INSTANCE_ID):
                        logger.info("Shutdown sequence initiated. Exiting monitoring script.")
                        break
                    else:
                        # Reset timer and try again later if shutdown failed
                        logger.error("Shutdown failed. Resetting idle timer and will try again later.")
                        idle_start_time = None
        else:
            # Reset the idle timer if GPU is being used
            if idle_start_time is not None:
                logger.info("GPU activity detected. Resetting idle timer.")
                idle_start_time = None
        
        # Sleep before next check
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
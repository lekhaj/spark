import boto3
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ec2_gpu_control")

def get_ec2_instances(region, filters=None):
    ec2 = boto3.resource('ec2', region_name=region)
    if filters:
        instances = ec2.instances.filter(Filters=filters)
    else:
        instances = ec2.instances.all()
    return list(instances)

def start_instances(instance_ids, region):
    ec2 = boto3.client('ec2', region_name=region)
    logger.info(f"Starting instances: {instance_ids}")
    ec2.start_instances(InstanceIds=instance_ids)

def stop_instances(instance_ids, region):
    ec2 = boto3.client('ec2', region_name=region)
    logger.info(f"Stopping instances: {instance_ids}")
    ec2.stop_instances(InstanceIds=instance_ids)

def list_gpu_instances(region):
    filters = [
        {'Name': 'instance-state-name', 'Values': ['running', 'stopped']},
        {'Name': 'instance-type', 'Values': ['g4dn.*', 'p3.*', 'p4.*', 'g5.*', 'g2.*', 'g3.*', 'p2.*']}
    ]
    instances = get_ec2_instances(region, filters)
    for inst in instances:
        print(f"ID: {inst.id}, State: {inst.state['Name']}, Type: {inst.instance_type}, Public IP: {inst.public_ip_address}")

def main():
    parser = argparse.ArgumentParser(description="Control AWS EC2 GPU Instances")
    parser.add_argument('--region', type=str, required=True, help='AWS region')
    parser.add_argument('--list', action='store_true', help='List GPU instances')
    parser.add_argument('--start', nargs='+', help='Instance IDs to start')
    parser.add_argument('--stop', nargs='+', help='Instance IDs to stop')
    args = parser.parse_args()

    if args.list:
        list_gpu_instances(args.region)
    if args.start:
        start_instances(args.start, args.region)
    if args.stop:
        stop_instances(args.stop, args.region)

if __name__ == "__main__":
    main()

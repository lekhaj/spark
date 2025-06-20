#!/bin/bash
# update_gpu_ip.sh

NEW_GPU_IP="$1"

if [ -z "$NEW_GPU_IP" ]; then
    echo "Usage: $0 <new_gpu_ip>"
    exit 1
fi

echo "Updating GPU IP to: $NEW_GPU_IP"

# Update .env.cpu
sed -i "s/GPU_SPOT_INSTANCE_IP=.*/GPU_SPOT_INSTANCE_IP=$NEW_GPU_IP/" .env.cpu
sed -i "s/redis:\/\/[^:]*:6379/redis:\/\/$NEW_GPU_IP:6379/g" .env.cpu

echo "✅ Updated .env.cpu"

# Test connection
python3 -c "
import redis
try:
    r = redis.Redis.from_url('redis://$NEW_GPU_IP:6379/0')
    r.ping()
    print('✅ Redis connection successful')
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
"
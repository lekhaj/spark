#!/bin/bash
# update_gpu_ip.sh - Comprehensive GPU IP update script
# Updates all configuration files when GPU instance IP changes

NEW_GPU_IP="$1"

if [ -z "$NEW_GPU_IP" ]; then
    echo "Usage: $0 <new_gpu_ip>"
    echo "Example: $0 13.201.23.51"
    exit 1
fi

echo "🔄 Updating GPU IP to: $NEW_GPU_IP"
echo "=================================="

# 1. Update .env.cpu
if [ -f ".env.cpu" ]; then
    echo "📝 Updating .env.cpu..."
    sed -i "s/GPU_SPOT_INSTANCE_IP=.*/GPU_SPOT_INSTANCE_IP=$NEW_GPU_IP/" .env.cpu
    sed -i "s/redis:\/\/[^:]*:6379/redis:\/\/$NEW_GPU_IP:6379/g" .env.cpu
    echo "✅ Updated .env.cpu"
else
    echo "⚠️  .env.cpu not found, skipping..."
fi

# 2. Update src/config.py
if [ -f "src/config.py" ]; then
    echo "📝 Updating src/config.py..."
    # Update both occurrences of GPU_SPOT_INSTANCE_IP
    sed -i "s/GPU_SPOT_INSTANCE_IP = os\.getenv('GPU_SPOT_INSTANCE_IP', '[^']*')/GPU_SPOT_INSTANCE_IP = os.getenv('GPU_SPOT_INSTANCE_IP', '$NEW_GPU_IP')/g" src/config.py
    echo "✅ Updated src/config.py"
else
    echo "⚠️  src/config.py not found, skipping..."
fi

# 3. Update scripts/start_cpu_worker.sh
if [ -f "scripts/start_cpu_worker.sh" ]; then
    echo "📝 Updating scripts/start_cpu_worker.sh..."
    # Update Redis URLs and GPU IP fallbacks
    sed -i "s/redis:\/\/[^:]*:6379/redis:\/\/$NEW_GPU_IP:6379/g" scripts/start_cpu_worker.sh
    sed -i "s/GPU_SPOT_INSTANCE_IP:-[^}]*/GPU_SPOT_INSTANCE_IP:-$NEW_GPU_IP/g" scripts/start_cpu_worker.sh
    echo "✅ Updated scripts/start_cpu_worker.sh"
else
    echo "⚠️  scripts/start_cpu_worker.sh not found, skipping..."
fi

# 4. Update tests/test_gpu_ip.py
if [ -f "tests/test_gpu_ip.py" ]; then
    echo "📝 Updating tests/test_gpu_ip.py..."
    sed -i "s/os\.environ\['GPU_SPOT_INSTANCE_IP'\] = '[^']*'/os.environ['GPU_SPOT_INSTANCE_IP'] = '$NEW_GPU_IP'/g" tests/test_gpu_ip.py
    echo "✅ Updated tests/test_gpu_ip.py"
else
    echo "⚠️  tests/test_gpu_ip.py not found, skipping..."
fi

# 5. Update any other potential config files
echo "📝 Checking for other configuration files..."

# Update any .env files that might contain the old IP
for env_file in .env.* ; do
    if [ -f "$env_file" ] && [ "$env_file" != ".env.gpu" ]; then
        if grep -q "redis://" "$env_file" 2>/dev/null; then
            echo "📝 Updating $env_file..."
            sed -i "s/redis:\/\/[^:]*:6379/redis:\/\/$NEW_GPU_IP:6379/g" "$env_file"
            sed -i "s/GPU_SPOT_INSTANCE_IP=.*/GPU_SPOT_INSTANCE_IP=$NEW_GPU_IP/" "$env_file"
            echo "✅ Updated $env_file"
        fi
    fi
done

echo ""
echo "🧪 Testing Redis connection..."

# Test connection with proper error handling
python3 -c "
import redis
import sys
try:
    r = redis.Redis.from_url('redis://$NEW_GPU_IP:6379/0', socket_timeout=5, socket_connect_timeout=5)
    r.ping()
    print('✅ Redis connection successful')
    sys.exit(0)
except redis.ConnectionError as e:
    print(f'❌ Redis connection failed: Connection error - {e}')
    print('   Make sure Redis is running on the GPU instance and port 6379 is open')
    sys.exit(1)
except redis.TimeoutError as e:
    print(f'❌ Redis connection failed: Timeout - {e}')
    print('   Check network connectivity and firewall settings')
    sys.exit(1)
except Exception as e:
    print(f'❌ Redis connection failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 GPU IP update completed successfully!"
    echo "📋 Summary of changes:"
    echo "   • Updated .env.cpu"
    echo "   • Updated src/config.py (2 locations)"
    echo "   • Updated scripts/start_cpu_worker.sh (3 locations)"
    echo "   • Updated tests/test_gpu_ip.py"
    echo ""
    echo "📌 Next steps:"
    echo "   1. Restart CPU workers: ./scripts/start_cpu_worker.sh"
    echo "   2. Restart main app: cd src && python3 merged_gradio_app.py"
    echo "   3. Test 3D generation to ensure GPU connection works"
else
    echo ""
    echo "⚠️  Configuration updated but Redis connection failed"
    echo "📌 Please check:"
    echo "   1. GPU instance is running: $NEW_GPU_IP"
    echo "   2. Redis is running on GPU instance: sudo systemctl status redis-server"
    echo "   3. Port 6379 is open: sudo ufw status"
    echo "   4. Redis accepts external connections: check /etc/redis/redis.conf"
fi
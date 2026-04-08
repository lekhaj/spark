#!/bin/bash
# setup_redis_readwrite.sh - Configure Redis for read/write operations
# This script sets up Redis to support both read and write operations

set -e

echo "=== Redis Read/Write Configuration Setup ==="

# Function to test Redis connection
test_redis() {
    local url=$1
    local name=$2
    echo "Testing $name connection to $url..."
    
    if redis-cli -u "$url" ping > /dev/null 2>&1; then
        echo "✅ $name connection successful"
        
        # Test write capability
        if redis-cli -u "$url" set test_write_key "test_value" EX 10 > /dev/null 2>&1; then
            echo "✅ $name write operation successful"
            redis-cli -u "$url" del test_write_key > /dev/null 2>&1
            return 0
        else
            echo "❌ $name write operation failed"
            return 1
        fi
    else
        echo "❌ $name connection failed"
        return 1
    fi
}

# Function to configure Redis as master
configure_redis_master() {
    local redis_url=$1
    echo "Configuring Redis at $redis_url as master..."
    
    # Convert from slave to master if needed
    redis-cli -u "$redis_url" SLAVEOF NO ONE
    
    # Disable read-only mode
    redis-cli -u "$redis_url" CONFIG SET slave-read-only no
    
    # Verify configuration
    local role=$(redis-cli -u "$redis_url" INFO replication | grep "role:" | cut -d: -f2 | tr -d '\r')
    if [ "$role" = "master" ]; then
        echo "✅ Redis successfully configured as master"
        return 0
    else
        echo "❌ Failed to configure Redis as master (role: $role)"
        return 1
    fi
}

# Main configuration
echo "1. Checking local Redis instance..."
LOCAL_REDIS="redis://127.0.0.1:6379/0"

if test_redis "$LOCAL_REDIS" "Local Redis"; then
    echo "✅ Local Redis is working properly"
else
    echo "⚠️ Local Redis needs configuration"
    configure_redis_master "$LOCAL_REDIS"
fi

echo ""
echo "2. Checking remote Redis instance..."
REMOTE_REDIS="redis://13.203.200.155:6379/0"

if test_redis "$REMOTE_REDIS" "Remote Redis"; then
    echo "✅ Remote Redis is working properly"
else
    echo "⚠️ Remote Redis needs configuration"
    configure_redis_master "$REMOTE_REDIS"
fi

echo ""
echo "3. Testing application Redis configuration..."

# Test with CPU worker configuration
export WORKER_TYPE=cpu
export USE_CELERY=True

cd "$(dirname "$0")"
if [ -f ".env.cpu" ]; then
    echo "Loading CPU environment configuration..."
    export $(cat .env.cpu | grep -v '^#' | grep -v '^$' | xargs)
fi

# Test the Python configuration
python3 -c "
import sys
sys.path.insert(0, 'src')
try:
    from config import REDIS_CONFIG
    result = REDIS_CONFIG.test_connection()
    
    write_ok = result.get('write', {}).get('success', False)
    read_ok = result.get('read', {}).get('success', False)
    
    print(f'Application Redis Configuration:')
    print(f'  Worker Type: {REDIS_CONFIG.worker_type}')
    print(f'  Write URL: {REDIS_CONFIG.write_url}')
    print(f'  Read URL: {REDIS_CONFIG.read_url}')
    print(f'  Write Status: {\"✅ OK\" if write_ok else \"❌ FAILED\"}')
    print(f'  Read Status: {\"✅ OK\" if read_ok else \"❌ FAILED\"}')
    
    if write_ok and read_ok:
        print('✅ Application Redis configuration is working properly')
        exit(0)
    else:
        print('❌ Application Redis configuration has issues')
        exit(1)
except Exception as e:
    print(f'❌ Error testing application configuration: {e}')
    exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Redis read/write setup completed successfully!"
    echo ""
    echo "You can now:"
    echo "  - Submit Celery tasks (write operations)"
    echo "  - Read task results and status"
    echo "  - Use both CPU and GPU workers"
    echo ""
    echo "To test the setup:"
    echo "  python3 test_redis_config.py --test-env"
else
    echo ""
    echo "❌ Redis setup encountered issues. Please check the error messages above."
    exit 1
fi

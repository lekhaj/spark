#!/usr/bin/env python3
"""
Test script for Redis read/write configuration.
Run this script to verify Redis connectivity for both read and write operations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_redis_config(worker_type=None):
    """Test Redis configuration for specified worker type."""
    
    # Set worker type if specified
    if worker_type:
        os.environ['WORKER_TYPE'] = worker_type
        print(f"Testing Redis configuration for worker type: {worker_type}")
    
    try:
        from config import REDIS_CONFIG
        
        print(f"\n=== Redis Configuration Test ===")
        print(f"Worker Type: {REDIS_CONFIG.worker_type}")
        print(f"GPU Instance IP: {REDIS_CONFIG.gpu_ip}")
        print(f"Write URL: {REDIS_CONFIG.write_url}")
        print(f"Read URL: {REDIS_CONFIG.read_url}")
        
        # Test connections
        print(f"\n=== Connection Tests ===")
        results = REDIS_CONFIG.test_connection()
        
        # Write test
        write_result = results.get('write', {})
        if write_result.get('success'):
            print(f"✅ Write connection successful")
            print(f"   URL: {write_result['url']}")
        else:
            print(f"❌ Write connection failed")
            print(f"   URL: {write_result['url']}")
            print(f"   Error: {write_result.get('error')}")
        
        # Read test
        read_result = results.get('read', {})
        if read_result.get('success'):
            print(f"✅ Read connection successful")
            print(f"   URL: {read_result['url']}")
        else:
            print(f"❌ Read connection failed")
            print(f"   URL: {read_result['url']}")
            print(f"   Error: {read_result.get('error')}")
        
        # Overall status
        write_ok = write_result.get('success', False)
        read_ok = read_result.get('success', False)
        
        print(f"\n=== Overall Status ===")
        if write_ok and read_ok:
            print(f"✅ All Redis connections working properly")
            return True
        else:
            print(f"❌ Redis connection issues detected")
            if not write_ok:
                print(f"   - Write operations will fail")
            if not read_ok:
                print(f"   - Read operations will fail")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Redis configuration: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_environment_files():
    """Test environment file configurations."""
    print(f"\n=== Environment Files Test ===")
    
    # Test CPU configuration
    print(f"\nTesting CPU worker configuration...")
    os.environ.clear()
    if os.path.exists('.env.cpu'):
        with open('.env.cpu', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    cpu_result = test_redis_config('cpu')
    
    # Test GPU configuration
    print(f"\nTesting GPU worker configuration...")
    os.environ.clear()
    if os.path.exists('.env.gpu'):
        with open('.env.gpu', 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
    
    gpu_result = test_redis_config('gpu')
    
    return cpu_result, gpu_result

def main():
    parser = argparse.ArgumentParser(description='Test Redis read/write configuration')
    parser.add_argument('--worker-type', choices=['cpu', 'gpu'], help='Worker type to test')
    parser.add_argument('--test-env', action='store_true', help='Test environment files')
    args = parser.parse_args()
    
    if args.test_env:
        cpu_ok, gpu_ok = test_environment_files()
        print(f"\n=== Final Results ===")
        print(f"CPU configuration: {'✅ OK' if cpu_ok else '❌ FAILED'}")
        print(f"GPU configuration: {'✅ OK' if gpu_ok else '❌ FAILED'}")
    else:
        result = test_redis_config(args.worker_type)
        sys.exit(0 if result else 1)

if __name__ == '__main__':
    main()

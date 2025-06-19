#!/bin/bash
# Setup script for GPU spot instance at 13.203.200.155
# This script configures the spot instance with Redis, dependencies, and auto-start services

set -e  # Exit on any error

echo "Setting up GPU spot instance..."

# Update system packages
echo "Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Redis server
echo "Installing Redis server..."
sudo apt install -y redis-server

# Configure Redis for external connections
echo "Configuring Redis for external connections..."
sudo cp /etc/redis/redis.conf /etc/redis/redis.conf.backup

# Allow external connections and disable protected mode
sudo sed -i 's/^bind 127.0.0.1/#bind 127.0.0.1/' /etc/redis/redis.conf
sudo sed -i 's/^protected-mode yes/protected-mode no/' /etc/redis/redis.conf

# Set Redis to start on boot
sudo systemctl enable redis-server
sudo systemctl restart redis-server

# Configure firewall for Redis (if UFW is enabled)
if sudo ufw status | grep -q "Status: active"; then
    echo "Configuring firewall for Redis..."
    sudo ufw allow 6379
fi

# Verify Redis is working
echo "Testing Redis server..."
redis-cli ping

# Install Python dependencies if requirements.txt exists
if [ -f "/home/ubuntu/spark/requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip3 install -r /home/ubuntu/spark/requirements.txt
fi

# Install additional GPU dependencies
echo "Installing GPU-specific dependencies..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Create environment file for GPU worker
echo "Creating GPU worker environment file..."
cat > /home/ubuntu/spark/.env.gpu << EOF
USE_CELERY=True
REDIS_BROKER_URL=redis://127.0.0.1:6379/0
REDIS_RESULT_BACKEND=redis://127.0.0.1:6379/0
WORKER_TYPE=gpu
HUNYUAN3D_DEVICE=cuda
AWS_GPU_IS_SPOT_INSTANCE=True
SPOT_INSTANCE_HANDLING_ENABLED=True
EOF

# Make the GPU worker script executable
chmod +x /home/ubuntu/spark/scripts/start_gpu_worker.sh

# Create systemd service for Celery GPU worker
echo "Creating systemd service for Celery GPU worker..."
sudo tee /etc/systemd/system/celery-gpu-worker.service > /dev/null <<EOF
[Unit]
Description=Celery GPU Worker for Spot Instance
After=network.target redis.service
Requires=redis.service

[Service]
Type=simple
User=ubuntu
Group=ubuntu
EnvironmentFile=/home/ubuntu/spark/.env.gpu
WorkingDirectory=/home/ubuntu/spark/src
ExecStart=/home/ubuntu/spark/scripts/start_gpu_worker.sh
Restart=always
RestartSec=10
KillMode=mixed
TimeoutStopSec=30

# Resource limits for GPU worker
MemoryMax=16G
CPUQuota=400%

[Install]
WantedBy=multi-user.target
EOF

# Enable and start the service
sudo systemctl daemon-reload
sudo systemctl enable celery-gpu-worker

# Create a startup script that runs on boot
echo "Creating startup script for spot instance..."
sudo tee /etc/systemd/system/spot-instance-startup.service > /dev/null <<EOF
[Unit]
Description=Spot Instance Startup Tasks
After=network.target

[Service]
Type=oneshot
User=root
ExecStart=/bin/bash -c 'sleep 30 && systemctl start celery-gpu-worker'
RemainAfterExit=true

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl enable spot-instance-startup

# Create monitoring script for spot instance health
echo "Creating spot instance monitoring script..."
cat > /home/ubuntu/spot_monitor.sh << 'EOF'
#!/bin/bash
# Simple monitoring script for spot instance

while true; do
    echo "$(date): Spot instance monitoring check"
    
    # Check Redis status
    if ! redis-cli ping > /dev/null 2>&1; then
        echo "$(date): Redis is down, restarting..."
        sudo systemctl restart redis-server
    fi
    
    # Check Celery worker status
    if ! systemctl is-active --quiet celery-gpu-worker; then
        echo "$(date): Celery worker is down, restarting..."
        sudo systemctl restart celery-gpu-worker
    fi
    
    # Check GPU availability
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "$(date): GPU not accessible"
    fi
    
    sleep 300  # Check every 5 minutes
done
EOF

chmod +x /home/ubuntu/spot_monitor.sh

echo "GPU spot instance setup completed!"
echo ""
echo "Summary:"
echo "  ✅ Redis server installed and configured"
echo "  ✅ Firewall configured for Redis access"
echo "  ✅ Python dependencies installed"
echo "  ✅ GPU worker systemd service created"
echo "  ✅ Auto-start services enabled"
echo "  ✅ Monitoring script created"
echo ""
echo "To start the GPU worker manually:"
echo "  sudo systemctl start celery-gpu-worker"
echo ""
echo "To monitor the worker:"
echo "  sudo systemctl status celery-gpu-worker"
echo "  sudo journalctl -u celery-gpu-worker -f"
echo ""
echo "To start monitoring script:"
echo "  nohup /home/ubuntu/spot_monitor.sh > /home/ubuntu/spot_monitor.log 2>&1 &"

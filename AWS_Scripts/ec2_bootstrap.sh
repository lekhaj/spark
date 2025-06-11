#!/bin/bash
# EC2 Bootstrap Script for S3 Upload Configuration
# Use this in the User Data section when launching an EC2 instance

# Set up logging
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1
echo "Starting bootstrap script..."

# Update system packages
yum update -y || apt-get update -y

# Install required packages
yum install -y python3 python3-pip git || apt-get install -y python3 python3-pip git

# Create app directory
APP_DIR="/opt/text-to-image-pipeline"
mkdir -p $APP_DIR
cd $APP_DIR

# Clone the repository (replace with your actual repository URL)
# git clone https://github.com/your-username/text-to-image-pipeline.git .

# Install dependencies
pip3 install boto3 requests

# Create directories for output if they don't exist
mkdir -p output/images
mkdir -p output/3d_assets

# ------------------------------------
# OPTION 1: Set up credentials manually (not recommended for production)
# ------------------------------------
# mkdir -p /home/ec2-user/.aws
# cat > /home/ec2-user/.aws/credentials << EOL
# [default]
# aws_access_key_id = YOUR_ACCESS_KEY_HERE
# aws_secret_access_key = YOUR_SECRET_KEY_HERE
# EOL

# cat > /home/ec2-user/.aws/config << EOL
# [default]
# region = us-west-2
# output = json
# EOL

# # Set proper permissions
# chmod 600 /home/ec2-user/.aws/credentials
# chown -R ec2-user:ec2-user /home/ec2-user/.aws

# ------------------------------------
# OPTION 2: Use environment variables in a systemd service (better than plain text)
# ------------------------------------
# Create a service file for automatic uploads
cat > /etc/systemd/system/s3-upload.service << EOL
[Unit]
Description=S3 Upload Service
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_DIR
# Environment variables option (use IAM role instead when possible)
# Environment="AWS_ACCESS_KEY_ID=YOUR_ACCESS_KEY_HERE"
# Environment="AWS_SECRET_ACCESS_KEY=YOUR_SECRET_KEY_HERE"
# Environment="AWS_DEFAULT_REGION=us-west-2"
Environment="S3_BUCKET_NAME=your-bucket-name"
ExecStart=/usr/bin/python3 $APP_DIR/AWS_Scripts/s3_upload.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
EOL

# ------------------------------------
# OPTION 3: Set up a cron job to periodically upload files
# ------------------------------------
cat > /etc/cron.d/s3-upload << EOL
# Run every hour to upload new files to S3
0 * * * * ec2-user cd $APP_DIR && /usr/bin/python3 $APP_DIR/AWS_Scripts/s3_upload.py --bucket your-bucket-name
EOL

# Set proper permissions
chmod 644 /etc/cron.d/s3-upload

echo "Bootstrap script completed"

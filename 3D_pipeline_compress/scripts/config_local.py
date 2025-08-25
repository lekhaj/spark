# scripts/config_local.py

# Your local development settings for Blender
BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe" # Your specific local path
COMPRESSION_LEVELS = {
    'high': 10000,
    'medium': 50000,
    'low': 100000
}

# --- ADD THESE PLACEHOLDERS ---
# These are needed to prevent a crash when main_processor.py is imported.
# The values can be None since they won't be used in the local test.
AWS_REGION = "us-east-1"  # A dummy region string is safer
MONGO_URI = "mongodb://localhost:27017/" # A dummy connection string
MONGO_DB_NAME = "test_db" # MUST be a string
MONGO_COLLECTION_NAME = "test_collection" # MUST be a string
SQS_QUEUE_URL = "dummy_queue_url"
S3_BUCKET_NAME = "dummy_bucket_name"
S3_PROCESSED_PREFIX = "processed/"
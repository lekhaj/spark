import os
AWS_REGION = "ap-south-1"
SQS_QUEUE_URL = "https://sqs.ap-south-1.amazonaws.com/073643077764/3d-model-processing-queue"
S3_BUCKET_NAME = "sparkassets"
S3_PROCESSED_PREFIX = "processed/"

# MongoDB Configuration
MONGO_URI = "mongodb://kartik:Kartikg421@15.206.99.66:27017/?authSource=admin"

MONGO_DB_NAME = "World_builder"
MONGO_COLLECTION_NAME = "biomes"

# Blender Configuration
#BLENDER_EXECUTABLE = "C:\\Program Files\\Blender Foundation\\Blender 4.5\\blender.exe"
BLENDER_EXECUTABLE = os.getenv('BLENDER_PATH', 'blender')
# Processing Configuration
COMPRESSION_LEVELS = {
    "5k": 5000,
    "8k": 8000,
    "10k": 10000
    
}
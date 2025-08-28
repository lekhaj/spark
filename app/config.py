from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "spackassets")
S3_IMAGE_FOLDER = os.getenv("S3_IMAGE_FOLDER", "images")
S3_3D_ASSETS_FOLDER = os.getenv("S3_3D_ASSETS_FOLDER", "3d_assets")

# Redis Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
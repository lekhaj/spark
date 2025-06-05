# config.py
import os

# OpenRouter LLM settings
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


DB_USER = "sagar"
DB_PASSWORD = "Password"
DB_HOST = "ec2-13-203-200-155.ap-south-1.compute.amazonaws.com"
DB_PORT = 27017
DB_NAME = "World_builder"
COLLECTION_NAME = "biomes"
MONGO_URI = "mongodb://sagar:Password@ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017/World_builder?authSource=admin"

LOG_FILE = "logs/biome_generator.log"
GRID_SIZE = 10

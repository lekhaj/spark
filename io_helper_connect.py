from pymongo import MongoClient

# MongoDB URI (replace this with your EC2 hostname if different)
uri = "mongodb://ec2-13-203-200-155.ap-south-1.compute.amazonaws.com:27017/"

try:
    client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    # Try to get server info to verify connection
    server_info = client.server_info()
    print("✅ Connected to MongoDB server!")
    print("Server Info:", server_info)

    # List databases
    dbs = client.list_database_names()
    print("Databases:", dbs)

except Exception as e:
    print("❌ Failed to connect to MongoDB:", e)

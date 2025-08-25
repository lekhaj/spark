
import json
import os
import sys


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.aws_service import upload_to_s3
from app.services.mongo_service import update_task_status, get_task_by_id

temp_file_path = "test_file.txt"
with open(temp_file_path, "w") as f:
    f.write("This is a test file for S3 upload.")

# Simulate a task ID
test_task_id = "test-12345-abcde"

# Upload the test file to S3
print("Uploading test file to S3...")
s3_link = upload_to_s3(temp_file_path, f"test_assets/{test_task_id}.txt")
print(f"S3 Link: {s3_link}")

# Update the database
print("Updating MongoDB with the S3 link...")
success = update_task_status(test_task_id, "completed", s3_link)
print(f"Update successful: {success}")

# Verify the result by retrieving from MongoDB
task = get_task_by_id(test_task_id)
print("\nRetrieved task from MongoDB:")
print(json.dumps(task, indent=2))

# Clean up the test file after the script runs
os.remove(temp_file_path)

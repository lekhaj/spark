import time
from app import generate_2d_image_task  # Assuming your Celery tasks are in app.py

print("Attempting to push a test task to the Celery queue...")
try:
    # Use dummy arguments to push the task
    task = generate_2d_image_task.delay(
        text_prompt="test prompt",
        width=512,
        height=512,
        num_images=1,
        s3_bucket_name="test-bucket",
        base_filename="test-file"
    )
    print(f"Task successfully pushed with ID: {task.id}")
    print("Waiting for 10 seconds to give the worker time to process...")
    time.sleep(10)
    
    # Check the status of the task
    result = task.status
    print(f"Status of task {task.id} is: {result}")
    
except Exception as e:
    print(f"An error occurred: {e}")

print("Script finished.")

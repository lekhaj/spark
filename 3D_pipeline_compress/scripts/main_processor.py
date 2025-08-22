import os
import sys
import subprocess
import json
import boto3
from pymongo import MongoClient
from urllib.parse import unquote_plus

# Import configuration
import config

# Add the project's root directory to the Python path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Setup local directories
INPUT_DIR = os.path.join(PROJECT_ROOT, "input_models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output_models")
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize clients
sqs = boto3.client('sqs', region_name=config.AWS_REGION)
s3 = boto3.client('s3', region_name=config.AWS_REGION)
mongo_client = MongoClient(config.MONGO_URI)
db = mongo_client[config.MONGO_DB_NAME]
collection = db[config.MONGO_COLLECTION_NAME]

def run_blender_script(input_path, output_dir):
    """Calls Blender to process the model."""
    print("Starting Blender processing...")
    
    blender_script_path = os.path.join(os.path.dirname(__file__), "process_model.py")
    
    # Construct command-line arguments for compression levels
    levels_args = []
    for name, faces in config.COMPRESSION_LEVELS.items():
        levels_args.extend([name, str(faces)])

    command = [
        config.BLENDER_EXECUTABLE,
        '--background',
        '--python', blender_script_path,
        '--',        input_path,
        output_dir,
        *levels_args
    ]
    
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        print("Blender processing completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print("Blender script failed!")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return False

  

def main():
    """Main loop to poll SQS and process messages."""
    print("Worker started. Polling SQS queue for messages...")
    while True:
        response = sqs.receive_message(
            QueueUrl=config.SQS_QUEUE_URL,
            MaxNumberOfMessages=1,
            WaitTimeSeconds=20
        )
        
        if "Messages" not in response:
            continue

        message = response['Messages'][0]
        receipt_handle = message['ReceiptHandle']
    for message in messages:
        receipt_handle = message['ReceiptHandle']    
        try:
            body = json.loads(message['Body'])
            
            # Check if the message is a valid S3 event
            if 'Records' not in body or not body['Records']:
                print("Warning: Received a malformed message (e.g., a test message), skipping.")
                sqs.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                continue

            s3_key = body['Records'][0]['s3']['object']['key']
            # Decode the filename to handle spaces and special characters
            s3_key = unquote_plus(s3_key)
            print(f"Received message for S3 object: {s3_key}")
            run_blender_processing(s3_key)
            print("Blender processing completed successfully.")   # ✅ same indent as run_blender_processing

            # Delete message if processed successfully
            sqs.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)

        except Exception as e:
            print(f"An error occurred: {e}. Message will not be deleted.")
            # 1. Download the file from S3
            local_input_path = os.path.join(INPUT_DIR, os.path.basename(s3_key))
            s3.download_file(config.S3_BUCKET_NAME, s3_key, local_input_path)
            
            # Extract asset ID from filename (e.g., '123_car.glb' -> '123')
            asset_id_from_filename = os.path.basename(s3_key).split('_')[0]

            # 2. Run Blender processing
            if not run_blender_script(local_input_path, OUTPUT_DIR):
                raise Exception("Blender processing failed.")

            # 3. Upload processed files to S3
            processed_urls = {}
            print(f"Checking for output files in local directory: {OUTPUT_DIR}")
            for filename in os.listdir(OUTPUT_DIR):
                if filename.endswith(".glb"):
                    local_output_path = os.path.join(OUTPUT_DIR, filename)
                    s3_output_key = f"{config.S3_PROCESSED_PREFIX}{filename}"
                    
                    print(f"Found local file: {local_output_path}")
                    print(f"Attempting to upload to S3 at: s3://{config.S3_BUCKET_NAME}/{s3_output_key}")
                    
                    try:
                        s3.upload_file(local_output_path, config.S3_BUCKET_NAME, s3_output_key)
                        
                        url = f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_REGION}.amazonaws.com/{s3_output_key}"
                        level_name = filename.split('_')[-1].replace('.glb', '')
                        processed_urls[f"url_{level_name}"] = url
                        print(f"Successfully uploaded {filename}")

                    except Exception as e:
                        print(f"!!! ERROR DURING UPLOAD of {filename}: {e}")


            # 4. Update MongoDB
            print(f"Updating MongoDB for asset ID: {asset_id_from_filename}...")
            if processed_urls:
                collection.update_one(
                    {'asset_id': asset_id_from_filename},
                    {'$set': processed_urls}
                )
                print("MongoDB update successful.")
            else:
                print("Warning: No processed files were found to upload or update in MongoDB.")


            # 5. Clean up local files
            os.remove(local_input_path)
            for filename in os.listdir(OUTPUT_DIR):
                os.remove(os.path.join(OUTPUT_DIR, filename))
            
            # 6. Delete message from SQS
            sqs.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            print(f"Successfully processed and deleted message for {s3_key}")

        except Exception as e:
            print(f"An error occurred: {e}. Message will not be deleted.")

if __name__ == "__main__":
    main()

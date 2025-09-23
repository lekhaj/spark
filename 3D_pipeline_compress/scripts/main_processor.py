import os
import sys
import subprocess
import json
import boto3
from pymongo import MongoClient
from urllib.parse import unquote_plus
import datetime
from PIL import Image # NEW IMPORT FOR PNG PROCESSING

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
        '--',
        input_path,
        output_dir,
        *levels_args
    ]
    
    try:
        # Run the command and capture output for debugging
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Blender processing completed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print("Blender script failed!")
        print(f"Stderr: {e.stderr}")
        print(f"Stdout: {e.stdout}")
        return False

# --- NEW FUNCTION FOR PNG COMPRESSION ---
def compress_png(input_path, output_path):
    """Compresses a PNG file using Pillow."""
    try:
        print(f"Compressing PNG: {input_path}")
        with Image.open(input_path) as img:
            # The 'optimize' flag and 'compress_level' reduce file size.
            # compress_level=9 is the highest compression.
            img.save(output_path, "PNG", optimize=True, compress_level=9)
        print("PNG compression successful.")
        return True
    except Exception as e:
        print(f"Error compressing PNG: {e}")
        return False
# --- END NEW FUNCTION ---

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
        
        try:
            # Clean up local directories from any previous runs
            print("Cleaning up local directories for a new job...")
            for directory in [INPUT_DIR, OUTPUT_DIR]:
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    os.remove(file_path)

            body = json.loads(message['Body'])
            
            if 'Records' not in body or not body['Records']:
                print("Warning: Received a malformed message, skipping.")
                sqs.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                continue

            s3_key = unquote_plus(body['Records'][0]['s3']['object']['key'])
            print(f"Received message for S3 object: {s3_key}")

            # 1. Download the file from S3
            local_input_path = os.path.join(INPUT_DIR, os.path.basename(s3_key))
            s3.download_file(config.S3_BUCKET_NAME, s3_key, local_input_path)
            
            asset_id_from_filename = os.path.basename(s3_key).split('_')[0]

            # --- MODIFIED LOGIC: CHOOSE PROCESSOR BASED ON FILE TYPE ---
            processing_successful = False
            file_type = ""

            if s3_key.lower().endswith(('.glb', '.obj')):
                file_type = "3D_MODEL"
                print("Detected 3D model, running Blender processor...")
                processing_successful = run_blender_script(local_input_path, OUTPUT_DIR)
            
            elif s3_key.lower().endswith('.png'):
                file_type = "PNG_IMAGE"
                print("Detected PNG image, running PNG compressor...")
                local_output_path = os.path.join(OUTPUT_DIR, os.path.basename(s3_key))
                processing_successful = compress_png(local_input_path, local_output_path)
            
            else:
                print(f"Unsupported file type: {s3_key}. Skipping.")
                sqs.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
                continue

            if not processing_successful:
                raise Exception(f"Processing failed for file type: {file_type}")
            # --- END MODIFIED LOGIC ---

            # 3. Upload processed files to S3
            processed_urls = {}
            print(f"Checking for output files in local directory: {OUTPUT_DIR}")
            for filename in os.listdir(OUTPUT_DIR):
                local_output_path = os.path.join(OUTPUT_DIR, filename)
                s3_output_key = f"{config.S3_PROCESSED_PREFIX}{filename}"
                
                print(f"Uploading {local_output_path} to s3://{config.S3_BUCKET_NAME}/{s3_output_key}")
                s3.upload_file(local_output_path, config.S3_BUCKET_NAME, s3_output_key)
                
                url = f"https://{config.S3_BUCKET_NAME}.s3.{config.AWS_REGION}.amazonaws.com/{s3_output_key}"
                
                # Adjust how the URL key is stored based on file type
                if file_type == "3D_MODEL":
                    level_name = filename.split('_')[-1].replace('.glb', '')
                    processed_urls[f"url_{level_name}"] = url
                elif file_type == "PNG_IMAGE":
                    processed_urls["url_compressed"] = url
                
                print(f"Successfully uploaded {filename}")

            # 4. Update MongoDB
            print(f"Updating MongoDB for asset ID: {asset_id_from_filename}...")
            if processed_urls:
                update_document = {
                    'biome_id': 'unique_biome_id_here', # Replace with real data
                    'name': f'Asset_{asset_id_from_filename}', # Replace with real data
                    'climate': 'temperate', # Replace with real data
                    'description': f'Generated asset for biome {asset_id_from_filename}', # Replace with real data
                    'structures': [],
                    'layout_matrix': [],
                    'last_updated': datetime.datetime.now(),
                    **processed_urls
                }

                collection.update_one(
                    {'asset_id': asset_id_from_filename},
                    {'$set': update_document},
                    upsert=True
                )
                print("MongoDB update successful.")
            else:
                print("Warning: No processed files were found to upload or update in MongoDB.")
            
            # 5. Delete message from SQS
            sqs.delete_message(QueueUrl=config.SQS_QUEUE_URL, ReceiptHandle=receipt_handle)
            print(f"Successfully processed and deleted message for {s3_key}")

        except Exception as e:
            print(f"An error occurred: {e}. Message will not be deleted.")

if __name__ == "__main__":
    main()
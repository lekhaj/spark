import redis, json, boto3, requests, os
from hy3dgen.shapeogen import Hunyuan3DDiTFlowMatchingPipeline

# Redis config
REDIS_HOST = "15.206.99.66"
REDIS_PORT = 6380
r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# AWS S3 client
s3 = boto3.client("s3", region_name="ap-south-1")
BUCKET_NAME = "sparkassets"

def download_from_s3_url(s3_url, local_path):
    """Download file from a pre-signed/public S3 URL"""
    response = requests.get(s3_url)
    if response.status_code != 200:
        raise Exception(f"Failed to download from {s3_url}, status={response.status_code}")
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

def upload_to_s3(local_file, s3_key):
    """Upload file to S3 and return public URL"""
    s3.upload_file(local_file, BUCKET_NAME, s3_key)
    return f"https://{BUCKET_NAME}.s3.ap-south-1.amazonaws.com/{s3_key}"

def process_task(task):
    print(f"[GPU]  Starting task: {task}")

    
    global shape_pipeline
    if "shape_pipeline" not in globals():
        print("[GPU] Loading Hunyuan3D model...")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2"
        )
        print("[GPU] Model loaded successfully!")

    # 2. Download input image from S3 URL
    local_input = "input.jpg"
    download_from_s3_url(task["image_s3_url"], local_input)
    print(f"[GPU] Downloaded input image from {task['image_s3_url']}")

    # 3. Generate mesh
    output_file = "mesh.obj"
    print("[GPU] Generating 3D mesh...")
    mesh = shape_pipeline(local_input)  
    with open(output_file, "w") as f:
        f.write(str(mesh))  
    print("[GPU] Mesh saved to mesh.obj")

    # 4. Upload output mesh to S3
    s3_key = f"3d_assets/{task['job_id']}_mesh.obj"
    output_url = upload_to_s3(output_file, s3_key)
    print(f"[GPU]  Uploaded mesh to: {output_url}")

    # 5. Dummy MongoDB update later      
    print(f"[MongoDB] Updated job {task['job_id']} status='completed', output_url={output_url}")

    return output_url

def main():
    print("[GPU] Worker started. Listening for tasks...")
    while True:
        task_data = r.brpop("tasks", timeout=30)
        if task_data:
            _, raw_task = task_data
            task = json.loads(raw_task)
            print("[GPU] Got new task from Redis queue.")
            
            # MongoDB update → processing
            print(f"[MongoDB] Updated job {task['job_id']} status='processing'")

            try:
                output_url = process_task(task)

                # Push back result to results queue
                r.lpush("results", json.dumps({"job_id": task["job_id"], "output_url": output_url}))
            except Exception as e:
                print(f"[GPU]  Error processing task {task['job_id']}: {e}")
                print(f"[MongoDB] Updated job {task['job_id']} status='failed'")
        else:
            print("[GPU] No new tasks. Worker shutting down...")
            break

if __name__ == "__main__":
    main()
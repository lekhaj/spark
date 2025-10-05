import os, json, redis, boto3, requests, torch, gc, subprocess
import multiprocessing as mp
from rembg import remove
from PIL import Image
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ------------ Config ------------
REDIS_HOST = os.getenv("REDIS_HOST", "15.206.99.66")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6380))
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME", "sparkassets")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")

MODEL_QUEUE = "model_tasks"
# --------------------------------

def init_s3_client():
    """Initialize S3 client inside each process"""
    return boto3.client("s3", region_name=os.getenv("S3_REGION", "ap-south-1"))

def update_mongo(biome_id, update_key, mesh_url, painted_url, status="model_generated"):
    """Update MongoDB with 3D asset URLs"""
    try:
        mongo = MongoClient(os.getenv("MONGO_URI"))
        db = mongo[os.getenv("DB_NAME")]
        
        result = db.biomes.update_one(
            {"_id": biome_id},
            {"$set": {
                f"{update_key}.mesh_url": mesh_url,
                f"{update_key}.painted_url": painted_url,
                f"{update_key}.status": status
            }}
        )
        
        mongo.close()
        return result.modified_count > 0
    except Exception as e:
        print(f"❌ MongoDB update failed: {e}")
        return False

def download_image(s3_url, local_path):
    """Download image from S3 URL"""
    response = requests.get(s3_url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path

def remove_background(input_path, output_path):
    """Remove image background"""
    with Image.open(input_path) as img:
        img_no_bg = remove(img)
        img_no_bg.save(output_path)
    return output_path

def upload_to_s3(local_file, s3_key, s3_client):
    """Upload file to S3"""
    s3_client.upload_file(local_file, os.getenv("BUCKET_NAME", "sparkassets"), s3_key)
    return f"https://{os.getenv('BUCKET_NAME', 'sparkassets')}.s3.{os.getenv('S3_REGION', 'ap-south-1')}.amazonaws.com/{s3_key}"

def isolated_generation_worker(task):
    """
    COMPLETELY ISOLATED worker process for 3D generation
    This process will be destroyed after each task, freeing ALL memory
    """
    print(f"🔄 [{task['job_id']}] Starting isolated generation process...")
    
    try:
        # Initialize clients INSIDE the worker process
        s3_client = init_s3_client()
        
        # Download image
        input_img = f"/tmp/{task['job_id']}_input.jpg"
        download_image(task["image_s3_url"], input_img)
        
        # Remove background
        nobg_img = f"/tmp/{task['job_id']}_nobg.png"
        remove_background(input_img, nobg_img)
        
        # Load models FRESH in this process
        print(f"📦 [{task['job_id']}] Loading models...")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained(
            "tencent/Hunyuan3D-2",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        # Generate 3D mesh
        print(f"🎯 [{task['job_id']}] Generating 3D mesh...")
        results = shape_pipeline(nobg_img)
        mesh = results[0] if isinstance(results, (list, tuple)) else results
        
        mesh_file = f"/tmp/{task['job_id']}_mesh.glb"
        mesh.export(mesh_file)
        
        # Generate texture
        print(f"🎨 [{task['job_id']}] Painting texture...")
        textured_mesh = paint_pipeline(mesh, image=nobg_img)
        textured_file = f"/tmp/{task['job_id']}_textured.glb"
        textured_mesh.export(textured_file)
        
        # Upload to S3
        mesh_url = upload_to_s3(mesh_file, f"3d_assets/{task['job_id']}_mesh.glb", s3_client)
        painted_url = upload_to_s3(textured_file, f"3d_assets/{task['job_id']}_textured.glb", s3_client)
        
        # Update MongoDB
        if task.get('biome_id') and task.get('update_key'):
            update_mongo(task['biome_id'], task['update_key'], mesh_url, painted_url, "completed")
        
        # Cleanup local files
        for temp_file in [input_img, nobg_img, mesh_file, textured_file]:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        
        # Explicit cleanup (though process exit will handle this)
        del shape_pipeline, paint_pipeline, mesh, textured_mesh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "job_id": task['job_id'],
            "mesh_url": mesh_url,
            "painted_url": painted_url,
            "status": "completed"
        }
        
    except Exception as e:
        print(f"❌ [{task['job_id']}] Error in worker process: {e}")
        
        # Update MongoDB with error
        if task.get('biome_id') and task.get('update_key'):
            update_mongo(task['biome_id'], task['update_key'], "", "", "error")
        
        raise

def process_with_isolation(task, use_timeout=True):
    """
    Process task with complete memory isolation using separate processes
    """
    job_id = task['job_id']
    
    try:
        # Strategy 1: Use process pool with maxtasksperchild=1 (most effective)
        print(f"🔄 [{job_id}] Starting isolated generation...")
        
        # Create a fresh process for each task
        with mp.get_context("spawn").Pool(processes=1, maxtasksperchild=1) as pool:
            if use_timeout:
                # With timeout to prevent hanging
                result = pool.apply_async(isolated_generation_worker, (task,))
                try:
                    output = result.get(timeout=1800)  # 30 minute timeout
                    print(f"✅ [{job_id}] Generation completed successfully")
                    return output
                except mp.TimeoutError:
                    print(f"⏰ [{job_id}] Generation timed out after 30 minutes")
                    raise Exception("Generation timeout")
            else:
                # Without timeout
                output = pool.apply(isolated_generation_worker, (task,))
                print(f"✅ [{job_id}] Generation completed successfully")
                return output
                
    except Exception as e:
        print(f"❌ [{job_id}] Process isolation failed: {e}")
        raise

def lightweight_main():
    """
    Main process that stays lightweight - only handles queue management
    """
    print("🚀 3D Model Worker Started (Process Isolation Mode)")
    print("💡 Each generation runs in isolated process with fresh memory")
    print(f"📡 Listening on queue: {MODEL_QUEUE}")
    
    # Initialize Redis in main process only
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "15.206.99.66"),
        port=int(os.getenv("REDIS_PORT", 6380)),
        db=0, 
        decode_responses=True
    )
    
    task_count = 0
    
    while True:
        try:
            # Get task from queue
            task_data = r.blpop(MODEL_QUEUE, timeout=30)
            if not task_data:
                print("⏰ 30 seconds without tasks. Still listening...")
                continue

            _, raw_task = task_data
            task = json.loads(raw_task)
            task_count += 1
            
            print(f"\n{'='*60}")
            print(f"🎯 [{task['job_id']}] Starting task #{task_count}")
            print(f"{'='*60}")
            
            # Process with complete memory isolation
            result = process_with_isolation(task)
            
            print(f"✅ [{result['job_id']}] Task completed successfully")
            print(f"📊 Memory freed automatically by process termination")
            
        except json.JSONDecodeError as e:
            print(f"❌ JSON decode error: {e}")
        except Exception as e:
            job_id = task.get('job_id', 'unknown') if 'task' in locals() else 'unknown'
            print(f"❌ [{job_id}] Main loop error: {e}")
        
        print(f"{'-'*60}\n")

if __name__ == "__main__":
    # Critical for multiprocessing to work correctly
    mp.set_start_method('spawn', force=True)
    lightweight_main()
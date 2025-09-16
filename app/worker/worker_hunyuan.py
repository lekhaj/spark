
import os
import json
import redis
import requests
from rembg import remove
from PIL import Image
import trimesh
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.texgen import Hunyuan3DPaintPipeline
from .worker_config import redis_client as r, s3, BUCKET_NAME

# Redis config
MODEL_QUEUE = "model_tasks"

# Global model caches
shape_pipeline = None
paint_pipeline = None


def download_from_s3_url(s3_url, local_path):
    response = requests.get(s3_url)
    response.raise_for_status()
    with open(local_path, "wb") as f:
        f.write(response.content)
    return local_path


def upload_to_s3(local_file, s3_key):
    s3.upload_file(local_file, BUCKET_NAME, s3_key)
    return f"https://{BUCKET_NAME}.s3.{s3.meta.region_name}.amazonaws.com/{s3_key}"


def remove_bg(input_path, output_path):
    with Image.open(input_path) as img:
        img_no_bg = remove(img)
        img_no_bg.save(output_path)
    return output_path


def process_task(task):
    global shape_pipeline, paint_pipeline
    print(f"[GPU] Starting job_id={task['job_id']}")

    # Load pipelines lazily
    if shape_pipeline is None:
        print("[GPU] Loading shape pipeline...")
        shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
    if paint_pipeline is None:
        print("[GPU] Loading paint pipeline...")
        paint_pipeline = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2")
    print("[GPU] Pipelines loaded successfully!")

    # 1. Download + remove bg
    local_input = "input.jpg"
    download_from_s3_url(task["image_s3_url"], local_input)
    local_input_nobg = "input_nobg.png"
    remove_bg(local_input, local_input_nobg)
    print("[GPU] Background removed")

    # 2. Generate 3D shape (untextured)
    shape_file = "mesh_output.glb"
    print("[GPU] Generating 3D mesh...")
    results = shape_pipeline(local_input_nobg)
    mesh = results[0] if isinstance(results, (list, tuple)) else results
    mesh.export(shape_file)
    print(f"[GPU] Untextured mesh saved: {shape_file}")

    # 3. Paint texture
    print("[GPU] Painting texture...")
    textured_mesh = paint_pipeline(mesh, image=local_input_nobg)
    textured_file = "painted_output.glb"
    textured_mesh.export(textured_file)
    print(f"[GPU] Textured mesh saved: {textured_file}")

    # 4. Upload both to S3
    s3_key_untextured = f"3d_assets/{task['job_id']}_mesh.glb"
    s3_key_textured = f"3d_assets/{task['job_id']}_painted.glb"

    url_untextured = upload_to_s3(shape_file, s3_key_untextured)
    url_textured = upload_to_s3(textured_file, s3_key_textured)

    print(f"[GPU] Uploaded untextured: {url_untextured}")
    print(f"[GPU] Uploaded textured: {url_textured}")

    return {"mesh_url": url_untextured, "painted_url": url_textured}


def main():
    print(f"[GPU] Worker started. Listening on queue: {MODEL_QUEUE}...")
    while True:
        task_data = r.brpop(MODEL_QUEUE, timeout=30)
        if not task_data:
            print("[GPU] No new tasks. Exiting...")
            break

        _, raw_task = task_data
        task = json.loads(raw_task)

        try:
            print(f"[MongoDB] job_id={task['job_id']} → status='processing'")
            output_urls = process_task(task)
            print(f"[MongoDB] job_id={task['job_id']} → status='completed', outputs={output_urls}")
        except Exception as e:
            print(f"[GPU] Error in job_id={task['job_id']}: {e}")
            print(f"[MongoDB] job_id={task['job_id']} → status='failed'")


if __name__ == "__main__":
    main()

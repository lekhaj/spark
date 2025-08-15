#gpu worker
import os
import sys
import time
import redis
import json
from datetime import datetime


sys.path.insert(0, '/opt/Hunyuan3D/hy3dshape')
sys.path.insert(0, '/opt/Hunyuan3D/hy3dpaint')

# shape and paint pipeline
from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained('tencent/Hunyuan3D-2.1')
paint_pipeline = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=6, resolution=512))

# Redis connection
r = redis.Redis(host=os.getenv('REDIS_HOST'), port=6379)

def process_task(task_data):
    try:
        task = json.loads(task_data)
        print(f"Processing task {task['id']}")
        
        # generating 3D asset
        mesh = shape_pipeline(image=task['image_path'])[0]
        textured_mesh = paint_pipeline(mesh, image_path=task['image_path'])

        
        print(f"Completed task {task['id']}")
        return True
    except Exception as e:
        print(f"Error processing task: {str(e)}")
        return False

def main():
    print("Worker started ")
    while True:
        try:
            _, task_data = r.blpop('3d_generation_queue', timeout=30)
            if task_data:
                process_task(task_data)
        except Exception as e:
            print(f"Redis error: {str(e)}")
            time.sleep(5)

if __name__ == "__main__":
    main()
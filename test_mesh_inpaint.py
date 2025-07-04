#!/usr/bin/env python3
"""
Test the mesh_inpaint_processor module
"""

import os
import sys

# Add the DifferentiableRenderer to the path
script_dir = os.path.dirname(os.path.abspath(__file__))
mesh_inpaint_dir = os.path.join(script_dir, "Hunyuan3D-2.1", "hy3dpaint", "DifferentiableRenderer")
sys.path.insert(0, mesh_inpaint_dir)

print(f"Looking for mesh_inpaint_processor in: {mesh_inpaint_dir}")

try:
    import numpy as np
    import cv2
    
    # Try to import the mesh_inpaint_processor
    try:
        from mesh_inpaint_processor import meshVerticeInpaint
        print("✅ Successfully imported meshVerticeInpaint")
    except ImportError as e:
        print(f"❌ Failed to import meshVerticeInpaint: {e}")
        
        # Check if files exist
        print("\nChecking if files exist:")
        files_to_check = [
            "mesh_inpaint_processor.py",
            "mesh_inpaint_fallback.py",
            "mesh_inpaint_processor.so",
        ]
        
        for filename in files_to_check:
            path = os.path.join(mesh_inpaint_dir, filename)
            if os.path.exists(path):
                print(f"  ✅ {filename} exists")
            else:
                print(f"  ❌ {filename} does not exist")
                
        # Try the fallback directly
        print("\nTrying fallback directly:")
        try:
            from mesh_inpaint_fallback import meshVerticeInpaint
            print("✅ Successfully imported fallback meshVerticeInpaint")
        except ImportError as e:
            print(f"❌ Failed to import fallback: {e}")
        
        sys.exit(1)
    
    # Test the function with a simple example
    print("\nTesting meshVerticeInpaint function...")
    # Create a simple test texture and mask
    texture = np.ones((64, 64, 3), dtype=np.float32)  # White texture
    mask = np.ones((64, 64), dtype=np.uint8) * 255     # Valid everywhere
    
    # Make a hole in the mask
    mask[20:40, 20:40] = 0  # Hole in the center
    
    # Test the function
    result_texture, result_mask = meshVerticeInpaint(texture, mask)
    
    print(f"✅ Function executed successfully")
    print(f"  Input texture shape: {texture.shape}, dtype: {texture.dtype}")
    print(f"  Input mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Output texture shape: {result_texture.shape}, dtype: {result_texture.dtype}")
    print(f"  Output mask shape: {result_mask.shape}, dtype: {result_mask.dtype}")
    
    print("\n✅ Mesh inpaint processor is working!")
    
except Exception as e:
    print(f"❌ Error testing mesh_inpaint_processor: {e}")
    sys.exit(1)

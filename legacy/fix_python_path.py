#!/usr/bin/env python3
"""
Fix Python path for Hunyuan3D imports.
This script adds the Hunyuan3D directory to the Python path.
Updated for Hunyuan3D-2.1 compatibility.
"""

import os
import sys

# Find the project root
project_root = os.path.dirname(os.path.abspath(__file__))

# Potential Hunyuan3D directories (prioritize 2.1)
hunyuan_dirs = [
    os.path.join(project_root, "Hunyuan3D-2.1"),
    os.path.join(project_root, "Hunyuan3D-2"),
    os.path.join(project_root, "Hunyuan3D-1"),
    os.path.join(project_root, "models", "Hunyuan3D-2.1"),
    os.path.join(project_root, "models", "Hunyuan3D-1"),
    os.path.join(project_root, "models", "Hunyuan3D-2")
]

# Add the first existing Hunyuan3D directory to Python path
for hunyuan_dir in hunyuan_dirs:
    if os.path.exists(hunyuan_dir):
        if hunyuan_dir not in sys.path:
            sys.path.insert(0, hunyuan_dir)
            print(f"Added {hunyuan_dir} to Python path")
        
        # Check for hy3dshape subdirectory (2.1)
        hy3dshape_dir = os.path.join(hunyuan_dir, "hy3dshape")
        if os.path.exists(hy3dshape_dir):
            print(f"Found hy3dshape module at {hy3dshape_dir}")
            # Add specific paths for 2.1
            if hy3dshape_dir not in sys.path:
                sys.path.insert(0, hy3dshape_dir)
        
        # Check for hy3dpaint subdirectory (2.1)
        hy3dpaint_dir = os.path.join(hunyuan_dir, "hy3dpaint")
        if os.path.exists(hy3dpaint_dir):
            print(f"Found hy3dpaint module at {hy3dpaint_dir}")
            if hy3dpaint_dir not in sys.path:
                sys.path.insert(0, hy3dpaint_dir)
        
        # Also check for hy3dgen subdirectory (legacy)
        hy3dgen_dir = os.path.join(hunyuan_dir, "hy3dgen")
        if os.path.exists(hy3dgen_dir):
            print(f"Found hy3dgen module at {hy3dgen_dir}")
        
        break
else:
    print("Warning: Hunyuan3D directory not found")
    print("Expected locations:", hunyuan_dirs)

if __name__ == "__main__":
    print("Python path configuration completed")
    print("Current Python path:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")

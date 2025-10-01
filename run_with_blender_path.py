#!/usr/bin/env python3
"""
Run the pipeline with a specific Blender path
Use this if Blender is not in your system PATH
"""

import os
import sys
from pipeline_lib import BlenderPipelineLib

def main():
    print("Blender Pipeline with Custom Path")
    print("=" * 50)
    
    # Common Blender installation paths
    blender_paths = [
        r"C:\Program Files\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe",
        r"C:\Program Files\Blender Foundation\Blender 3.5\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender 4.5\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender 4.0\blender.exe",
        r"C:\Program Files (x86)\Blender Foundation\Blender 3.6\blender.exe",
        "blender"  # System PATH
    ]
    
    # Find Blender
    blender_path = None
    for path in blender_paths:
        if path == "blender":
            # Check if blender is in PATH
            import shutil
            if shutil.which("blender"):
                blender_path = "blender"
                break
        elif os.path.exists(path):
            blender_path = path
            break
    
    if not blender_path:
        print("Blender not found! Please install Blender or specify the path manually.")
        print("\nCommon installation paths:")
        for path in blender_paths[:-1]:  # Exclude "blender" from display
            print(f"  - {path}")
        
        manual_path = input("\nEnter Blender path manually (or press Enter to exit): ").strip()
        if manual_path and os.path.exists(manual_path):
            blender_path = manual_path
        else:
            print("Invalid path or cancelled.")
            return False
    
    print(f"Using Blender: {blender_path}")
    
    # Get input file
    if len(sys.argv) > 1:
        input_glb = sys.argv[1]
    else:
        input_glb = input("Enter path to your GLB file: ").strip()
    
    if not os.path.exists(input_glb):
        print(f"Error: Input file not found: {input_glb}")
        return False
    
    # Generate output filename
    from pathlib import Path
    input_path = Path(input_glb)
    output_obj = input_path.with_suffix('.obj')
    
    print(f"Input file: {input_glb}")
    print(f"Output file: {output_obj}")
    print("\nStarting pipeline...")
    
    # Create pipeline with custom Blender path
    pipeline = BlenderPipelineLib(blender_path=blender_path)
    
    # Run the pipeline
    success = pipeline.convert_glb_to_obj(
        input_glb=input_glb,
        output_obj=str(output_obj),
        verbose=True
    )
    
    if success:
        print(f"\n[SUCCESS] Pipeline completed successfully!")
        print(f"Output saved to: {output_obj}")
        return True
    else:
        print(f"\n[FAILED] Pipeline failed!")
        return False

if __name__ == "__main__":
    main()

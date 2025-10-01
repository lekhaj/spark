#!/usr/bin/env python3
"""
Test script to run the Blender pipeline
This script demonstrates how to use the pipeline with your own GLB files.
"""

import os
import sys
from pathlib import Path
from pipeline_lib import BlenderPipelineLib, convert_glb_to_obj

def main():
    print("Blender Pipeline Test")
    print("=" * 50)
    
    # Check if input file is provided as command line argument
    if len(sys.argv) > 1:
        input_glb = sys.argv[1]
    else:
        # Prompt user for input file
        input_glb = input("Enter path to your GLB file: ").strip()
    
    # Validate input file
    if not os.path.exists(input_glb):
        print(f"Error: Input file not found: {input_glb}")
        print("\nPlease provide a valid GLB file path.")
        print("Example: python test_pipeline.py my_model.glb")
        return False
    
    # Generate output filename
    input_path = Path(input_glb)
    output_obj = input_path.with_suffix('.obj')
    
    print(f"Input file: {input_glb}")
    print(f"Output file: {output_obj}")
    print("\nStarting pipeline...")
    
    # Method 1: Simple one-line conversion
    print("\n--- Method 1: Simple Conversion ---")
    success = convert_glb_to_obj(
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

def test_with_sample_files():
    """Test the pipeline with sample file paths"""
    print("\n" + "=" * 50)
    print("Testing with sample file paths")
    print("=" * 50)
    
    # Common locations where GLB files might be found
    sample_paths = [
        "input.glb",
        "model.glb", 
        "test.glb",
        "./models/input.glb",
        "./input/input.glb",
        "C:/Users/shubh/Desktop/input.glb"
    ]
    
    print("Looking for GLB files in common locations...")
    
    found_files = []
    for path in sample_paths:
        if os.path.exists(path):
            found_files.append(path)
            print(f"[FOUND] {path}")
    
    if not found_files:
        print("No GLB files found in common locations.")
        print("\nPlease place your GLB file in one of these locations:")
        for path in sample_paths:
            print(f"  - {path}")
        print("\nOr run: python test_pipeline.py path/to/your/file.glb")
        return False
    
    # Test with the first found file
    test_file = found_files[0]
    print(f"\nTesting with: {test_file}")
    
    input_path = Path(test_file)
    output_obj = input_path.with_suffix('.obj')
    
    success = convert_glb_to_obj(
        input_glb=test_file,
        output_obj=str(output_obj),
        verbose=True
    )
    
    return success

if __name__ == "__main__":
    # Check if we should run sample test
    if len(sys.argv) > 1 and sys.argv[1] == "--sample":
        test_with_sample_files()
    else:
        main()

#!/usr/bin/env python3
"""
Standalone script to run the Blender pipeline without Blender GUI.
This script can be used to run the pipeline from Python directly.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_blender_pipeline(input_glb, output_obj, config_file=None, blender_path="blender"):
    """
    Run the Blender pipeline using subprocess.
    
    Args:
        input_glb (str): Path to input GLB file
        output_obj (str): Path to output OBJ file
        config_file (str, optional): Path to configuration file
        blender_path (str): Path to Blender executable
    
    Returns:
        bool: True if successful, False otherwise
    """
    
    # Validate input file
    if not os.path.exists(input_glb):
        print(f"Error: Input file not found: {input_glb}")
        return False
    
    # Create output directory
    output_path = Path(output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        blender_path,
        "--background",
        "--python", "blender_pipeline.py",
        "--",
        input_glb,
        output_obj
    ]
    
    if config_file:
        cmd.append(config_file)
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the pipeline
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        print("Pipeline completed successfully!")
        print(f"Output: {output_obj}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Pipeline failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print(f"Error: Blender not found at '{blender_path}'")
        print("Please install Blender or specify the correct path")
        return False

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Run Blender GLB to OBJ Pipeline")
    parser.add_argument("input_glb", help="Input GLB file path")
    parser.add_argument("output_obj", help="Output OBJ file path")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("-b", "--blender", default="blender", help="Blender executable path")
    
    args = parser.parse_args()
    
    success = run_blender_pipeline(
        args.input_glb,
        args.output_obj,
        args.config,
        args.blender
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

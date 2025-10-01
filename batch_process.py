#!/usr/bin/env python3
"""
Batch processing script for multiple GLB files.
Processes all GLB files in a directory through the Blender pipeline.
"""

import os
import sys
import argparse
from pathlib import Path
from run_pipeline import run_blender_pipeline

def batch_process_glb_files(input_dir, output_dir, config_file=None, blender_path="blender"):
    """
    Process all GLB files in a directory.
    
    Args:
        input_dir (str): Directory containing GLB files
        output_dir (str): Directory for output OBJ files
        config_file (str, optional): Configuration file path
        blender_path (str): Path to Blender executable
    
    Returns:
        dict: Results summary
    """
    
    input_path = Path("C:\Users\shubh\OneDrive\Desktop\input.glb")
    output_path = Path()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all GLB files
    glb_files = list(input_path.glob("*.glb")) + list(input_path.glob("*.GLB"))
    
    if not glb_files:
        print(f"No GLB files found in {input_dir}")
        return {"total": 0, "success": 0, "failed": 0}
    
    print(f"Found {len(glb_files)} GLB files to process")
    
    results = {"total": len(glb_files), "success": 0, "failed": 0}
    failed_files = []
    
    for i, glb_file in enumerate(glb_files, 1):
        print(f"\nProcessing {i}/{len(glb_files)}: {glb_file.name}")
        
        # Generate output filename
        output_obj = output_path / f"{glb_file.stem}.obj"
        
        # Run pipeline
        success = run_blender_pipeline(
            str(glb_file),
            str(output_obj),
            config_file,
            blender_path
        )
        
        if success:
            results["success"] += 1
            print(f"✓ Success: {output_obj}")
        else:
            results["failed"] += 1
            failed_files.append(glb_file.name)
            print(f"✗ Failed: {glb_file.name}")
    
    # Print summary
    print(f"\n{'='*50}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Total files: {results['total']}")
    print(f"Successful: {results['success']}")
    print(f"Failed: {results['failed']}")
    
    if failed_files:
        print(f"\nFailed files:")
        for file in failed_files:
            print(f"  - {file}")
    
    return results

def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Batch process GLB files through Blender pipeline")
    parser.add_argument("input_dir", help="Directory containing GLB files")
    parser.add_argument("output_dir", help="Directory for output OBJ files")
    parser.add_argument("-c", "--config", help="Configuration file path")
    parser.add_argument("-b", "--blender", default="blender", help="Blender executable path")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Run batch processing
    results = batch_process_glb_files(
        args.input_dir,
        args.output_dir,
        args.config,
        args.blender
    )
    
    # Exit with error code if any files failed
    sys.exit(1 if results["failed"] > 0 else 0)

if __name__ == "__main__":
    main()

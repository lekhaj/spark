#!/usr/bin/env python3
"""
Blender Pipeline Library
A simple library interface for the GLB to OBJ conversion pipeline.
Use this to integrate the pipeline into your own applications.
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import json

class BlenderPipelineLib:
    """
    Library interface for the Blender GLB to OBJ pipeline.
    
    This class provides a simple API to convert GLB files to optimized OBJ files
    without requiring manual interaction with Blender.
    """
    
    def __init__(self, blender_path: str = "blender", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the pipeline library.
        
        Args:
            blender_path (str): Path to Blender executable
            config (dict, optional): Configuration dictionary
        """
        self.blender_path = blender_path
        self.config = config or self._get_default_config()
        self.temp_dir = None
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "remesh_voxel_size": 0.001,
            "instantmesh_poly_ratio": 0.3,
            "decimate_ratio": 0.3,
            "export_format": "OBJ",
            "preserve_materials": True,
            "preserve_uvs": True,
            "use_smooth_shade": True,
            "use_python_mesh_processing": True,
            "quadify_faces": True,
            "preserve_normals": True
        }
    
    def _create_temp_config(self) -> str:
        """Create a temporary configuration file."""
        if not self.temp_dir:
            self.temp_dir = tempfile.mkdtemp(prefix="blender_pipeline_")
        
        config_path = os.path.join(self.temp_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return config_path
    
    def _cleanup_temp_files(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def convert_glb_to_obj(self, input_glb: str, output_obj: str, 
                          verbose: bool = False) -> bool:
        """
        Convert a GLB file to an optimized OBJ file.
        
        Args:
            input_glb (str): Path to input GLB file
            output_obj (str): Path to output OBJ file
            verbose (bool): Enable verbose output
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            # Validate input file
            if not os.path.exists(input_glb):
                print(f"Error: Input file not found: {input_glb}")
                return False
            
            # Create output directory
            output_path = Path(output_obj)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create temporary configuration file
            config_file = self._create_temp_config()
            
            # Build Blender command
            cmd = [
                self.blender_path,
                "--background",
                "--python", "blender_pipeline.py",
                "--",
                input_glb,
                output_obj,
                config_file
            ]
            
            if verbose:
                print(f"Running: {' '.join(cmd)}")
            
            # Run the pipeline
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if verbose:
                print("Pipeline completed successfully!")
                print(f"Output: {output_obj}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Pipeline failed with return code {e.returncode}")
            if verbose:
                print(f"Error output: {e.stderr}")
            return False
        except FileNotFoundError:
            print(f"Error: Blender not found at '{self.blender_path}'")
            print("Please install Blender or specify the correct path")
            return False
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
            return False
        finally:
            # Clean up temporary files
            self._cleanup_temp_files()
    
    def batch_convert(self, input_files: list, output_dir: str, 
                     verbose: bool = False) -> Dict[str, bool]:
        """
        Convert multiple GLB files to OBJ files.
        
        Args:
            input_files (list): List of input GLB file paths
            output_dir (str): Directory for output OBJ files
            verbose (bool): Enable verbose output
            
        Returns:
            dict: Dictionary mapping input files to success status
        """
        results = {}
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        for i, input_file in enumerate(input_files, 1):
            if verbose:
                print(f"Processing {i}/{len(input_files)}: {os.path.basename(input_file)}")
            
            # Generate output filename
            input_path = Path(input_file)
            output_file = os.path.join(output_dir, f"{input_path.stem}.obj")
            
            # Convert file
            success = self.convert_glb_to_obj(input_file, output_file, verbose)
            results[input_file] = success
            
            if verbose:
                status = "✓" if success else "✗"
                print(f"{status} {os.path.basename(input_file)}")
        
        return results
    
    def set_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        self.config.update(kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config.copy()

# Convenience functions for simple usage
def convert_glb_to_obj(input_glb: str, output_obj: str, 
                      blender_path: str = "blender", 
                      config: Optional[Dict[str, Any]] = None,
                      verbose: bool = False) -> bool:
    """
    Simple function to convert a single GLB file to OBJ.
    
    Args:
        input_glb (str): Path to input GLB file
        output_obj (str): Path to output OBJ file
        blender_path (str): Path to Blender executable
        config (dict, optional): Configuration dictionary
        verbose (bool): Enable verbose output
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    pipeline = BlenderPipelineLib(blender_path, config)
    return pipeline.convert_glb_to_obj(input_glb, output_obj, verbose)

def batch_convert_glb_to_obj(input_files: list, output_dir: str,
                           blender_path: str = "blender",
                           config: Optional[Dict[str, Any]] = None,
                           verbose: bool = False) -> Dict[str, bool]:
    """
    Simple function to convert multiple GLB files to OBJ files.
    
    Args:
        input_files (list): List of input GLB file paths
        output_dir (str): Directory for output OBJ files
        blender_path (str): Path to Blender executable
        config (dict, optional): Configuration dictionary
        verbose (bool): Enable verbose output
        
    Returns:
        dict: Dictionary mapping input files to success status
    """
    pipeline = BlenderPipelineLib(blender_path, config)
    return pipeline.batch_convert(input_files, output_dir, verbose)

# Example usage
if __name__ == "__main__":
    # Example 1: Simple conversion
    success = convert_glb_to_obj("input.glb", "output.obj", verbose=True)
    print(f"Conversion successful: {success}")
    
    # Example 2: Using the library class
    pipeline = BlenderPipelineLib()
    pipeline.set_config(remesh_voxel_size=0.002, decimate_ratio=0.5)
    success = pipeline.convert_glb_to_obj("input.glb", "output.obj")
    print(f"Conversion successful: {success}")
    
    # Example 3: Batch conversion
    input_files = ["model1.glb", "model2.glb", "model3.glb"]
    results = batch_convert_glb_to_obj(input_files, "./output", verbose=True)
    print(f"Batch results: {results}")

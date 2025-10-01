#!/usr/bin/env python3
"""
Example usage of the Blender Pipeline Library
Demonstrates how to use the pipeline as a library in your own applications.
"""

from pipeline_lib import BlenderPipelineLib, convert_glb_to_obj, batch_convert_glb_to_obj
import os

def example_simple_conversion():
    """Example 1: Simple one-line conversion"""
    print("=== Example 1: Simple Conversion ===")
    
    # Convert a single GLB file to OBJ
    success = convert_glb_to_obj(
        input_glb="input.glb",
        output_obj="output.obj",
        verbose=True
    )
    
    print(f"Conversion successful: {success}")
    return success

def example_custom_config():
    """Example 2: Using custom configuration"""
    print("\n=== Example 2: Custom Configuration ===")
    
    # Create pipeline with custom settings
    pipeline = BlenderPipelineLib()
    
    # Update configuration
    pipeline.set_config(
        remesh_voxel_size=0.002,  # Larger voxels for faster processing
        decimate_ratio=0.5,       # Less aggressive decimation
        instantmesh_poly_ratio=0.5  # Keep 50% of polygons
    )
    
    # Convert with custom settings
    success = pipeline.convert_glb_to_obj(
        input_glb="input.glb",
        output_obj="output_custom.obj",
        verbose=True
    )
    
    print(f"Custom conversion successful: {success}")
    return success

def example_batch_processing():
    """Example 3: Batch processing multiple files"""
    print("\n=== Example 3: Batch Processing ===")
    
    # List of input files
    input_files = [
        "model1.glb",
        "model2.glb", 
        "model3.glb"
    ]
    
    # Batch convert all files
    results = batch_convert_glb_to_obj(
        input_files=input_files,
        output_dir="./batch_output",
        verbose=True
    )
    
    # Print results
    print("\nBatch processing results:")
    for input_file, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {os.path.basename(input_file)}")
    
    return results

def example_integration():
    """Example 4: Integration into existing application"""
    print("\n=== Example 4: Application Integration ===")
    
    class My3DApp:
        def __init__(self):
            # Initialize pipeline with custom settings
            self.pipeline = BlenderPipelineLib()
            self.pipeline.set_config(
                remesh_voxel_size=0.001,
                decimate_ratio=0.3,
                preserve_materials=True
            )
        
        def process_model(self, input_path, output_path):
            """Process a 3D model through the pipeline"""
            print(f"Processing: {input_path} -> {output_path}")
            
            success = self.pipeline.convert_glb_to_obj(
                input_glb=input_path,
                output_obj=output_path,
                verbose=False  # Silent processing
            )
            
            if success:
                print("✓ Model processed successfully")
                return True
            else:
                print("✗ Model processing failed")
                return False
        
        def process_folder(self, input_folder, output_folder):
            """Process all GLB files in a folder"""
            import glob
            
            # Find all GLB files
            glb_files = glob.glob(os.path.join(input_folder, "*.glb"))
            
            if not glb_files:
                print("No GLB files found in input folder")
                return {}
            
            # Create output folder
            os.makedirs(output_folder, exist_ok=True)
            
            # Process each file
            results = {}
            for glb_file in glb_files:
                filename = os.path.basename(glb_file)
                output_file = os.path.join(output_folder, filename.replace('.glb', '.obj'))
                
                success = self.process_model(glb_file, output_file)
                results[glb_file] = success
            
            return results
    
    # Use the application
    app = My3DApp()
    
    # Process single file
    success = app.process_model("input.glb", "output.obj")
    
    # Process folder
    results = app.process_folder("./input_models", "./output_models")
    
    return results

def example_error_handling():
    """Example 5: Proper error handling"""
    print("\n=== Example 5: Error Handling ===")
    
    pipeline = BlenderPipelineLib()
    
    # Test with non-existent file
    success = pipeline.convert_glb_to_obj(
        input_glb="nonexistent.glb",
        output_obj="output.obj",
        verbose=True
    )
    
    if not success:
        print("✓ Error handling working correctly")
    
    # Test with invalid Blender path
    invalid_pipeline = BlenderPipelineLib(blender_path="invalid_blender_path")
    success = invalid_pipeline.convert_glb_to_obj(
        input_glb="input.glb",
        output_obj="output.obj",
        verbose=True
    )
    
    if not success:
        print("✓ Blender path validation working correctly")
    
    return True

def main():
    """Run all examples"""
    print("Blender Pipeline Library Examples")
    print("=" * 50)
    
    # Note: These examples assume you have GLB files to test with
    # Replace "input.glb" with actual file paths for real testing
    
    try:
        # Run examples
        example_simple_conversion()
        example_custom_config()
        example_batch_processing()
        example_integration()
        example_error_handling()
        
        print("\n" + "=" * 50)
        print("All examples completed!")
        print("\nTo use in your own code:")
        print("1. Import: from pipeline_lib import BlenderPipelineLib")
        print("2. Create: pipeline = BlenderPipelineLib()")
        print("3. Convert: pipeline.convert_glb_to_obj('input.glb', 'output.obj')")
        
    except Exception as e:
        print(f"Example failed: {str(e)}")
        print("Make sure you have GLB files to test with")

if __name__ == "__main__":
    main()

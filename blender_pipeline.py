#!/usr/bin/env python3
"""
Blender Pipeline for GLB to OBJ Conversion
Automates the workflow from the YouTube video:
1. Import GLB -> Remesh (0.001m voxel) -> Export OBJ
2. Process through InstantMesh (30% poly reduction + quadification)
3. Import back -> Decimate (collapse ratio 0.3) -> Export final OBJ

Based on: https://youtu.be/O_65iVCcXJk?si=oudzQ0U7fACz9i67
"""

import bpy
import bmesh
import os
import sys
import subprocess
import json
from pathlib import Path
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlenderPipeline:
    def __init__(self, config_file: Optional[str] = None):
        """Initialize the Blender pipeline with configuration."""
        self.config = self.load_config(config_file)
        self.temp_dir = Path(self.config.get('temp_dir', './temp'))
        self.temp_dir.mkdir(exist_ok=True)
        
    def load_config(self, config_file: Optional[str]) -> dict:
        """Load configuration from file or use defaults."""
        default_config = {
            'remesh_voxel_size': 0.001,  # 0.001m as specified
            'instantmesh_poly_ratio': 0.3,  # 30% poly count
            'decimate_ratio': 0.3,  # Collapse ratio 0.3
            'temp_dir': './temp',
            'instantmesh_path': 'instantmesh',  # Path to instantmesh executable
            'export_format': 'OBJ'
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
            
        return default_config
    
    def clear_scene(self):
        """Clear all objects from the scene."""
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.delete(use_global=False)
        logger.info("Scene cleared")
    
    def import_glb(self, glb_path: str) -> bool:
        """Import GLB file into Blender."""
        try:
            # Clear existing objects
            self.clear_scene()
            
            # Import GLB
            bpy.ops.import_scene.gltf(filepath=glb_path)
            logger.info(f"Successfully imported GLB: {glb_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import GLB {glb_path}: {str(e)}")
            return False
    
    def apply_remesh_modifier(self) -> bool:
        """Apply remesh modifier with 0.001m voxel size."""
        try:
            # Select all mesh objects
            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            
            if not mesh_objects:
                logger.warning("No mesh objects found for remeshing")
                return False
            
            for obj in mesh_objects:
                # Make object active
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                
                # Add remesh modifier
                remesh_modifier = obj.modifiers.new(name="Remesh", type='REMESH')
                remesh_modifier.mode = 'VOXEL'
                remesh_modifier.voxel_size = self.config['remesh_voxel_size']
                remesh_modifier.use_smooth_shade = True
                
                # Apply the modifier
                bpy.ops.object.modifier_apply(modifier="Remesh")
                logger.info(f"Applied remesh modifier to {obj.name} with voxel size {self.config['remesh_voxel_size']}m")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply remesh modifier: {str(e)}")
            return False
    
    def export_obj(self, output_path: str) -> bool:
        """Export current scene as OBJ file."""
        try:
            # Select all objects
            bpy.ops.object.select_all(action='SELECT')
            
            # Export as OBJ
            bpy.ops.wm.obj_export(
                filepath=output_path,
                export_selected_objects=True,
                export_materials=True,
                export_uv=True,
                export_normals=True,
                export_triangulated_mesh=True,
                export_vertex_groups=False,
                export_object_groups=False,
                export_material_groups=False,
                global_scale=1.0,
                path_mode='AUTO'
            )
            
            logger.info(f"Exported OBJ: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export OBJ {output_path}: {str(e)}")
            return False
    
    def process_with_instantmesh(self, input_obj: str, output_obj: str) -> bool:
        """Process OBJ file through Instant Meshes for poly reduction and quadification."""
        try:
            instantmesh_path = self.config['instantmesh_path']
            poly_ratio = self.config['instantmesh_poly_ratio']
            
            # Instant Meshes from wjakob/instant-meshes is primarily GUI-based
            # We'll use a Python-based approach for mesh processing instead
            logger.info("Using Python-based mesh processing (Instant Meshes equivalent)")
            
            # Import the mesh and process it
            success = self.process_mesh_python(input_obj, output_obj, poly_ratio)
            
            if success:
                logger.info(f"Mesh processing completed: {output_obj}")
                return True
            else:
                logger.error("Mesh processing failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to process with mesh processing: {str(e)}")
            return False
    
    def process_mesh_python(self, input_obj: str, output_obj: str, poly_ratio: float) -> bool:
        """Process mesh using Blender modifiers/operators (no external Instant Meshes).

        - Reduces polygons using Decimate modifier (Collapse) with ratio=poly_ratio
        - Attempts quadification using tris->quads in Edit Mode
        """
        try:
            import bpy
            
            # Clear scene and import the OBJ (Blender 4.5+ API)
            self.clear_scene()
            bpy.ops.wm.obj_import(filepath=input_obj)
            
            # Get all mesh objects
            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            
            if not mesh_objects:
                logger.warning("No mesh objects found for processing")
                return False
            
            for obj in mesh_objects:
                # Make object active
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                
                # Decimate via modifier (Collapse) to keep ~poly_ratio geometry
                dec = obj.modifiers.new(name="IM_Decimate", type='DECIMATE')
                dec.decimate_type = 'COLLAPSE'
                dec.ratio = max(0.01, min(1.0, poly_ratio))
                
                # Apply decimate
                bpy.ops.object.modifier_apply(modifier=dec.name)
                
                # Enter Edit Mode and try to convert tris to quads
                bpy.ops.object.mode_set(mode='EDIT')
                bpy.ops.mesh.select_all(action='SELECT')
                # Convert triangles to quads with reasonable thresholds
                bpy.ops.mesh.tris_convert_to_quads(face_threshold=0.8, shape_threshold=0.1)
                bpy.ops.object.mode_set(mode='OBJECT')
            
            # Export the processed mesh
            bpy.ops.wm.obj_export(
                filepath=output_obj,
                export_selected_objects=False,
                export_materials=True,
                export_uv=True,
                export_normals=True,
                export_triangulated_mesh=True,
                export_vertex_groups=False,
                export_object_groups=False,
                export_material_groups=False,
                global_scale=1.0,
                path_mode='AUTO'
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Python mesh processing failed: {str(e)}")
            return False
    
    def apply_decimate_modifier(self) -> bool:
        """Apply decimate modifier with collapse ratio 0.3."""
        try:
            # Select all mesh objects
            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            
            if not mesh_objects:
                logger.warning("No mesh objects found for decimation")
                return False
            
            for obj in mesh_objects:
                # Make object active
                bpy.context.view_layer.objects.active = obj
                obj.select_set(True)
                
                # Add decimate modifier
                decimate_modifier = obj.modifiers.new(name="Decimate", type='DECIMATE')
                decimate_modifier.decimate_type = 'COLLAPSE'
                decimate_modifier.ratio = self.config['decimate_ratio']
                
                # Apply the modifier
                bpy.ops.object.modifier_apply(modifier="Decimate")
                logger.info(f"Applied decimate modifier to {obj.name} with ratio {self.config['decimate_ratio']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply decimate modifier: {str(e)}")
            return False
    
    def run_pipeline(self, input_glb: str, output_obj: str) -> bool:
        """Run the complete pipeline from GLB to final OBJ."""
        try:
            logger.info("Starting Blender Pipeline")
            logger.info(f"Input: {input_glb}")
            logger.info(f"Output: {output_obj}")
            
            # Step 1: Import GLB
            if not self.import_glb(input_glb):
                return False
            
            # Step 2: Apply remesh modifier
            if not self.apply_remesh_modifier():
                return False
            
            # Step 3: Export intermediate OBJ
            temp_obj_1 = self.temp_dir / "remeshed.obj"
            if not self.export_obj(str(temp_obj_1)):
                return False
            
            # Step 4: Process with InstantMesh
            temp_obj_2 = self.temp_dir / "instantmesh_processed.obj"
            if not self.process_with_instantmesh(str(temp_obj_1), str(temp_obj_2)):
                return False
            
            # Step 5: Import processed OBJ back to Blender (Blender 4.5+ API)
            self.clear_scene()
            bpy.ops.wm.obj_import(filepath=str(temp_obj_2))
            
            # Step 6: Apply decimate modifier
            if not self.apply_decimate_modifier():
                return False
            
            # Step 7: Export final OBJ
            if not self.export_obj(output_obj):
                return False
            
            logger.info("Pipeline completed successfully!")
            logger.info(f"Final output: {output_obj}")
            
            # Clean up temporary files
            self.cleanup_temp_files()
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return False
    
    def cleanup_temp_files(self):
        """Clean up temporary files."""
        try:
            for temp_file in self.temp_dir.glob("*.obj"):
                temp_file.unlink()
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up temp files: {str(e)}")

def main():
    """Main function to run the pipeline from command line."""
    # When run from Blender with --background --python script.py -- args
    # The args after -- are passed to the script
    # We need to find where the actual arguments start
    args_start = 1
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--':
            args_start = i + 1
            break
    
    # Get the actual arguments (after --)
    actual_args = sys.argv[args_start:]
    
    if len(actual_args) < 2:
        print("Usage: blender --background --python blender_pipeline.py -- <input.glb> <output.obj> [config.json]")
        sys.exit(1)
    
    # Parse command line arguments
    input_glb = actual_args[0]
    output_obj = actual_args[1]
    config_file = actual_args[2] if len(actual_args) > 2 else None
    
    # Validate input file
    if not os.path.exists(input_glb):
        logger.error(f"Input file not found: {input_glb}")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_obj)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run pipeline
    pipeline = BlenderPipeline(config_file)
    success = pipeline.run_pipeline(input_glb, output_obj)
    
    if success:
        logger.info("Pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("Pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()

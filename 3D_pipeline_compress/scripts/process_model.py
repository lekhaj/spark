import bpy
import sys
import os

def process_and_save(input_path, output_dir, levels):
    """Imports a GLB, decimates it to multiple levels, and saves each version."""
    
   
    bpy.ops.wm.read_factory_settings(use_empty=True)

    
    try:
        bpy.ops.import_scene.gltf(filepath=input_path)
    except Exception as e:
        print(f"Error importing {input_path}: {e}")
        return

    
    mesh_obj = next((obj for obj in bpy.context.scene.objects if obj.type == 'MESH'), None)
    if not mesh_obj:
        print("No mesh object found in the scene.")
        return

    original_faces = len(mesh_obj.data.polygons)
    print(f"Original model face count: {original_faces}")
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    asset_id, asset_name = base_name.split('_', 1)

    
    for level_name, target_faces in levels.items():
        
        bpy.ops.object.select_all(action='DESELECT')
        mesh_obj.select_set(True)
        bpy.context.view_layer.objects.active = mesh_obj
        bpy.ops.object.duplicate()
        decimated_obj = bpy.context.active_object
        
        
        ratio = target_faces / original_faces if original_faces > target_faces else 1.0
        print(f"Processing for {level_name} ({target_faces} faces). Ratio: {ratio:.4f}")
        
        
        mod = decimated_obj.modifiers.new(name='Decimate', type='DECIMATE')
        mod.ratio = ratio
        bpy.ops.object.modifier_apply(modifier=mod.name)

        
        output_filename = f"{asset_id}_{asset_name}_{level_name}.glb"
        output_path = os.path.join(output_dir, output_filename)
        
        bpy.ops.object.select_all(action='DESELECT')
        decimated_obj.select_set(True)
        bpy.ops.export_scene.gltf(
            filepath=output_path,
            export_format='GLB',
            use_selection=True
        )
        print(f"Saved: {output_filename}")
        
        
        bpy.ops.object.delete()

if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:]
    input_file = args[0]
    output_directory = args[1]
    
    
    compression_levels = {args[i]: int(args[i+1]) for i in range(2, len(args), 2)}
    
    process_and_save(input_file, output_directory, compression_levels)
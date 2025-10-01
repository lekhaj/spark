# Blender GLB to OBJ Pipeline

This Python pipeline automates the 3D model processing workflow described in the [YouTube video](https://youtu.be/O_65iVCcXJk?si=oudzQ0U7fACz9i67), converting GLB files to optimized OBJ files through a multi-step process.

## Workflow Overview

The pipeline follows these steps:

1. **Import GLB** → Load the input 3D model
2. **Remesh Modifier** → Apply voxel remeshing with 0.001m resolution
3. **Export Intermediate OBJ** → Save for InstantMesh processing
4. **InstantMesh Processing** → Reduce polygon count to 30% and quadify
5. **Re-import to Blender** → Load the processed model back
6. **Decimate Modifier** → Apply collapse decimation with 0.3 ratio
7. **Export Final OBJ** → Save the optimized result

## Prerequisites

### Required Software
- **Blender** (3.0+ recommended)
- **Python** (comes with Blender)

### Instant Meshes Integration
The pipeline uses Python-based mesh processing to replicate Instant Meshes functionality:
- **Polygon reduction** to 30% of original count
- **Quadification** of triangular faces where possible
- **Field-aligned mesh generation** using Blender's built-in tools

For the original Instant Meshes GUI tool, visit: [wjakob/instant-meshes](https://github.com/wjakob/instant-meshes)

## Installation

1. Clone or download this repository
2. Ensure Blender and InstantMesh are installed
3. Update the configuration file if needed

## Usage

### Library Usage (Recommended)

Use the pipeline as a library in your own applications:

```python
from pipeline_lib import BlenderPipelineLib, convert_glb_to_obj

# Simple one-line conversion
success = convert_glb_to_obj("input.glb", "output.obj")

# Using the library class with custom settings
pipeline = BlenderPipelineLib()
pipeline.set_config(remesh_voxel_size=0.002, decimate_ratio=0.5)
success = pipeline.convert_glb_to_obj("input.glb", "output.obj")

# Batch processing
from pipeline_lib import batch_convert_glb_to_obj
results = batch_convert_glb_to_obj(
    input_files=["model1.glb", "model2.glb"],
    output_dir="./output"
)
```

### Command Line Usage

```bash
# Basic usage
blender --background --python blender_pipeline.py -- input.glb output.obj

# With custom configuration
blender --background --python blender_pipeline.py -- input.glb output.obj config.json
```

### Python Script Usage

```python
from blender_pipeline import BlenderPipeline

# Initialize pipeline
pipeline = BlenderPipeline('config.json')

# Run the pipeline
success = pipeline.run_pipeline('input.glb', 'output.obj')

if success:
    print("Pipeline completed successfully!")
else:
    print("Pipeline failed!")
```

## Configuration

Edit `config.json` to customize the pipeline parameters:

```json
{
  "remesh_voxel_size": 0.001,        // Voxel size for remeshing (meters)
  "instantmesh_poly_ratio": 0.3,     // Target polygon reduction ratio (30%)
  "decimate_ratio": 0.3,             // Decimate collapse ratio
  "temp_dir": "./temp",              // Temporary files directory
  "instantmesh_path": "instantmesh", // Path to InstantMesh executable
  "export_format": "OBJ",            // Output format
  "log_level": "INFO",               // Logging level
  "preserve_materials": true,        // Keep material information
  "preserve_uvs": true,              // Keep UV coordinates
  "use_smooth_shade": true           // Use smooth shading
}
```

## Pipeline Steps Explained

### 1. Remesh Modifier (0.001m voxel)
- Converts the mesh to a uniform voxel-based structure
- Ensures consistent topology for better processing
- Uses 0.001m voxel size for high detail preservation

### 2. InstantMesh Processing (30% poly reduction)
- Reduces polygon count to 30% of original
- Converts triangles to quads where possible
- Maintains mesh quality while reducing complexity

### 3. Decimate Modifier (0.3 collapse ratio)
- Further reduces polygon count using edge collapse
- Preserves important mesh features
- Final optimization step

## Output

The pipeline produces:
- **Final OBJ file** with optimized geometry
- **Log file** with processing details
- **Temporary files** (automatically cleaned up)

## Known Limitations

- **Texture information is lost** during the process (as mentioned in the original workflow)
- Materials and UV coordinates may be affected
- Processing time depends on model complexity

## Troubleshooting

### Common Issues

1. **InstantMesh not found**
   - Ensure InstantMesh is installed and in PATH
   - Update `instantmesh_path` in config.json

2. **Memory issues with large models**
   - Reduce `remesh_voxel_size` for smaller voxels
   - Process models in smaller batches

3. **Import/Export errors**
   - Check file permissions
   - Ensure output directory exists
   - Verify GLB file is not corrupted

### Logging

The pipeline provides detailed logging. Check the console output for:
- Processing steps
- Error messages
- Performance metrics

## Performance Tips

- Use SSD storage for temporary files
- Close other applications to free up memory
- Process multiple files in batch for efficiency

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source. Please check the original InstantMesh license for their components.

## References

- [Original YouTube Tutorial](https://youtu.be/O_65iVCcXJk?si=oudzQ0U7fACz9i67)
- [InstantMesh GitHub Repository](https://github.com/TencentARC/InstantMesh)
- [Blender Python API Documentation](https://docs.blender.org/api/current/)

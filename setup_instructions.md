# How to Run the Blender Pipeline

## Quick Start

### 1. Place Your GLB File
Put your GLB file in the same directory as the pipeline scripts, or provide the full path.

**Example file locations:**
```
C:\Users\shubh\OneDrive\Desktop\blender pipeline\
├── your_model.glb          ← Put your GLB file here
├── blender_pipeline.py
├── pipeline_lib.py
└── test_pipeline.py
```

### 2. Run the Pipeline

**Option A: Using the test script (Recommended)**
```bash
python test_pipeline.py your_model.glb
```

**Option B: Using the library directly**
```python
from pipeline_lib import convert_glb_to_obj
success = convert_glb_to_obj("your_model.glb", "output.obj")
```

**Option C: Command line with Blender**
```bash
blender --background --python blender_pipeline.py -- your_model.glb output.obj
```

## Step-by-Step Instructions

### Step 1: Prepare Your Input
1. Make sure you have a GLB file to convert
2. Note the full path to your GLB file
3. Example: `C:\Users\shubh\OneDrive\Desktop\my_model.glb`

### Step 2: Run the Test Script
```bash
# Navigate to the pipeline directory
cd "C:\Users\shubh\OneDrive\Desktop\blender pipeline"

# Run the test script
python test_pipeline.py "C:\Users\shubh\OneDrive\Desktop\my_model.glb"
```

### Step 3: Check the Output
- The pipeline will create an OBJ file in the same location as your input
- Example: `my_model.glb` → `my_model.obj`
- Check the console output for success/failure messages

## Input File Requirements

### Supported Formats
- **Input**: GLB files (.glb)
- **Output**: OBJ files (.obj)

### File Size Considerations
- Large files (>100MB) may take longer to process
- Ensure you have enough disk space for temporary files
- The pipeline creates temporary files during processing

## Troubleshooting

### Common Issues

**1. "Blender not found" error**
```bash
# Solution: Install Blender and add to PATH, or specify path
python test_pipeline.py --blender "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" your_model.glb
```

**2. "Input file not found" error**
- Check the file path is correct
- Make sure the file exists
- Use quotes around paths with spaces

**3. Permission errors**
- Run as administrator if needed
- Check file permissions
- Ensure output directory is writable

### Example Commands

**Basic usage:**
```bash
python test_pipeline.py model.glb
```

**With custom Blender path:**
```bash
python test_pipeline.py --blender "C:\Program Files\Blender Foundation\Blender 4.0\blender.exe" model.glb
```

**Batch processing:**
```python
from pipeline_lib import batch_convert_glb_to_obj
results = batch_convert_glb_to_obj(
    input_files=["model1.glb", "model2.glb"],
    output_dir="./output"
)
```

## Expected Output

When successful, you'll see:
```
Blender Pipeline Test
==================================================
Input file: your_model.glb
Output file: your_model.obj

Starting pipeline...
Running: blender --background --python blender_pipeline.py -- your_model.glb your_model.obj config.json
Pipeline completed successfully!
Output: your_model.obj

✓ Pipeline completed successfully!
Output saved to: your_model.obj
```

## File Structure After Running

```
C:\Users\shubh\OneDrive\Desktop\blender pipeline\
├── your_model.glb          ← Your input file
├── your_model.obj          ← Generated output file
├── temp\                   ← Temporary files (auto-cleaned)
├── blender_pipeline.py
├── pipeline_lib.py
└── test_pipeline.py
```

## Next Steps

1. **Test with your GLB file**: Place it in the pipeline directory and run the test script
2. **Check the output**: Verify the OBJ file was created successfully
3. **Customize settings**: Edit `config.json` if you need different parameters
4. **Integrate into your workflow**: Use the library functions in your own code

## Need Help?

If you encounter issues:
1. Check the console output for error messages
2. Verify Blender is installed and accessible
3. Ensure your GLB file is not corrupted
4. Try with a smaller test file first

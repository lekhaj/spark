# Merged 2D-to-3D Gradio Application

This document provides detailed information about the merged Gradio application that combines 2D image generation with 3D model creation.

## Overview

The `merged_gradio_app.py` combines the functionality of two separate applications:

1. **2D Image Generation**: Creates images from text prompts or terrain grid data using various AI models.
2. **3D Model Generation**: Converts 2D images into 3D models with optional textures using Hunyuan3D-2.

## System Requirements

### Basic Requirements (2D-only mode)
- Python 3.8+
- Dependencies in `requirements.txt`
- API keys for OpenAI and/or Stability AI

### Full Requirements (2D + 3D)
- All basic requirements
- GPU recommended for 3D generation (CPU mode is available but slow)
- Dependencies in `Hunyuan3D-2/requirements.txt`
- System OpenGL libraries:
  - Ubuntu/Debian: `libgl1-mesa-glx` and `xvfb`
  - CentOS/RHEL: `mesa-libGL`

## Running the Application

### Basic Usage

```bash
python merged_gradio_app.py
```

### Command-line Options

```
--model_path           Path to Hunyuan3D-2 model [default: tencent/Hunyuan3D-2mini]
--subfolder            Specific model subfolder [default: hunyuan3d-dit-v2-mini-turbo]
--texgen_model_path    Path to texture generation model [default: tencent/Hunyuan3D-2]
--port                 Port number [default: 8080]
--host                 Host IP address [default: 0.0.0.0]
--device               Device to run on (cuda, cpu) [default: cuda]
--share                Create a public shareable link
--disable_3d           Disable 3D generation functionality
```

### Examples

```bash
# Run with CPU only
python merged_gradio_app.py --device cpu

# Run on a specific port with a public link
python merged_gradio_app.py --port 7860 --share

# Run with 3D features disabled
python merged_gradio_app.py --disable_3d

# Use a different model
python merged_gradio_app.py --model_path tencent/Hunyuan3D-2 --subfolder hunyuan3d-dit-v2-0
```

## Using the Application

### 2D Image Generation

#### Text to Image
1. Navigate to the "Text to Image" tab
2. Enter your text prompt in the input box
3. Adjust image settings as needed
4. Select the model to use (openai, stability, local)
5. Click "Generate Image from Text"
6. View the generated image in the output panel
7. Click "Convert to 3D" to send this image to the 3D generation tab

#### Grid to Image
1. Navigate to the "Grid to Image" tab
2. Enter a grid of numbers in the input box, or use "Load Sample Grid"
3. Adjust settings as needed
4. Click "Generate Image from Grid"
5. View the generated terrain and grid visualization
6. Click "Convert to 3D" to send this image to the 3D generation tab

#### File Upload
1. Navigate to the "File Upload" tab
2. Upload a file containing text or grid data
3. The system will automatically detect the content type
4. Click "Process File" to generate an image
5. View the output and click "Convert to 3D" if desired

### 3D Model Generation

1. Navigate to the "3D Generation" tab
   - Either directly, or by clicking "Convert to 3D" after generating a 2D image
2. Adjust 3D generation parameters:
   - Steps: Controls detail level and quality
   - Guidance Scale: Controls adherence to the input image
   - Seed: Controls randomness
   - Octree Resolution: Controls mesh resolution
   - Num Chunks: Controls processing division
3. Choose whether to remove background and generate texture
4. Click "Generate 3D Model"
5. Once complete, view the 3D model in the interactive viewer

### Additional Features

- The 3D model viewer allows rotation, zoom, and pan
- Download options for the generated models
- Statistics view shows details about the generation process

## Troubleshooting

### 3D Generation Issues

If you see errors related to `libGL.so.1` or similar OpenGL errors:

1. Ensure you have the required system libraries installed:
   ```bash
   sudo apt-get install libgl1-mesa-glx xvfb
   ```

2. If you still have issues, run with 3D disabled:
   ```bash
   python merged_gradio_app.py --disable_3d
   ```

### API Key Issues

If you get authentication errors:

1. Check your `.env` file has the correct API keys
2. Verify the API keys are valid and have sufficient credits

### Performance Issues

For slow performance:

1. Lower the resolution parameters
2. Use a smaller model variant
3. Enable GPU acceleration if available
4. Reduce the number of steps for 3D generation

## Extending the Application

This application is designed to be modular and extensible:

1. Additional models can be added to the text/grid processors
2. The 3D generation pipeline can be customized with different models
3. New tabs or features can be added to the Gradio interface

See the codebase documentation for more details on implementing extensions.

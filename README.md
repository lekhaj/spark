# Text to grid ,2D-to-3D Generation Pipeline

A versatile pipeline that converts text prompts and terrain grids into images, and then transforms those images into 3D models using Hunyuan3D-2.

## Project Structure

```
2d-to-3d-pipeline
├── src
│   ├── app.py                # Command line app entry point
│   ├── config.py             # Configuration settings
│   ├── gradio_app.py         # Original Gradio web interface
│   ├── merged_gradio_app.py  # NEW! Unified 2D & 3D Gradio interface
│   ├── pipeline               # Contains processing logic
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Main pipeline class
│   │   ├── text_processor.py   # Text processing logic
│   │   └── grid_processor.py   # Grid processing logic
│   ├── models                 # Contains model implementations
│   │   ├── __init__.py
│   │   ├── api_client.py      # API client for external LLMs
│   │   └── local_model.py     # Local model implementation
│   ├── terrain                # Terrain type definitions and grid parsing
│   │   ├── __init__.py
│   │   ├── terrain_types.py    # Definitions of terrain types
│   │   └── grid_parser.py
│   ├── text_to_grid                # Terrain type definitions and grid parsing
│   │   ├── __init__.py
│   │   ├── grid_generator.py    grid logic -> grid
│   │   ├── grid_placement_logic.py    # logic -> grid logic
|   |   ├── placement_suggestor.py    # llm suggestions -> logic
│   │   ├── utils.py             # utility functions
|   |   ├── llm.py             # llm calls
|   |   └── structure_registry.py # Grid parsing logic
│   └── utils                  # Utility functions
│       ├── __init__.py
│       └── image_utils.py      # Image manipulation utilities
├── Hunyuan3D-2               # 3D generation component
│   ├── examples              # Example scripts for 3D generation
│   ├── hy3dgen               # 3D generation library
│   ├── assets                # Templates and example files
│   └── ...                   # Other 3D model files and configs
├── tests                      # Unit tests for the application
├── examples                   # Example input files
├── requirements.txt           # Project dependencies
├── .env                       # Environment variables
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository with submodules:

```bash
git clone --recursive [repository_url]
```

2. Install the required dependencies:

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install 3D generation dependencies (optional but required for 3D features)
pip install -r Hunyuan3D-2/requirements.txt
pip install -e Hunyuan3D-2
```

3. Install system dependencies for 3D features (if needed):

```bash
# For Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx xvfb

# For CentOS/RHEL
sudo yum install mesa-libGL
```

4. Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
DALLE_API_KEY=your_dalle_api_key_here
```

## Usage

### Command Line Interface

#### Text-to-Image

Convert a text prompt to an image:

```bash
python src/app.py --mode text --prompt "A beautiful sunset over the mountains" --num-images 1
```

#### Grid-to-Image

Convert a grid of terrain types to an image:

```bash
python src/app.py --mode grid --grid "0 1 1 0 0
1 1 0 0 1
0 0 1 1 0
0 1 1 0 0
1 0 0 1 1" --num-images 1
```

#### File Input

Process a file containing either a text prompt or a grid:

```bash
python src/app.py --mode file --file examples/text_prompts.txt
python src/app.py --mode file --file examples/grid_samples.txt
```

#### Additional CLI Options

```
--width          Width of the generated image (default: 512)
--height         Height of the generated image (default: 512)
--num-images     Number of images to generate (default: 1)
--text-model     Model for text-to-image (openai, stability, local) (default: openai)
--grid-model     Model for grid-to-image (openai, stability, local) (default: stability)
--output-dir     Directory to save generated images
```

### Web Interface (NEW!)

The project now includes a unified Gradio web interface that combines the 2D image generation capabilities with 3D model generation using Hunyuan3D-2.

#### Starting the Web Interface

```bash
# Launch with default settings
python src/merged_gradio_app.py

# Launch with CPU-only mode (for systems without GPU)
python src/merged_gradio_app.py --device cpu

# Launch with a specific port and host
python src/merged_gradio_app.py --port 7860 --host 0.0.0.0

# Launch with a public shareable link
python src/merged_gradio_app.py --share

# Launch with 3D generation features disabled
python src/merged_gradio_app.py --disable_3d
```

#### Web Interface Features

1. **Text to Image**: Generate images from text descriptions
2. **Grid to Image**: Create terrain images from grid inputs
3. **File Upload**: Process text or grid files to create images
4. **3D Generation**: Convert generated images to 3D models
   - Requires system OpenGL libraries (see Requirements section)
   - Includes mesh generation and texturing capabilities
   - Exports in various 3D formats (GLB, OBJ, etc.)

#### System Requirements for 3D Generation

3D generation features require additional system libraries. If you're running into the `libGL.so.1` error, install:

For Ubuntu/Debian:
```bash
sudo apt-get install libgl1-mesa-glx xvfb
```

For CentOS/RHEL:
```bash
sudo yum install mesa-libGL
```

## Terrain Types

The grid processor supports the following terrain types:

- 0: Plains - Flat, grassy plains
- 1: Forest - Dense forest with tall trees
- 2: Mountain - Rugged mountains with snow-capped peaks
- 3: Water - Clear blue water
- 4: Desert - Sandy desert with dunes
- 5: Snow - Snowy landscape
- 6: Swamp - Murky swampland
- 7: Hills - Rolling hills
- 8: Urban - Bustling urban area
- 9: Ruins - Ancient ruins

## Tutorial: Using the Web Interface

### Step 1: Generate a 2D Image
1. Choose a tab based on your input type (Text, Grid, or File)
2. For text input:
   - Enter a descriptive prompt (e.g., "A majestic castle on a hilltop")
   - Select model type, image dimensions, etc.
   - Click "Generate Image from Text"
3. For grid input:
   - Enter a grid of numbers (use "Load Sample Grid" for an example)
   - Click "Generate Image from Grid"

### Step 2: Convert to 3D Model
1. After generating an image, click the "Convert to 3D" button
2. The application will switch to the 3D Generation tab with your image loaded
3. Adjust 3D generation parameters as needed:
   - Steps: Controls generation quality (higher = better but slower)
   - Guidance Scale: Controls adherence to the input image
   - Octree Resolution: Controls mesh detail
   - Seed: Controls randomness (useful for reproducibility)
4. Click "Generate 3D Model" to create the 3D representation
5. View and download the resulting 3D model

### Step 3: Export and Use
1. Download the generated 3D model files
2. Use in 3D software like Blender, Unity, or other 3D applications

## Current Features

- Text-to-image generation using OpenAI, Stability AI, or local models
- Grid-to-image terrain generation with multiple terrain types
- 3D model generation from 2D images using Hunyuan3D-2
- Texture generation for 3D models
- Web interface for easy use and visualization
- Command line interface for batch processing or automation

## Future Improvements

- Implement additional local model support for offline usage
- Add more terrain types and biome combinations
- Support for image-to-image transformations and editing
- Advanced 3D model editing capabilities
- Animation support for 3D models
- Virtual environment creation from text descriptions

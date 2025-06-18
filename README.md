# SPARK: Text-to-Grid, 2D-to-3D Generation Pipeline

A comprehensive pipeline that converts text prompts and terrain grids into images, and then transforms those images into 3D models using Hunyuan3D-2. Features include MongoDB integration, Celery task queue for scalability, and a unified web interface for seamless workflow management.

## Project Structure

```
spark/
├── src/
│   ├── app.py                     # Command line app entry point
│   ├── config.py                  # Configuration settings & environment variables
│   ├── gradio_app.py              # Original Gradio web interface
│   ├── merged_gradio_app.py       # Unified 2D & 3D Gradio interface with Celery
│   ├── tasks.py                   # Celery task definitions for async processing
│   ├── aws_manager.py             # AWS S3 integration for asset storage
│   ├── db_helper.py               # MongoDB helper functions
│   ├── hunyuan3d_worker.py        # 3D generation worker integration
│   ├── mongodb_explorer.py        # MongoDB data exploration utilities
│   ├── pipeline/                  # Processing logic modules
│   │   ├── __init__.py
│   │   ├── pipeline.py            # Main pipeline orchestration
│   │   ├── text_processor.py      # Text-to-image processing
│   │   ├── grid_processor.py      # Grid-to-image processing
│   │   └── biome_processor.py     # Biome-specific processing logic
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── api_client.py          # External API clients (OpenAI, Stability)
│   │   └── local_model.py         # Local model implementations
│   ├── terrain/                   # Terrain and biome definitions
│   │   ├── __init__.py
│   │   ├── terrain_types.py       # Terrain type definitions
│   │   ├── biome_parser.py        # Biome parsing logic
│   │   └── grid_parser.py         # Grid parsing utilities
│   ├── text_grid/                 # Text-to-grid conversion system
│   │   ├── __init__.py
│   │   ├── grid_generator.py      # Grid generation logic
│   │   ├── grid_placement_logic.py # Grid placement algorithms
│   │   ├── placement_suggestor.py  # LLM-based placement suggestions
│   │   ├── llm.py                 # LLM integration utilities
│   │   ├── structure_registry.py  # MongoDB structure registry
│   │   └── utils.py               # Text-to-grid utilities
│   └── utils/                     # General utility functions
│       ├── __init__.py
│       └── image_utils.py         # Image manipulation utilities
├── Hunyuan3D-2/                   # 3D generation component (submodule)
│   ├── examples/                  # Example scripts for 3D generation
│   ├── hy3dgen/                   # 3D generation library
│   ├── assets/                    # Templates and example files
│   ├── gradio_app.py              # Original Hunyuan3D Gradio interface
│   └── requirements.txt           # 3D-specific dependencies
├── tests/                         # Comprehensive test suite
│   ├── test_aws_manager.py        # AWS integration tests
│   ├── test_celery_tasks.py       # Celery task tests
│   ├── test_config.py             # Configuration tests
│   ├── test_3d_integration.py     # 3D pipeline tests
│   ├── test_frontend_integration.py # Frontend integration tests
│   └── ...                       # Additional test modules
├── AWS_Scripts/                   # AWS deployment and management
│   ├── ec2_bootstrap.sh           # EC2 instance setup script
│   ├── gpu_auto_sleep.py          # GPU instance management
│   ├── s3_upload.py               # S3 upload utilities
│   └── requirements.txt           # AWS-specific dependencies
├── examples/                      # Example input files and demos
│   ├── biome_example.py           # Biome generation example
│   ├── grid_samples.txt           # Sample grid inputs
│   └── text_prompts.txt           # Sample text prompts
├── generated_assets/              # Output directory for generated content
│   ├── images/                    # Generated 2D images
│   └── 3d_assets/                 # Generated 3D models
├── logs/                          # Application logs
├── output/                        # Legacy output directory
├── main.py                        # Main application entry point
├── viewer.py                      # 3D model viewer utility
├── pytest.ini                    # Pytest configuration
├── requirements.txt               # Main project dependencies
├── .env                           # Environment variables (create from template)
└── README.md                      # This documentation
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster 3D generation)
- MongoDB (for data persistence)
- Redis (for Celery task queue)

### System Dependencies

For Ubuntu/Debian:
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install libgl1-mesa-glx xvfb redis-server mongodb

# Start services
sudo systemctl start redis-server
sudo systemctl enable redis-server
sudo systemctl start mongod
sudo systemctl enable mongod
```

For CentOS/RHEL:
```bash
# Install system dependencies
sudo yum install mesa-libGL redis mongodb-server

# Start services
sudo systemctl start redis
sudo systemctl enable redis
sudo systemctl start mongod
sudo systemctl enable mongod
```

### Python Dependencies

1. Clone the repository with submodules:

```bash
git clone --recursive [repository_url]
cd spark
```

2. Create and activate a virtual environment:

```bash
python -m venv txt23d
source txt23d/bin/activate  # On Windows: txt23d\Scripts\activate
```

3. Install the required dependencies:

```bash
# Install main project dependencies
pip install -r requirements.txt

# Install 3D generation dependencies
pip install -r Hunyuan3D-2/requirements.txt
pip install -e Hunyuan3D-2

# Install Celery with Redis support
pip install celery[redis]
```

### Environment Configuration

1. Create a `.env` file in the project root:

```bash
cp .env.example .env  # If you have a template
```

2. Configure your environment variables in `.env`:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Database Configuration
MONGO_DB_NAME=World_builder
MONGO_BIOME_COLLECTION=biomes
MONGODB_URI=mongodb://localhost:27017/

# Celery Configuration
USE_CELERY=true
REDIS_BROKER_URL=redis://localhost:6379/0

# AWS Configuration (optional)
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_S3_BUCKET=your_bucket_name
AWS_REGION=us-east-1

# Application Settings
OUTPUT_DIR=generated_assets
GRADIO_SERVER_PORT=7860
GRADIO_SERVER_HOST=0.0.0.0
```

### Verify Installation

Test that all components are working:

```bash
# Test Redis connection
redis-cli ping

# Test MongoDB connection
mongo --eval "db.runCommand('ping')"

# Test the application
python main.py --help
```

## Usage

## Usage

### Quick Start

The easiest way to use SPARK is through the unified web interface:

```bash
# Start the application (includes web interface and Celery workers)
python main.py

# Or start with specific configuration
python main.py --port 7860 --host 0.0.0.0
```

This will launch:
- The Gradio web interface at `http://localhost:7860`
- Celery workers for background task processing
- All necessary services for 2D and 3D generation

### Web Interface Features

#### 1. **Text to Image Generation**
- Enter descriptive text prompts
- Generate high-quality 2D images using various AI models
- Support for multiple image dimensions and styles

#### 2. **Grid to Image Generation**
- Create terrain maps from numerical grid inputs
- Support for complex biome configurations
- MongoDB integration for storing and retrieving biome definitions

#### 3. **3D Model Generation**
- Convert 2D images to 3D models using Hunyuan3D-2
- Customizable generation parameters (steps, guidance scale, resolution)
- Multiple export formats (GLB, OBJ, PLY, STL)

#### 4. **Batch Processing**
- Process multiple prompts or grids simultaneously
- MongoDB-based task queue management
- Real-time progress tracking

#### 5. **Asset Management**
- Automatic file organization and storage
- AWS S3 integration for cloud storage (optional)
- Download and sharing capabilities

### Command Line Interface

For automated workflows and batch processing:

#### Text-to-Image

```bash
python src/app.py --mode text --prompt "A beautiful sunset over the mountains" --num-images 1
```

#### Grid-to-Image

```bash
python src/app.py --mode grid --grid "0 1 1 0 0
1 1 0 0 1
0 0 1 1 0
0 1 1 0 0
1 0 0 1 1" --num-images 1
```

#### File Processing

```bash
python src/app.py --mode file --file examples/text_prompts.txt
python src/app.py --mode file --file examples/grid_samples.txt
```

#### CLI Options

```
--width          Width of the generated image (default: 512)
--height         Height of the generated image (default: 512)
--num-images     Number of images to generate (default: 1)
--text-model     Model for text-to-image (openai, stability, local)
--grid-model     Model for grid-to-image (openai, stability, local)
--output-dir     Directory to save generated images
--device         Device for 3D generation (cuda, cpu, mps)
--disable-3d     Disable 3D generation features
--disable-celery Run without Celery task queue
```

### Production Deployment

#### Using Celery for Scalability

Start Celery workers separately for production environments:

```bash
# Start Celery worker
celery -A tasks worker --loglevel=info

# Start Celery flower for monitoring (optional)
celery -A tasks flower

# Start the web application
python src/merged_gradio_app.py --port 8080
```

#### Docker Deployment (Coming Soon)

```bash
# Build and run with Docker Compose
docker-compose up -d
```

## Troubleshooting

### Common Issues

#### Redis Connection Errors
```
AttributeError: 'NoneType' object has no attribute 'Redis'
```

**Solution:**
```bash
# Install Redis client
pip install redis celery[redis]

# Verify Redis is running
sudo systemctl status redis-server
redis-cli ping

# Restart Redis if needed
sudo systemctl restart redis-server
```

#### Port Already in Use
```
Cannot find empty port in range: 7860-7860
```

**Solution:**
```bash
# Kill process using the port
sudo lsof -ti:7860 | xargs kill -9

# Or use a different port
python main.py --port 7861
```

#### MongoDB Connection Issues

**Solution:**
```bash
# Start MongoDB
sudo systemctl start mongod

# Check MongoDB status
sudo systemctl status mongod

# Test connection
mongo --eval "db.runCommand('ping')"
```

#### 3D Generation Errors (libGL.so.1)

**Solution:**
```bash
# Ubuntu/Debian
sudo apt-get install libgl1-mesa-glx xvfb

# CentOS/RHEL  
sudo yum install mesa-libGL

# For headless servers
export DISPLAY=:99
Xvfb :99 -screen 0 1024x768x24 &
```

#### CUDA/GPU Issues

**Solution:**
```bash
# Check GPU availability
nvidia-smi

# Use CPU mode if needed
python main.py --device cpu

# Or specify GPU device
export CUDA_VISIBLE_DEVICES=0
```

### Performance Optimization

#### Memory Management
- Use `--low-vram-mode` for systems with limited GPU memory
- Enable model CPU offloading for large models
- Monitor memory usage with `nvidia-smi` or `htop`

#### Speed Optimization
- Use Turbo mode for faster 3D generation
- Reduce octree resolution for faster processing
- Enable compilation with `--compile` flag

### Logging and Debugging

Enable detailed logging:
```bash
# Set log level
export LOG_LEVEL=DEBUG

# View application logs
tail -f logs/application.log

# View Celery logs
celery -A tasks worker --loglevel=debug
```

## Architecture Overview

### System Components

#### Frontend Layer
- **Gradio Web Interface**: User-friendly interface for all features
- **FastAPI Backend**: RESTful API for programmatic access
- **Static File Serving**: Efficient delivery of generated assets

#### Processing Layer
- **Pipeline Framework**: Modular processing system for different input types
- **Model Integrations**: Support for multiple AI/ML models
- **Biome System**: Advanced terrain and biome generation logic

#### Data Layer
- **MongoDB**: Persistent storage for biomes, configurations, and metadata
- **Redis**: Task queue and caching layer
- **File System**: Local storage for generated assets
- **AWS S3**: Optional cloud storage integration

#### Worker Layer
- **Celery Workers**: Distributed task processing
- **3D Generation Pipeline**: Hunyuan3D-2 integration
- **Background Jobs**: Async processing for long-running tasks

### Data Flow

1. **Input Processing**: Text prompts or grids are validated and preprocessed
2. **Task Queuing**: Requests are queued using Celery/Redis for scalability
3. **Model Execution**: AI models generate 2D images from inputs
4. **3D Conversion**: Images are optionally converted to 3D models
5. **Asset Storage**: Generated content is stored locally and/or in cloud
6. **Result Delivery**: Users can view, download, and share results

### Supported Models

#### Text-to-Image Models
- **OpenAI DALL-E**: High-quality image generation
- **Stability AI**: Stable Diffusion models
- **HunyuanDiT**: Tencent's advanced text-to-image model
- **Local Models**: Custom or open-source implementations

#### 3D Generation Models
- **Hunyuan3D-2**: State-of-the-art image-to-3D conversion
- **Hunyuan3D-2mini**: Lightweight version for faster processing
- **Multi-view Support**: 1-4 view image processing

## Feature Highlights

### Current Features ✅

#### Core Generation Capabilities
- ✅ **Text-to-Image Generation**: Multiple AI models (OpenAI, Stability AI, HunyuanDiT)
- ✅ **Grid-to-Image Generation**: Advanced terrain and biome mapping
- ✅ **Image-to-3D Conversion**: Hunyuan3D-2 integration with multiple export formats
- ✅ **Batch Processing**: Handle multiple requests simultaneously
- ✅ **Real-time Preview**: Live preview of generation progress

#### Advanced Features
- ✅ **MongoDB Integration**: Persistent storage for biomes and configurations
- ✅ **Celery Task Queue**: Scalable async processing
- ✅ **AWS S3 Integration**: Cloud storage for generated assets
- ✅ **Multi-format Export**: GLB, OBJ, PLY, STL support for 3D models
- ✅ **Biome System**: Complex terrain generation with custom biome definitions
- ✅ **Web Interface**: Comprehensive Gradio-based UI
- ✅ **CLI Tools**: Command-line interface for automation

#### Technical Features
- ✅ **GPU Acceleration**: CUDA support for faster processing
- ✅ **Memory Optimization**: Low VRAM mode for resource-constrained systems
- ✅ **Error Handling**: Comprehensive error management and recovery
- ✅ **Logging System**: Detailed logging for debugging and monitoring
- ✅ **Configuration Management**: Environment-based configuration
- ✅ **Testing Suite**: Comprehensive test coverage

### Terrain Types Supported

The grid processor supports the following terrain types:

- **0: Plains** - Flat, grassy plains with wildflowers
- **1: Forest** - Dense forest with tall trees and undergrowth
- **2: Mountain** - Rugged mountains with snow-capped peaks
- **3: Water** - Clear blue water bodies (lakes, rivers, oceans)
- **4: Desert** - Sandy desert with dunes and sparse vegetation
- **5: Snow** - Snowy landscape with ice formations
- **6: Swamp** - Murky swampland with wetland vegetation
- **7: Hills** - Rolling hills with gentle slopes
- **8: Urban** - Bustling urban areas with buildings and infrastructure
- **9: Ruins** - Ancient ruins and archaeological sites

### Planned Features 🚧

#### Short-term (v1.1)
- 🚧 **Docker Containerization**: Full Docker support for easy deployment
- 🚧 **Enhanced Biome Editor**: Visual biome creation and editing tools
- 🚧 **Animation Support**: Basic 3D model animation capabilities
- 🚧 **Improved Error Handling**: Better user feedback and error recovery

#### Medium-term (v1.2)
- 🚧 **Real-time Collaboration**: Multi-user editing and sharing
- 🚧 **Advanced 3D Editing**: In-browser 3D model editing capabilities
- 🚧 **Template System**: Pre-defined templates for common use cases
- 🚧 **Performance Dashboard**: Real-time performance monitoring

#### Long-term (v2.0)
- 🚧 **VR/AR Integration**: Virtual and augmented reality viewing
- 🚧 **Game Engine Plugins**: Direct integration with Unity, Unreal Engine
- 🚧 **Advanced AI Models**: Integration with latest generation models
- 🚧 **Marketplace**: Community sharing of biomes and templates

## Tutorial: Complete Workflow Guide

### Tutorial 1: Text to 3D Model

#### Step 1: Generate a 2D Image from Text
1. Launch the application: `python main.py`
2. Open your browser to `http://localhost:7860`
3. Navigate to the **Text to Image** tab
4. Enter a descriptive prompt:
   ```
   "A medieval castle on a hilltop with surrounding forests and mountains"
   ```
5. Configure settings:
   - **Model**: OpenAI (for high quality) or Stability AI (for speed)
   - **Dimensions**: 512x512 (recommended for 3D conversion)
   - **Images**: 1-4 (try multiple variations)
6. Click **"Generate Image from Text"**
7. Wait for processing (30-60 seconds depending on model)

#### Step 2: Convert to 3D Model
1. Select your favorite generated image
2. Click **"Convert to 3D"** button
3. The application switches to the **3D Generation** tab
4. Adjust 3D parameters:
   - **Steps**: 30 (balanced quality/speed) or 50 (high quality)
   - **Guidance Scale**: 7.5 (recommended)
   - **Octree Resolution**: 256 (standard) or 384 (high detail)
   - **Seed**: Keep default or change for variation
5. Click **"Generate 3D Model"**
6. Wait for 3D processing (2-5 minutes depending on settings)

#### Step 3: Export and Use
1. Preview the 3D model in the interactive viewer
2. Choose export format:
   - **GLB**: For web viewing and game engines
   - **OBJ**: For 3D software like Blender
   - **STL**: For 3D printing
3. Download the generated files
4. Import into your preferred 3D application

### Tutorial 2: Grid-Based Terrain Generation

#### Step 1: Create a Terrain Grid
1. Navigate to the **Grid to Image** tab
2. Design your terrain using numbers 0-9:
   ```
   2 2 1 1 0
   2 1 1 0 0
   1 1 0 0 3
   0 0 0 3 3
   0 4 4 3 3
   ```
   (Mountains → Forest → Plains → Water → Desert)
3. Or use **"Load Sample Grid"** for inspiration
4. Click **"Generate Image from Grid"**

#### Step 2: Enhance with Biome Details
1. Access the **Biome Configuration** section
2. Select biomes from the MongoDB database
3. Customize biome properties:
   - Vegetation density
   - Color schemes
   - Texture patterns
4. Apply biome settings to your grid

#### Step 3: Generate and Convert
1. Generate the terrain image
2. Convert to 3D following the same process as Tutorial 1
3. Export for use in games, simulations, or visualization

### Tutorial 3: Batch Processing

#### Using the Web Interface
1. Navigate to **Batch Processing** tab
2. Upload a file with multiple prompts:
   ```
   Ancient temple in jungle
   Futuristic city skyline
   Mountain lake at sunrise
   Desert oasis with palms
   ```
3. Configure batch settings
4. Start batch processing
5. Monitor progress in real-time
6. Download all results as a ZIP file

#### Using CLI for Automation
```bash
# Batch process text prompts
python src/app.py --mode file --file examples/text_prompts.txt --output-dir batch_output

# Batch process grids
python src/app.py --mode file --file examples/grid_samples.txt --output-dir terrain_output
```

## API Documentation

### REST API Endpoints

The application exposes RESTful API endpoints for programmatic access:

#### Text-to-Image Generation
```http
POST /api/v1/generate/text
Content-Type: application/json

{
    "prompt": "A beautiful landscape",
    "model": "openai",
    "width": 512,
    "height": 512,
    "num_images": 1
}
```

#### Grid-to-Image Generation
```http
POST /api/v1/generate/grid
Content-Type: application/json

{
    "grid": [[0,1,1],[1,2,2],[2,2,3]],
    "model": "stability",
    "width": 512,
    "height": 512
}
```

#### 3D Model Generation
```http
POST /api/v1/generate/3d
Content-Type: application/json

{
    "image_path": "/path/to/image.png",
    "steps": 30,
    "guidance_scale": 7.5,
    "octree_resolution": 256
}
```

### Python SDK Example

```python
import requests
import json

# Initialize client
base_url = "http://localhost:7860"

# Generate image from text
response = requests.post(f"{base_url}/api/v1/generate/text", 
    json={
        "prompt": "A mystical forest with glowing trees",
        "model": "openai",
        "width": 512,
        "height": 512
    }
)

result = response.json()
image_path = result["image_path"]

# Convert to 3D
response = requests.post(f"{base_url}/api/v1/generate/3d",
    json={
        "image_path": image_path,
        "steps": 30,
        "guidance_scale": 7.5
    }
)

model_result = response.json()
model_path = model_result["model_path"]
```




















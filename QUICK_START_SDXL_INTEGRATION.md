# Quick Start GPU Worker Script - SDXL Turbo Integration

## Overview

The `quick_start_gpu_worker.sh` script has been comprehensively updated to include full SDXL Turbo integration alongside Hunyuan3D-2.1. This creates a unified GPU worker that handles both 3D model generation and fast image generation.

## Key Updates

### 1. Dual Model Setup
- **Hunyuan3D-2.1**: For 3D model generation from images
- **SDXL Turbo**: For fast text-to-image generation
- Both models optimized for 15-20GB VRAM with memory management

### 2. SDXL Turbo Testing
- Created `test_sdxl_turbo_setup.py` for comprehensive SDXL testing
- Tests CUDA availability, dependencies, model loading, and image generation
- Integrated into the quick start workflow

### 3. Environment Configuration
Updated `.env.gpu` with SDXL-specific settings:
```bash
# SDXL Turbo Configuration
SDXL_MODEL_PATH=stabilityai/sdxl-turbo
SDXL_DEVICE=cuda
SDXL_ENABLE_CPU_OFFLOAD=True
SDXL_ENABLE_ATTENTION_SLICING=True
SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD=True
SDXL_MEMORY_EFFICIENT=True
DEFAULT_TEXT_MODEL=sdxl-turbo
DEFAULT_GRID_MODEL=sdxl-turbo
```

### 4. Celery Worker Configuration
- **Updated Queues**: `gpu_tasks,sdxl_tasks,image_generation`
- **Worker Hostname**: `gpu-worker-hunyuan3d-sdxl@hostname`
- **Task Routes**: Added SDXL tasks to routing configuration in `config.py`

### 5. Testing Workflow
The script now tests both models comprehensively:

1. **Environment Setup**
   - Virtual environment activation
   - Redis connection testing
   - Basic module imports

2. **Hunyuan3D-2.1 Testing**
   - Model availability check
   - GPU compatibility verification
   - 3D generation capabilities

3. **SDXL Turbo Testing**
   - CUDA and VRAM availability
   - Diffusers library compatibility
   - Model loading and memory management
   - Test image generation
   - Celery task registration

4. **Final Integration Check**
   - All modules imported successfully
   - Both workers ready for production
   - Memory optimization settings applied

### 6. Memory Optimization
Environment variables for optimal GPU usage:
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:2048,expandable_segments:True
export HUNYUAN3D_LOW_VRAM_MODE=False
export SDXL_DEVICE=cuda
export SDXL_ENABLE_CPU_OFFLOAD=True
export SDXL_ENABLE_ATTENTION_SLICING=True
export SDXL_ENABLE_SEQUENTIAL_CPU_OFFLOAD=True
```

## File Changes

### New Files
- `test_sdxl_turbo_setup.py`: Comprehensive SDXL testing script

### Updated Files
- `quick_start_gpu_worker.sh`: Full SDXL integration
- `src/config.py`: Added SDXL task routes and worker queues

## Usage

Run the updated script:
```bash
./quick_start_gpu_worker.sh
```

The script will:
1. Set up the environment
2. Test both Hunyuan3D-2.1 and SDXL Turbo
3. Start a unified Celery worker handling both 3D and image generation tasks

## Task Types Handled

The worker now handles these task queues:
- **`gpu_tasks`**: Hunyuan3D-2.1 3D model generation
- **`sdxl_tasks`**: SDXL Turbo image generation 
- **`image_generation`**: General image processing tasks

## Error Handling

Enhanced error messages and troubleshooting guidance for both models:
- CUDA driver installation
- PyTorch and Diffusers setup
- GPU memory checking
- Dependency verification

## Benefits

1. **Unified Workflow**: Single worker handles both 3D and image generation
2. **Memory Efficient**: Optimized for 15-20GB VRAM with proper offloading
3. **Fast Setup**: Automated testing and configuration
4. **Production Ready**: Comprehensive error handling and validation
5. **Local-Only**: No external API dependencies

This creates a complete local pipeline for both high-quality image generation and 3D model creation, optimized for GPU performance.

# Comprehensive Text-to-3D Pipeline

🚀 **Complete local AI pipeline for generating 3D assets from text descriptions using SDXL Turbo and HunyuanDi-3D**

## 🎯 **Pipeline Overview**

This is a complete text-to-3D generation pipeline that converts text descriptions into high-quality 3D models through a streamlined workflow:

```
📝 Text Prompt → 🖼️ SDXL Turbo Image → 🏗️ HunyuanDi-3D → 🎨 3D Model
```

### **Key Features**
- ✅ **100% Local Processing**: No API keys or cloud dependencies required
- ⚡ **Ultra-Fast Generation**: SDXL Turbo (2-4 steps) + GPU acceleration
- 🎯 **3D-Optimized**: Automatic prompt enhancement for better 3D assets
- 📊 **Scalable Architecture**: Celery + Redis for distributed processing
- 🗄️ **Database Integration**: MongoDB for prompt storage and batch processing
- ☁️ **Cloud Storage**: S3 integration for asset management
- 🌐 **Web Interface**: Intuitive Gradio interface for all operations

---

## 🏗️ **Architecture**

### **Distributed Processing**
```
Frontend (Gradio) → Redis Queue → Workers (CPU/GPU) → S3 Storage → MongoDB
```

### **Component Layout**
```
text-to-image-pipeline/
├── src/                           # Core application code
│   ├── merged_gradio_app.py       # Main web interface
│   ├── tasks.py                   # Celery task definitions
│   ├── sdxl_turbo_worker.py       # SDXL image generation worker
│   ├── hunyuan3d_worker.py        # 3D model generation worker
│   ├── config.py                  # Configuration management
│   ├── s3_manager.py              # AWS S3 integration
│   ├── db_helper.py               # MongoDB operations
│   └── ...
├── Hunyuan3D-2.1/                 # 3D generation engine
├── scripts/                       # Deployment and worker scripts
├── AWS_Scripts/                   # Cloud deployment utilities
├── docs/                          # Detailed documentation
├── tests/                         # Comprehensive test suite
└── generated_assets/              # Local output directory
```

---

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- CUDA-compatible GPU (recommended, 15-20GB VRAM for full pipeline)
- MongoDB (for data persistence)
- Redis (for task queuing)

### **Installation**

1. **Clone the repository**:
```bash
git clone --recursive [repository_url]
cd text-to-image-pipeline
```

2. **Install dependencies**:
```bash
# Create virtual environment
python -m venv txt23d
source txt23d/bin/activate  # Windows: txt23d\Scripts\activate

# Install requirements
pip install -r requirements.txt
pip install -r Hunyuan3D-2.1/requirements.txt
pip install celery[redis]
```

3. **Configure environment**:
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
MONGO_DB_NAME=your_database
REDIS_BROKER_URL=redis://localhost:6379/0
USE_CELERY=true
S3_BUCKET_NAME=your-bucket  # Optional
```

4. **Start services**:
```bash
# Start MongoDB and Redis
sudo systemctl start mongod redis-server

# Start the application
python main.py
```

**Access the interface at**: `http://localhost:7860`

---

## 🎯 **Usage**

### **1. Text-to-3D Generation**

#### **Simple Workflow**:
1. Open the web interface
2. Navigate to "Text to Image" tab
3. Enter a description: `"a wooden chair with carved details"`
4. Click "🚀 Generate Image (3D-Optimized)"
5. Go to "3D Generation" tab
6. Click "Generate 3D Model"
7. Download your GLB/OBJ file!

#### **Advanced Options**:
- **Model Selection**: SDXL Turbo (local, fast) or API models
- **3D Settings**: Texture generation, output format (GLB, OBJ, STL)
- **Batch Processing**: Process multiple prompts simultaneously

### **2. MongoDB Integration**

#### **Description-Based Workflow**:
```
MongoDB Documents → Extract "description" → Generate Image → Create 3D Model
```

1. **Load MongoDB Prompts**:
   - Navigate to "Text to Image" → "MongoDB Text Prompts"
   - Click "Fetch Prompts from Database"
   - Select from dropdown list

2. **Batch Processing**:
   - Go to "Batch Processing" section
   - Set limit (e.g., 10 prompts)
   - Choose SDXL Turbo for GPU processing
   - Monitor progress in real-time

### **3. 3D Asset Management**

#### **S3 Integration** (Optional):
- Automatic upload to cloud storage
- Organized folder structure
- Public/private access control
- CDN distribution

---

## ⚙️ **Configuration**

### **Environment Variables**

```bash
# Core Settings
USE_CELERY=true                    # Enable distributed processing
MONGO_DB_NAME=World_builder        # MongoDB database name
MONGO_BIOME_COLLECTION=biomes      # Collection for prompts

# Redis Configuration
REDIS_BROKER_URL=redis://localhost:6379/0

# GPU/CPU Workers
GPU_WORKER_HOST=localhost          # GPU instance IP
CPU_WORKER_HOST=localhost          # CPU instance IP

# S3 Storage (Optional)
S3_BUCKET_NAME=your-assets-bucket
S3_REGION=us-east-1
USE_S3_STORAGE=false

# Model Paths
SDXL_MODEL_PATH=stabilityai/sdxl-turbo
HUNYUAN3D_MODEL_PATH=tencent/Hunyuan3D-2mini
```

### **MongoDB Document Structure**

The pipeline processes documents with `description` fields:

```json
{
  "_id": "document_id",
  "description": "a red sports car with racing stripes",
  "category": "vehicles",
  "generated_image_path": "s3://bucket/images/car_123.png",
  "model_generated": true,
  "model_s3_url": "s3://bucket/3d_assets/car_123.glb"
}
```

---

## 🔧 **Technical Specifications**

### **SDXL Turbo Features**
- **Model**: `stabilityai/sdxl-turbo`
- **Speed**: 2-4 inference steps (ultra-fast)
- **Resolution**: 1024x1024 (adjustable)
- **Memory**: 6-8GB VRAM
- **Privacy**: 100% local processing

### **3D Generation Features**
- **Engine**: HunyuanDi-3D 2.1
- **Formats**: GLB, OBJ, PLY, STL
- **Texture**: Optional PBR texture generation
- **Memory**: 8-12GB VRAM additional
- **Quality**: Production-ready meshes

### **Performance Benchmarks**
| Task | Time | Hardware |
|------|------|----------|
| SDXL Image Generation | 3-5 seconds | RTX 4090 |
| 3D Model Generation | 30-60 seconds | RTX 4090 |
| Batch Processing (10 prompts) | 2-3 minutes | RTX 4090 |

---

## 🌐 **Deployment**

### **Single Machine Setup**
```bash
# Start all services locally
python main.py --host 0.0.0.0 --port 7860
```

### **Distributed Setup**

#### **GPU Worker (EC2 Spot Instance)**:
```bash
# Setup GPU instance
cd scripts/
bash setup_gpu_spot_instance.sh

# Start GPU worker
sudo systemctl start celery-gpu-worker
```

#### **CPU Worker**:
```bash
# Start CPU worker
./scripts/start_cpu_worker.sh
```

#### **Frontend Server**:
```bash
# Start web interface
python src/merged_gradio_app.py --port 8080
```

### **Task Routing**
- **SDXL Tasks** → GPU Worker (`sdxl_tasks` queue)
- **Text Processing** → CPU Worker (`cpu_tasks` queue)
- **3D Generation** → GPU Worker (`gpu_tasks` queue)

---

## 📊 **Features**

### **Image Generation**
- [x] SDXL Turbo (local, ultra-fast)
- [x] 3D-optimized prompt enhancement
- [x] Automatic background removal
- [x] Multiple output formats
- [x] Batch processing support

### **3D Model Generation**
- [x] HunyuanDi-3D integration
- [x] Texture generation
- [x] Multiple export formats
- [x] GPU acceleration
- [x] Memory optimization

### **Data Management**
- [x] MongoDB integration
- [x] S3 cloud storage
- [x] Automatic file organization
- [x] Progress tracking
- [x] Error handling

### **User Interface**
- [x] Intuitive web interface
- [x] Real-time progress updates
- [x] Gallery view for results
- [x] Download management
- [x] Mobile-responsive design

---

## 🔍 **Workflow Examples**

### **Example 1: Single Object Generation**
```python
# Input
prompt = "a ceramic vase with blue patterns"

# Enhanced for 3D
enhanced = "a ceramic vase with blue patterns, 3d render, photorealistic, clean white background, studio lighting, product photography style, sharp details"

# Output
- High-quality image (1024x1024)
- 3D GLB model with textures
- S3 URLs for sharing
```

### **Example 2: Batch Processing**
```python
# MongoDB collection with 100 furniture descriptions
# Batch process 10 at a time
# Generate 3D models for e-commerce catalog
# Automatic S3 upload and MongoDB updates
```

### **Example 3: Grid-to-3D Generation**
```python
# Input: Terrain grid
grid = [[0,1,1,0], [1,2,2,1], [0,1,1,0]]

# Output: 
- Terrain visualization
- Height-mapped 3D model
- Ready for game engine import
```

---

## 🛠️ **Troubleshooting**

### **Common Issues**

#### **Redis Connection Errors**
```bash
# Check Redis status
sudo systemctl status redis-server
redis-cli ping

# Restart if needed
sudo systemctl restart redis-server
```

#### **GPU Memory Issues**
```bash
# Monitor GPU usage
nvidia-smi

# Use CPU fallback
python main.py --device cpu
```

#### **MongoDB Connection**
```bash
# Check MongoDB
sudo systemctl status mongod
mongo --eval "db.runCommand('ping')"
```

#### **Port Already in Use**
```bash
# Find and kill process
sudo lsof -ti:7860 | xargs kill -9

# Use different port
python main.py --port 7861
```

### **Performance Optimization**
- Use GPU workers for SDXL and 3D generation
- Enable S3 storage for better file management
- Configure Redis with persistent storage
- Monitor memory usage during batch processing

---

## 📚 **Documentation**

### **Core Documentation**
- [SDXL Implementation](SDXL_IMPLEMENTATION.md) - Local image generation
- [S3 Workflow](S3_WORKFLOW_README.md) - Cloud storage integration
- [3D Optimized Prompts](docs/3D_OPTIMIZED_PROMPTS.md) - Prompt enhancement
- [MongoDB Integration](docs/MONGODB_IMAGE_INTEGRATION.md) - Database workflows

### **Deployment Guides**
- [AWS Scripts](AWS_Scripts/README.md) - Cloud deployment
- [Worker Scripts](scripts/README.md) - Distributed setup
- [Source Code](src/Readme.md) - Development guide

### **API Reference**
- [Task Definitions](src/tasks.py) - Celery task documentation
- [Configuration](src/config.py) - Environment setup
- [S3 Manager](src/s3_manager.py) - Cloud storage API

---

## 🎉 **Benefits**

### **For Developers**
- 🔧 **Modular Architecture**: Easy to extend and customize
- 📈 **Scalable**: Distribute across multiple machines
- 🧪 **Well Tested**: Comprehensive test suite
- 📝 **Documented**: Detailed documentation and examples

### **For Users**
- 🚀 **Fast**: Ultra-fast image generation (3-5 seconds)
- 🎯 **Quality**: Production-ready 3D assets
- 💰 **Cost-Effective**: No API fees, local processing
- 🔒 **Private**: All data stays on your infrastructure

### **For Businesses**
- 📊 **Batch Processing**: Handle large volumes efficiently
- ☁️ **Cloud Ready**: S3 integration for CDN distribution
- 🗄️ **Database Driven**: MongoDB for data management
- 📈 **Monitoring**: Built-in progress tracking and logging

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Code formatting
black src/
flake8 src/
```

### **Adding New Features**
1. Create feature branch
2. Add tests for new functionality
3. Update documentation
4. Submit pull request

---

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🆘 **Support**

### **Community**
- 📧 Email: [support@yourproject.com]
- 💬 Discord: [Your Discord Server]
- 🐛 Issues: [GitHub Issues]

### **Enterprise Support**
- 🏢 Custom deployment assistance
- 🎓 Training and workshops
- 🔧 Custom feature development
- 📞 Priority support

---

**🚀 Ready to start generating amazing 3D content? Follow the Quick Start guide above!**

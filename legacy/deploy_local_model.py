#!/usr/bin/env python3
"""
Local Image Generation Model Deployment Script

This script helps deploy and configure the local image generation model
for use with the Celery worker system.
"""

import os
import sys
import subprocess
import logging
import torch
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LocalModelDeployer:
    """Handles deployment and setup of local image generation models"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.requirements_installed = False
        
    def check_system_requirements(self):
        """Check if system meets requirements for local model deployment"""
        logger.info("🔍 Checking system requirements...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version.major != 3 or python_version.minor < 8:
            logger.error(f"❌ Python 3.8+ required, found {python_version.major}.{python_version.minor}")
            return False
        logger.info(f"✅ Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"✅ CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
            
            if gpu_memory < 4.0:
                logger.warning("⚠️  GPU has less than 4GB memory, consider enabling CPU offload")
        else:
            logger.warning("⚠️  CUDA not available, will use CPU (slower)")
        
        # Check disk space
        disk_usage = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
        logger.info(f"📁 Disk space: {disk_usage.stdout.split()[10]} available")
        
        return True
    
    def install_dependencies(self):
        """Install required dependencies for local model"""
        logger.info("📦 Installing dependencies...")
        
        try:
            # Core dependencies
            dependencies = [
                "torch>=2.0.0",
                "torchvision>=0.15.0", 
                "diffusers>=0.21.0",
                "transformers>=4.25.0",
                "accelerate>=0.20.0",
                "safetensors>=0.3.0",
                "pillow>=9.0.0",
                "numpy>=1.21.0"
            ]
            
            # Optional performance dependencies
            optional_deps = [
                "xformers>=0.0.20",  # For memory efficient attention
                "bitsandbytes>=0.39.0",  # For 8-bit optimization
                "triton>=2.0.0"  # For torch.compile
            ]
            
            logger.info("Installing core dependencies...")
            for dep in dependencies:
                logger.info(f"  Installing {dep}")
                result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"❌ Failed to install {dep}: {result.stderr}")
                    return False
            
            logger.info("Installing optional performance dependencies...")
            for dep in optional_deps:
                logger.info(f"  Installing {dep} (optional)")
                result = subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"⚠️  Failed to install optional dependency {dep}")
            
            self.requirements_installed = True
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def download_model(self, model_id="runwayml/stable-diffusion-v1-5"):
        """Pre-download model to cache"""
        logger.info(f"⬇️  Pre-downloading model: {model_id}")
        
        try:
            from diffusers import StableDiffusionPipeline
            
            # Download model to cache
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            logger.info(f"✅ Model {model_id} downloaded and cached")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error downloading model: {e}")
            return False
    
    def test_local_model(self):
        """Test local model functionality"""
        logger.info("🧪 Testing local model...")
        
        try:
            # Import local model class
            sys.path.insert(0, str(self.project_root / "src"))
            from models.local_model import LocalModel
            
            # Initialize and test model
            model = LocalModel()
            
            # Load model
            if not model.load_model():
                logger.error("❌ Failed to load model")
                return False
            
            # Generate test image
            logger.info("Generating test image...")
            test_prompt = "a simple red apple on a white background"
            images = model.generate_image(
                input_data=test_prompt,
                width=256,
                height=256,
                num_images=1,
                num_inference_steps=10  # Quick test
            )
            
            if images and len(images) > 0:
                # Save test image
                test_output_dir = self.project_root / "outputs" / "images"
                test_output_dir.mkdir(parents=True, exist_ok=True)
                test_image_path = test_output_dir / "test_local_model.png"
                images[0].save(test_image_path)
                logger.info(f"✅ Test image saved to: {test_image_path}")
                
                # Cleanup
                model.cleanup()
                return True
            else:
                logger.error("❌ No images generated during test")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error testing local model: {e}")
            return False
    
    def setup_celery_config(self):
        """Setup Celery configuration for local model tasks"""
        logger.info("⚙️  Setting up Celery configuration...")
        
        try:
            config_path = self.project_root / "src" / "config.py"
            
            # Read current config
            with open(config_path, 'r') as f:
                config_content = f.read()
            
            # Add local model configuration if not present
            local_config = """
# Local Model Configuration
LOCAL_MODEL_ENABLED = True
LOCAL_MODEL_ID = "runwayml/stable-diffusion-v1-5"
LOCAL_MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LOCAL_MODEL_MEMORY_EFFICIENT = True
LOCAL_MODEL_CPU_OFFLOAD = False  # Enable for limited VRAM
LOCAL_MODEL_TORCH_COMPILE = False  # Enable for PyTorch 2.0+
"""
            
            if "LOCAL_MODEL_ENABLED" not in config_content:
                with open(config_path, 'a') as f:
                    f.write(local_config)
                logger.info("✅ Added local model configuration to config.py")
            else:
                logger.info("ℹ️  Local model configuration already exists")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error setting up Celery config: {e}")
            return False
    
    def create_startup_script(self):
        """Create startup script for local model worker"""
        logger.info("📝 Creating startup script...")
        
        startup_script = f"""#!/bin/bash
# Local Model Celery Worker Startup Script

echo "🚀 Starting local model Celery worker..."

# Set environment variables
export PYTHONPATH="{self.project_root}/src:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

# Start Celery worker with local model support
cd {self.project_root}
celery -A src.tasks worker \\
    --loglevel=info \\
    --queues=local_tasks,cpu_tasks \\
    --hostname=local-worker@%h \\
    --concurrency=1 \\
    --max-tasks-per-child=10 \\
    --prefetch-multiplier=1

echo "✅ Local model worker stopped"
"""
        
        script_path = self.project_root / "start_local_worker.sh"
        with open(script_path, 'w') as f:
            f.write(startup_script)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"✅ Startup script created: {script_path}")
        return True
    
    def deploy(self):
        """Run complete deployment process"""
        logger.info("🚀 Starting local model deployment...")
        
        steps = [
            ("Checking system requirements", self.check_system_requirements),
            ("Installing dependencies", self.install_dependencies),
            ("Downloading default model", lambda: self.download_model()),
            ("Testing local model", self.test_local_model),
            ("Setting up Celery config", self.setup_celery_config),
            ("Creating startup script", self.create_startup_script)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            if not step_func():
                logger.error(f"❌ Failed at step: {step_name}")
                return False
        
        logger.info("\n🎉 Local model deployment completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start Redis server: redis-server")
        logger.info("2. Start local worker: ./start_local_worker.sh")
        logger.info("3. Test local generation in your application")
        
        return True

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy local image generation model")
    parser.add_argument("--skip-deps", action="store_true", help="Skip dependency installation")
    parser.add_argument("--skip-download", action="store_true", help="Skip model download")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5", 
                       help="Model ID to download")
    
    args = parser.parse_args()
    
    deployer = LocalModelDeployer()
    
    if args.skip_deps:
        deployer.requirements_installed = True
    
    # Custom deployment steps
    if not deployer.check_system_requirements():
        return 1
    
    if not args.skip_deps and not deployer.install_dependencies():
        return 1
    
    if not args.skip_download and not deployer.download_model(args.model_id):
        return 1
    
    if not deployer.test_local_model():
        return 1
    
    if not deployer.setup_celery_config():
        return 1
    
    if not deployer.create_startup_script():
        return 1
    
    logger.info("\n🎉 Deployment completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())

import os
import argparse
import logging
import sys
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
from config import OUTPUT_DIR
from pipeline.text_processor import TextProcessor
from pipeline.grid_processor import GridProcessor
from pipeline.pipeline import Pipeline
from terrain.grid_parser import GridParser
from utils.image_utils import save_image, create_image_grid

def setup_arg_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(description='Text-to-Image Pipeline')
    
    # Main arguments
    parser.add_argument('--mode', type=str, choices=['text', 'grid', 'file'], default='text',
                        help='Processing mode: text, grid, or file')
    
    # Text mode arguments
    parser.add_argument('--prompt', type=str, 
                        help='Text prompt for image generation')
    
    # Grid mode arguments
    parser.add_argument('--grid', type=str, 
                        help='Grid string for terrain generation')
    
    # File mode arguments
    parser.add_argument('--file', type=str, 
                        help='File path for input')
    
    # Common arguments
    parser.add_argument('--width', type=int, default=512,
                        help='Width of the generated image')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of the generated image')
    parser.add_argument('--num-images', type=int, default=1,
                        help='Number of images to generate')
    parser.add_argument('--text-model', type=str, default='openai',
                        choices=['openai', 'stability', 'local'],
                        help='Model to use for text-to-image generation')
    parser.add_argument('--grid-model', type=str, default='stability',
                        choices=['openai', 'stability', 'local'],
                        help='Model to use for grid-to-image generation')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR,
                        help='Directory to save generated images')
    
    return parser

def process_text(pipeline, args):
    """Process text prompt and generate images"""
    if not args.prompt:
        logger.error("No prompt provided for text mode")
        return False
    
    logger.info(f"Processing text prompt: {args.prompt}")
    images = pipeline.process_text(args.prompt)
    
    if not images:
        logger.error("No images were generated")
        return False
    
    logger.info(f"Generated {len(images)} images from text prompt")
    
    # Create a grid of images if multiple were generated
    if len(images) > 1:
        grid = create_image_grid(images)
        if grid:
            save_image(grid, f"text_grid_{args.prompt[:20]}")
    
    return True

def process_grid(pipeline, args):
    """Process grid data and generate terrain images"""
    if not args.grid:
        logger.error("No grid provided for grid mode")
        return False
    
    logger.info(f"Processing grid: {args.grid}")
    images, grid_viz = pipeline.process_grid(args.grid)
    
    if not images:
        logger.error("No images were generated")
        return False
    
    logger.info(f"Generated {len(images)} images from grid")
    
    # Create a grid of images if multiple were generated
    if len(images) > 1:
        grid = create_image_grid(images)
        if grid:
            save_image(grid, "terrain_grid")
    
    return True

def process_file(pipeline, args):
    """Process a file containing text or grid data"""
    if not args.file:
        logger.error("No file provided for file mode")
        return False
    
    if not os.path.exists(args.file):
        logger.error(f"File not found: {args.file}")
        return False
    
    try:
        with open(args.file, 'r') as f:
            content = f.read().strip()
        
        # Determine if this is a text prompt or a grid
        # (Simple heuristic: if it contains numbers and whitespace primarily, it's a grid)
        is_grid = True
        non_grid_chars = [c for c in content if not (c.isdigit() or c.isspace())]
        if len(non_grid_chars) > len(content) * 0.1:  # More than 10% non-grid chars
            is_grid = False
        
        if is_grid:
            logger.info("File content detected as grid data")
            return process_grid(pipeline, argparse.Namespace(
                grid=content,
                width=args.width,
                height=args.height,
                num_images=args.num_images
            ))
        else:
            logger.info("File content detected as text prompt")
            return process_text(pipeline, argparse.Namespace(
                prompt=content,
                width=args.width,
                height=args.height,
                num_images=args.num_images
            ))
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return False

def main():
    """Main entry point for the application"""
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize processors with specified models
    text_processor = TextProcessor(model_type=args.text_model)
    grid_processor = GridProcessor(model_type=args.grid_model)
    
    # Create the pipeline
    pipeline = Pipeline(text_processor, grid_processor)
    
    # Process based on mode
    if args.mode == 'text':
        process_text(pipeline, args)
    elif args.mode == 'grid':
        process_grid(pipeline, args)
    elif args.mode == 'file':
        process_file(pipeline, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
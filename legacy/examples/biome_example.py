#!/usr/bin/env python3
# filepath: /Users/shashwat/Documents/Internships/Red Panda/Pipeline/text-to-image-pipeline/examples/biome_example.py

import os
import sys
import json
import logging

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.biome_processor import BiomeProcessor
from src.pipeline.pipeline import Pipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Example script demonstrating how to use the biome processor"""
    
    # Path to a sample biome JSON file
    biome_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../medival_city.json'))
    
    if not os.path.exists(biome_file):
        # If the file doesn't exist at the expected path, try another common location
        biome_file = os.path.abspath('/Users/shashwat/Downloads/medival_city.json')
        
        if not os.path.exists(biome_file):
            logger.error(f"Could not find the biome file. Please provide the correct path.")
            return
    
    logger.info(f"Using biome file: {biome_file}")
    
    # Initialize the biome processor (default model is specified in config.py)
    biome_processor = BiomeProcessor()
    
    # Process the biome file
    images, grid_viz = biome_processor.process_biome_file(
        biome_file,
        width=1024,
        height=1024,
        num_images=1,
        save=True,
        visualize_grid=True
    )
    
    if images:
        logger.info(f"Successfully generated {len(images)} images from the biome")
    else:
        logger.error("Failed to generate images from the biome")
    
    # Alternatively, you can process the biome data directly
    try:
        with open(biome_file, 'r') as f:
            biome_data = json.load(f)
            
        # For debugging - print the biome data
        logger.info(f"Biome data keys: {biome_data.keys()}")
        logger.info(f"Biome name: {biome_data.get('biome_name', 'Unknown')}")
        logger.info(f"Number of structures: {len(biome_data.get('possible_structures', {}).get('buildings', {}))}")
        logger.info(f"Number of grids: {len(biome_data.get('possible_grids', []))}")
        
        # Initialize a new pipeline with a new biome processor
        pipeline = Pipeline(biome_processor=BiomeProcessor())
        
        # Process the biome data
        images, grid_viz = pipeline.process_biome_data(
            biome_data,
            width=1024,
            height=1024,
            num_images=1,
            save=True,
            visualize_grid=True
        )
        
        if images:
            logger.info(f"Successfully generated {len(images)} images from the biome using the pipeline")
        else:
            logger.error("Failed to generate images from the biome using the pipeline")
            
    except Exception as e:
        logger.error(f"Error processing biome data: {str(e)}")

if __name__ == "__main__":
    main()

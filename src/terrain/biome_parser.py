import json
import numpy as np
from PIL import Image, ImageDraw
import logging
import re
import os
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiomeParser:
    """Parse and process biome data from JSON files"""
    
    def __init__(self):
        self.biome_name = None
        self.structures = {}
        self.grids = []
        self.current_grid = None
        self.grid_layout = None
        self.width = 0
        self.height = 0
    
    def parse_biome_file(self, file_path):
        """
        Parse a biome from a JSON file
        
        Args:
            file_path (str): Path to the biome JSON file
        
        Returns:
            dict: Parsed biome data
        """
        try:
            with open(file_path, 'r') as f:
                biome_data = json.load(f)
            return self.parse_biome_data(biome_data)
        except Exception as e:
            logger.error(f"Error reading biome file {file_path}: {str(e)}")
            return None
    
    def parse_biome_data(self, biome_data):
        """
        Parse biome data from a dict
        
        Args:
            biome_data (dict): Biome data containing structures and grids
        
        Returns:
            dict: Parsed biome data with additional metadata
        """
        try:
            # Clear existing data to avoid conflicts
            self.structures = {}
            self.grids = []
            
            # Extract basic biome information
            self.biome_name = biome_data.get('biome_name', 'Unknown Biome')
            logger.info(f"Parsing biome: {self.biome_name}")
            
            # Parse structures
            structures_data = biome_data.get('possible_structures', {}).get('buildings', {})
            for structure_id, structure_info in structures_data.items():
                self.structures[int(structure_id)] = {
                    'type': structure_info.get('type', 'unknown'),
                    'attributes': structure_info.get('attributes', {}),
                    'description': structure_info.get('description', '')
                }
            
            logger.info(f"Parsed {len(self.structures)} structures")
            
            # Parse grids
            self.grids = biome_data.get('possible_grids', [])
            logger.info(f"Found {len(self.grids)} possible grid layouts")
            
            # Set the first grid as the current one if available
            if self.grids:
                self.set_current_grid(0)
            
            return {
                'biome_name': self.biome_name,
                'structures': self.structures,
                'grids': self.grids
            }
            
        except Exception as e:
            logger.error(f"Error parsing biome data: {str(e)}")
            return None
    
    def set_current_grid(self, grid_index=0):
        """
        Set the current grid to a specific index
        
        Args:
            grid_index (int): Index of the grid to use
        
        Returns:
            numpy.ndarray: The selected grid layout
        """
        if not self.grids or grid_index >= len(self.grids):
            logger.error(f"Invalid grid index: {grid_index}")
            return None
        
        self.current_grid = self.grids[grid_index]
        self.grid_layout = np.array(self.current_grid.get('layout', []))
        
        if self.grid_layout.size > 0:
            self.height, self.width = self.grid_layout.shape
            logger.info(f"Set current grid to index {grid_index} with dimensions {self.height}x{self.width}")
            return self.grid_layout
        else:
            logger.error("Selected grid has no valid layout")
            return None
    
    def get_structure_description(self, structure_id):
        """
        Get the description of a structure by its ID
        
        Args:
            structure_id (int): Structure ID
        
        Returns:
            str: Structure description or 'unknown structure' if not found
        """
        structure = self.structures.get(structure_id, {})
        return structure.get('description', 'unknown structure')
    
    def get_structure_type(self, structure_id):
        """
        Get the type of a structure by its ID
        
        Args:
            structure_id (int): Structure ID
        
        Returns:
            str: Structure type or 'unknown' if not found
        """
        structure = self.structures.get(structure_id, {})
        return structure.get('type', 'unknown')
    
    def get_structure_color(self, structure_id):
        """
        Get a distinct color for visualization based on structure ID
        
        Args:
            structure_id (int): Structure ID
        
        Returns:
            tuple: RGB color tuple
        """
        # Define a color palette for structures
        colors = {
            0: (200, 200, 200),  # Empty/road (gray)
            1: (165, 42, 42),    # Mill (brown)
            2: (105, 105, 105),  # Blacksmith (dark gray)
            3: (210, 180, 140),  # Shop (tan)
            4: (255, 255, 224),  # School (light yellow)
            5: (160, 82, 45)     # Inn (sienna)
        }
        
        return colors.get(structure_id, (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    
    def get_prompt_from_biome(self):
        """
        Generate a text prompt describing the biome based on structures and layout
        
        Returns:
            str: A text prompt for image generation
        """
        if self.grid_layout is None or self.biome_name is None:
            logger.error("No biome data available")
            return None
        
        # Count occurrences of each structure type
        unique, counts = np.unique(self.grid_layout, return_counts=True)
        structure_counts = dict(zip(unique, counts))
        
        # Sort structures by occurrence (most common first, excluding empty spaces)
        sorted_structures = [(s_id, count) for s_id, count in sorted(structure_counts.items(), 
                                                                     key=lambda x: x[1], reverse=True)
                             if s_id != 0]  # Exclude empty/road spaces (ID 0)
        
        # Generate description
        structure_descriptions = []
        for structure_id, count in sorted_structures:
            percentage = count / (self.width * self.height) * 100
            if percentage > 3:  # Only include structures that make up more than 3% of the grid
                structure_type = self.get_structure_type(structure_id)
                structure_descriptions.append(f"{structure_type}")
        
        # Generate a prompt based on the biome composition
        prompt = f"A detailed 3D model of a {self.biome_name.lower()} featuring "
        
        if structure_descriptions:
            prompt += ", ".join(structure_descriptions[:-1])
            if len(structure_descriptions) > 1:
                prompt += f", and {structure_descriptions[-1]}"
            else:
                prompt += structure_descriptions[0]
        else:
            prompt += "various medieval structures"
        
        # Add additional context about the overall landscape
        prompt += ". Isometric view, detailed architecture, medieval style, realistic textures."
        
        return prompt
    
    def visualize_biome_grid(self, cell_size=30):
        """
        Create a visual representation of the biome grid
        
        Args:
            cell_size (int): Size of each cell in pixels
        
        Returns:
            PIL.Image: Colored grid visualization
        """
        if self.grid_layout is None:
            logger.error("No grid layout available")
            return None
        
        # Create a new image
        img_width = self.width * cell_size
        img_height = self.height * cell_size
        image = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Fill each cell with the corresponding structure color
        for y in range(self.height):
            for x in range(self.width):
                structure_id = self.grid_layout[y, x]
                color = self.get_structure_color(structure_id)
                
                # Draw the cell
                draw.rectangle(
                    [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size],
                    fill=color,
                    outline=(0, 0, 0)
                )
                
                # Add structure type label if not empty
                if structure_id != 0:
                    structure_type = self.get_structure_type(structure_id)
                    # Shorten the text if needed
                    label = structure_type[:1].upper()
                    # Draw the label
                    text_bbox = draw.textbbox((0, 0), label)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    text_x = x * cell_size + (cell_size - text_width) // 2
                    text_y = y * cell_size + (cell_size - text_height) // 2
                    draw.text((text_x, text_y), label, fill=(255, 255, 255))
        
        return image

import numpy as np
from PIL import Image, ImageDraw
import logging
from .terrain_types import get_terrain_name, get_terrain_description, get_terrain_color

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GridParser:
    """Parse and process grid data representing terrain types"""
    
    def __init__(self):
        self.grid = None
        self.width = 0
        self.height = 0
    
    def parse_grid_string(self, grid_string):
        """
        Parse a string representation of a grid
        
        Args:
            grid_string (str): String with space-separated numbers representing terrain types
        
        Returns:
            numpy.ndarray: 2D array of terrain type IDs
        """
        try:
            # Split the string into rows
            rows = [row.strip() for row in grid_string.strip().split('\n')]
            
            # Parse each row into numbers
            grid = []
            for row in rows:
                if not row:  # Skip empty rows
                    continue
                # Convert string numbers to integers
                grid.append([int(cell) for cell in row.split() if cell.strip()])
            
            # Validate grid
            if not grid:
                logger.error("Empty grid")
                return None
            
            # Make sure all rows have the same length
            row_lengths = [len(row) for row in grid]
            if len(set(row_lengths)) > 1:
                logger.warning(f"Inconsistent row lengths in grid: {row_lengths}")
                max_length = max(row_lengths)
                # Pad shorter rows with 0 (plain terrain)
                grid = [row + [0] * (max_length - len(row)) for row in grid]
            
            # Convert to numpy array
            self.grid = np.array(grid)
            self.height, self.width = self.grid.shape
            logger.info(f"Parsed grid with dimensions {self.height}x{self.width}")
            
            return self.grid
            
        except Exception as e:
            logger.error(f"Error parsing grid: {str(e)}")
            return None
    
    def parse_grid_file(self, file_path):
        """
        Parse a grid from a file
        
        Args:
            file_path (str): Path to the grid file
        
        Returns:
            numpy.ndarray: 2D array of terrain type IDs
        """
        try:
            with open(file_path, 'r') as f:
                grid_string = f.read()
            return self.parse_grid_string(grid_string)
        except Exception as e:
            logger.error(f"Error reading grid file {file_path}: {str(e)}")
            return None
    
    def get_prompt_from_grid(self):
        """
        Generate a text prompt describing the terrain based on the grid
        
        Returns:
            str: A text prompt for image generation
        """
        if self.grid is None:
            logger.error("No grid data available")
            return None
        
        # Count occurrences of each terrain type
        unique, counts = np.unique(self.grid, return_counts=True)
        terrain_counts = dict(zip(unique, counts))
        
        # Sort terrain types by occurrence (most common first)
        sorted_terrains = sorted(terrain_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Generate description
        terrain_descriptions = []
        for terrain_id, count in sorted_terrains:
            percentage = count / (self.width * self.height) * 100
            if percentage > 5:  # Only include terrains that make up more than 5% of the grid
                terrain_descriptions.append(f"{get_terrain_description(terrain_id)} ({percentage:.1f}%)")
        
        # Generate a prompt based on the terrain composition
        if not terrain_descriptions:
            return "A landscape with various terrain types"
        
        prompt = "A scenic landscape featuring " + ", ".join(terrain_descriptions[:-1])
        if len(terrain_descriptions) > 1:
            prompt += f", and {terrain_descriptions[-1]}"
        else:
            prompt = f"A scenic landscape featuring {terrain_descriptions[0]}"
        
        # Add additional context about the overall landscape
        prompt += ". Aerial view, detailed terrain, realistic, natural colors."
        
        return prompt
    
    def visualize_grid(self, cell_size=20):
        """
        Create a visual representation of the grid using terrain colors
        
        Args:
            cell_size (int): Size of each cell in pixels
        
        Returns:
            PIL.Image: Colored grid visualization
        """
        if self.grid is None:
            logger.error("No grid data available")
            return None
        
        # Create a new image
        img_width = self.width * cell_size
        img_height = self.height * cell_size
        image = Image.new('RGB', (img_width, img_height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Fill each cell with the corresponding terrain color
        for y in range(self.height):
            for x in range(self.width):
                terrain_id = self.grid[y, x]
                color = get_terrain_color(terrain_id)
                
                # Draw the cell
                draw.rectangle(
                    [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size],
                    fill=color
                )
        
        return image
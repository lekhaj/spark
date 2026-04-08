import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, text_processor=None, grid_processor=None, biome_processor=None):
        self.text_processor = text_processor
        self.grid_processor = grid_processor
        self.biome_processor = biome_processor

    def process_text(self, text_prompt):
        if not self.text_processor:
            logger.error("Text processor not initialized")
            return None
        return self.text_processor.convert_to_image(text_prompt)

    def process_grid(self, grid):
        if not self.grid_processor:
            logger.error("Grid processor not initialized")
            return None
        return self.grid_processor.convert_grid_to_image(grid)
        
    def process_biome_file(self, biome_file_path, **kwargs):
        if not self.biome_processor:
            logger.error("Biome processor not initialized")
            return None
        return self.biome_processor.process_biome_file(biome_file_path, **kwargs)
        
    def process_biome_data(self, biome_data, **kwargs):
        if not self.biome_processor:
            logger.error("Biome processor not initialized")
            return None
        return self.biome_processor.process_biome_data(biome_data, **kwargs)
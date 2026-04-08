import os
import logging
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
from datetime import datetime
import numpy as np
from config import OUTPUT_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_output_dir():
    """Create the output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR

def save_image(image, name_prefix="image", subfolder=None):
    """
    Save an image to the output directory
    
    Args:
        image (PIL.Image): The image to save
        name_prefix (str): Prefix for the filename
        subfolder (str): Optional subfolder in the output directory
        
    Returns:
        str: Path to the saved image
    """
    try:
        # Create the output directory
        output_path = OUTPUT_DIR
        if subfolder:
            output_path = os.path.join(output_path, subfolder)
            os.makedirs(output_path, exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name_prefix}_{timestamp}.png"
        filepath = os.path.join(output_path, filename)
        
        # Save the image
        image.save(filepath)
        logger.info(f"Saved image to {filepath}")
        
        return filepath
    except Exception as e:
        logger.error(f"Error saving image: {str(e)}")
        return None

def load_image(path):
    from PIL import Image
    return Image.open(path)

def enhance_image(image, brightness=1.0, contrast=1.0, saturation=1.0, sharpness=1.0):
    """
    Enhance an image with various adjustments
    
    Args:
        image (PIL.Image): The image to enhance
        brightness (float): Brightness factor (1.0 = original)
        contrast (float): Contrast factor (1.0 = original)
        saturation (float): Saturation factor (1.0 = original)
        sharpness (float): Sharpness factor (1.0 = original)
        
    Returns:
        PIL.Image: The enhanced image
    """
    try:
        # Apply enhancements in sequence
        if brightness != 1.0:
            image = ImageEnhance.Brightness(image).enhance(brightness)
        
        if contrast != 1.0:
            image = ImageEnhance.Contrast(image).enhance(contrast)
            
        if saturation != 1.0:
            image = ImageEnhance.Color(image).enhance(saturation)
            
        if sharpness != 1.0:
            image = ImageEnhance.Sharpness(image).enhance(sharpness)
            
        return image
    
    except Exception as e:
        logger.error(f"Error enhancing image: {str(e)}")
        return image  # Return original image if enhancement fails

def apply_filter(image, filter_type="sharpen"):
    """
    Apply a filter to an image
    
    Args:
        image (PIL.Image): The image to filter
        filter_type (str): The type of filter to apply
        
    Returns:
        PIL.Image: The filtered image
    """
    try:
        filters = {
            "blur": ImageFilter.BLUR,
            "sharpen": ImageFilter.SHARPEN,
            "contour": ImageFilter.CONTOUR,
            "detail": ImageFilter.DETAIL,
            "edge_enhance": ImageFilter.EDGE_ENHANCE,
            "emboss": ImageFilter.EMBOSS,
            "smooth": ImageFilter.SMOOTH
        }
        
        if filter_type not in filters:
            logger.warning(f"Unknown filter type: {filter_type}. Using 'sharpen' instead.")
            filter_type = "sharpen"
            
        return image.filter(filters[filter_type])
        
    except Exception as e:
        logger.error(f"Error applying filter: {str(e)}")
        return image  # Return original image if filtering fails

def resize_image(image, width, height, maintain_aspect=True):
    """
    Resize an image
    
    Args:
        image (PIL.Image): The image to resize
        width (int): Target width
        height (int): Target height
        maintain_aspect (bool): Whether to maintain the aspect ratio
        
    Returns:
        PIL.Image: The resized image
    """
    try:
        if maintain_aspect:
            image = ImageOps.contain(image, (width, height))
        else:
            image = image.resize((width, height))
            
        return image
        
    except Exception as e:
        logger.error(f"Error resizing image: {str(e)}")
        return image  # Return original image if resizing fails

def create_image_grid(images, rows=None, cols=None, spacing=10):
    """
    Create a grid of images
    
    Args:
        images (list): List of PIL.Image objects
        rows (int): Number of rows in the grid
        cols (int): Number of columns in the grid
        spacing (int): Spacing between images in pixels
        
    Returns:
        PIL.Image: A single image containing the grid
    """
    try:
        if not images:
            logger.error("No images provided for grid")
            return None
            
        # Determine grid dimensions if not specified
        num_images = len(images)
        if rows is None and cols is None:
            # Default to square grid
            cols = int(np.ceil(np.sqrt(num_images)))
            rows = int(np.ceil(num_images / cols))
        elif rows is None:
            rows = int(np.ceil(num_images / cols))
        elif cols is None:
            cols = int(np.ceil(num_images / rows))
            
        # Ensure all images are the same size
        max_width = max(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Create a blank canvas for the grid
        grid_width = cols * max_width + (cols - 1) * spacing
        grid_height = rows * max_height + (rows - 1) * spacing
        grid_img = Image.new('RGB', (grid_width, grid_height), color=(255, 255, 255))
        
        # Paste images into the grid
        for i, img in enumerate(images):
            if i >= rows * cols:
                break  # Skip extra images
                
            # Resize image to fit in the grid cell
            if img.width != max_width or img.height != max_height:
                img = resize_image(img, max_width, max_height)
                
            # Calculate position in the grid
            row = i // cols
            col = i % cols
            x = col * (max_width + spacing)
            y = row * (max_height + spacing)
            
            # Paste the image
            grid_img.paste(img, (x, y))
            
        return grid_img
        
    except Exception as e:
        logger.error(f"Error creating image grid: {str(e)}")
        return None
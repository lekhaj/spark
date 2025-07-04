"""
Patch file to create a fallback meshVerticeInpaint function
when the original C++ implementation fails to load.

This file should be imported in MeshRender.py after attempting
to import from mesh_inpaint_processor.
"""

import numpy as np
import cv2

def meshVerticeInpaint(texture_np, mask, vtx_pos=None, vtx_uv=None, pos_idx=None, uv_idx=None):
    """
    Fallback implementation of meshVerticeInpaint that uses OpenCV inpainting
    instead of mesh vertex connectivity.
    
    This implementation ignores the mesh data and just uses standard OpenCV inpainting.
    
    Args:
        texture_np: Texture as numpy array
        mask: Binary mask (255=valid, 0=needs inpainting)
        vtx_pos, vtx_uv, pos_idx, uv_idx: Ignored in this implementation
        
    Returns:
        inpainted_texture, mask
    """
    print("Using fallback meshVerticeInpaint implementation (no C++ extension)")
    
    # Make sure texture is in correct format
    if texture_np.dtype != np.uint8:
        texture_uint8 = (texture_np * 255).astype(np.uint8)
    else:
        texture_uint8 = texture_np.copy()
    
    # Create invert mask for inpainting (OpenCV wants 0=valid, 255=inpaint area)
    inpaint_mask = 255 - mask
    
    # Use OpenCV inpainting
    if np.any(inpaint_mask > 0):
        try:
            # Try different inpainting methods if available
            result = cv2.inpaint(texture_uint8, inpaint_mask, 3, cv2.INPAINT_NS)
        except Exception as e:
            print(f"Fallback inpainting failed: {e}")
            # If all fails, just return the original
            result = texture_uint8
    else:
        result = texture_uint8
        
    # Return the result in same format as input
    if texture_np.dtype != np.uint8:
        result = result.astype(np.float32) / 255.0
        
    return result, mask

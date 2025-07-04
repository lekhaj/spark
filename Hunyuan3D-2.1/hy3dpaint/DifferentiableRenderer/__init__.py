# DifferentiableRenderer package
try:
    from .mesh_inpaint_processor import meshVerticeInpaint
except ImportError:
    try:
        from .mesh_inpaint_fallback import meshVerticeInpaint
        print("Using Python fallback for mesh inpainting")
    except ImportError:
        def meshVerticeInpaint(texture, mask, *args, **kwargs):
            print("❌ No mesh inpainting implementation available")
            return texture, mask

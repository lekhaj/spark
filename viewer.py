"""
Viewer Compatibility Module
This module provides compatibility for the original viewer functionality
while integrating with the new text-to-image pipeline.
"""

import os
import sys

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_ui():
    """
    Compatibility function for the original run_ui() call.
    This now launches the merged Gradio app instead.
    """
    try:
        from merged_gradio_app import build_app
        
        print("🚀 Launching Text-to-Image Pipeline (Viewer Mode)...")
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output", exist_ok=True) 
        os.makedirs("generated_assets", exist_ok=True)
        
        # Build and launch the app
        demo = build_app()
        
        # Launch with viewer-specific settings
        demo.launch(
            server_name="0.0.0.0",  # Allow external connections
            server_port=7860,       # Default port
            share=False,            # No public tunnel
            debug=True,             # Enable debug mode
            show_error=True,        # Show detailed errors
            inbrowser=True          # Auto-open browser
        )
        
    except ImportError as e:
        print(f"❌ Error importing merged_gradio_app: {e}")
        print("Make sure src/merged_gradio_app.py exists and is properly configured.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching viewer: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_ui()

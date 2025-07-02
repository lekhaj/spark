# main.py
import os
import sys
import argparse

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def run_ui():
    """Launch the main Gradio application"""
    try:
        from merged_gradio_app import build_app
        
        # Create necessary directories
        os.makedirs("logs", exist_ok=True)
        os.makedirs("output", exist_ok=True)
        os.makedirs("generated_assets", exist_ok=True)
        
        print("🚀 Launching Text-to-Image Pipeline...")
        
        # Build the Gradio app
        demo = build_app()
        
        # Launch with configuration suitable for EC2
        demo.launch(
            server_name="0.0.0.0",  # Allow external connections for EC2
            server_port=7860,       # Default Gradio port
            share=False,            # Don't create public tunnel
            debug=True,             # Enable debug mode
            show_error=True         # Show detailed errors
        )
    except ImportError as e:
        print("❌ Error: merged_gradio_app not found. Make sure you're in the correct project directory.")
        print("Expected file: src/merged_gradio_app.py")
        print(f"Import error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error launching application: {e}")
        print("Make sure all dependencies are installed and configuration is correct.")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text-to-Image Pipeline')
    parser.add_argument('--mode', choices=['app', 'viewer'], default='app',
                       help='Launch mode: app (full pipeline) or viewer (compatibility mode)')
    
    args = parser.parse_args()
    
    if args.mode == 'viewer':
        # Use compatibility mode (original implementation style)
        print("🎯 Running in viewer compatibility mode...")
        run_ui()
    else:
        # Default mode - full pipeline
        run_ui()

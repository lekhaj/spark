# main.py
import os
from viewer import run_ui

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    print("🚀 Launching Biome Viewer...")
    run_ui()

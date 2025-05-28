# Text & Grid to Image Generator

A Gradio web application that serves as an interface to the text-to-image and grid-to-image generation pipeline.

## Features

- **Text to Image**: Generate images from text descriptions
- **Grid to Image**: Generate terrain images from grid data
- **File Upload**: Process text or grid data from uploaded files
- **Multiple Models**: Choose between OpenAI, Stability AI, or local models
- **Configuration**: Adjust image size and number of images

## Setup

1. Install the requirements:
```bash
pip install -r requirements.txt
```

2. Set up your environment variables in a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key
STABILITY_API_KEY=your_stability_api_key
DEFAULT_TEXT_MODEL=openai
DEFAULT_GRID_MODEL=stability
```

3. Run the Gradio app:
```bash
cd src
python gradio_app.py
```

4. Open the provided URL in your browser to access the web interface.

## Grid Format

For grid-based terrain generation, use numbers to represent different terrain types:
- 0: Plain
- 1: Forest
- 2: Mountain
- 3: Water
- 4: Desert

Example grid:
```
0 0 0 1 1 1 0 0 0 0
0 0 1 1 1 1 1 0 0 0
0 1 1 1 1 1 1 1 0 0
0 0 1 1 2 2 1 0 0 0
0 0 0 2 2 2 2 0 0 0
0 0 0 2 2 2 0 0 0 0
0 0 3 3 3 3 3 0 0 0
0 3 3 3 3 3 3 3 0 0
3 3 3 3 3 3 3 3 3 0
0 0 0 0 0 0 0 0 0 0
```

## Model Selection

- **OpenAI**: Uses DALL-E models from OpenAI for image generation
- **Stability**: Uses Stability AI models for image generation
- **Local**: Uses locally deployed diffusion models (requires appropriate setup)

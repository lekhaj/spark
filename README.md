# Text-to-Image Pipeline

A versatile pipeline that converts text prompts and terrain grids into images using various AI image generation models.

## Project Structure

```
text-to-image-pipeline
├── src
│   ├── app.py                # Entry point of the application
│   ├── config.py             # Configuration settings
│   ├── pipeline               # Contains processing logic
│   │   ├── __init__.py
│   │   ├── pipeline.py        # Main pipeline class
│   │   ├── text_processor.py   # Text processing logic
│   │   └── grid_processor.py   # Grid processing logic
│   ├── models                 # Contains model implementations
│   │   ├── __init__.py
│   │   ├── api_client.py      # API client for external LLMs
│   │   └── local_model.py     # Local model implementation
│   ├── terrain                # Terrain type definitions and grid parsing
│   │   ├── __init__.py
│   │   ├── terrain_types.py    # Definitions of terrain types
│   │   └── grid_parser.py      # Grid parsing logic
│   └── utils                  # Utility functions
│       ├── __init__.py
│       └── image_utils.py      # Image manipulation utilities
├── tests                      # Unit tests for the application
│   ├── __init__.py
│   ├── test_pipeline.py       # Tests for the pipeline
│   ├── test_text_processor.py  # Tests for text processing
│   └── test_grid_processor.py  # Tests for grid processing
├── examples                   # Example input files
│   ├── text_prompts.txt      # Sample text prompts
│   └── grid_samples.txt      # Sample grids of numbers
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variable template
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository


2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your API keys:

```
OPENAI_API_KEY=your_openai_api_key_here
STABILITY_API_KEY=your_stability_api_key_here
```

## Usage

### Text-to-Image

Convert a text prompt to an image:

```bash
python src/app.py --mode text --prompt "A beautiful sunset over the mountains" --num-images 1
```

### Grid-to-Image

Convert a grid of terrain types to an image:

```bash
python src/app.py --mode grid --grid "0 1 1 0 0
1 1 0 0 1
0 0 1 1 0
0 1 1 0 0
1 0 0 1 1" --num-images 1
```

### File Input

Process a file containing either a text prompt or a grid:

```bash
python src/app.py --mode file --file examples/text_prompts.txt
python src/app.py --mode file --file examples/grid_samples.txt
```

### Additional Options

```
--width          Width of the generated image (default: 512)
--height         Height of the generated image (default: 512)
--num-images     Number of images to generate (default: 1)
--text-model     Model for text-to-image (openai, stability, local) (default: openai)
--grid-model     Model for grid-to-image (openai, stability, local) (default: stability)
--output-dir     Directory to save generated images
```

## Terrain Types

The grid processor supports the following terrain types:

- 0: Plains - Flat, grassy plains
- 1: Forest - Dense forest with tall trees
- 2: Mountain - Rugged mountains with snow-capped peaks
- 3: Water - Clear blue water
- 4: Desert - Sandy desert with dunes
- 5: Snow - Snowy landscape
- 6: Swamp - Murky swampland
- 7: Hills - Rolling hills
- 8: Urban - Bustling urban area
- 9: Ruins - Ancient ruins

## Future Improvements

- Implement local model support
- Add more terrain types
- Develop a web interface
- Support for image-to-image transformations
- 3D terrain visualization

import gradio as gr
from structure_registry import get_biome_names, fetch_biome
from grid_generator import generate_biome
from llm import call_structure_generator  # Assuming this function is defined to get structure definitions

def handler(theme):
    structure_defs =call_structure_generator(theme)  # Assuming this function is defined to get structure definitions
    msg = generate_biome(theme, structure_defs)
    names = get_biome_names()
    return msg, gr.update(choices=names, value=names[-1] if names else None)

def display_selected_biome(name):
    biome = fetch_biome(name)
    return biome

def run_ui():

    with gr.Blocks() as demo:
        gr.Markdown("#  Pipeline Inspector")

        theme_input = gr.Textbox(label="Enter Biome Theme")
        generate_button = gr.Button("Generate Biome")
        biome_selector = gr.Dropdown(choices=get_biome_names(), label="Select Biome")
        biome_display = gr.JSON(label="Biome Details")

        generate_button.click(fn=handler, inputs=theme_input, outputs=[gr.Textbox(), biome_selector])
        biome_selector.change(fn=display_selected_biome, inputs=biome_selector, outputs=biome_display)

    demo.launch(share=True)

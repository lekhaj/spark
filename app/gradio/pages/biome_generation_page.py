import gradio as gr
import json
from typing import Any, Dict

from app.src_biome_gen.biome_generator import create_new_biome

# Best-effort import of DB save helper; if DB isn't configured, provide a noop stub.
try:
    from app.src_biome_gen.database import save_biome_document
except Exception:  # pragma: no cover - allow local dev without DB
    def save_biome_document(_doc: dict) -> str | None:  # type: ignore[misc]
        return None


def _attach_generator_widgets() -> Dict[str, object]:
    """Create generator widgets and wire their callbacks in the current Blocks context.

    Returns a dict of created widgets so callers can reference them if needed.
    This function must be called from inside a `with gr.Blocks():` context.
    """
    gr.Markdown("# Procedural Biome Generator")

    with gr.Accordion("System prompt (temporary)", open=False):
        sys_prompt = gr.Textbox(
            lines=4,
            placeholder="Optional system prompt / instruction (fill for custom system_prompt else use default)",
            label="System Prompt (temporary)",
        )
        clear_btn = gr.Button("Clear current system prompt")

    with gr.Row():
        inp = gr.Textbox(lines=4, placeholder="Enter a theme prompt or paste JSON here...", label="Prompt")
        out = gr.Textbox(lines=20, label="Generated Biome (JSON - Editable)", interactive=True)

    with gr.Row():
        btn = gr.Button("Generate Biome")
        save_btn = gr.Button("Save to Database")

    save_status = gr.Textbox(lines=2, label="Save Status", interactive=False)
    current_sys_prompt = gr.State("")

    def generate_prompt(prompt: Any, system_prompt: str | None = None) -> str:
        # Coerce prompt to a string
        if not isinstance(prompt, str):
            try:
                prompt_text = json.dumps(prompt, ensure_ascii=False)
            except Exception:
                prompt_text = str(prompt)
        else:
            prompt_text = prompt

        try:
            result = create_new_biome(prompt_text, system_prompt)
        except Exception:
            import traceback
            return """Failed to run generator:\n""" + traceback.format_exc()

        out_obj = {
            "success": getattr(result, "success", False),
            "message": getattr(result, "message", None),
            "biome_name": getattr(result, "biome_name", None),
            "biome_document": getattr(result, "biome_document", None),
        }
        try:
            return json.dumps(out_obj, indent=2, ensure_ascii=False)
        except Exception:
            return str(out_obj)

    def on_generate(p, s_input, stored_sys):
        """Generate using s_input if non-empty (and update stored prompt); otherwise use stored_sys.
        Returns (result_text, new_stored_sys).
        """
        new_stored = stored_sys
        effective_system = stored_sys
        if s_input and s_input.strip():
            new_stored = s_input
            effective_system = s_input

        result_text = generate_prompt(p, effective_system)
        return result_text, new_stored

    def on_clear(stored_sys):
        return ""

    def on_save(displayed_text):
        """Save the JSON content (whether original or edited) to database"""
        try:
            # Parse the JSON content
            parsed = json.loads(displayed_text)
            
            # Handle both the wrapper object and direct biome document
            if isinstance(parsed, dict):
                # If it's the wrapper object with biome_document inside
                if 'biome_document' in parsed and parsed['biome_document']:
                    # Use the actual biome document for saving
                    biome_doc = parsed['biome_document']
                else:
                    # Use the parsed object directly
                    biome_doc = parsed
                
                # Ensure we have a proper document structure
                if not isinstance(biome_doc, dict):
                    biome_doc = {"content": biome_doc}
                
                # Save to database
                res = save_biome_document(biome_doc)
                if res is None:
                    return "Save failed (no DB connection or error)."
                return f"Successfully saved to database: {res}"
            else:
                return "Error: Content must be a valid JSON object"
                
        except json.JSONDecodeError as e:
            return f"Invalid JSON format: {e}"
        except Exception as e:
            return f"Save error: {e}"

    # Wire events
    btn.click(
        on_generate, 
        inputs=[inp, sys_prompt, current_sys_prompt], 
        outputs=[out, current_sys_prompt]
    )
    save_btn.click(
        on_save, 
        inputs=[out], 
        outputs=[save_status]
    )
    clear_btn.click(
        on_clear, 
        inputs=[current_sys_prompt], 
        outputs=[current_sys_prompt]
    )

    return {
        "sys_prompt": sys_prompt,
        "clear_btn": clear_btn,
        "inp": inp,
        "out": out,
        "btn": btn,
        "save_btn": save_btn,
        "save_status": save_status,
        "current_sys_prompt": current_sys_prompt,
    }


def c_gradio_ui():
    """Return a Gradio Blocks object (standalone launcher).

    Use this to run the generator UI by itself (it returns a `gr.Blocks` instance).
    """
    with gr.Blocks() as demo:
        _attach_generator_widgets()
    return demo


def mount_into():
    """Mount the generator UI into the surrounding Blocks context.

    Call this from inside an existing `with gr.Blocks():` or `with gr.TabItem(...):`.
    """
    _attach_generator_widgets()
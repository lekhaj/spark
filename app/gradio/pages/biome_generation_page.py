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
        out = gr.Textbox(lines=20, label="Result (JSON)")

    btn = gr.Button("Generate")
    auto_save = gr.Checkbox(value=True, label="Auto-save generated biome to MongoDB")
    save_btn = gr.Button("Save to DB")
    save_status = gr.Textbox(lines=2, label="Save status", interactive=False)
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

    def on_generate(p, s_input, stored_sys, auto_save_val: bool | None = True):
        """Generate using s_input if non-empty (and update stored prompt); otherwise use stored_sys.
        Returns (result_text, new_stored_sys, save_status_message).
        """
        new_stored = stored_sys
        effective_system = stored_sys
        if s_input and s_input.strip():
            new_stored = s_input
            effective_system = s_input

        result_text = generate_prompt(p, effective_system)

        save_msg = ""
        # If auto-save enabled, try to parse and save the returned JSON/dict
        try:
            if auto_save_val:
                parsed = None
                try:
                    parsed = json.loads(result_text)
                except Exception:
                    parsed = {"biome_name": None, "generated": result_text}

                if isinstance(parsed, dict):
                    res = save_biome_document(parsed)
                    save_msg = f"Auto-save: {res or 'failed or no DB connection'}"
                else:
                    save_msg = "Auto-save: could not parse output"
        except Exception as e:
            save_msg = f"Auto-save error: {e}"

        return result_text, new_stored, save_msg

    def on_clear(stored_sys):
        return ""

    def on_save(displayed_text):
        try:
            parsed = json.loads(displayed_text)
        except Exception:
            parsed = {"biome_name": f"unsaved_{int(__import__('time').time())}", "generated": displayed_text}

        try:
            res = save_biome_document(parsed)
            if res is None:
                return "Save failed (no DB connection or error)."
            return f"Saved: {res}"
        except Exception as e:
            return f"Save error: {e}"

    # Wire events
    btn.click(on_generate, inputs=[inp, sys_prompt, current_sys_prompt, auto_save], outputs=[out, current_sys_prompt, save_status])
    save_btn.click(on_save, inputs=[out], outputs=[save_status])
    clear_btn.click(on_clear, inputs=[current_sys_prompt], outputs=[current_sys_prompt])

    return {
        "sys_prompt": sys_prompt,
        "clear_btn": clear_btn,
        "inp": inp,
        "out": out,
        "btn": btn,
        "auto_save": auto_save,
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

# app/gradio/pages/orchestrator_control_page.py

import gradio as gr
import requests
import time

API_BASE_URL = "http://localhost:8000"

def get_orchestrator_status():
    """Get orchestrator status from /orchestrate/status"""
    try:
        response = requests.get(f"{API_BASE_URL}/orchestrate/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"auto_mode": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"auto_mode": False, "error": str(e)}

def set_orchestrator_mode(auto_mode: bool):
    """Set orchestrator mode via /orchestrate/mode"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/orchestrate/mode",
            params={"auto_mode": auto_mode},
            timeout=5
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def create_orchestrator_ui():
    """Create a minimal orchestrator control UI"""
    with gr.Blocks() as control_panel:
        gr.Markdown("# ⚙️ Orchestrator Control")

        # Status Display
        with gr.Group():
            gr.Markdown("### Status")
            status_display = gr.Markdown("Loading...")

        # Mode Control
        with gr.Group():
            gr.Markdown("### Auto Scaling Mode")

            auto_mode_check = gr.Checkbox(
                label="Enable Auto Scaling",
                value=False,
                info="Automatically manage GPU instances based on queue"
            )

            mode_status = gr.Textbox(
                label="Status",
                value="",
                interactive=False
            )

        # Refresh Button
        refresh_btn = gr.Button("🔄 Refresh", variant="secondary")

        # Update function
        def update_display():
            """Update the status display"""
            status = get_orchestrator_status()

            if "error" in status:
                status_text = f"### ❌ Error\n{status['error']}"
                auto_mode = False
                mode_msg = "Failed to get status"
            else:
                auto_mode = status.get("auto_mode", False)
                mode_icon = "✅" if auto_mode else "⏸️"
                mode_text = "ENABLED" if auto_mode else "DISABLED"

                # Create status display
                status_text = f"### 🎛️ Orchestrator Status\n\n"
                status_text += f"**Auto Scaling:** {mode_icon} **{mode_text}**\n\n"

                # Add GPU status if available
                if "gpu_t4" in status:
                    gpu_t4 = status["gpu_t4"]
                    if isinstance(gpu_t4, dict):
                        state = gpu_t4.get("state", "unknown")
                        icon = "🟢" if state == "running" else "🔴" if state in ["stopped", "terminated"] else "🟡"
                        status_text += f"**T4 GPU:** {icon} {state.capitalize()}\n"

                if "gpu_a10" in status:
                    gpu_a10 = status["gpu_a10"]
                    if isinstance(gpu_a10, dict):
                        state = gpu_a10.get("state", "unknown")
                        icon = "🟢" if state == "running" else "🔴" if state in ["stopped", "terminated"] else "🟡"
                        status_text += f"**A10 GPU:** {icon} {state.capitalize()}\n"

                # Add queue status if available
                if "queues" in status:
                    queues = status["queues"]
                    if isinstance(queues, dict):
                        status_text += f"\n**Queues:** Images: {queues.get('image_tasks', 0)} | 3D: {queues.get('model_tasks', 0)}\n"

                status_text += f"\n**Updated:** {time.strftime('%H:%M:%S')}"
                mode_msg = f"Auto mode is {mode_text.lower()}"

            return status_text, auto_mode, mode_msg

        # Toggle callback
        def handle_mode_toggle(auto_mode: bool):
            """Handle auto mode toggle"""
            result = set_orchestrator_mode(auto_mode)

            if "error" in result:
                status_text, current_mode, _ = update_display()
                mode_msg = f"❌ Failed to update mode: {result['error']}"
                # Revert checkbox
                auto_mode = not auto_mode
            else:
                status_text, current_mode, _ = update_display()
                mode_msg = result.get("message", f"Auto mode {'enabled' if auto_mode else 'disabled'}")

            return status_text, auto_mode, mode_msg

        # Refresh callback
        def handle_refresh():
            """Handle refresh"""
            status_text, auto_mode, mode_msg = update_display()
            return status_text, auto_mode, mode_msg

        # Connect callbacks
        auto_mode_check.change(
            fn=handle_mode_toggle,
            inputs=[auto_mode_check],
            outputs=[status_display, auto_mode_check, mode_status]
        )

        refresh_btn.click(
            fn=handle_refresh,
            outputs=[status_display, auto_mode_check, mode_status]
        )

        # Initial load
        control_panel.load(
            fn=handle_refresh,
            outputs=[status_display, auto_mode_check, mode_status]
        )

    return control_panel

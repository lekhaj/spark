# app/gradio/pages/orchestrator_control_page.py

import gradio as gr
import requests
import time

API_BASE_URL = "http://localhost:8000"


def get_orchestrator_status():
    try:
        response = requests.get(f"{API_BASE_URL}/orchestrate/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"auto_mode": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"auto_mode": False, "error": str(e)}


def set_orchestrator_mode(auto_mode: bool):
    try:
        response = requests.post(
            f"{API_BASE_URL}/orchestrate/mode",
            params={"auto_mode": auto_mode},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_autoshutdown_state():
    try:
        response = requests.get(f"{API_BASE_URL}/orchestrate/autoshutdown", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"enabled": True, "idle_minutes": 60, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"enabled": True, "idle_minutes": 60, "error": str(e)}


def set_autoshutdown(enabled: bool, idle_minutes: int):
    try:
        response = requests.post(
            f"{API_BASE_URL}/orchestrate/autoshutdown",
            params={"enabled": enabled, "idle_minutes": int(idle_minutes)},
            timeout=5,
        )
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def create_orchestrator_ui():
    with gr.Blocks() as control_panel:
        gr.Markdown("# Orchestrator Control")

        # ── Status ────────────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Status")
            status_display = gr.Markdown("Loading...")

        # ── Auto Scaling ──────────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### Auto Scaling Mode")
            auto_mode_check = gr.Checkbox(
                label="Enable Auto Scaling",
                value=False,
                info="Automatically manage GPU instances based on queue",
            )
            mode_status = gr.Textbox(label="Status", value="", interactive=False)

        # ── GPU AutoShutdown ──────────────────────────────────────────────────
        with gr.Group():
            gr.Markdown("### GPU AutoShutdown")
            gr.Markdown(
                "Controls both the CPU orchestrator and the GPU-side shutdown watcher. "
                "When disabled, the GPU will not be stopped automatically regardless of idle time."
            )
            with gr.Row():
                autoshutdown_toggle = gr.Checkbox(
                    label="Enable AutoShutdown",
                    value=True,
                    info="Toggle GPU idle-shutdown on both CPU orchestrator and GPU worker",
                )
                idle_minutes_input = gr.Number(
                    label="Idle Timeout (minutes)",
                    value=15,
                    minimum=1,
                    maximum=480,
                    step=1,
                    interactive=True,
                    info="GPU shuts down after this many idle minutes",
                )
            apply_shutdown_btn = gr.Button("Apply AutoShutdown Settings", variant="primary")
            shutdown_status = gr.Textbox(label="AutoShutdown Status", value="", interactive=False)

        # ── Refresh ───────────────────────────────────────────────────────────
        refresh_btn = gr.Button("Refresh", variant="secondary")

        # ── Callbacks ─────────────────────────────────────────────────────────

        def update_display():
            status = get_orchestrator_status()
            sd_state = get_autoshutdown_state()

            if "error" in status:
                status_text = f"### Error\n{status['error']}"
                auto_mode = False
                mode_msg = "Failed to get status"
            else:
                auto_mode = status.get("auto_mode", False)
                mode_icon = "✅" if auto_mode else "⏸️"
                mode_text = "ENABLED" if auto_mode else "DISABLED"

                status_text = f"### Orchestrator Status\n\n"
                status_text += f"**Auto Scaling:** {mode_icon} **{mode_text}**\n\n"

                if "gpu_instance" in status:
                    gpu = status["gpu_instance"]
                    state = gpu.get("state", "unknown")
                    icon = "🟢" if state == "running" else "🔴" if state in ["stopped", "terminated"] else "🟡"
                    status_text += f"**GPU:** {icon} {state.capitalize()} ({gpu.get('public_ip', '?')})\n"

                if "queues" in status:
                    queues = status["queues"]
                    if isinstance(queues, dict):
                        parts = " | ".join(f"{k}: {v}" for k, v in queues.items())
                        status_text += f"\n**Queues:** {parts}\n"

                # autoshutdown from status or separate call
                as_info = status.get("autoshutdown", sd_state)
                as_on = as_info.get("enabled", True)
                as_min = as_info.get("idle_minutes", 15)
                as_icon = "✅" if as_on else "🔴"
                status_text += f"\n**AutoShutdown:** {as_icon} {'ON' if as_on else 'OFF'} — idle threshold: {as_min} min\n"

                if status.get("idle_seconds") is not None:
                    idle_sec = status["idle_seconds"]
                    status_text += f"**Current Idle:** {idle_sec // 60}m {idle_sec % 60}s\n"

                status_text += f"\n**Updated:** {time.strftime('%H:%M:%S')}"
                mode_msg = f"Auto mode is {mode_text.lower()}"

            # populate autoshutdown controls
            as_on = sd_state.get("enabled", True)
            as_min = sd_state.get("idle_minutes", 15)

            return status_text, auto_mode, mode_msg, as_on, as_min

        def handle_mode_toggle(auto_mode: bool):
            if auto_mode:
                gr.Warning(
                    "⚠️ spark_l4 instance has been DELETED. "
                    "GPU management now uses the spot instance only (spark_gpu_spot). "
                    "Fixed-instance start/stop calls will fail — use the spot flow.",
                    duration=10,
                )
            result = set_orchestrator_mode(auto_mode)
            if "error" in result:
                status_text, current_mode, _, as_on, as_min = update_display()
                return status_text, not auto_mode, f"Failed: {result['error']}", as_on, as_min
            status_text, current_mode, _, as_on, as_min = update_display()
            return status_text, current_mode, result.get("message", ""), as_on, as_min

        def handle_apply_autoshutdown(enabled: bool, idle_minutes: float):
            result = set_autoshutdown(enabled, int(idle_minutes))
            if "error" in result:
                msg = f"Failed: {result['error']}"
            else:
                state_word = "enabled" if result.get("enabled") else "disabled"
                msg = f"AutoShutdown {state_word} — idle threshold: {result.get('idle_minutes')} min"
            status_text, auto_mode, mode_msg, as_on, as_min = update_display()
            return status_text, auto_mode, mode_msg, as_on, as_min, msg

        def handle_refresh():
            status_text, auto_mode, mode_msg, as_on, as_min = update_display()
            return status_text, auto_mode, mode_msg, as_on, as_min, ""

        all_outputs = [
            status_display, auto_mode_check, mode_status,
            autoshutdown_toggle, idle_minutes_input,
        ]

        auto_mode_check.change(
            fn=handle_mode_toggle,
            inputs=[auto_mode_check],
            outputs=all_outputs,
        )

        apply_shutdown_btn.click(
            fn=handle_apply_autoshutdown,
            inputs=[autoshutdown_toggle, idle_minutes_input],
            outputs=all_outputs + [shutdown_status],
        )

        refresh_btn.click(
            fn=handle_refresh,
            outputs=all_outputs + [shutdown_status],
        )

        control_panel.load(
            fn=handle_refresh,
            outputs=all_outputs + [shutdown_status],
        )

    return control_panel

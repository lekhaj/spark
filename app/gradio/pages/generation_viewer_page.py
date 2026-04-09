#!/usr/bin/env python3
"""
Generation Viewer V2
====================
Purpose-built viewer for the SD1.5 → TRELLIS.2 → Auto-Rig Pro pipeline.

Supports the 2-layer MongoDB structure written by sd15_image_worker.py:
  biome.possible_structures.characters.{char_name}:
    stage1  : { prompt, negative, status, image_key, image_url }
    stage2  : { prompt, negative, status, image_key, image_url }
    images  : { base, refined, final }
    image_url  (top-level, Gradio legacy)
    model_path / model_url
    rigged_model_url
    status / generation_stage

Works with both string _id (new: "claudetest002") and ObjectId (old biomes).
"""

import os
import urllib.parse

import gradio as gr
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL", "")
MONGO_DB  = os.getenv("MONGO_DB")  or os.getenv("MONGODB_DB_NAME", "World_builder")


def _get_db():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)
        return client[MONGO_DB]
    except Exception as e:
        print(f"[GenViewer] MongoDB connect error: {e}")
        return None


def _s3_https(url):
    if url and url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        if len(parts) == 2:
            return f"https://{parts[0]}.s3.amazonaws.com/{parts[1]}"
    return url or ""


def _list_biomes():
    """Return [(display_label, biome_id)] for every doc in biomes collection."""
    db = _get_db()
    if db is None:
        return []
    try:
        docs = list(db.biomes.find({}, {"_id": 1, "biome_name": 1, "biome_type": 1}))
        choices = []
        for doc in docs:
            raw_id = doc["_id"]
            biome_id = str(raw_id)
            name = doc.get("biome_name") or biome_id
            btype = doc.get("biome_type", "")
            label = f"{name}  [{biome_id}]" if btype == "" else f"{name}  ({btype})  [{biome_id}]"
            choices.append((label, biome_id))
        return choices
    except Exception as e:
        print(f"[GenViewer] list_biomes error: {e}")
        return []


def _get_biome_doc(biome_id: str):
    """Fetch biome document by string or ObjectId."""
    db = _get_db()
    if db is None:
        return None
    try:
        doc = db.biomes.find_one({"_id": biome_id})
        if doc is None:
            # try ObjectId
            from bson import ObjectId
            doc = db.biomes.find_one({"_id": ObjectId(biome_id)})
        return doc
    except Exception:
        return None


def _extract_characters(biome_doc) -> dict:
    """Return {char_name: char_data} from possible_structures.characters."""
    if not biome_doc:
        return {}
    ps = biome_doc.get("possible_structures", {})
    # New pipeline: possible_structures.characters
    chars = ps.get("characters", {})
    if chars:
        return chars
    # Fallback: flatten all categories (old pipeline)
    flat = {}
    for cat_data in ps.values():
        if isinstance(cat_data, dict):
            flat.update(cat_data)
    return flat


# ── Status badge helper ──────────────────────────────────────────────────────

def _badge(status):
    status = (status or "").lower()
    if status in ("complete", "image_complete", "stage1_complete", "model_complete"):
        return "✅"
    if status in ("generating", "model_generating"):
        return "⏳"
    if status in ("not_started", "none", ""):
        return "⭕"
    return "🔷"


# ── Gradio page factory ──────────────────────────────────────────────────────

def generation_viewer_ui():

    with gr.Blocks() as page:
        gr.Markdown("## 🎨 Generation Viewer V2")
        gr.Markdown(
            "Live view of the **SD1.5 → TRELLIS.2 → Auto-Rig Pro** pipeline. "
            "Shows Stage 1 (ControlNet structure) and Stage 2 (img2img detail) "
            "images side-by-side, plus 3D model status for each character."
        )

        # ── Biome selector ────────────────────────────────────────────────────
        with gr.Row():
            biome_dropdown = gr.Dropdown(
                label="Select Biome",
                choices=_list_biomes(),
                interactive=True,
                scale=4,
            )
            refresh_btn = gr.Button("🔄 Refresh", scale=1)

        biome_info_md = gr.Markdown("", visible=False)

        # ── Load button ───────────────────────────────────────────────────────
        load_btn = gr.Button("Load Characters", variant="primary")

        # ── Per-character output area ─────────────────────────────────────────
        status_table_md = gr.Markdown("", visible=False)

        with gr.Row(visible=False) as gallery_row:
            with gr.Column():
                gr.Markdown("### Stage 1 — ControlNet Structure")
                gallery_s1 = gr.Gallery(
                    show_label=False,
                    columns=2, rows=2,
                    height=420,
                    object_fit="contain",
                )
            with gr.Column():
                gr.Markdown("### Stage 2 — img2img Detail")
                gallery_s2 = gr.Gallery(
                    show_label=False,
                    columns=2, rows=2,
                    height=420,
                    object_fit="contain",
                )

        # ── Character detail accordion ────────────────────────────────────────
        char_detail_md = gr.Markdown("", visible=False)

        # ── 3D / Rigged model section ─────────────────────────────────────────
        with gr.Accordion("🧊 3D Models & Rigging", open=False, visible=False) as models_accordion:
            models_md   = gr.Markdown("")
            model_links = gr.HTML("")

        # ── Handlers ─────────────────────────────────────────────────────────

        def on_refresh():
            choices = _list_biomes()
            return gr.update(choices=choices, value=choices[0][1] if choices else None)

        def on_biome_change(biome_id):
            """Show biome description when biome is selected."""
            if not biome_id:
                return gr.update(visible=False, value="")
            doc = _get_biome_doc(biome_id)
            if not doc:
                return gr.update(visible=True, value=f"⚠️ Biome `{biome_id}` not found in MongoDB.")
            name  = doc.get("biome_name", biome_id)
            btype = doc.get("biome_type", "—")
            desc  = doc.get("description", "")
            chars = _extract_characters(doc)
            md = (
                f"**{name}** · type: `{btype}` · id: `{biome_id}`\n\n"
                f"{desc}\n\n"
                f"**Characters**: {', '.join(f'`{c}`' for c in chars) or '_(none yet)_'}"
            )
            return gr.update(visible=True, value=md)

        def on_load(biome_id):
            """Load all characters and return gallery images + status table."""
            empty = (
                gr.update(visible=False, value=""),  # status_table_md
                gr.update(visible=False),             # gallery_row
                [],                                   # gallery_s1
                [],                                   # gallery_s2
                gr.update(visible=False, value=""),   # char_detail_md
                gr.update(visible=False),             # models_accordion
                "",                                   # models_md
                "",                                   # model_links
            )

            if not biome_id:
                return empty

            doc = _get_biome_doc(biome_id)
            if not doc:
                return (
                    gr.update(visible=True, value=f"⚠️ Biome `{biome_id}` not found."),
                    gr.update(visible=False), [], [],
                    gr.update(visible=False, value=""),
                    gr.update(visible=False), "", "",
                )

            chars = _extract_characters(doc)
            if not chars:
                return (
                    gr.update(visible=True, value="⚠️ No characters found in `possible_structures.characters`."),
                    gr.update(visible=False), [], [],
                    gr.update(visible=False, value=""),
                    gr.update(visible=False), "", "",
                )

            # ── Build gallery images ──────────────────────────────────────────
            s1_imgs, s2_imgs = [], []
            table_rows = ["| Character | Type | Stage 1 | Stage 2 | 3D | Rigged |", "|---|---|---|---|---|---|"]
            detail_sections = []
            model_lines = []
            model_html_parts = []

            for char_name, char_data in chars.items():
                if not isinstance(char_data, dict):
                    continue

                stage1 = char_data.get("stage1") or {}
                stage2 = char_data.get("stage2") or {}
                s1_url    = _s3_https(stage1.get("image_url", ""))
                s2_url    = _s3_https(stage2.get("image_url", ""))
                s1_status = stage1.get("status") or "—"
                s2_status = stage2.get("status") or "—"
                overall   = char_data.get("status") or "—"
                gen_stage = char_data.get("generation_stage") or "—"
                ctype     = char_data.get("character_type") or char_data.get("creature_category") or "—"
                model_url = _s3_https(char_data.get("model_url") or char_data.get("model_path") or "")
                rigged_url= _s3_https(char_data.get("rigged_model_url") or "")
                desc      = char_data.get("description", "")

                if s1_url:
                    s1_imgs.append((s1_url, char_name))
                if s2_url:
                    s2_imgs.append((s2_url, char_name))

                s1_b = _badge(s1_status)
                s2_b = _badge(s2_status)
                m_b  = "✅" if model_url else "⭕"
                r_b  = "✅" if rigged_url else "⭕"
                table_rows.append(
                    f"| **{char_name}** | {ctype} | {s1_b} {s1_status} "
                    f"| {s2_b} {s2_status} | {m_b} | {r_b} |"
                )

                # Detail section per character
                s1_prompt_txt = stage1.get("prompt", "_not set_")
                s2_prompt_txt = stage2.get("prompt", "_not set_")
                detail_sections.append(
                    f"\n---\n### `{char_name}` · {ctype}\n"
                    f"**Status**: {_badge(overall)} `{overall}` · stage: `{gen_stage}`\n\n"
                    f"**Description**:\n> {desc}\n\n"
                    f"**Stage 1 prompt** ({s1_b} {s1_status}):\n```\n{s1_prompt_txt}\n```\n\n"
                    f"**Stage 2 prompt** ({s2_b} {s2_status}):\n```\n{s2_prompt_txt}\n```"
                )

                # 3D model links
                if model_url or rigged_url:
                    model_lines.append(f"**{char_name}**")
                    if model_url:
                        enc = urllib.parse.quote(model_url, safe=':/')
                        model_lines.append(f"  - [TRELLIS.2 GLB]({model_url})")
                        model_html_parts.append(
                            f'<a href="https://3dviewer.net/#model={enc}" target="_blank" '
                            f'style="padding:6px 12px;background:#0b5fff;color:#fff;'
                            f'border-radius:5px;text-decoration:none;margin:4px;display:inline-block">'
                            f'🔍 {char_name} — View 3D</a>'
                            f'<a href="{model_url}" target="_blank" '
                            f'style="padding:6px 12px;background:#1f7f46;color:#fff;'
                            f'border-radius:5px;text-decoration:none;margin:4px;display:inline-block">'
                            f'⬇️ Download GLB</a>'
                        )
                    if rigged_url:
                        enc_r = urllib.parse.quote(rigged_url, safe=':/')
                        model_lines.append(f"  - [Rigged GLB]({rigged_url})")
                        model_html_parts.append(
                            f'<a href="https://3dviewer.net/#model={enc_r}" target="_blank" '
                            f'style="padding:6px 12px;background:#7c3aed;color:#fff;'
                            f'border-radius:5px;text-decoration:none;margin:4px;display:inline-block">'
                            f'🦴 {char_name} — View Rigged</a>'
                        )

            has_models = bool(model_lines)
            status_md  = "\n".join(table_rows)
            detail_md  = "\n".join(detail_sections) if detail_sections else "No character details available."
            models_md_val = "\n".join(model_lines) if model_lines else "No 3D models generated yet."
            html_val   = "<div>" + "".join(model_html_parts) + "</div>" if model_html_parts else ""

            return (
                gr.update(visible=True, value=status_md),
                gr.update(visible=True),
                s1_imgs if s1_imgs else [],
                s2_imgs if s2_imgs else [],
                gr.update(visible=True, value=detail_md),
                gr.update(visible=has_models),
                models_md_val,
                html_val,
            )

        # Wire events
        refresh_btn.click(fn=on_refresh, outputs=[biome_dropdown])

        biome_dropdown.change(
            fn=on_biome_change,
            inputs=[biome_dropdown],
            outputs=[biome_info_md],
        )

        load_btn.click(
            fn=on_load,
            inputs=[biome_dropdown],
            outputs=[
                status_table_md,
                gallery_row,
                gallery_s1,
                gallery_s2,
                char_detail_md,
                models_accordion,
                models_md,
                model_links,
            ],
        )

    return page

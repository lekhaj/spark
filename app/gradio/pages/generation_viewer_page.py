#!/usr/bin/env python3
"""
Generation Viewer V2
====================
Purpose-built for the SD1.5 → TRELLIS.2 → Auto-Rig Pro character pipeline.

Reads possible_structures.characters.{char_name} from MongoDB biomes collection.
Supports 2-layer structure:
  stage1 : { prompt, negative, status, image_key, image_url }
  stage2 : { prompt, negative, status, image_key, image_url }
  model_path / model_url / rigged_model_url

Works with Gradio 5.x — no Row/Accordion visibility hacks.
"""

import os
import urllib.parse
import logging

import gradio as gr
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("generation_viewer")

MONGO_URI = os.getenv("MONGO_URI") or os.getenv("MONGODB_URL", "")
MONGO_DB  = os.getenv("MONGO_DB")  or os.getenv("MONGODB_DB_NAME", "World_builder")

# ── DB helpers ────────────────────────────────────────────────────────────────

def _db():
    try:
        return MongoClient(MONGO_URI, serverSelectionTimeoutMS=4000)[MONGO_DB]
    except Exception as e:
        logger.error(f"MongoDB connect error: {e}")
        return None


def _s3(url):
    if url and url.startswith("s3://"):
        parts = url[5:].split("/", 1)
        if len(parts) == 2:
            return f"https://{parts[0]}.s3.amazonaws.com/{parts[1]}"
    return url or ""


def _badge(status):
    s = (status or "").lower()
    if s in ("complete", "image_complete", "stage1_complete", "model_complete"):
        return "✅"
    if "generat" in s or "pending" in s:
        return "⏳"
    return "⭕"


# ── Data fetchers ─────────────────────────────────────────────────────────────

def list_biomes():
    db = _db()
    if db is None:
        return []
    try:
        docs = list(db.biomes.find({}, {"_id": 1, "biome_name": 1, "biome_type": 1}))
        out = []
        for d in docs:
            bid   = str(d["_id"])
            name  = d.get("biome_name") or bid
            btype = d.get("biome_type", "")
            label = f"{name}  [{bid}]" if not btype else f"{name}  ({btype})  [{bid}]"
            out.append((label, bid))
        return out
    except Exception as e:
        logger.error(f"list_biomes: {e}")
        return []


def get_biome(biome_id):
    db = _db()
    if db is None:
        return None
    try:
        doc = db.biomes.find_one({"_id": biome_id})
        if doc is None:
            from bson import ObjectId
            try:
                doc = db.biomes.find_one({"_id": ObjectId(biome_id)})
            except Exception:
                pass
        return doc
    except Exception as e:
        logger.error(f"get_biome: {e}")
        return None


def get_characters(doc):
    if not doc:
        return {}
    return doc.get("possible_structures", {}).get("characters", {})


# ── UI builder ────────────────────────────────────────────────────────────────

def generation_viewer_ui():

    with gr.Blocks() as page:

        gr.Markdown(
            "## 🎨 Generation Viewer V2\n"
            "Live view of the **SD1.5 → TRELLIS.2 → Auto-Rig Pro** character pipeline.\n"
            "Reads directly from MongoDB — refresh anytime to check progress."
        )

        # ── Biome selector row ────────────────────────────────────────────────
        with gr.Row():
            biome_dd = gr.Dropdown(
                label="Biome",
                choices=list_biomes(),
                interactive=True,
                scale=5,
            )
            gr.Button("🔄 Refresh list", scale=1).click(
                fn=lambda: gr.update(choices=list_biomes()),
                outputs=[biome_dd],
            )

        biome_meta = gr.Markdown("")

        # ── Load + Auto-poll ──────────────────────────────────────────────────
        with gr.Row():
            load_btn    = gr.Button("▶ Load / Refresh", variant="primary", scale=3)
            autopoll_cb = gr.Checkbox(label="Auto-refresh every 15s", value=False, scale=1)

        # ── Status summary ────────────────────────────────────────────────────
        status_md = gr.Markdown("_Select a biome and click Load._")

        # ── Stage 1 / Stage 2 galleries (always visible, empty when no images) ─
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Stage 1 — ControlNet structure")
                gallery_s1 = gr.Gallery(
                    show_label=False,
                    columns=2,
                    height=400,
                    object_fit="contain",
                    allow_preview=True,
                )
            with gr.Column():
                gr.Markdown("### Stage 2 — img2img detail")
                gallery_s2 = gr.Gallery(
                    show_label=False,
                    columns=2,
                    height=400,
                    object_fit="contain",
                    allow_preview=True,
                )

        # ── Character details (prompts + descriptions) ────────────────────────
        with gr.Accordion("📋 Character Details & Prompts", open=False):
            char_detail_md = gr.Markdown("_Load a biome to see character details._")

        # ── 3D & Rigged models ────────────────────────────────────────────────
        with gr.Accordion("🧊 3D Models", open=False):
            models_md   = gr.Markdown("_No 3D models yet._")
            model_links = gr.HTML("")

        # ── Core load function ────────────────────────────────────────────────

        def load(biome_id):
            if not biome_id:
                return (
                    "",                # biome_meta
                    "⚠️ No biome selected.",  # status_md
                    [],                # gallery_s1
                    [],                # gallery_s2
                    "_No data._",      # char_detail_md
                    "_No data._",      # models_md
                    "",                # model_links
                )

            doc = get_biome(biome_id)
            if not doc:
                return (
                    f"⚠️ Biome `{biome_id}` not found in MongoDB.",
                    "Run `python worker/queue_claudetest002.py --stage1-only` on the CPU server first.",
                    [], [],
                    "_Biome not created yet._",
                    "_No data._", "",
                )

            # ── Biome metadata ────────────────────────────────────────────────
            name  = doc.get("biome_name", biome_id)
            btype = doc.get("biome_type", "—")
            desc  = doc.get("description", "")
            chars = get_characters(doc)
            meta_md = (
                f"**{name}**  ·  type: `{btype}`  ·  id: `{biome_id}`\n\n"
                f"{desc}\n\n"
                f"**Characters in pipeline**: {', '.join(f'`{c}`' for c in chars) or '_(none)_'}"
            )

            if not chars:
                return (
                    meta_md,
                    "⚠️ No characters found under `possible_structures.characters`.",
                    [], [],
                    "_No characters._",
                    "_No data._", "",
                )

            # ── Build output per character ────────────────────────────────────
            s1_imgs, s2_imgs    = [], []
            table_rows          = ["| Character | Type | Stage 1 | Stage 2 | 3D | Rigged |",
                                   "|---|---|---|---|---|---|"]
            detail_parts        = []
            model_md_parts      = []
            model_html_parts    = []

            for char_name, cd in chars.items():
                if not isinstance(cd, dict):
                    continue

                stage1 = cd.get("stage1") or {}
                stage2 = cd.get("stage2") or {}
                s1_url = _s3(stage1.get("image_url") or "")
                s2_url = _s3(stage2.get("image_url") or "")
                s1_st  = stage1.get("status") or "not_started"
                s2_st  = stage2.get("status") or "not_started"
                gen_st = cd.get("generation_stage") or "—"
                ctype  = cd.get("character_type") or cd.get("creature_category") or "—"
                m_url  = _s3(cd.get("model_url") or "")
                r_url  = _s3(cd.get("rigged_model_url") or "")
                desc   = cd.get("description", "")

                if s1_url:
                    s1_imgs.append((s1_url, char_name))
                if s2_url:
                    s2_imgs.append((s2_url, char_name))

                table_rows.append(
                    f"| **{char_name}** | {ctype} "
                    f"| {_badge(s1_st)} `{s1_st}` "
                    f"| {_badge(s2_st)} `{s2_st}` "
                    f"| {'✅' if m_url else '⭕'} "
                    f"| {'✅' if r_url else '⭕'} |"
                )

                # Prompt details
                s1_p = stage1.get("prompt") or cd.get("stage1_prompt") or "_not set_"
                s2_p = stage2.get("prompt") or cd.get("stage2_prompt") or "_not set_"
                s1_n = stage1.get("negative") or cd.get("stage1_negative") or "_not set_"
                detail_parts.append(
                    f"\n---\n### `{char_name}` · {ctype}\n"
                    f"**Stage**: `{gen_st}`\n\n"
                    f"> {desc}\n\n"
                    f"**Stage 1 prompt** ({_badge(s1_st)} {s1_st}):\n"
                    f"```\n{s1_p}\n```\n\n"
                    f"**Stage 1 negative**:\n"
                    f"```\n{s1_n}\n```\n\n"
                    f"**Stage 2 prompt** ({_badge(s2_st)} {s2_st}):\n"
                    f"```\n{s2_p}\n```"
                )

                # 3D model links
                if m_url:
                    enc = urllib.parse.quote(m_url, safe=':/')
                    model_md_parts.append(f"- **{char_name}** TRELLIS.2 GLB: [view]({m_url})")
                    model_html_parts.append(
                        f'<a href="https://3dviewer.net/#model={enc}" target="_blank" '
                        f'style="display:inline-block;margin:4px;padding:7px 14px;'
                        f'background:#0b5fff;color:#fff;border-radius:5px;text-decoration:none;">'
                        f'🔍 {char_name} — 3D View</a>'
                        f'<a href="{m_url}" target="_blank" '
                        f'style="display:inline-block;margin:4px;padding:7px 14px;'
                        f'background:#1f7f46;color:#fff;border-radius:5px;text-decoration:none;">'
                        f'⬇️ Download GLB</a> '
                    )
                if r_url:
                    enc_r = urllib.parse.quote(r_url, safe=':/')
                    model_md_parts.append(f"- **{char_name}** Rigged GLB: [view]({r_url})")
                    model_html_parts.append(
                        f'<a href="https://3dviewer.net/#model={enc_r}" target="_blank" '
                        f'style="display:inline-block;margin:4px;padding:7px 14px;'
                        f'background:#7c3aed;color:#fff;border-radius:5px;text-decoration:none;">'
                        f'🦴 {char_name} — Rigged View</a> '
                    )

            # ── Assemble outputs ──────────────────────────────────────────────
            s1_count = len(s1_imgs)
            s2_count = len(s2_imgs)
            table_md = "\n".join(table_rows)
            table_md += (
                f"\n\n**Stage 1 images ready**: {s1_count} / {len(chars)}  "
                f"| **Stage 2 images ready**: {s2_count} / {len(chars)}"
            )
            if s1_count == 0:
                table_md += "\n\n⚠️ _No Stage 1 images yet — GPU workers may still be processing._"

            return (
                meta_md,
                table_md,
                s1_imgs or [],
                s2_imgs or [],
                "\n".join(detail_parts) or "_No details available._",
                "\n".join(model_md_parts) or "_No 3D models generated yet._",
                "<div style='margin:8px 0'>" + "".join(model_html_parts) + "</div>" if model_html_parts else "",
            )

        # ── Wire load button ──────────────────────────────────────────────────
        _outputs = [biome_meta, status_md, gallery_s1, gallery_s2,
                    char_detail_md, models_md, model_links]

        load_btn.click(fn=load, inputs=[biome_dd], outputs=_outputs)

        # ── Auto-refresh via Gradio timer (every 15s when checkbox on) ────────
        timer = gr.Timer(value=15, active=False)
        autopoll_cb.change(
            fn=lambda active: gr.Timer(active=active),
            inputs=[autopoll_cb],
            outputs=[timer],
        )
        timer.tick(fn=load, inputs=[biome_dd], outputs=_outputs)

        # ── Biome meta on dropdown change ─────────────────────────────────────
        def on_biome_select(bid):
            if not bid:
                return ""
            doc = get_biome(bid)
            if not doc:
                return f"⚠️ Biome `{bid}` not found in MongoDB — run the queue script first."
            name  = doc.get("biome_name", bid)
            btype = doc.get("biome_type", "—")
            chars = get_characters(doc)
            statuses = {
                cn: (cd.get("status") or "not_started")
                for cn, cd in chars.items()
                if isinstance(cd, dict)
            }
            status_str = "  ".join(f"`{n}` {_badge(s)}" for n, s in statuses.items()) or "_none_"
            return (
                f"**{name}** · type: `{btype}` · `{len(chars)}` character(s)\n\n"
                f"{status_str}"
            )

        biome_dd.change(fn=on_biome_select, inputs=[biome_dd], outputs=[biome_meta])

    return page

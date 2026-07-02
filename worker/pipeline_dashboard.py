#!/usr/bin/env python3
"""
pipeline_dashboard.py — Gradio UI for the Flux + SD1.5 character pipeline
==========================================================================

Tabs:
  1. Pipeline Control  — view biome characters, trigger Flux/Stage1/Stage2/Stage3
  2. Character Specs   — view/edit structured character data
  3. Style Library     — view/edit reusable style-DNA blocks
  4. Reference Bank    — browse reference images by group
  5. Prompt Templates  — view/test prompt templates
  6. Prompt Builder    — assemble prompts from spec + template + style
  7. MongoDB Viewer    — raw biome document viewer

Usage:
  python worker/pipeline_dashboard.py          # local
  python worker/pipeline_dashboard.py --share   # public tunnel
"""

import io
import json
import os
import sys
import time
from datetime import datetime

import gradio as gr
import pymongo

# Ensure worker/lib is importable
sys.path.insert(0, os.path.dirname(__file__))
from lib.spec_schema import (
    get_db,
    list_character_specs, get_character_spec, upsert_character_spec, delete_character_spec,
    list_styles, get_style, upsert_style,
    list_references, get_reference, search_references, upsert_reference,
    list_prompt_templates, get_prompt_template,
    assemble_prompt, assemble_negative,
    build_body_desc, build_face_desc, build_clothing_desc, build_creature_desc,
)

# ── Config ────────────────────────────────────────────────────────────────────
MONGO_URI   = (os.getenv("MONGO_URI") or os.getenv("MONGODB_URL") or "")
MONGO_DB    = os.getenv("MONGO_DB",  "World_builder")
S3_BUCKET   = "sparkassets-us"
S3_BASE     = f"https://{S3_BUCKET}.s3.us-east-1.amazonaws.com"
# g7e GPU boxes are Amazon Linux 2023 → user is ec2-user (NOT ubuntu); EIP 52.91.128.47.
GPU_HOST    = os.getenv("GPU_HOST", "ec2-user@52.91.128.47")
GPU_SSH_KEY = os.getenv("GPU_SSH_KEY", "/Users/lekhaj/Documents/us_cpu_key.pem")


def _db():
    return get_db(MONGO_URI, MONGO_DB)


def _json_pretty(obj):
    """Pretty-print a dict, handling datetime/ObjectId."""
    def default(o):
        if isinstance(o, (datetime, float)):
            return str(o)
        return str(o)
    return json.dumps(obj, indent=2, default=default, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PIPELINE CONTROL
# ═══════════════════════════════════════════════════════════════════════════════

def load_biome_characters(biome_id):
    """Load all characters from a biome and return summary + images."""
    db = _db()
    biome = db.biomes.find_one({"_id": biome_id})
    if not biome:
        return "Biome not found.", [], ""

    chars = (biome.get("possible_structures") or {}).get("characters", {})
    rows = []
    details = {}
    for name, doc in chars.items():
        status = doc.get("status", "unknown")
        gen_stage = doc.get("generation_stage", "none")
        flux_status = (doc.get("flux_concept") or {}).get("status", "not_started")
        s1_status = (doc.get("stage1") or {}).get("status", "not_started")
        s2_status = (doc.get("stage2") or {}).get("status", "not_started")
        s3_status = (doc.get("stage3") or {}).get("status", "not_started")

        rows.append({
            "Character": name,
            "Category": doc.get("creature_category", doc.get("character_type", "?")),
            "Flux": flux_status,
            "Stage1": s1_status or "not_started",
            "Stage2": s2_status or "not_started",
            "Stage3": s3_status or "not_started",
            "Status": status,
        })
        details[name] = doc

    import pandas as pd
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return df, details


def get_character_images(biome_id, char_name):
    """Get all images for a character across pipeline stages."""
    db = _db()
    biome = db.biomes.find_one({"_id": biome_id})
    if not biome:
        return None, None, None, None, "Biome not found"

    chars = (biome.get("possible_structures") or {}).get("characters", {})
    doc = chars.get(char_name, {})
    if not doc:
        return None, None, None, None, "Character not found"

    # Collect image URLs
    flux_url = (doc.get("flux_concept") or {}).get("image_url")
    s1_url = (doc.get("stage1") or {}).get("image_url")
    s2_url = (doc.get("stage2") or {}).get("image_url")
    s3_url = (doc.get("stage3") or {}).get("front_url")

    summary = _json_pretty({
        "flux_concept": doc.get("flux_concept"),
        "stage1": doc.get("stage1"),
        "stage2": doc.get("stage2"),
        "stage3": doc.get("stage3"),
        "images": doc.get("images"),
    })

    return flux_url, s1_url, s2_url, s3_url, summary


def list_biomes():
    """List all biome IDs."""
    db = _db()
    return [b["_id"] for b in db.biomes.find({}, {"_id": 1}).sort("_id", 1)]


def trigger_gpu_command(biome_id, char_name, stage, extra_args=""):
    """Build SSH command to trigger a pipeline stage on GPU."""
    if not biome_id:
        return "Select a biome first."

    if stage == "flux":
        cmd = "cd /home/ubuntu/worker && python flux_concept_generator.py"
    elif stage == "stage1_2":
        cmd = "cd /home/ubuntu/worker && python run_sd15_direct.py"
    elif stage == "stage3":
        cmd = "cd /home/ubuntu/worker && python run_multiview_direct.py"
    else:
        return f"Unknown stage: {stage}"

    ssh_cmd = f'ssh -i {GPU_SSH_KEY} {GPU_HOST} "{cmd}"'
    return f"Run this on GPU (or copy to terminal):\n\n{ssh_cmd}\n\nOr run directly on GPU:\n  {cmd}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — CHARACTER SPECS
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_specs():
    db = _db()
    specs = list_character_specs(db)
    return _json_pretty(specs)


def load_spec_detail(char_id):
    db = _db()
    spec = get_character_spec(db, char_id)
    if not spec:
        return "Not found", "", "", "", ""
    body = build_body_desc(spec) if spec.get("category") != "fantastical_animal" else build_creature_desc(spec)
    face = build_face_desc(spec)
    clothing = build_clothing_desc(spec)
    return _json_pretty(spec), body, face, clothing, spec.get("notes", "")


def list_spec_ids():
    db = _db()
    return [s["_id"] for s in list_character_specs(db)]


def save_spec_from_json(json_str):
    try:
        spec = json.loads(json_str)
        if "_id" not in spec:
            return "Error: spec must have '_id' field."
        db = _db()
        upsert_character_spec(db, spec)
        return f"Saved: {spec['_id']}"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — STYLE LIBRARY
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_styles():
    db = _db()
    return _json_pretty(list_styles(db))


def load_style_detail(style_id):
    db = _db()
    style = get_style(db, style_id)
    if not style:
        return "Not found", "", ""
    return _json_pretty(style), style.get("positive_tags", ""), style.get("negative_tags", "")


def list_style_ids():
    db = _db()
    return [s["_id"] for s in list_styles(db)]


def save_style_from_json(json_str):
    try:
        style = json.loads(json_str)
        if "_id" not in style:
            return "Error: style must have '_id' field."
        db = _db()
        upsert_style(db, style)
        return f"Saved: {style['_id']}"
    except Exception as e:
        return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — REFERENCE IMAGE BANK
# ═══════════════════════════════════════════════════════════════════════════════

def load_references_by_group(group):
    db = _db()
    if group == "all":
        refs = list_references(db)
    else:
        refs = list_references(db, group=group)

    rows = []
    for r in refs:
        rows.append({
            "ID": r["_id"],
            "Group": r.get("group", ""),
            "Subgroup": r.get("subgroup", ""),
            "Pose": r.get("pose", ""),
            "View": r.get("view", ""),
            "Description": r.get("description", ""),
            "S3 Key": r.get("s3_key", ""),
            "Source": r.get("metadata", {}).get("source", ""),
        })
    import pandas as pd
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def search_refs(query):
    db = _db()
    refs = search_references(db, query)
    rows = []
    for r in refs:
        rows.append({
            "ID": r["_id"],
            "Group": r.get("group", ""),
            "Pose": r.get("pose", ""),
            "View": r.get("view", ""),
            "Description": r.get("description", ""),
            "Tags": ", ".join(r.get("tags", [])),
        })
    import pandas as pd
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def get_ref_groups():
    db = _db()
    groups = db.reference_image_bank.distinct("group")
    return ["all"] + sorted(groups)


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 5 — PROMPT TEMPLATES
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_templates():
    db = _db()
    return _json_pretty(list_prompt_templates(db))


def load_template_detail(tpl_id):
    db = _db()
    tpl = get_prompt_template(db, tpl_id)
    if not tpl:
        return "Not found", "", "", ""
    return (
        _json_pretty(tpl),
        tpl.get("template", ""),
        tpl.get("negative_template", ""),
        ", ".join(tpl.get("variables", [])),
    )


def list_template_ids():
    db = _db()
    return [t["_id"] for t in list_prompt_templates(db)]


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 6 — PROMPT BUILDER
# ═══════════════════════════════════════════════════════════════════════════════

def build_prompt_preview(char_id, template_id, style_override, extra_tags):
    db = _db()
    try:
        prompt = assemble_prompt(db, char_id, template_id,
                                 style_override=style_override or None,
                                 extra_tags=extra_tags or "")
        negative = assemble_negative(db, char_id, template_id,
                                      style_override=style_override or None)
        word_count = len(prompt.split())
        clip_estimate = int(word_count * 1.3)
        info = f"Words: {word_count} | Est. CLIP tokens: {clip_estimate}"
        if clip_estimate > 77:
            info += " WARNING: Exceeds 77 CLIP token limit for SD1.5!"
        return prompt, negative, info
    except Exception as e:
        return f"Error: {e}", "", ""


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 7 — MONGODB VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

def load_biome_raw(biome_id):
    db = _db()
    doc = db.biomes.find_one({"_id": biome_id})
    if not doc:
        return "Not found"
    return _json_pretty(doc)


def load_collection_raw(collection_name, limit=20):
    db = _db()
    docs = list(db[collection_name].find().limit(limit))
    return _json_pretty(docs)


def list_collections():
    db = _db()
    return sorted(db.list_collection_names())


# ═══════════════════════════════════════════════════════════════════════════════
#  BUILD APP
# ═══════════════════════════════════════════════════════════════════════════════

def build_app():
    # Pre-load choices
    try:
        biome_ids = list_biomes()
    except Exception:
        biome_ids = ["claudetest002"]
    try:
        spec_ids = list_spec_ids()
    except Exception:
        spec_ids = []
    try:
        style_ids = list_style_ids()
    except Exception:
        style_ids = []
    try:
        template_ids = list_template_ids()
    except Exception:
        template_ids = []
    try:
        ref_groups = get_ref_groups()
    except Exception:
        ref_groups = ["all"]
    try:
        collections = list_collections()
    except Exception:
        collections = ["biomes"]

    with gr.Blocks(title="Spark Pipeline Dashboard", theme=gr.themes.Soft()) as app:
        gr.Markdown("# Spark Pipeline Dashboard\n"
                     "Character specs, style library, reference bank, prompt builder, pipeline control.")

        # ── TAB 1: Pipeline Control ───────────────────────────────────────────
        with gr.Tab("Pipeline Control"):
            gr.Markdown("### View biome characters and trigger pipeline stages")

            with gr.Row():
                biome_dd = gr.Dropdown(choices=biome_ids, value="claudetest002",
                                       label="Biome ID", interactive=True)
                refresh_btn = gr.Button("Refresh", size="sm")

            pipeline_table = gr.Dataframe(label="Characters Pipeline Status", interactive=False)
            char_details_json = gr.State({})

            def refresh_pipeline(biome_id):
                df, details = load_biome_characters(biome_id)
                return df, details

            refresh_btn.click(refresh_pipeline, [biome_dd], [pipeline_table, char_details_json])
            biome_dd.change(refresh_pipeline, [biome_dd], [pipeline_table, char_details_json])

            gr.Markdown("---\n#### Character Images")
            with gr.Row():
                char_name_input = gr.Textbox(label="Character Name", placeholder="cultivation_youth")
                load_imgs_btn = gr.Button("Load Images")

            with gr.Row():
                flux_img = gr.Image(label="Flux Concept (Stage 0)", type="filepath")
                s1_img = gr.Image(label="Stage 1 (Base)", type="filepath")
                s2_img = gr.Image(label="Stage 2 (Refined)", type="filepath")
                s3_img = gr.Image(label="Stage 3 (Multi-view)", type="filepath")

            img_summary = gr.Textbox(label="Image Metadata (JSON)", lines=10)

            load_imgs_btn.click(
                get_character_images,
                [biome_dd, char_name_input],
                [flux_img, s1_img, s2_img, s3_img, img_summary],
            )

            gr.Markdown("---\n#### Trigger Pipeline Stage")
            gr.Markdown("*Generates the command to run on GPU. Copy & paste to terminal or run via SSH.*")

            with gr.Row():
                stage_dd = gr.Dropdown(
                    choices=["flux", "stage1_2", "stage3"],
                    value="flux", label="Stage to Run",
                )
                trigger_btn = gr.Button("Generate Command", variant="primary")

            trigger_output = gr.Textbox(label="GPU Command", lines=4)
            trigger_btn.click(trigger_gpu_command, [biome_dd, char_name_input, stage_dd], [trigger_output])

        # ── TAB 2: Character Specs ────────────────────────────────────────────
        with gr.Tab("Character Specs"):
            gr.Markdown("### Structured character data — body, face, hair, clothing")

            with gr.Row():
                spec_dd = gr.Dropdown(choices=spec_ids, label="Character Spec", interactive=True)
                spec_refresh = gr.Button("Refresh List", size="sm")

            with gr.Row():
                with gr.Column(scale=2):
                    spec_json = gr.Textbox(label="Full Spec (JSON — editable)", lines=25, interactive=True)
                with gr.Column(scale=1):
                    spec_body = gr.Textbox(label="Body Description (auto)", lines=2, interactive=False)
                    spec_face = gr.Textbox(label="Face Description (auto)", lines=2, interactive=False)
                    spec_clothing = gr.Textbox(label="Clothing Description (auto)", lines=2, interactive=False)
                    spec_notes = gr.Textbox(label="Notes", lines=3, interactive=False)

            spec_save_btn = gr.Button("Save Spec (from JSON)", variant="primary")
            spec_save_msg = gr.Textbox(label="Status", interactive=False)

            def on_spec_select(char_id):
                return load_spec_detail(char_id)

            def on_spec_refresh():
                ids = list_spec_ids()
                return gr.update(choices=ids)

            spec_dd.change(on_spec_select, [spec_dd], [spec_json, spec_body, spec_face, spec_clothing, spec_notes])
            spec_refresh.click(on_spec_refresh, [], [spec_dd])
            spec_save_btn.click(save_spec_from_json, [spec_json], [spec_save_msg])

        # ── TAB 3: Style Library ──────────────────────────────────────────────
        with gr.Tab("Style Library"):
            gr.Markdown("### Reusable style-DNA blocks\n"
                         "**Rule:** Use the EXACT same style block in EVERY generation for consistency.")

            with gr.Row():
                style_dd = gr.Dropdown(choices=style_ids, label="Style", interactive=True)
                style_refresh = gr.Button("Refresh", size="sm")

            style_json = gr.Textbox(label="Style JSON (editable)", lines=12, interactive=True)
            with gr.Row():
                style_pos = gr.Textbox(label="Positive Tags", lines=3, interactive=False)
                style_neg = gr.Textbox(label="Negative Tags", lines=3, interactive=False)

            style_save_btn = gr.Button("Save Style (from JSON)", variant="primary")
            style_save_msg = gr.Textbox(label="Status", interactive=False)

            def on_style_select(sid):
                return load_style_detail(sid)

            def on_style_refresh():
                return gr.update(choices=list_style_ids())

            style_dd.change(on_style_select, [style_dd], [style_json, style_pos, style_neg])
            style_refresh.click(on_style_refresh, [], [style_dd])
            style_save_btn.click(save_style_from_json, [style_json], [style_save_msg])

        # ── TAB 4: Reference Bank ─────────────────────────────────────────────
        with gr.Tab("Reference Image Bank"):
            gr.Markdown("### Grouped reference images — Humans, Fantastical Animals\n"
                         "Placeholders with S3 keys. Upload actual images to fill them.")

            with gr.Row():
                ref_group_dd = gr.Dropdown(choices=ref_groups, value="all",
                                           label="Group Filter", interactive=True)
                ref_search_box = gr.Textbox(label="Search (tags, description)", placeholder="lion side quadruped")
                ref_search_btn = gr.Button("Search", size="sm")

            ref_table = gr.Dataframe(label="Reference Images", interactive=False)

            ref_group_dd.change(load_references_by_group, [ref_group_dd], [ref_table])
            ref_search_btn.click(search_refs, [ref_search_box], [ref_table])

            gr.Markdown("---\n#### Reference Detail")
            ref_detail_id = gr.Textbox(label="Reference ID", placeholder="ref_feline_standing_side")
            ref_detail_btn = gr.Button("Load Detail")
            ref_detail_json = gr.Textbox(label="Reference JSON", lines=12, interactive=False)

            def load_ref_detail(ref_id):
                db = _db()
                ref = get_reference(db, ref_id)
                return _json_pretty(ref) if ref else "Not found"

            ref_detail_btn.click(load_ref_detail, [ref_detail_id], [ref_detail_json])

        # ── TAB 5: Prompt Templates ───────────────────────────────────────────
        with gr.Tab("Prompt Templates"):
            gr.Markdown("### Tested prompt templates with variable slots\n"
                         "Templates act like functions — reuse instead of rewriting.")

            with gr.Row():
                tpl_dd = gr.Dropdown(choices=template_ids, label="Template", interactive=True)
                tpl_refresh = gr.Button("Refresh", size="sm")

            tpl_json = gr.Textbox(label="Template JSON", lines=10, interactive=False)
            tpl_template = gr.Textbox(label="Template String", lines=4, interactive=False)
            tpl_negative = gr.Textbox(label="Negative Template", lines=2, interactive=False)
            tpl_vars = gr.Textbox(label="Variables", lines=1, interactive=False)

            def on_tpl_select(tid):
                return load_template_detail(tid)

            def on_tpl_refresh():
                return gr.update(choices=list_template_ids())

            tpl_dd.change(on_tpl_select, [tpl_dd], [tpl_json, tpl_template, tpl_negative, tpl_vars])
            tpl_refresh.click(on_tpl_refresh, [], [tpl_dd])

        # ── TAB 6: Prompt Builder ─────────────────────────────────────────────
        with gr.Tab("Prompt Builder"):
            gr.Markdown("### Assemble prompts from character spec + template + style\n"
                         "Preview the final prompt before sending to Flux or SD1.5.")

            with gr.Row():
                pb_char = gr.Dropdown(choices=spec_ids, label="Character Spec", interactive=True)
                pb_tpl = gr.Dropdown(choices=template_ids, label="Prompt Template", interactive=True)
                pb_style = gr.Dropdown(choices=[""] + style_ids, value="",
                                       label="Style Override (blank = use spec default)", interactive=True)

            pb_extra = gr.Textbox(label="Extra Tags (appended)", placeholder="glowing eyes, battle stance")
            pb_build_btn = gr.Button("Build Prompt", variant="primary")

            pb_prompt = gr.Textbox(label="Assembled Prompt", lines=5, interactive=False)
            pb_negative = gr.Textbox(label="Assembled Negative", lines=3, interactive=False)
            pb_info = gr.Textbox(label="Token Info", lines=1, interactive=False)

            pb_build_btn.click(
                build_prompt_preview,
                [pb_char, pb_tpl, pb_style, pb_extra],
                [pb_prompt, pb_negative, pb_info],
            )

        # ── TAB 7: MongoDB Viewer ─────────────────────────────────────────────
        with gr.Tab("MongoDB Viewer"):
            gr.Markdown("### Raw MongoDB document viewer\n"
                         "Browse biomes and any collection without needing Compass/shell.")

            with gr.Row():
                mv_biome_dd = gr.Dropdown(choices=biome_ids, label="Biome Document", interactive=True)
                mv_biome_btn = gr.Button("Load Biome")

            mv_biome_json = gr.Textbox(label="Biome JSON", lines=20, interactive=False)
            mv_biome_btn.click(load_biome_raw, [mv_biome_dd], [mv_biome_json])

            gr.Markdown("---\n#### Browse Any Collection")
            with gr.Row():
                mv_coll_dd = gr.Dropdown(choices=collections, label="Collection", interactive=True)
                mv_coll_limit = gr.Slider(1, 50, value=10, step=1, label="Limit")
                mv_coll_btn = gr.Button("Load")

            mv_coll_json = gr.Textbox(label="Collection Documents", lines=20, interactive=False)
            mv_coll_btn.click(load_collection_raw, [mv_coll_dd, mv_coll_limit], [mv_coll_json])

    return app


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spark Pipeline Dashboard")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7861, help="Port (default 7861)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Spark Pipeline Dashboard")
    print(f"  MongoDB: {MONGO_URI}")
    print(f"  Port:    {args.port}")
    print("=" * 60)

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()

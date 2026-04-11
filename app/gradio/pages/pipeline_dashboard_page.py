"""
pipeline_dashboard_page.py — Pipeline Dashboard tab for the main Gradio app
============================================================================
Mounts the character spec library, style DNA, reference bank, prompt builder,
and pipeline control as a tab inside the existing web_app.py on port 7860.
"""

import json
import os
import sys

import gradio as gr

# Ensure worker/lib is importable from project root
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_WORKER_DIR = os.path.join(_PROJECT_ROOT, "worker")
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from lib.spec_schema import (
    get_db,
    list_character_specs, get_character_spec, upsert_character_spec,
    list_styles, get_style, upsert_style,
    list_references, search_references, get_reference,
    list_prompt_templates, get_prompt_template,
    assemble_prompt, assemble_negative,
    build_body_desc, build_face_desc, build_clothing_desc, build_creature_desc,
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB", "World_builder")
S3_BASE   = "https://sparkassets-us.s3.us-east-1.amazonaws.com"


def _db():
    return get_db(MONGO_URI, MONGO_DB)


def _json_pretty(obj):
    from datetime import datetime
    def default(o):
        return str(o)
    return json.dumps(obj, indent=2, default=default, ensure_ascii=False)


# ── data loaders ──────────────────────────────────────────────────────────────

def _spec_ids():
    try:
        return [s["_id"] for s in list_character_specs(_db())]
    except Exception:
        return []

def _style_ids():
    try:
        return [s["_id"] for s in list_styles(_db())]
    except Exception:
        return []

def _template_ids():
    try:
        return [t["_id"] for t in list_prompt_templates(_db())]
    except Exception:
        return []

def _ref_groups():
    try:
        groups = _db().reference_image_bank.distinct("group")
        return ["all"] + sorted(groups)
    except Exception:
        return ["all"]

def _biome_ids():
    try:
        return [b["_id"] for b in _db().biomes.find({}, {"_id": 1}).sort("_id", 1)]
    except Exception:
        return ["claudetest002"]


# ── pipeline control ─────────────────────────────────────────────────────────

def load_pipeline_status(biome_id):
    try:
        db = _db()
        biome = db.biomes.find_one({"_id": biome_id})
        if not biome:
            return None, "Biome not found."
        chars = (biome.get("possible_structures") or {}).get("characters", {})
        rows = []
        for name, doc in chars.items():
            rows.append({
                "Character": name,
                "Category": doc.get("creature_category", doc.get("character_type", "?")),
                "Flux": (doc.get("flux_concept") or {}).get("status", "not_started"),
                "Stage1": (doc.get("stage1") or {}).get("status") or "not_started",
                "Stage2": (doc.get("stage2") or {}).get("status") or "not_started",
                "Stage3": (doc.get("stage3") or {}).get("status") or "not_started",
                "Overall": doc.get("status", "unknown"),
            })
        import pandas as pd
        return pd.DataFrame(rows) if rows else pd.DataFrame(), "OK"
    except Exception as e:
        return None, str(e)


def load_char_images(biome_id, char_name):
    try:
        db = _db()
        biome = db.biomes.find_one({"_id": biome_id})
        chars = (biome.get("possible_structures") or {}).get("characters", {})
        doc = chars.get(char_name, {})
        flux_url = (doc.get("flux_concept") or {}).get("image_url")
        s1_url   = (doc.get("stage1") or {}).get("image_url")
        s2_url   = (doc.get("stage2") or {}).get("image_url")
        summary  = _json_pretty({k: doc.get(k) for k in ("flux_concept","stage1","stage2","stage3","images")})
        return flux_url, s1_url, s2_url, summary
    except Exception as e:
        return None, None, None, str(e)


def gpu_command(biome_id, stage):
    cmds = {
        "flux":     "cd /home/ec2-user/worker && python flux_concept_generator.py",
        "stage1_2": "cd /home/ec2-user/worker && python run_sd15_direct.py",
        "stage3":   "cd /home/ec2-user/worker && python run_multiview_direct.py",
    }
    cmd = cmds.get(stage, "Unknown stage")
    return f"Run on GPU instance:\n\n  {cmd}"


# ── character specs ───────────────────────────────────────────────────────────

def load_spec(char_id):
    try:
        spec = get_character_spec(_db(), char_id)
        if not spec:
            return "Not found", "", "", ""
        is_creature = spec.get("category") in ("fantastical_animal", "quadruped", "creature")
        body = build_creature_desc(spec) if is_creature else build_body_desc(spec)
        clothing = build_clothing_desc(spec)
        return _json_pretty(spec), body, clothing, spec.get("notes", "")
    except Exception as e:
        return str(e), "", "", ""


def save_spec(json_str):
    try:
        spec = json.loads(json_str)
        if "_id" not in spec:
            return "Error: missing '_id'"
        upsert_character_spec(_db(), spec)
        return f"Saved: {spec['_id']}"
    except Exception as e:
        return f"Error: {e}"


# ── style library ─────────────────────────────────────────────────────────────

def load_style(style_id):
    try:
        s = get_style(_db(), style_id)
        if not s:
            return "Not found", "", ""
        return _json_pretty(s), s.get("positive_tags", ""), s.get("negative_tags", "")
    except Exception as e:
        return str(e), "", ""


def save_style(json_str):
    try:
        style = json.loads(json_str)
        if "_id" not in style:
            return "Error: missing '_id'"
        upsert_style(_db(), style)
        return f"Saved: {style['_id']}"
    except Exception as e:
        return f"Error: {e}"


# ── reference bank ────────────────────────────────────────────────────────────

def load_refs(group):
    try:
        db = _db()
        refs = list_references(db) if group == "all" else list_references(db, group=group)
        rows = [{"ID": r["_id"], "Group": r.get("group",""), "Pose": r.get("pose",""),
                 "View": r.get("view",""), "Description": r.get("description",""),
                 "Source": r.get("metadata",{}).get("source","")} for r in refs]
        import pandas as pd
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        return str(e)


def search_refs(query):
    try:
        refs = search_references(_db(), query)
        rows = [{"ID": r["_id"], "Group": r.get("group",""), "Pose": r.get("pose",""),
                 "View": r.get("view",""), "Tags": ", ".join(r.get("tags",[]))} for r in refs]
        import pandas as pd
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        return str(e)


# ── prompt builder ────────────────────────────────────────────────────────────

def build_preview(char_id, tpl_id, style_override, extra):
    try:
        db = _db()
        prompt = assemble_prompt(db, char_id, tpl_id,
                                 style_override=style_override or None,
                                 extra_tags=extra or "")
        neg    = assemble_negative(db, char_id, tpl_id,
                                   style_override=style_override or None)
        words  = len(prompt.split())
        clip   = int(words * 1.3)
        info   = f"Words: {words} | Est. CLIP tokens: {clip}"
        if clip > 77:
            info += "  ⚠ Exceeds 77-token SD1.5 limit"
        return prompt, neg, info
    except Exception as e:
        return f"Error: {e}", "", ""


# ── mongodb viewer ────────────────────────────────────────────────────────────

def load_biome_raw(biome_id):
    try:
        doc = _db().biomes.find_one({"_id": biome_id})
        return _json_pretty(doc) if doc else "Not found"
    except Exception as e:
        return str(e)


def load_collection(coll_name, limit):
    try:
        docs = list(_db()[coll_name].find().limit(int(limit)))
        return _json_pretty(docs)
    except Exception as e:
        return str(e)


def list_colls():
    try:
        return sorted(_db().list_collection_names())
    except Exception:
        return []


# ── main UI builder ───────────────────────────────────────────────────────────

def pipeline_dashboard_ui():
    """Mounts the full pipeline dashboard as a Gradio Blocks component."""

    spec_ids     = _spec_ids()
    style_ids    = _style_ids()
    tpl_ids      = _template_ids()
    ref_groups   = _ref_groups()
    biome_ids    = _biome_ids()
    collections  = list_colls()

    with gr.Blocks() as tab:
        gr.Markdown("## Character Pipeline Dashboard\n"
                    "Manage character specs, styles, references, prompts, and trigger generation stages.")

        # ── Pipeline Control ──────────────────────────────────────────────────
        with gr.Accordion("Pipeline Control", open=True):
            with gr.Row():
                biome_dd = gr.Dropdown(choices=biome_ids, value="claudetest002",
                                       label="Biome", interactive=True)
                refresh_btn = gr.Button("Refresh", size="sm")

            pipeline_table = gr.Dataframe(label="Pipeline Status", interactive=False)
            status_msg = gr.Textbox(label="", interactive=False, visible=False)

            with gr.Row():
                char_input = gr.Textbox(label="Character name", placeholder="cultivation_youth")
                load_img_btn = gr.Button("Load Images")

            with gr.Row():
                flux_img = gr.Image(label="Flux (Stage 0)")
                s1_img   = gr.Image(label="Stage 1")
                s2_img   = gr.Image(label="Stage 2")

            img_json = gr.Textbox(label="Image Metadata", lines=6, interactive=False)

            with gr.Row():
                stage_dd  = gr.Dropdown(choices=["flux", "stage1_2", "stage3"],
                                         value="flux", label="Stage")
                cmd_btn   = gr.Button("Get GPU Command", variant="primary")
            cmd_out = gr.Textbox(label="Command to run on GPU", lines=3, interactive=False)

            def _refresh(bid):
                df, msg = load_pipeline_status(bid)
                return df

            refresh_btn.click(_refresh, [biome_dd], [pipeline_table])
            biome_dd.change(_refresh, [biome_dd], [pipeline_table])
            load_img_btn.click(load_char_images, [biome_dd, char_input],
                               [flux_img, s1_img, s2_img, img_json])
            cmd_btn.click(gpu_command, [biome_dd, stage_dd], [cmd_out])

        # ── Character Specs ───────────────────────────────────────────────────
        with gr.Accordion("Character Specs", open=False):
            with gr.Row():
                spec_dd      = gr.Dropdown(choices=spec_ids, label="Spec", interactive=True)
                spec_refresh = gr.Button("↻", size="sm")

            with gr.Row():
                with gr.Column(scale=2):
                    spec_json = gr.Textbox(label="JSON (editable)", lines=20, interactive=True)
                with gr.Column(scale=1):
                    spec_body     = gr.Textbox(label="Body (auto)", lines=2, interactive=False)
                    spec_clothing = gr.Textbox(label="Clothing (auto)", lines=2, interactive=False)
                    spec_notes    = gr.Textbox(label="Notes", lines=3, interactive=False)

            spec_save_btn = gr.Button("Save Spec")
            spec_save_out = gr.Textbox(label="", interactive=False)

            spec_dd.change(load_spec, [spec_dd], [spec_json, spec_body, spec_clothing, spec_notes])
            spec_refresh.click(lambda: gr.update(choices=_spec_ids()), [], [spec_dd])
            spec_save_btn.click(save_spec, [spec_json], [spec_save_out])

        # ── Style Library ─────────────────────────────────────────────────────
        with gr.Accordion("Style Library", open=False):
            gr.Markdown("*Same style block in every generation = consistency.*")
            with gr.Row():
                style_dd      = gr.Dropdown(choices=style_ids, label="Style", interactive=True)
                style_refresh = gr.Button("↻", size="sm")

            style_json = gr.Textbox(label="JSON (editable)", lines=10, interactive=True)
            with gr.Row():
                style_pos = gr.Textbox(label="Positive tags", lines=2, interactive=False)
                style_neg = gr.Textbox(label="Negative tags", lines=2, interactive=False)

            style_save_btn = gr.Button("Save Style")
            style_save_out = gr.Textbox(label="", interactive=False)

            style_dd.change(load_style, [style_dd], [style_json, style_pos, style_neg])
            style_refresh.click(lambda: gr.update(choices=_style_ids()), [], [style_dd])
            style_save_btn.click(save_style, [style_json], [style_save_out])

        # ── Reference Image Bank ──────────────────────────────────────────────
        with gr.Accordion("Reference Image Bank", open=False):
            gr.Markdown("*Placeholder S3 keys — upload real images to fill them.*")
            with gr.Row():
                ref_group_dd  = gr.Dropdown(choices=ref_groups, value="all",
                                             label="Group", interactive=True)
                ref_search    = gr.Textbox(label="Search tags/description", placeholder="lion side quadruped")
                ref_search_btn = gr.Button("Search", size="sm")

            ref_table = gr.Dataframe(label="References", interactive=False)

            ref_group_dd.change(load_refs, [ref_group_dd], [ref_table])
            ref_search_btn.click(search_refs, [ref_search], [ref_table])

        # ── Prompt Builder ────────────────────────────────────────────────────
        with gr.Accordion("Prompt Builder", open=False):
            gr.Markdown("*Assemble prompts from spec + template + style — no hand-writing.*")
            with gr.Row():
                pb_char  = gr.Dropdown(choices=spec_ids,              label="Character Spec",  interactive=True)
                pb_tpl   = gr.Dropdown(choices=tpl_ids,               label="Template",         interactive=True)
                pb_style = gr.Dropdown(choices=[""] + style_ids,      label="Style override",   interactive=True)

            pb_extra = gr.Textbox(label="Extra tags", placeholder="battle stance, glowing eyes")
            pb_btn   = gr.Button("Build Prompt", variant="primary")

            pb_prompt = gr.Textbox(label="Prompt",    lines=4, interactive=False)
            pb_neg    = gr.Textbox(label="Negative",  lines=2, interactive=False)
            pb_info   = gr.Textbox(label="Token info",lines=1, interactive=False)

            pb_btn.click(build_preview, [pb_char, pb_tpl, pb_style, pb_extra],
                         [pb_prompt, pb_neg, pb_info])

        # ── MongoDB Viewer ────────────────────────────────────────────────────
        with gr.Accordion("MongoDB Viewer", open=False):
            with gr.Row():
                mv_biome_dd  = gr.Dropdown(choices=biome_ids, label="Biome", interactive=True)
                mv_biome_btn = gr.Button("Load")
            mv_biome_out = gr.Textbox(label="Biome JSON", lines=15, interactive=False)
            mv_biome_btn.click(load_biome_raw, [mv_biome_dd], [mv_biome_out])

            with gr.Row():
                mv_coll_dd    = gr.Dropdown(choices=collections, label="Collection", interactive=True)
                mv_coll_limit = gr.Slider(1, 50, value=10, step=1, label="Limit")
                mv_coll_btn   = gr.Button("Load")
            mv_coll_out = gr.Textbox(label="Documents", lines=15, interactive=False)
            mv_coll_btn.click(load_collection, [mv_coll_dd, mv_coll_limit], [mv_coll_out])

    return tab

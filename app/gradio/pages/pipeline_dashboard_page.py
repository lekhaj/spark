"""
pipeline_dashboard_page.py — Pipeline Dashboard tab (port 7860)
================================================================
Fully editable: character specs, styles, prompt templates, references.
All changes write back to MongoDB on Save.
"""

import json
import os
import sys

import gradio as gr

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_WORKER_DIR   = os.path.join(_PROJECT_ROOT, "worker")
if _WORKER_DIR not in sys.path:
    sys.path.insert(0, _WORKER_DIR)

from lib.spec_schema import (
    get_db,
    list_character_specs, get_character_spec, upsert_character_spec,
    list_styles,          get_style,           upsert_style,
    list_references,      search_references,   get_reference, upsert_reference,
    list_prompt_templates, get_prompt_template, upsert_prompt_template,
    assemble_prompt, assemble_negative,
    build_body_desc, build_face_desc, build_clothing_desc, build_creature_desc,
)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://kartik:Kartikg421@localhost:27017")
MONGO_DB  = os.getenv("MONGO_DB",  "World_builder")
S3_BASE   = "https://sparkassets-us.s3.us-east-1.amazonaws.com"


def _db():
    return get_db(MONGO_URI, MONGO_DB)


def _json_pretty(obj):
    def _default(o): return str(o)
    return json.dumps(obj, indent=2, default=_default, ensure_ascii=False)


# ── dropdown helpers ───────────────────────────────────────────────────────────

def _spec_ids():
    try:   return [s["_id"] for s in list_character_specs(_db())]
    except: return []

def _style_ids():
    try:   return [s["_id"] for s in list_styles(_db())]
    except: return []

def _tpl_ids():
    try:   return [t["_id"] for t in list_prompt_templates(_db())]
    except: return []

def _ref_ids():
    try:   return [r["_id"] for r in list_references(_db())]
    except: return []

def _ref_groups():
    try:
        groups = _db().reference_image_bank.distinct("group")
        return ["all"] + sorted(groups)
    except: return ["all"]

def _biome_ids():
    try:   return [b["_id"] for b in _db().biomes.find({}, {"_id": 1}).sort("_id", 1)]
    except: return ["claudetest002"]


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def load_pipeline_status(biome_id):
    try:
        db = _db()
        biome = db.biomes.find_one({"_id": biome_id})
        if not biome:
            return None, "Biome not found."
        chars = (biome.get("possible_structures") or {}).get("characters", {})
        rows  = []
        for name, doc in chars.items():
            rows.append({
                "Character": name,
                "Category":  doc.get("creature_category", doc.get("character_type", "?")),
                "Flux":      (doc.get("flux_concept") or {}).get("status", "not_started"),
                "Stage1":    (doc.get("stage1") or {}).get("status") or "not_started",
                "Stage2":    (doc.get("stage2") or {}).get("status") or "not_started",
                "Stage3":    (doc.get("stage3") or {}).get("status") or "not_started",
                "Overall":   doc.get("status", "unknown"),
            })
        import pandas as pd
        return pd.DataFrame(rows) if rows else pd.DataFrame(), "OK"
    except Exception as e:
        return None, str(e)


def load_char_images(biome_id, char_name):
    try:
        db    = _db()
        biome = db.biomes.find_one({"_id": biome_id})
        chars = (biome.get("possible_structures") or {}).get("characters", {})
        doc   = chars.get(char_name, {})
        flux_url = (doc.get("flux_concept") or {}).get("image_url")
        s1_url   = (doc.get("stage1")        or {}).get("image_url")
        s2_url   = (doc.get("stage2")        or {}).get("image_url")
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
    return f"Run on GPU instance:\n\n  {cmds.get(stage, 'Unknown stage')}"


# ══════════════════════════════════════════════════════════════════════════════
#  CHARACTER SPECS — load + save
# ══════════════════════════════════════════════════════════════════════════════

def load_spec(char_id):
    """Load spec → populate both Quick-Edit fields and raw JSON."""
    try:
        spec = get_character_spec(_db(), char_id)
        if not spec:
            return "Not found", "", "", "", "", ""
        is_creature = spec.get("category") in ("fantastical_animal", "quadruped", "creature")
        body     = build_creature_desc(spec) if is_creature else build_body_desc(spec)
        clothing = build_clothing_desc(spec)
        return (
            _json_pretty(spec),
            spec.get("display_name", ""),
            spec.get("notes", ""),
            spec.get("lore", ""),
            body,
            clothing,
        )
    except Exception as e:
        return str(e), "", "", "", "", ""


def save_spec_json(json_str):
    """Save spec from raw JSON textarea."""
    try:
        spec = json.loads(json_str)
        if "_id" not in spec:
            return "❌ Missing '_id' field in JSON"
        upsert_character_spec(_db(), spec)
        return f"✓ Saved: {spec['_id']}"
    except json.JSONDecodeError as e:
        return f"❌ Invalid JSON: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


def save_spec_quick(char_id, display_name, notes, lore):
    """Update only quick-edit fields without touching the rest of the spec."""
    try:
        db   = _db()
        spec = get_character_spec(db, char_id)
        if not spec:
            return f"❌ Spec '{char_id}' not found"
        spec["display_name"] = display_name
        spec["notes"]        = notes
        spec["lore"]         = lore
        upsert_character_spec(db, spec)
        return f"✓ Saved quick fields for: {char_id}"
    except Exception as e:
        return f"❌ Error: {e}"


def new_spec_template():
    """Return a blank spec template JSON for creating a new character."""
    template = {
        "_id": "new_character",
        "display_name": "New Character",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": [],
        "body":      {"gender": "male", "build": "athletic", "age_range": "20-25", "height": "175cm"},
        "face":      {"expression": "calm", "eyes": "dark eyes"},
        "hair":      {"style": "short black hair", "color": "black", "accessories": []},
        "clothing":  {"primary": "gray robe", "details": [], "accessories": []},
        "pose_default": "T-pose",
        "view_default": "front",
        "flux_generation": {"width": 768, "height": 1024, "steps": 4, "guidance_scale": 0.0},
        "notes": "",
        "lore": "",
    }
    return _json_pretty(template), "", "", "", "", ""


# ══════════════════════════════════════════════════════════════════════════════
#  STYLE LIBRARY — load + save (field-based, no JSON needed)
# ══════════════════════════════════════════════════════════════════════════════

def load_style_fields(style_id):
    """Load style into individual editable fields."""
    try:
        s = get_style(_db(), style_id)
        if not s:
            return "", "", "", ""
        return (
            s.get("name", ""),
            s.get("positive_tags", ""),
            s.get("negative_tags", ""),
            s.get("description", ""),
        )
    except Exception as e:
        return str(e), "", "", ""


def save_style_fields(style_id, name, pos_tags, neg_tags, description):
    """Save style from individual fields directly to MongoDB."""
    try:
        db       = _db()
        existing = get_style(db, style_id) or {"_id": style_id}
        existing.update({
            "_id":           style_id,
            "name":          name or style_id,
            "positive_tags": pos_tags.strip(),
            "negative_tags": neg_tags.strip(),
            "description":   description.strip(),
        })
        upsert_style(db, existing)
        return f"✓ Saved style: {style_id}"
    except Exception as e:
        return f"❌ Error: {e}"


def new_style_template():
    """Blank style fields for a new style."""
    return (
        "new_style",
        "New Style",
        "semi-realistic 3D render, game character design, PBR materials, clean topology, game-ready asset, soft studio lighting",
        "anime, 2D illustration, cel shading, cartoon, sketch, noisy, watermark",
        "New style description",
        "",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT TEMPLATES — load + save (field-based)
# ══════════════════════════════════════════════════════════════════════════════

def load_template(tpl_id):
    """Load template into editable fields."""
    try:
        tpl = get_prompt_template(_db(), tpl_id)
        if not tpl:
            return "", "", "", "", ""
        vars_hint = ", ".join(f"{{{v}}}" for v in tpl.get("variables", []))
        return (
            tpl.get("template", ""),
            tpl.get("negative_template", ""),
            tpl.get("notes", ""),
            tpl.get("stage", ""),
            vars_hint,
        )
    except Exception as e:
        return str(e), "", "", "", ""


def save_template_fields(tpl_id, template_str, negative_str, notes):
    """Save template string + negative + notes back to MongoDB."""
    try:
        db       = _db()
        existing = get_prompt_template(db, tpl_id)
        if not existing:
            return f"❌ Template '{tpl_id}' not found. Load it first."
        existing["template"]          = template_str.strip()
        existing["negative_template"] = negative_str.strip()
        existing["notes"]             = notes.strip()
        upsert_prompt_template(db, existing)
        return f"✓ Saved template: {tpl_id}"
    except Exception as e:
        return f"❌ Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  REFERENCE IMAGE BANK — load + save
# ══════════════════════════════════════════════════════════════════════════════

def load_refs(group):
    try:
        db   = _db()
        refs = list_references(db) if group == "all" else list_references(db, group=group)
        rows = [{"ID": r["_id"], "Group": r.get("group",""), "Subgroup": r.get("subgroup",""),
                 "Pose": r.get("pose",""), "View": r.get("view",""),
                 "Description": r.get("description","")[:80],
                 "Source": r.get("metadata",{}).get("source","")} for r in refs]
        import pandas as pd
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        return str(e)


def search_refs(query):
    try:
        refs = search_references(_db(), query)
        rows = [{"ID": r["_id"], "Group": r.get("group",""),
                 "Pose": r.get("pose",""), "View": r.get("view",""),
                 "Tags": ", ".join(r.get("tags",[]))} for r in refs]
        import pandas as pd
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception as e:
        return str(e)


def load_ref_edit(ref_id):
    """Load a reference entry for editing."""
    try:
        ref = get_reference(_db(), ref_id)
        if not ref:
            return "", "", "", "", ""
        return (
            ref.get("description", ""),
            ", ".join(ref.get("tags", [])),
            ref.get("usage_notes", ""),
            ref.get("s3_key", ""),
            _json_pretty(ref.get("metadata", {})),
        )
    except Exception as e:
        return str(e), "", "", "", ""


def save_ref_edit(ref_id, description, tags_str, usage_notes, s3_key):
    """Save reference edits back to MongoDB."""
    try:
        db       = _db()
        existing = get_reference(db, ref_id)
        if not existing:
            return f"❌ Reference '{ref_id}' not found."
        existing["description"] = description.strip()
        existing["tags"]        = [t.strip() for t in tags_str.split(",") if t.strip()]
        existing["usage_notes"] = usage_notes.strip()
        if s3_key.strip():
            existing["s3_key"] = s3_key.strip()
            existing["s3_url"] = f"{S3_BASE}/{s3_key.strip()}"
        upsert_reference(db, existing)
        return f"✓ Saved reference: {ref_id}"
    except Exception as e:
        return f"❌ Error: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_preview(char_id, tpl_id, style_override, extra):
    try:
        db     = _db()
        prompt = assemble_prompt(db, char_id, tpl_id,
                                 style_override=style_override or None,
                                 extra_tags=extra or "")
        neg    = assemble_negative(db, char_id, tpl_id,
                                   style_override=style_override or None)
        words  = len(prompt.split())
        clip   = int(words * 1.3)
        is_sd  = "stage" in tpl_id
        info   = f"Words: {words} | Est. CLIP tokens: ~{clip}"
        if is_sd and clip > 77:
            info += f"  ⚠️  SD1.5 limit is 77 — trim {clip - 77} tokens"
        elif is_sd:
            info += "  ✓ Within SD1.5 CLIP limit"
        else:
            info += "  [Flux T5 — no token limit]"
        return prompt, neg, info
    except Exception as e:
        return f"Error: {e}", "", ""


# ══════════════════════════════════════════════════════════════════════════════
#  MONGODB VIEWER
# ══════════════════════════════════════════════════════════════════════════════

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
    try:   return sorted(_db().list_collection_names())
    except: return []


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN UI
# ══════════════════════════════════════════════════════════════════════════════

def pipeline_dashboard_ui():
    """Full pipeline dashboard as a Gradio Blocks component."""

    spec_ids    = _spec_ids()
    style_ids   = _style_ids()
    tpl_ids     = _tpl_ids()
    ref_ids     = _ref_ids()
    ref_groups  = _ref_groups()
    biome_ids   = _biome_ids()
    collections = list_colls()

    _first = lambda lst: lst[0] if lst else None

    with gr.Blocks() as tab:
        gr.Markdown("## Character Pipeline Dashboard\n"
                    "Manage specs · styles · templates · references · trigger generation")

        # ══════════════════════════════════════════════════════════════════════
        #  PIPELINE CONTROL
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🚀 Pipeline Control", open=True):
            with gr.Row():
                biome_dd    = gr.Dropdown(choices=biome_ids, value=_first(biome_ids),
                                          label="Biome", interactive=True)
                refresh_btn = gr.Button("↻ Refresh", size="sm")

            pipeline_table = gr.Dataframe(label="Pipeline Status", interactive=False)

            with gr.Row():
                char_input   = gr.Textbox(label="Character name", placeholder="cultivation_youth")
                load_img_btn = gr.Button("Load Images", size="sm")

            with gr.Row():
                flux_img = gr.Image(label="Flux (Stage 0)")
                s1_img   = gr.Image(label="Stage 1")
                s2_img   = gr.Image(label="Stage 2")

            img_json = gr.Textbox(label="Image Metadata", lines=5, interactive=False)

            with gr.Row():
                stage_dd = gr.Dropdown(choices=["flux","stage1_2","stage3"],
                                       value="flux", label="Stage", interactive=True)
                cmd_btn  = gr.Button("Get GPU Command", variant="primary", size="sm")
            cmd_out = gr.Textbox(label="Command (run on GPU instance)", lines=2, interactive=False)

            def _refresh(bid):
                df, _ = load_pipeline_status(bid)
                return df

            refresh_btn.click(_refresh,          [biome_dd],                   [pipeline_table])
            biome_dd.change(_refresh,             [biome_dd],                   [pipeline_table])
            load_img_btn.click(load_char_images,  [biome_dd, char_input],       [flux_img, s1_img, s2_img, img_json])
            cmd_btn.click(gpu_command,            [biome_dd, stage_dd],         [cmd_out])

        # ══════════════════════════════════════════════════════════════════════
        #  CHARACTER SPECS — two tabs: Quick Edit + Raw JSON
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🧩 Character Specs", open=False):
            with gr.Row():
                spec_dd      = gr.Dropdown(choices=spec_ids, value=_first(spec_ids),
                                           label="Spec ID", interactive=True)
                spec_refresh = gr.Button("↻", size="sm")
                spec_new_btn = gr.Button("+ New Template", size="sm")

            spec_save_out = gr.Textbox(label="", interactive=False, lines=1)

            with gr.Tabs():
                # ── Quick Edit tab ──────────────────────────────────────
                with gr.TabItem("✏️ Quick Edit"):
                    gr.Markdown("*Edit display name, notes and lore — leaves body/face/clothing JSON intact.*")
                    with gr.Row():
                        with gr.Column():
                            sq_name  = gr.Textbox(label="Display Name", interactive=True)
                            sq_notes = gr.Textbox(label="Notes", lines=3, interactive=True)
                            sq_lore  = gr.Textbox(label="Lore / Backstory", lines=3, interactive=True)
                            sq_save  = gr.Button("Save Quick Fields", variant="primary")
                        with gr.Column():
                            sq_body     = gr.Textbox(label="Body (auto-parsed)", lines=2, interactive=False)
                            sq_clothing = gr.Textbox(label="Clothing (auto-parsed)", lines=3, interactive=False)
                            gr.Markdown("*Body/clothing shown for reference. Edit in Raw JSON tab to change.*")

                    sq_save.click(save_spec_quick,
                                  [spec_dd, sq_name, sq_notes, sq_lore],
                                  [spec_save_out])

                # ── Raw JSON tab ────────────────────────────────────────
                with gr.TabItem("📄 Raw JSON"):
                    gr.Markdown("*Edit full JSON structure — all fields including body, face, hair, clothing.*")
                    spec_json     = gr.Textbox(label="Full Spec JSON", lines=28, interactive=True)
                    spec_json_btn = gr.Button("Save Full JSON", variant="primary")
                    spec_json_btn.click(save_spec_json, [spec_json], [spec_save_out])

            # wire dropdown → load all outputs
            def _load_spec(sid):
                raw, name, notes, lore, body, clothing = load_spec(sid)
                return raw, name, notes, lore, body, clothing

            spec_dd.change(_load_spec, [spec_dd],
                           [spec_json, sq_name, sq_notes, sq_lore, sq_body, sq_clothing])
            spec_refresh.click(lambda: gr.update(choices=_spec_ids()), [], [spec_dd])
            spec_new_btn.click(new_spec_template, [],
                               [spec_json, sq_name, sq_notes, sq_lore, sq_body, sq_clothing])

        # ══════════════════════════════════════════════════════════════════════
        #  STYLE LIBRARY — fully field-based (no JSON)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🎨 Style Library", open=False):
            gr.Markdown("*Lock one style block and use it in every generation for consistency.*")
            with gr.Row():
                style_dd      = gr.Dropdown(choices=style_ids, value=_first(style_ids),
                                            label="Style ID", interactive=True)
                style_refresh = gr.Button("↻", size="sm")
                style_new_btn = gr.Button("+ New Style", size="sm")

            style_id_edit  = gr.Textbox(label="Style ID (only for new styles)", interactive=True, visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    st_name = gr.Textbox(label="Name", interactive=True)
                    st_desc = gr.Textbox(label="Description", lines=2, interactive=True)
                with gr.Column(scale=2):
                    st_pos  = gr.Textbox(label="Positive tags  ← edit directly here",
                                         lines=4, interactive=True)
                    st_neg  = gr.Textbox(label="Negative tags  ← edit directly here",
                                         lines=3, interactive=True)

            gr.Markdown("💡 *Tip: Keep SD1.5 style tags short (< 15 words total). Flux has no limit.*")

            with gr.Row():
                st_save = gr.Button("💾 Save Style", variant="primary")
                st_out  = gr.Textbox(label="", interactive=False, scale=2)

            def _load_style(sid):
                name, pos, neg, desc = load_style_fields(sid)
                return name, pos, neg, desc

            def _save_style(sid, name, pos, neg, desc):
                return save_style_fields(sid, name, pos, neg, desc)

            style_dd.change(_load_style,  [style_dd],                              [st_name, st_pos, st_neg, st_desc])
            style_refresh.click(lambda: gr.update(choices=_style_ids()), [],       [style_dd])
            style_new_btn.click(lambda: gr.update(visible=True),         [],       [style_id_edit])
            st_save.click(_save_style, [style_dd, st_name, st_pos, st_neg, st_desc], [st_out])

        # ══════════════════════════════════════════════════════════════════════
        #  PROMPT TEMPLATES — editable template strings
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("📝 Prompt Templates", open=False):
            gr.Markdown("*Edit the template strings directly. Variables like `{body_desc}` are filled at build time.*")
            with gr.Row():
                tpl_dd      = gr.Dropdown(choices=tpl_ids, value=_first(tpl_ids),
                                          label="Template ID", interactive=True)
                tpl_refresh = gr.Button("↻", size="sm")

            with gr.Row():
                tpl_stage    = gr.Textbox(label="Stage", interactive=False, scale=1)
                tpl_vars     = gr.Textbox(label="Available variables", interactive=False, scale=3)

            tpl_template = gr.Textbox(label="Template  ← edit here", lines=6, interactive=True)
            tpl_negative = gr.Textbox(label="Negative template  ← edit here", lines=3, interactive=True)
            tpl_notes    = gr.Textbox(label="Notes  ← edit here", lines=2, interactive=True)

            gr.Markdown(
                "**All available `{variable}` slots:** "
                "`{body_desc}` `{face_desc}` `{hair_desc}` `{clothing_desc}` "
                "`{creature_desc}` `{style_tags}` `{extra_tags}` `{char_name}` `{theme}`"
            )

            with gr.Row():
                tpl_save = gr.Button("💾 Save Template", variant="primary")
                tpl_out  = gr.Textbox(label="", interactive=False, scale=2)

            def _load_tpl(tid):
                tmpl, neg, notes, stage, vars_hint = load_template(tid)
                return tmpl, neg, notes, stage, vars_hint

            tpl_dd.change(_load_tpl, [tpl_dd], [tpl_template, tpl_negative, tpl_notes, tpl_stage, tpl_vars])
            tpl_refresh.click(lambda: gr.update(choices=_tpl_ids()), [], [tpl_dd])
            tpl_save.click(save_template_fields,
                           [tpl_dd, tpl_template, tpl_negative, tpl_notes],
                           [tpl_out])

        # ══════════════════════════════════════════════════════════════════════
        #  REFERENCE IMAGE BANK
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🖼️ Reference Image Bank", open=False):
            with gr.Row():
                ref_group_dd   = gr.Dropdown(choices=ref_groups, value="all",
                                              label="Filter by group", interactive=True)
                ref_search_box = gr.Textbox(label="Search tags / description",
                                            placeholder="lion side quadruped")
                ref_search_btn = gr.Button("Search", size="sm")

            ref_table = gr.Dataframe(label="References", interactive=False)

            ref_group_dd.change(load_refs,    [ref_group_dd], [ref_table])
            ref_search_btn.click(search_refs, [ref_search_box], [ref_table])

            gr.Markdown("---\n**Edit a reference entry:**")
            with gr.Row():
                ref_edit_dd      = gr.Dropdown(choices=ref_ids, value=_first(ref_ids),
                                               label="Reference ID", interactive=True)
                ref_edit_load    = gr.Button("Load", size="sm")
                ref_edit_refresh = gr.Button("↻", size="sm")

            with gr.Row():
                with gr.Column():
                    ref_desc  = gr.Textbox(label="Description", lines=3, interactive=True)
                    ref_tags  = gr.Textbox(label="Tags (comma-separated)", interactive=True)
                with gr.Column():
                    ref_usage = gr.Textbox(label="Usage notes", lines=2, interactive=True)
                    ref_s3    = gr.Textbox(label="S3 key (path after bucket)", interactive=True)

            with gr.Row():
                ref_save = gr.Button("💾 Save Reference", variant="primary")
                ref_out  = gr.Textbox(label="", interactive=False, scale=2)

            def _load_ref(rid):
                desc, tags, usage, s3, _meta = load_ref_edit(rid)
                return desc, tags, usage, s3

            ref_edit_load.click(_load_ref, [ref_edit_dd],
                                [ref_desc, ref_tags, ref_usage, ref_s3])
            ref_edit_dd.change(_load_ref, [ref_edit_dd],
                               [ref_desc, ref_tags, ref_usage, ref_s3])
            ref_edit_refresh.click(lambda: gr.update(choices=_ref_ids()), [], [ref_edit_dd])
            ref_save.click(save_ref_edit,
                           [ref_edit_dd, ref_desc, ref_tags, ref_usage, ref_s3],
                           [ref_out])

        # ══════════════════════════════════════════════════════════════════════
        #  PROMPT BUILDER — assemble + preview
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🔧 Prompt Builder", open=False):
            gr.Markdown("*Select character + template → assemble prompt automatically from spec data.*")
            with gr.Row():
                pb_char  = gr.Dropdown(choices=spec_ids,         value=_first(spec_ids),  label="Character Spec",  interactive=True)
                pb_tpl   = gr.Dropdown(choices=tpl_ids,          value=_first(tpl_ids),   label="Template",        interactive=True)
                pb_style = gr.Dropdown(choices=[""] + style_ids, value="",                label="Style override",  interactive=True)

            pb_extra = gr.Textbox(label="Extra tags (appended to prompt)",
                                  placeholder="battle stance, glowing sword, rain")
            pb_btn   = gr.Button("🔨 Build Prompt", variant="primary")

            pb_prompt = gr.Textbox(label="Assembled Prompt (copy → paste into Flux job)", lines=6, interactive=False)
            pb_neg    = gr.Textbox(label="Negative Prompt", lines=3, interactive=False)
            pb_info   = gr.Textbox(label="Token count", lines=1, interactive=False)

            pb_btn.click(build_preview,
                         [pb_char, pb_tpl, pb_style, pb_extra],
                         [pb_prompt, pb_neg, pb_info])

        # ══════════════════════════════════════════════════════════════════════
        #  MONGODB VIEWER
        # ══════════════════════════════════════════════════════════════════════
        with gr.Accordion("🗄️ MongoDB Viewer", open=False):
            with gr.Row():
                mv_biome_dd  = gr.Dropdown(choices=biome_ids, value=_first(biome_ids),
                                           label="Biome", interactive=True)
                mv_biome_btn = gr.Button("Load Biome", size="sm")
            mv_biome_out = gr.Textbox(label="Biome JSON", lines=15, interactive=False)
            mv_biome_btn.click(load_biome_raw, [mv_biome_dd], [mv_biome_out])

            gr.Markdown("---")
            with gr.Row():
                mv_coll_dd    = gr.Dropdown(choices=collections, value=_first(collections),
                                            label="Collection", interactive=True)
                mv_coll_limit = gr.Slider(1, 50, value=10, step=1, label="Limit")
                mv_coll_btn   = gr.Button("Load Collection", size="sm")
            mv_coll_out = gr.Textbox(label="Documents JSON", lines=15, interactive=False)
            mv_coll_btn.click(load_collection, [mv_coll_dd, mv_coll_limit], [mv_coll_out])

    return tab

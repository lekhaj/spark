import gradio as gr
import os
import json
import redis
from dotenv import load_dotenv
from app.services.mongo_service import (
    get_biome_choices_live,
    biome_assets_for_task,
    get_biome_asset_update_key,
)
try:
    from app.config import settings
except Exception:
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
    from app.config import settings

# Load .env to pick up Redis/Mongo settings when the page code runs
load_dotenv()

# Derive Redis host/port by parsing broker URL (no REDIS_HOST env expected)
broker = os.getenv('CELERY_BROKER_URL') or os.getenv('REDIS_URL') or os.getenv('CELERY_BROKER')
if broker:
    try:
        from urllib.parse import urlparse
        parsed = urlparse(broker)
        REDIS_HOST = parsed.hostname or getattr(settings, 'redis_host', getattr(settings, 'REDIS_HOST', 'localhost'))
        REDIS_PORT = parsed.port or getattr(settings, 'redis_port', getattr(settings, 'REDIS_PORT', 6379))
    except Exception:
        REDIS_HOST = getattr(settings, 'redis_host', getattr(settings, 'REDIS_HOST', 'localhost'))
        REDIS_PORT = getattr(settings, 'redis_port', getattr(settings, 'REDIS_PORT', 6379))
else:
    REDIS_HOST = getattr(settings, 'redis_host', getattr(settings, 'REDIS_HOST', 'localhost'))
    REDIS_PORT = getattr(settings, 'redis_port', getattr(settings, 'REDIS_PORT', 6379))

REDIS_PORT = int(REDIS_PORT)


def _connect_redis():
    # Prefer using the broker URL (CELERY_BROKER_URL / REDIS_URL) so credentials/db are preserved
    broker_url = os.getenv('CELERY_BROKER_URL') or os.getenv('REDIS_URL') or os.getenv('CELERY_BROKER')
    try:
        if broker_url:
            return redis.from_url(broker_url, db=None)
        return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)
    except Exception:
        return None


def _collect_variants_for_asset(asset_name: str, asset: dict):
    """Return list of (variant_key, model_url, rigged_asset_name) for an asset.

    Preference: all decimated profiles (dynamic), else painted_url, else s3_3d_url.
    """
    variants = []
    dec = asset.get('decimated_assets') or {}
    if isinstance(dec, dict) and dec:
        for prof, entry in dec.items():
            if isinstance(entry, dict) and entry.get('url'):
                # derive rigged asset name: e.g., 'knight_5k' from 'decimated_5k' or keep full profile suffix
                suffix = prof.split('_', 1)[-1] if '_' in prof else prof
                rigged_name = f"{asset_name}_{suffix}"
                variants.append((prof, entry.get('url'), rigged_name))
    if variants:
        return variants
    # fallback painted_url
    painted = asset.get('painted_url') or (asset.get('attributes') or {}).get('painted_url')
    if painted:
        variants.append(('painted_raw', painted, asset_name))
        return variants
    # fallback raw 3d
    raw = asset.get('s3_3d_url') or (asset.get('attributes') or {}).get('s3_3d_url')
    if raw:
        variants.append(('raw_3d', raw, asset_name))
    return variants


def rigging_ui():
    with gr.Blocks() as rigging_ui:
        gr.Markdown('## 🛠️ Rigging')
        gr.Markdown('Select a biome and optionally an asset, then push rigging jobs into the `rig_model` Redis list.\n\nNote: this page only queues jobs; the worker on the Unirig instance will perform rigging and update Mongo when finished.')

        try:
            biome_choices = get_biome_choices_live(settings.MONGODB_DB_NAME, 'biomes') if get_biome_choices_live else []
        except Exception:
            biome_choices = []

        if biome_choices:
            biome_dropdown = gr.Dropdown(choices=[(name, _id) for name, _id in biome_choices], label='Select Biome', value=biome_choices[0][1], interactive=True)
        else:
            biome_dropdown = gr.Dropdown(choices=[], label='Select Biome', value=None, interactive=False)

        asset_dropdown = gr.Dropdown(label='Select Asset (optional)', choices=[('All Assets', 'all')], value='all', interactive=True)
        push_btn = gr.Button('Queue Rigging')
        result_box = gr.Textbox(label='Result', interactive=False)
        jobs_json = gr.JSON(label='Queued Rig Jobs')
        job_count = gr.Number(label='Queue Count', value=0, interactive=False)
        refresh_count_btn = gr.Button('Refresh Count')

        def update_assets_for_biome(biome_id):
            try:
                assets = biome_assets_for_task(biome_id, status_filter='3d_generated')
                choices = [('All Assets', 'all')] + [(k, k) for k in assets.keys()]
                return gr.update(choices=choices, value='all')
            except Exception:
                return gr.update(choices=[('All Assets', 'all')], value='all')

        def _read_rig_queue(max_items: int = 200):
            r = _connect_redis()
            if r is None:
                return []
            try:
                raw = r.lrange('rig_model', 0, max_items - 1)
                items = []
                for b in raw:
                    try:
                        items.append(json.loads(b.decode('utf-8') if isinstance(b, (bytes, bytearray)) else b))
                    except Exception:
                        continue
                return items
            except Exception:
                return []

        def _get_queue_count():
            r = _connect_redis()
            if r is None:
                return 0
            try:
                return int(r.llen('rig_model'))
            except Exception:
                return 0


        def push_rigging_jobs(biome_id, selected_asset):
            if not biome_id:
                return 'No biome selected.', [], 0
            r = _connect_redis()
            if r is None:
                return 'Redis not available.', [], 0
            assets = biome_assets_for_task(biome_id, status_filter='3d_generated')
            if not assets:
                return 'No assets ready for rigging.', [], 0

            to_queue = []
            if selected_asset and selected_asset != 'all':
                if selected_asset in assets:
                    to_queue = [(selected_asset, assets[selected_asset])]
                else:
                    return f"Asset {selected_asset} not available for rigging.", [], 0
            else:
                to_queue = list(assets.items())

            pushed = 0
            # Only enqueue assets from allowed riggable categories
            allowed_categories = {'creatures', 'props'}
            skipped = 0
            for asset_name, asset in to_queue:
                update_key = get_biome_asset_update_key(biome_id, asset_name)
                # derive category from update_key like 'possible_structures.creatures.asset'
                category = None
                if update_key and isinstance(update_key, str) and update_key.count('.') >= 2:
                    parts = update_key.split('.')
                    if len(parts) >= 2:
                        category = parts[1]
                if category not in allowed_categories:
                    skipped += 1
                    continue
                variants = _collect_variants_for_asset(asset_name, asset)
                for variant_key, model_url, rigged_name in variants:
                    payload = {
                        'biome_id': str(biome_id),
                        'asset_name': asset_name,
                        'variant_key': variant_key,
                        'model_url': model_url,
                        'update_key': update_key,
                        'rigged_asset_name': rigged_name,
                    }
                    try:
                        r.lpush('rig_model', json.dumps(payload))
                        pushed += 1
                    except Exception as e:
                        # return current queue state along with error
                        queued_list = _read_rig_queue()
                        count = _get_queue_count()
                        return f'Failed to push {asset_name}:{variant_key} -> {e}', queued_list, count

            queued_list = _read_rig_queue()
            count = _get_queue_count()
            msg = f'Pushed {pushed} rig jobs to rig_model.'
            if skipped:
                msg += f' Skipped {skipped} non-riggable assets.'
            return msg, queued_list, count

        biome_dropdown.change(fn=update_assets_for_biome, inputs=[biome_dropdown], outputs=[asset_dropdown])
        push_btn.click(fn=push_rigging_jobs, inputs=[biome_dropdown, asset_dropdown], outputs=[result_box, jobs_json, job_count])
        refresh_count_btn.click(fn=lambda: _get_queue_count(), inputs=[], outputs=[job_count])

    return rigging_ui

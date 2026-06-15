"""
CycleZero game-authoring backend (module inside the spark repo).

Anyone can create a game (a persisted resource with a URI/slug) and author it as
a generic graph of content: `entities` (typed by `layer` — scene, character,
system, story, gameplay_loop, …) connected by `relations`. Asset generation is
triggered via `asset_jobs` that reuse the existing GPU pipeline; results surface
as S3 links. A `contract` builder turns the authored graph into the engine-
agnostic scene JSON the Babylon (and later Unity) runtime plays, and a `matching`
pass reports how well a built game covers the authored spec.

Stores:
- Postgres `cyclezero` — the design graph (games/entities/relations/asset_jobs).
- Mongo `cyclezero`   — freeform / generated content + job artifacts.

Routes mounted at ``/cyclezero`` (see app/main.py).
"""

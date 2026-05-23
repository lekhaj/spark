"""
proxy_generators.py — programmatic ControlNet proxy renderers for non-humanoid
creatures.

Each morphology bucket gets one skeleton/silhouette renderer.  Output is a
white-line drawing on a black background, sized 1024 by default.  These are
designed to be consumed by FLUX.1-dev-ControlNet-Union-Pro-2.0 in either
``soft_edge`` mode (preferred for B2-B7) or ``pose`` mode (B1 only — that
case is handled by the existing ``generate_apose.py``).

Buckets
-------
  B1_humanoid       — bipedal humanoid          (orc, goblin, minotaur)
                      → use ``apose_canonical.png`` from generate_apose.py
  B2_centaur        — humanoid torso + quadruped lower
  B3_naga           — humanoid torso + serpent tail
  B4_quadruped      — horse/tiger/raccoon/rabbit
  B5_winged_quad    — dragon, pegasus (B4 + extended wings)
  B6_hydra          — quadruped + fan of necks
  B7_arthropod      — spider (8 legs), ant (6 legs)

Each renderer accepts a ``view`` argument.  ``primary`` is the canonical
single-best-view for image-to-3D (see plan).  Alt views (``front_34``,
``back_34``, ``top``, …) are coarse 2D foreshortening approximations of the
primary skeleton — good enough to give the model a directional hint when the
user triggers the multi-view expand pass.

Run from the repo root to bake all PNGs::

    python worker/controlnet_refs/bake_proxies.py
"""

from __future__ import annotations

import math
from typing import Callable

from PIL import Image, ImageDraw

# ── Common constants ─────────────────────────────────────────────────────────

_LINE_COLOR = (255, 255, 255)
_BG_COLOR = (0, 0, 0)


def _bone_w(size: int) -> int:
    return max(4, size // 110)


def _joint_r(size: int) -> int:
    return max(4, size // 160)


def _new_canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (size, size), _BG_COLOR)
    return img, ImageDraw.Draw(img, "RGBA")


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[float, float]],
                   size: int, color=_LINE_COLOR) -> None:
    """Draw a connected polyline plus a joint dot at each vertex."""
    bw = _bone_w(size)
    jr = _joint_r(size)
    pts_px = [(int(x * size), int(y * size)) for x, y in pts]
    for a, b in zip(pts_px, pts_px[1:]):
        draw.line([a, b], fill=(*color, 230), width=bw)
    for (x, y) in pts_px:
        draw.ellipse((x - jr, y - jr, x + jr, y + jr), fill=(*color, 255))


def _draw_segments(draw: ImageDraw.ImageDraw,
                   segments: list[tuple[tuple[float, float], tuple[float, float]]],
                   size: int, color=_LINE_COLOR) -> None:
    bw = _bone_w(size)
    jr = _joint_r(size)
    for (a, b) in segments:
        ax, ay = int(a[0] * size), int(a[1] * size)
        bx, by = int(b[0] * size), int(b[1] * size)
        draw.line([(ax, ay), (bx, by)], fill=(*color, 230), width=bw)
        for (x, y) in ((ax, ay), (bx, by)):
            draw.ellipse((x - jr, y - jr, x + jr, y + jr), fill=(*color, 255))


def _foreshorten_x(pt: tuple[float, float], factor: float,
                   axis_x: float = 0.5) -> tuple[float, float]:
    """Compress x-distance from a vertical axis by ``factor`` — cheap rotation proxy.

    factor = 1.0 → unchanged (pure side view); 0.0 → fully collapsed (front view).
    """
    x, y = pt
    return (axis_x + (x - axis_x) * factor, y)


# ─────────────────────────────────────────────────────────────────────────────
# B2 — Centaur
# ─────────────────────────────────────────────────────────────────────────────

def render_centaur(size: int = 1024, view: str = "primary") -> Image.Image:
    """
    Front view (primary): human A-pose torso on top, four splayed legs below.
    Quadruped body axis runs left-right (the horse half stands facing camera).
    """
    img, draw = _new_canvas(size)

    # Human upper body — A-pose (mirrors generate_apose.py proportions)
    arm_deg = 35.0
    arm_seg = 0.13
    c, s = math.cos(math.radians(arm_deg)), math.sin(math.radians(arm_deg))
    sho_y = 0.18
    r_sho_x, l_sho_x = 0.44, 0.56

    head = [(0.50, 0.07), (0.50, sho_y)]                       # head→neck
    r_arm = [(r_sho_x, sho_y),
             (r_sho_x - arm_seg * c, sho_y + arm_seg * s),
             (r_sho_x - 2 * arm_seg * c, sho_y + 2 * arm_seg * s)]
    l_arm = [(l_sho_x, sho_y),
             (l_sho_x + arm_seg * c, sho_y + arm_seg * s),
             (l_sho_x + 2 * arm_seg * c, sho_y + 2 * arm_seg * s)]
    spine = [(0.50, sho_y), (0.50, 0.42)]                      # neck→hip-anchor
    shoulders = [(r_sho_x, sho_y), (l_sho_x, sho_y)]

    # Horse half — body slab between front shoulders (~0.42) and rear hips (~0.42)
    # Front shoulders pair on either side of midline; rear hips pair wider.
    f_sho_y, h_hip_y = 0.42, 0.42
    f_sho = [(0.40, f_sho_y), (0.60, f_sho_y)]
    h_hip = [(0.35, h_hip_y), (0.65, h_hip_y)]
    body_top = [(0.40, f_sho_y), (0.35, h_hip_y)]   # left flank line (visible on front view)
    body_bot = [(0.60, f_sho_y), (0.65, h_hip_y)]   # right flank line

    # Legs — splayed outward so hooves are clearly separated
    front_l = [(0.40, f_sho_y), (0.35, 0.72), (0.32, 0.95)]
    front_r = [(0.60, f_sho_y), (0.65, 0.72), (0.68, 0.95)]
    hind_l = [(0.35, h_hip_y), (0.30, 0.72), (0.27, 0.95)]
    hind_r = [(0.65, h_hip_y), (0.70, 0.72), (0.73, 0.95)]

    # Tail (hint)
    tail = [(0.50, h_hip_y + 0.02), (0.50, 0.58)]

    polylines = [head, spine, r_arm, l_arm,
                 front_l, front_r, hind_l, hind_r, tail]
    segments = [(shoulders[0], shoulders[1]),
                (f_sho[0], f_sho[1]),
                (h_hip[0], h_hip[1]),
                (body_top[0], body_top[1]),
                (body_bot[0], body_bot[1])]

    if view in ("back_34", "front_34"):
        fx = 0.45 if view == "front_34" else 0.55
        polylines = [[_foreshorten_x(p, fx) for p in pl] for pl in polylines]
        segments = [(_foreshorten_x(a, fx), _foreshorten_x(b, fx)) for a, b in segments]

    for pl in polylines:
        _draw_polyline(draw, pl, size)
    _draw_segments(draw, segments, size)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# B3 — Naga
# ─────────────────────────────────────────────────────────────────────────────

def render_naga(size: int = 1024, view: str = "primary") -> Image.Image:
    """
    Front view: human A-pose torso ending at hips (~0.5), serpent tail coils
    down-and-right in a clear S-curve.  Tail stays off the body silhouette.
    """
    img, draw = _new_canvas(size)

    # Human upper body
    arm_deg = 35.0
    arm_seg = 0.14
    c, s = math.cos(math.radians(arm_deg)), math.sin(math.radians(arm_deg))
    sho_y = 0.16
    r_sho_x, l_sho_x = 0.44, 0.56

    head = [(0.50, 0.05), (0.50, sho_y)]
    spine = [(0.50, sho_y), (0.50, 0.48)]
    r_arm = [(r_sho_x, sho_y),
             (r_sho_x - arm_seg * c, sho_y + arm_seg * s),
             (r_sho_x - 2 * arm_seg * c, sho_y + 2 * arm_seg * s)]
    l_arm = [(l_sho_x, sho_y),
             (l_sho_x + arm_seg * c, sho_y + arm_seg * s),
             (l_sho_x + 2 * arm_seg * c, sho_y + 2 * arm_seg * s)]
    shoulders = [(r_sho_x, sho_y), (l_sho_x, sho_y)]
    hips = [(0.45, 0.48), (0.55, 0.48)]

    # Serpent tail — sampled S-curve from hip downward
    tail = []
    n = 32
    for i in range(n + 1):
        t = i / n
        # x: 0.50 -> 0.30 -> 0.72 -> 0.42  (S-shape)
        x = 0.50 + 0.22 * math.sin(t * math.pi * 1.5)
        y = 0.48 + 0.50 * t
        tail.append((x, y))

    if view == "front_34":
        polylines_pre = [head, spine, r_arm, l_arm, tail]
        polylines_pre = [[_foreshorten_x(p, 0.55) for p in pl] for pl in polylines_pre]
        head, spine, r_arm, l_arm, tail = polylines_pre

    for pl in (head, spine, r_arm, l_arm, tail):
        _draw_polyline(draw, pl, size)
    _draw_segments(draw, [(shoulders[0], shoulders[1]), (hips[0], hips[1])], size)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# B4 — Quadruped (horse / tiger / raccoon / rabbit)
# ─────────────────────────────────────────────────────────────────────────────

def render_quadruped(size: int = 1024, view: str = "primary") -> Image.Image:
    """
    Primary = side view, head on left, tail on right.  All four legs visible
    with offset front/back pairs so they don't merge in image-to-3D.

    Proportion knobs:  body slab y in [0.45, 0.58], leg ground line y=0.92.
    Identical baseline serves horse/tiger/raccoon; rabbit-ish proportions can
    be handled via prompt without changing the skeleton.
    """
    img, draw = _new_canvas(size)

    # Body slab (side profile)
    f_sho = (0.30, 0.50)    # front shoulder
    h_hip = (0.72, 0.50)    # rear hip
    chest = (0.24, 0.55)    # underside front
    belly = (0.72, 0.58)    # underside rear

    # Head + neck
    neck = [(0.30, 0.50), (0.20, 0.38), (0.13, 0.30)]
    muzzle = [(0.13, 0.30), (0.06, 0.32)]
    ears = [(0.12, 0.30), (0.14, 0.22)]

    # Tail
    tail = [(0.72, 0.50), (0.85, 0.42), (0.94, 0.40)]

    # Legs — offset the near/far pair slightly along x so they read as 4 legs
    # in 3D-gen, not 2.
    front_near = [(0.30, 0.55), (0.30, 0.74), (0.30, 0.92)]
    front_far = [(0.36, 0.55), (0.36, 0.74), (0.36, 0.92)]
    hind_near = [(0.70, 0.58), (0.70, 0.76), (0.70, 0.92)]
    hind_far = [(0.64, 0.58), (0.64, 0.76), (0.64, 0.92)]

    body_segments = [(f_sho, h_hip), (chest, belly),
                     (f_sho, chest), (h_hip, belly)]

    polylines = [neck, muzzle, ears, tail,
                 front_near, front_far, hind_near, hind_far]

    if view == "front_34":
        # Compress x toward midline of body (~0.5) — gives a 3/4 hint
        polylines = [[_foreshorten_x(p, 0.55) for p in pl] for pl in polylines]
        body_segments = [(_foreshorten_x(a, 0.55), _foreshorten_x(b, 0.55))
                         for a, b in body_segments]
    elif view == "back_34":
        # Mirror around x=0.5 then 3/4 compress — gives a rear-three-quarter
        polylines = [[_foreshorten_x((1.0 - x, y), 0.55) for x, y in pl] for pl in polylines]
        body_segments = [(_foreshorten_x((1.0 - a[0], a[1]), 0.55),
                          _foreshorten_x((1.0 - b[0], b[1]), 0.55))
                         for a, b in body_segments]

    for pl in polylines:
        _draw_polyline(draw, pl, size)
    _draw_segments(draw, body_segments, size)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# B5 — Winged quadruped (dragon, pegasus)
# ─────────────────────────────────────────────────────────────────────────────

def render_winged_quad(size: int = 1024, view: str = "primary") -> Image.Image:
    """
    Primary = side view of B4 with two wing skeletons extending up-and-back
    from the shoulders.  Wings are drawn as triple-finger fans so the
    silhouette has clear positive shape, not just a flat triangle.
    """
    img, draw = _new_canvas(size)

    # Reuse the B4 side skeleton inline (slightly compressed body to leave
    # room for wings up top).
    f_sho = (0.32, 0.55)
    h_hip = (0.72, 0.55)
    chest = (0.26, 0.60)
    belly = (0.72, 0.62)

    neck = [(0.32, 0.55), (0.22, 0.45), (0.14, 0.38)]
    muzzle = [(0.14, 0.38), (0.06, 0.40)]
    horns = [(0.13, 0.38), (0.15, 0.28)]

    tail = [(0.72, 0.55), (0.85, 0.48), (0.96, 0.50)]

    front_near = [(0.32, 0.60), (0.32, 0.78), (0.32, 0.93)]
    front_far = [(0.38, 0.60), (0.38, 0.78), (0.38, 0.93)]
    hind_near = [(0.70, 0.62), (0.70, 0.80), (0.70, 0.93)]
    hind_far = [(0.64, 0.62), (0.64, 0.80), (0.64, 0.93)]

    # Wings — shoulder pivot at (0.36, 0.52), three "fingers" splaying back-up
    wing_root = (0.36, 0.52)
    wing_a = [wing_root, (0.20, 0.20), (0.10, 0.08)]
    wing_b = [wing_root, (0.35, 0.18), (0.30, 0.05)]
    wing_c = [wing_root, (0.50, 0.20), (0.55, 0.05)]
    wing_membrane = [
        ((0.10, 0.08), (0.30, 0.05)),
        ((0.30, 0.05), (0.55, 0.05)),
        ((0.55, 0.05), (0.50, 0.20)),
        ((0.20, 0.20), (0.35, 0.18)),
        ((0.35, 0.18), (0.50, 0.20)),
    ]

    body_segments = [(f_sho, h_hip), (chest, belly),
                     (f_sho, chest), (h_hip, belly)]

    polylines = [neck, muzzle, horns, tail,
                 front_near, front_far, hind_near, hind_far,
                 wing_a, wing_b, wing_c]

    if view == "front_34":
        polylines = [[_foreshorten_x(p, 0.55) for p in pl] for pl in polylines]
        body_segments = [(_foreshorten_x(a, 0.55), _foreshorten_x(b, 0.55))
                         for a, b in body_segments]
        wing_membrane = [(_foreshorten_x(a, 0.55), _foreshorten_x(b, 0.55))
                         for a, b in wing_membrane]

    for pl in polylines:
        _draw_polyline(draw, pl, size)
    _draw_segments(draw, body_segments + wing_membrane, size)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# B6 — Hydra (quadruped + fan of necks)
# ─────────────────────────────────────────────────────────────────────────────

def render_hydra(size: int = 1024, view: str = "primary",
                 neck_count: int = 5) -> Image.Image:
    """
    Primary = front-3/4 view.  Quadruped base with a fan of necks sprouting
    from the shoulders.  Necks splay outward and never cross.
    """
    img, draw = _new_canvas(size)

    # Compressed-side quadruped body, sitting square on
    f_sho = (0.32, 0.58)
    h_hip = (0.74, 0.58)
    chest = (0.26, 0.64)
    belly = (0.74, 0.66)

    tail = [(0.74, 0.58), (0.88, 0.50), (0.96, 0.52)]
    front_near = [(0.32, 0.64), (0.32, 0.80), (0.32, 0.94)]
    front_far = [(0.40, 0.64), (0.40, 0.80), (0.40, 0.94)]
    hind_near = [(0.72, 0.66), (0.72, 0.82), (0.72, 0.94)]
    hind_far = [(0.64, 0.66), (0.64, 0.82), (0.64, 0.94)]

    body_segments = [(f_sho, h_hip), (chest, belly),
                     (f_sho, chest), (h_hip, belly)]

    polylines = [tail, front_near, front_far, hind_near, hind_far]

    # Necks fan from a single shoulder pivot (front of body)
    pivot = (0.34, 0.58)
    # Spread necks across an angle range — outermost go furthest out & up
    span = math.radians(150)  # 150° fan
    start = math.radians(-90) - span / 2  # centered on straight up
    for i in range(neck_count):
        t = i / max(neck_count - 1, 1)
        ang = start + t * span
        mid = (pivot[0] + 0.20 * math.cos(ang),
               pivot[1] + 0.20 * math.sin(ang))
        tip = (pivot[0] + 0.34 * math.cos(ang),
               pivot[1] + 0.34 * math.sin(ang))
        head_tip = (tip[0] + 0.04 * math.cos(ang),
                    tip[1] + 0.04 * math.sin(ang))
        polylines.append([pivot, mid, tip, head_tip])

    if view == "front":
        polylines = [[_foreshorten_x(p, 0.4) for p in pl] for pl in polylines]
        body_segments = [(_foreshorten_x(a, 0.4), _foreshorten_x(b, 0.4))
                         for a, b in body_segments]

    for pl in polylines:
        _draw_polyline(draw, pl, size)
    _draw_segments(draw, body_segments, size)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# B7 — Arthropod (spider, ant)
# ─────────────────────────────────────────────────────────────────────────────

def render_arthropod(size: int = 1024, view: str = "primary",
                     leg_count: int = 8, segmented_body: bool = False) -> Image.Image:
    """
    High 3/4 view (camera ~45° elevation).  Body is an ellipse near center,
    legs splay radially.

    leg_count=8 → spider; leg_count=6, segmented_body=True → ant.
    """
    img, draw = _new_canvas(size)

    # Body
    if segmented_body:
        # Ant: head + thorax + abdomen as three nodes along the body axis
        head_node = (0.50, 0.40)
        thorax = (0.50, 0.50)
        abdomen = (0.50, 0.63)
        body_pts = [head_node, thorax, abdomen]
        antennae_l = [head_node, (0.42, 0.30), (0.38, 0.22)]
        antennae_r = [head_node, (0.58, 0.30), (0.62, 0.22)]
        leg_origin = thorax
    else:
        # Spider: single cephalothorax + abdomen
        cephalo = (0.50, 0.48)
        abdomen = (0.50, 0.62)
        body_pts = [cephalo, abdomen]
        antennae_l = antennae_r = None
        leg_origin = cephalo

    # Legs splayed radially in two banks (left and right of body axis)
    legs: list[list[tuple[float, float]]] = []
    per_side = leg_count // 2
    # Angles for each leg, fan from forward-side to backward-side
    angle_start = math.radians(-60)   # forward-out
    angle_end = math.radians(110)     # backward-out
    for side in (-1, +1):
        for i in range(per_side):
            t = i / max(per_side - 1, 1)
            ang = angle_start + t * (angle_end - angle_start)
            # Mirror across body axis: side=-1 → flip x
            cx = math.cos(ang) * side * -1  # leftward when side=-1
            sy = math.sin(ang)
            knee = (leg_origin[0] + 0.16 * cx, leg_origin[1] + 0.16 * sy)
            tip = (leg_origin[0] + 0.32 * cx, leg_origin[1] + 0.30 * sy)
            legs.append([leg_origin, knee, tip])

    polylines = [body_pts] + legs
    if antennae_l:
        polylines += [antennae_l, antennae_r]

    if view == "top":
        # Identity — top-down already
        pass

    for pl in polylines:
        _draw_polyline(draw, pl, size)
    return img


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

MORPHOLOGIES: tuple[str, ...] = (
    "B1_humanoid",
    "B2_centaur",
    "B3_naga",
    "B4_quadruped",
    "B5_winged_quad",
    "B6_hydra",
    "B7_arthropod",
)

# Per-morphology available views.  Index 0 is always the canonical "primary"
# single-best-view.  Alt views are used by the UI "Expand to multi-view" button.
MORPHOLOGY_VIEWS: dict[str, tuple[str, ...]] = {
    "B1_humanoid":     ("primary",),                          # apose handled separately
    "B2_centaur":      ("primary", "front_34", "back_34"),
    "B3_naga":         ("primary", "front_34"),
    "B4_quadruped":    ("primary", "front_34", "back_34"),
    "B5_winged_quad":  ("primary", "front_34"),
    "B6_hydra":        ("primary", "front"),
    "B7_arthropod":    ("primary",),                          # top-3/4 only sane view
}

# Recommended ControlNet mode per morphology.  B1 keeps OpenPose (pose);
# everything else falls back to soft_edge, which is the most forgiving mode
# for white-line skeleton inputs that don't match the trained OpenPose layout.
RECOMMENDED_MODE: dict[str, str] = {
    "B1_humanoid":     "pose",
    "B2_centaur":      "soft_edge",
    "B3_naga":         "soft_edge",
    "B4_quadruped":    "soft_edge",
    "B5_winged_quad":  "soft_edge",
    "B6_hydra":        "soft_edge",
    "B7_arthropod":    "soft_edge",
}

# Stance phrases appended to the user prompt by the worker.  These name the
# pose constraints that the proxy renderer assumes — keeping prompt + proxy
# aligned reduces the chance of FLUX fighting the skeleton.
STANCE_PHRASE: dict[str, str] = {
    "B1_humanoid":
        "in relaxed A-pose, arms angled below shoulders, "
        "clear silhouette gap between arms and torso",
    "B2_centaur":
        "centaur standing in front view, A-pose human torso, "
        "all four equine legs spread in stable stance and fully separated",
    "B3_naga":
        "naga in upright A-pose torso facing camera, "
        "serpentine tail in a clear S-curve below, tail not crossing the body",
    "B4_quadruped":
        "standing in pure side view, head left and tail right, "
        "all four legs visibly separated, far-side legs slightly offset, "
        "tail extended away from the body",
    "B5_winged_quad":
        "standing in side view, wings fully extended outward and back, "
        "all four legs separated, tail extended away from the body",
    "B6_hydra":
        "standing front-three-quarter view, multiple necks fanned outward, "
        "necks never crossing each other, four legs clearly separated",
    "B7_arthropod":
        "high three-quarter view from above, all legs splayed radially "
        "and clearly separated, no legs overlapping the body",
}

# Universal limb-separation suffix appended after the stance phrase.
LIMB_SEPARATION_SUFFIX: str = (
    "full body, neutral solid background, even studio lighting, "
    "all limbs fully separated and visible, no overlapping limbs, "
    "no crossed legs, no interwoven appendages, clear silhouette, "
    "character sheet reference"
)


_RENDERERS: dict[str, Callable[..., Image.Image]] = {
    "B2_centaur":     render_centaur,
    "B3_naga":        render_naga,
    "B4_quadruped":   render_quadruped,
    "B5_winged_quad": render_winged_quad,
    "B6_hydra":       render_hydra,
    "B7_arthropod":   render_arthropod,
}


def render_proxy(morphology: str, view: str = "primary",
                 size: int = 1024, **kwargs) -> Image.Image:
    """
    Dispatch to the right per-morphology renderer.

    B1_humanoid is intentionally NOT handled here — call ``render_apose`` from
    ``generate_apose.py`` for that.  Raising loudly avoids silent fallback.
    """
    if morphology == "B1_humanoid":
        raise ValueError(
            "B1_humanoid uses the OpenPose A-pose skeleton — "
            "call generate_apose.render_apose() or use the bundled "
            "apose_canonical.png instead of render_proxy()."
        )
    fn = _RENDERERS.get(morphology)
    if fn is None:
        raise ValueError(
            f"Unknown morphology {morphology!r}. "
            f"Known: {sorted(_RENDERERS) + ['B1_humanoid']}"
        )
    return fn(size=size, view=view, **kwargs)


def proxy_filename(morphology: str, view: str = "primary") -> str:
    """Canonical PNG filename for a given (morphology, view) pair."""
    return f"{morphology}_{view}.png"

#!/usr/bin/env python3
"""
seed_forest_creatures.py — Replace placeholder specs with real xianxia forest creature specs.

Run:
  python worker/seed_forest_creatures.py

Adds 4 detailed character specs:
  - qingmao_spirit   : Qing Mao — Green Fur Spirit Cat (humanoid/bipedal)
  - tielong_boar     : Tie Long — Iron-Ridge Spirit Boar (quadruped)
  - baihe_crane      : Bai He — White Bone Crane (avian/biped)
  - jinyuan_ape      : Jin Yuan — Gilded Spirit Ape (humanoid/quadruped hybrid)

Also removes old placeholder _id entries.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from lib.spec_schema import get_db, upsert_character_spec, delete_character_spec


def seed(db):
    # ── Remove placeholder stubs ──────────────────────────────────────────────
    for old_id in ["placeholder_male_warrior", "placeholder_female_mage",
                   "placeholder_dire_wolf", "placeholder_phoenix"]:
        delete_character_spec(db, old_id)
        print(f"  [removed] {old_id}")

    # ══════════════════════════════════════════════════════════════════════════
    #  1 — Qing Mao Spirit Cat (humanoid/bipedal)
    #  A forest spirit that adopted humanoid form: lithe, teal-furred, jade eyes
    #  Wears flowing nature-spirit robes. Tail always visible.
    # ══════════════════════════════════════════════════════════════════════════
    upsert_character_spec(db, {
        "_id": "qingmao_spirit",
        "display_name": "Qing Mao — Green Fur Spirit Cat",
        "category": "humanoid",
        "subcategory": "bipedal",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        "body": {
            "species": "spirit cat in humanoid form",
            "build": "slender lithe, long-limbed, slightly shorter than human average",
            "skin": "teal-green short dense fur covering entire body",
            "height": "160cm",
            "tail": "single long teal cat tail, darker stripe pattern, always visible behind robes",
            "ears": "large pointed cat ears on top of head, darker teal with pale inner fur",
            "hands": "four-fingered with small retracted curved claws, teal furred",
        },

        "face": {
            "structure": "triangular cat-like face, defined cheekbones, narrow chin",
            "eyes": "large almond-shaped jade green cat eyes with vertical slit pupils",
            "nose": "small flat feline nose, pale pink",
            "expression": "curious alert, slight tilt to head, ears perked",
            "whiskers": "four thin white whiskers, short, at corners of muzzle",
            "lips": "thin, closed, slight upturn at corners",
        },

        "hair": {
            "style": "short wild tuft of darker teal-green hair between ears, slightly unkempt",
            "color": "dark teal-green #1A5C5A",
            "length": "very short, barely covers forehead",
            "texture": "bristly, sticking up slightly",
        },

        "clothing": {
            "primary": "loose flowing outer robe in pale celadon green, wide sleeves",
            "inner": "thin off-white under-robe visible at collar",
            "belt": "woven vine belt with small carved jade stone clasp",
            "details": [
                "robe has subtle leaf-vein embroidery along hem and cuffs in darker green",
                "fabric is light and translucent at edges, like nature-spirit garment",
                "robe open at back to allow tail freedom of movement",
                "barefoot, small teal-furred feet with visible claws",
            ],
            "accessories": [
                "small string of carved green jade beads around left wrist",
                "single dried flower tucked behind right ear",
            ],
        },

        "color_palette": {
            "fur": "teal green #2E8B7A",
            "fur_dark_stripe": "deep forest green #1A5C4A",
            "eyes": "jade green #4CAF50",
            "robe_primary": "celadon pale green #A8D5B5",
            "robe_inner": "off-white #F0F0E8",
            "jade_accent": "translucent jade #7EC8A0",
        },

        "pose_default": "T-pose",
        "view_default": "front",

        "flux_generation": {
            "width": 768, "height": 1024,
            "steps": 4, "guidance_scale": 0.0, "model": "FLUX.1-schnell",
        },
        "sd_stage1": {
            "denoise": 0.20, "cfg": 7.5,
            "controlnet": "openpose + canny dual",
            "openpose_weight": 0.85, "canny_weight": 0.55,
        },
        "sd_stage2": {"denoise": 0.35, "cfg": 7.0},

        "notes": (
            "Spirit cat in humanoid form. ALWAYS has tail and cat ears visible. "
            "Teal fur on all exposed skin. No human skin visible. "
            "Gentle nature-spirit vibe, NOT fierce predator. "
            "Small, lithe, curious energy."
        ),
    })
    print("  [upserted] qingmao_spirit")

    # ══════════════════════════════════════════════════════════════════════════
    #  2 — Tie Long Spirit Boar (quadruped)
    #  Iron-scaled forest boar with crystalline ridge spine.
    #  Massive, low, heavily armored looking. Slow but imposing.
    # ══════════════════════════════════════════════════════════════════════════
    upsert_character_spec(db, {
        "_id": "tielong_boar",
        "display_name": "Tie Long — Iron-Ridge Spirit Boar",
        "category": "fantastical_animal",
        "subcategory": "quadruped",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        "body": {
            "species": "spirit boar",
            "build": "massive low-slung quadruped, barrel-chested, very wide body",
            "fur": "coarse dark iron-gray bristle fur, short and dense",
            "size": "shoulder height of a large draft horse, twice as wide",
            "legs": "four thick stubby powerful legs, ending in split cloven hooves",
            "skin_exposed": "dark charcoal hide under sparse bristle on belly",
        },

        "face": {
            "snout": "long broad flattened hog snout, prominent nostrils",
            "eyes": "small deep-set iron-gray eyes with a red glow at the iris edge",
            "tusks": "two upward-curving iron-black crystalline tusks, 40cm, slight metallic sheen",
            "expression": "heavy-lidded, calm but imposing, mouth closed",
            "ears": "large flat floppy ears drooping to the sides, inner veins visible",
        },

        "distinguishing_features": {
            "ridge_spine": "row of sharp iron-crystal spines erupting from neck to tail along spine ridge, graduated from 5cm at tail to 30cm at shoulders",
            "shoulder_plates": "two large flat iron-gray scale plates over shoulders, smooth natural armor",
            "knee_rings": "thick ring of iron-black hardened skin around each knee joint",
            "tail": "short thick tail with dense bristle tuft, constantly slow-swishing",
            "crystal_glow": "very faint blue-cold light emanating from between spine crystals",
            "hoof_marks": "each cloven hoof leaves faint mineral discoloration on ground",
        },

        "color_palette": {
            "fur": "iron gray #4A4A4F",
            "hide": "dark charcoal #2A2A2E",
            "tusks": "iron black with shimmer #1C1C22",
            "spines": "cold steel blue-gray #5A6A7A",
            "crystal_glow": "pale cold blue #A8C8E0",
            "eyes": "iron gray with red edge #8A2020",
        },

        "pose_default": "standing neutral quadruped",
        "view_default": "front-three-quarter",

        "flux_generation": {
            "width": 1024, "height": 768,
            "steps": 4, "guidance_scale": 0.0, "model": "FLUX.1-schnell",
        },
        "sd_stage1": {
            "denoise": 0.22, "cfg": 7.5,
            "controlnet": "animal_openpose + canny dual",
            "openpose_weight": 0.80, "canny_weight": 0.60,
        },
        "sd_stage2": {"denoise": 0.35, "cfg": 7.0},

        "notes": (
            "Massive iron-armored spirit boar. Quadruped — four legs ALWAYS on ground. "
            "Crystal spine ridge is the KEY visual signature. "
            "NOT a standard boar — iron-scale armor plates are grown-in, not worn. "
            "Calm, not aggressive-looking. Ancient forest guardian energy."
        ),
    })
    print("  [upserted] tielong_boar")

    # ══════════════════════════════════════════════════════════════════════════
    #  3 — Bai He White Bone Crane (avian/bipedal)
    #  Ancient spirit crane with bone-white plumage, scholar's spirit energy.
    #  Standing bipedal. Tall, elegant, slightly unsettling pale.
    # ══════════════════════════════════════════════════════════════════════════
    upsert_character_spec(db, {
        "_id": "baihe_crane",
        "display_name": "Bai He — White Bone Crane",
        "category": "fantastical_animal",
        "subcategory": "avian_biped",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        "body": {
            "species": "spirit crane",
            "build": "tall slender avian biped, long neck, standing on two legs",
            "plumage": "pure bone-white feathers, faint blue-gray undertone at wingtip edges",
            "size": "2m tall standing, wingspan 4m when extended (wings folded in T-pose)",
            "legs": "two long slender crane legs with elongated three-toed feet, black leg scales",
            "wings": "large folded wings held close to body in neutral stance",
        },

        "face": {
            "beak": "long straight slender white beak, tip slightly hooked, pale ivory-white",
            "eyes": "round pale silver-white eyes with fine black pupils, surrounded by faint red skin ring",
            "crown": "single long black-tipped crest feather standing vertical from crown",
            "neck_markings": "subtle row of slightly longer feathers along back of neck forming faint mane",
            "expression": "serene ancient composure, slightly otherworldly gaze",
        },

        "distinguishing_features": {
            "wing_primaries": "outermost 5 primary feathers on each wing are pure matte black, stark contrast to white",
            "tail_feathers": "long streaming white tail feathers trailing 60cm behind, two black-tipped",
            "leg_runes": "faint blue-glowing calligraphic rune marks on both legs, natural spirit markings",
            "chest_glow": "very subtle pale blue radiance from chest feathers when still, like inner light",
            "feather_tips": "each body feather has hair-thin black edge tracing, giving slight scale-like appearance",
            "bone_clarity": "skeletal structure slightly visible through translucent white plumage at wings",
        },

        "color_palette": {
            "primary_plumage": "bone white #F8F5F0",
            "wing_primaries": "matte black #1A1A1A",
            "leg_scales": "dark charcoal #2A2A30",
            "beak": "pale ivory #EDE8DC",
            "eye_ring": "faint vermillion #CC4444",
            "rune_glow": "cold pale blue #B0C8E8",
            "chest_radiance": "very pale blue-white #E8F0F8",
        },

        "pose_default": "standing T-pose bipedal, wings spread slightly for balance",
        "view_default": "front",

        "flux_generation": {
            "width": 768, "height": 1024,
            "steps": 4, "guidance_scale": 0.0, "model": "FLUX.1-schnell",
        },
        "sd_stage1": {
            "denoise": 0.20, "cfg": 7.5,
            "controlnet": "openpose + canny dual",
            "openpose_weight": 0.80, "canny_weight": 0.65,
        },
        "sd_stage2": {"denoise": 0.35, "cfg": 7.0},

        "notes": (
            "Ancient spirit crane. Standing bipedal on two long legs. "
            "Key features: bone-white plumage, black primary wing feathers, leg rune markings. "
            "Very tall and slender — NOT a stocky bird. "
            "Elegant, slightly eerie, scholar-spirit energy. "
            "Wings folded in neutral pose. No clothing."
        ),
    })
    print("  [upserted] baihe_crane")

    # ══════════════════════════════════════════════════════════════════════════
    #  4 — Jin Yuan Gilded Spirit Ape (quadruped/knuckle-walker)
    #  Golden-furred mountain ape, massive, wise elder energy.
    #  Knuckle-walking — NOT humanoid upright. Very large.
    # ══════════════════════════════════════════════════════════════════════════
    upsert_character_spec(db, {
        "_id": "jinyuan_ape",
        "display_name": "Jin Yuan — Gilded Spirit Ape",
        "category": "fantastical_animal",
        "subcategory": "knuckle_walker",
        "theme": "xianxia_cultivation",
        "style_ref": "xianxia_cultivation",
        "biome_ids": ["claudetest002"],

        "body": {
            "species": "spirit mountain ape",
            "build": "massive stocky ape, extremely broad shoulders, long arms for knuckle-walking",
            "fur": "deep burnished gold fur, thick and dense, slightly wavy",
            "size": "knuckle-walking height 180cm at shoulder, enormous mass",
            "arms": "arms significantly longer than legs, thick with defined muscle",
            "hands": "massive five-fingered hands with thick dark-bronze knuckle pads, strong black nails",
            "feet": "prehensile wide feet with opposable toe, dark bronze skin on soles",
            "face_exposed": "flat broad ape face, dark bronze skin",
        },

        "face": {
            "structure": "broad flat ape face, pronounced brow ridge, wide skull",
            "eyes": "deep-set large amber-gold eyes, calm wise gaze",
            "nose": "wide flat ape nose, dark bronze",
            "expression": "calm dignified elder, not aggressive, mouth loosely closed",
            "brow": "heavy brow ridge with sparse longer fur brow tufts",
            "ears": "round flat ape ears, small relative to head, fur-covered",
        },

        "distinguishing_features": {
            "mane": "thick voluminous golden mane around head and upper shoulders, long flowing, like a lion's mane but gold",
            "crown_marking": "natural diamond-shaped patch of paler cream-gold fur on top of head",
            "back_stripe": "pale cream stripe of longer fur running down center of back spine",
            "golden_shimmer": "fur has slight metallic golden shimmer in light, reflecting spirit energy",
            "tail": "no tail — great apes have no tail",
            "knuckle_pads": "large dark bronze calloused knuckle pads on both hands, worn smooth",
            "chest": "broad bare chest with dark bronze hairless skin, showing muscle definition",
        },

        "color_palette": {
            "fur_primary": "burnished gold #C5973A",
            "mane": "deep warm gold #D4A017",
            "crown_patch": "pale cream gold #E8D88A",
            "exposed_skin": "dark bronze #4A2C0A",
            "eyes": "amber gold #C8861A",
            "knuckle_pads": "very dark bronze #2A1A08",
            "back_stripe": "cream pale gold #F0E0A0",
        },

        "pose_default": "knuckle-walking stance, slightly risen on back legs, front knuckles on ground",
        "view_default": "front-three-quarter",

        "flux_generation": {
            "width": 1024, "height": 768,
            "steps": 4, "guidance_scale": 0.0, "model": "FLUX.1-schnell",
        },
        "sd_stage1": {
            "denoise": 0.22, "cfg": 7.5,
            "controlnet": "animal_openpose + canny dual",
            "openpose_weight": 0.78, "canny_weight": 0.62,
        },
        "sd_stage2": {"denoise": 0.38, "cfg": 7.0},

        "notes": (
            "Massive golden spirit ape. Knuckle-walker — front knuckles near ground. "
            "NOT upright humanoid. Very large and wide. "
            "KEY feature: the thick golden mane around head distinguishes from normal ape. "
            "Calm wise elder energy — NOT aggressive silverback. "
            "Spirit energy visible as subtle golden shimmer in fur. "
            "No clothing — full body fur."
        ),
    })
    print("  [upserted] jinyuan_ape")


if __name__ == "__main__":
    db = get_db()
    print("Seeding forest creature specs...")
    seed(db)
    print("Done. 4 placeholder specs replaced with real specs.")

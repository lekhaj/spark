"""
Define terrain types and their corresponding descriptions for image generation
"""

# Mapping of terrain type numbers to names
TERRAIN_TYPES = {
    0: "plains",
    1: "forest",
    2: "mountain",
    3: "water",
    4: "desert",
    5: "snow",
    6: "swamp",
    7: "hills",
    8: "urban",
    9: "ruins"
}

# Detailed descriptions for each terrain type for better image generation
TERRAIN_DESCRIPTIONS = {
    0: "flat, grassy plains with scattered wildflowers and occasional small shrubs",
    1: "dense forest with tall trees, lush undergrowth, and dappled sunlight filtering through the canopy",
    2: "rugged mountains with snow-capped peaks, rocky cliffs, and steep slopes",
    3: "clear blue water, gentle waves, and reflective surface",
    4: "sandy desert with dunes, sparse vegetation, and a dry, arid landscape",
    5: "snowy landscape with pristine white snow covering the ground and trees",
    6: "murky swampland with twisted trees, hanging moss, and shallow, stagnant water",
    7: "rolling hills with grassy slopes and gentle valleys",
    8: "bustling urban area with buildings, streets, and infrastructure",
    9: "ancient ruins with crumbling stone structures, fallen columns, and overgrown vegetation"
}

# Color mapping for visualization purposes
TERRAIN_COLORS = {
    0: (144, 238, 144),  # Light green for plains
    1: (34, 139, 34),    # Forest green
    2: (139, 137, 137),  # Gray for mountains
    3: (65, 105, 225),   # Blue for water
    4: (238, 214, 175),  # Beige for desert
    5: (255, 250, 250),  # White for snow
    6: (107, 142, 35),   # Olive green for swamp
    7: (160, 82, 45),    # Brown for hills
    8: (169, 169, 169),  # Dark gray for urban
    9: (112, 128, 144)   # Slate gray for ruins
}

def get_terrain_name(terrain_id):
    """Get the name of a terrain type by its ID"""
    return TERRAIN_TYPES.get(terrain_id, "unknown")

def get_terrain_description(terrain_id):
    """Get the detailed description of a terrain type by its ID"""
    return TERRAIN_DESCRIPTIONS.get(terrain_id, "unknown terrain")

def get_terrain_color(terrain_id):
    """Get the color associated with a terrain type by its ID"""
    return TERRAIN_COLORS.get(terrain_id, (200, 200, 200))  # Default gray
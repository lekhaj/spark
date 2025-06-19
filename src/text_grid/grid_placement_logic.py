# src/text_grid/grid_placement_logic.py

import random
import logging
import copy # Import the copy module

logger = logging.getLogger(__name__)

# Dictionary to keep track of placed structure coordinates for adjacency/avoidance checks
# This needs to be reset per call to generate_grid_from_hints
_placed_structures_coords: dict[int, list[tuple[int, int, int, int]]] = {}


def is_valid_placement(grid: list[list[int]], r: int, c: int, structure_width: int, structure_height: int, grid_width: int, grid_height: int) -> bool:
    """
    Checks if a multi-cell structure can be placed at (r, c) without overlap or going out of bounds.
    """
    # Check if placement is out of bounds
    if r < 0 or r + structure_height > grid_height or c < 0 or c + structure_width > grid_width:
        return False

    # Check for overlap with existing structures (non-zero cells)
    for row_offset in range(structure_height):
        for col_offset in range(structure_width):
            if grid[r + row_offset][c + col_offset] != 0:
                return False # Cell is already occupied
    return True

def apply_priority_zones(r: int, c: int, grid_width: int, grid_height: int, priority_zones: list[str]) -> bool:
    """
    Checks if a coordinate (r, c) satisfies the priority zone rules provided by the LLM.
    """
    if not priority_zones or "any" in priority_zones:
        return True # Default if no specific zones or 'any' is specified

    is_on_edge = (r == 0 or r == grid_height - 1 or c == 0 or c == grid_width - 1)
    is_on_corner = ((r == 0 and (c == 0 or c == grid_width - 1)) or
                    (r == grid_height - 1 and (c == 0 or c == grid_width - 1)))
    # Define center region as inner 50% for 10x10, inner 3x3 or 4x4
    is_in_center_region = (r >= grid_height * 0.3 and r < grid_height * 0.7 and
                            c >= grid_width * 0.3 and c < grid_width * 0.7)

    for zone in priority_zones:
        if zone == "corner" and is_on_corner:
            return True
        if zone == "edge" and is_on_edge and not is_on_corner: # Check for 'edge' only if it's not a corner
            return True
        if zone == "center" and is_in_center_region:
            return True
        # Custom "coastal strip" examples - assuming coastal strips are just edges for simplicity
        if zone == "coastal_strip_0" and (r == 0 or c == 0): # Near top or left edge
            return True
        if zone == f"coastal_strip_{grid_height-1}" and (r == grid_height - 1 or c == grid_width - 1): # Near bottom or right edge
            return True

    return False # No matching priority zone found


def calculate_manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Calculates Manhattan distance between two points."""
    return abs(r1 - r2) + abs(c1 - c2)

def place_structure(grid: list[list[int]], r: int, c: int, structure_id: int, size: list[int]):
    """
    Places a structure on the grid and records its coordinates.
    """
    height, width = size
    for row_offset in range(height):
        for col_offset in range(width):
            if 0 <= r + row_offset < len(grid) and 0 <= c + col_offset < len(grid[0]):
                grid[r + row_offset][c + col_offset] = structure_id
    # Record the placement along with its dimensions in the global tracking dict
    _placed_structures_coords.setdefault(structure_id, []).append((r, c, height, width))


def generate_grid_from_hints(llm_hints: dict, structured_buildings: dict) -> list[list[int]]:
    """
    Generates a 2D grid layout based on LLM-provided placement hints and structure definitions.
    
    Args:
        llm_hints (dict): Parsed hints from the LLM, including grid_dimensions, placement_rules, general_density.
        structured_buildings (dict): Dictionary mapping structure IDs (as strings) to their definitions.

    Returns:
        list[list[int]]: The generated 2D grid layout.
    """
    global _placed_structures_coords
    _placed_structures_coords = {} # Reset for each new grid generation

    grid_dims = llm_hints.get("grid_dimensions", {"width": 10, "height": 10})
    grid_width = grid_dims.get("width", 10)
    grid_height = grid_dims.get("height", 10)
    placement_rules = llm_hints.get("placement_rules", [])
    general_density = llm_hints.get("general_density", 0.3)

    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    
    logger.info(f"Grid dimensions: {grid_width}x{grid_height}. Total rules: {len(placement_rules)}")
    
    # Convert structure IDs in structured_buildings to integers for consistency
    # And augment with size/counts from rules if available
    augmented_buildings = {}
    for struct_id_str, struct_def in structured_buildings.items():
        try:
            struct_id_int = int(struct_id_str)
            # CRITICAL FIX: Use deepcopy to prevent modifying the original structured_buildings
            augmented_buildings[struct_id_int] = copy.deepcopy(struct_def) 
            
            # Find corresponding rule to get size and min/max counts
            rule = next((r for r in placement_rules if r.get('structure_id') == struct_id_int), {}) # Use structure_id from rule
            
            # Ensure size is a list of two integers [height, width]
            size_raw = rule.get('size', [1, 1])
            size = [max(1, int(s)) for s in size_raw] if isinstance(size_raw, list) and len(size_raw) == 2 else [1, 1]
            
            augmented_buildings[struct_id_int]['size'] = size
            augmented_buildings[struct_id_int]['min_count'] = rule.get('min_count', 2)
            augmented_buildings[struct_id_int]['max_count'] = rule.get('max_count', 7)
        except ValueError:
            logger.warning(f"Skipping invalid structure ID '{struct_id_str}' (not an integer) from structured_buildings.")
            continue
        except Exception as e:
            logger.error(f"Error processing structure {struct_id_str} for augmentation: {e}", exc_info=True)
            continue

    if not augmented_buildings:
        logger.warning("No valid structures found in augmented_buildings after processing. Returning empty grid.")
        return grid

    # Order structures by priority (e.g., larger structures first, or those with more constraints)
    structures_to_place = sorted(
        augmented_buildings.items(),
        key=lambda item: (
            len(next((r.get('priority_zones', []) for r in placement_rules if r.get('structure_id') == item[0]), [])), # Use structure_id
            item[1].get('min_count', 0),
            item[1]['size'][0] * item[1]['size'][1]
        ),
        reverse=True
    )

    if not structures_to_place:
        logger.warning("No structures to place after sorting and filtering. Returning empty grid.")
        return grid

    placed_counts = {struct_id: 0 for struct_id in augmented_buildings.keys()}

    logger.info("Phase 1: Attempting to place minimum required structures.")
    # Phase 1: Place minimum required structures
    for struct_id, struct_def in structures_to_place:
        min_count = struct_def.get('min_count', 0)
        size = struct_def.get('size', [1,1])
        
        for _ in range(min_count):
            possible_coords = []
            for r_cand in range(grid_height):
                for c_cand in range(grid_width):
                    if is_valid_placement(grid, r_cand, c_cand, size[0], size[1], grid_width, grid_height):
                        # Contextual rules (avoidance and adjacency) are checked here before adding to possible_coords
                        can_place_due_to_context = True

                        current_rules = next((r for r in placement_rules if r.get('structure_id') == struct_id), {})
                        avoid_ids = current_rules.get('avoid_ids', [])
                        if avoid_ids:
                            for avoid_id in avoid_ids:
                                avoid_id_int = int(avoid_id) 
                                if avoid_id_int in _placed_structures_coords and _placed_structures_coords[avoid_id_int]:
                                    for sr, sc, sh, sw in _placed_structures_coords[avoid_id_int]:
                                        # Check if any part of the *current* structure's footprint
                                        # is too close to any part of the *avoided* structure's footprint
                                        is_too_close = False
                                        for cur_r_offset in range(size[0]): # Iterate through current structure's cells
                                            for cur_c_offset in range(size[1]):
                                                for avoid_r_offset in range(sh): # Iterate through avoided structure's cells
                                                    for avoid_c_offset in range(sw):
                                                        if calculate_manhattan_distance(r_cand + cur_r_offset, c_cand + cur_c_offset, sr + avoid_r_offset, sc + avoid_c_offset) <= 2:
                                                            is_too_close = True
                                                            break
                                                    if is_too_close: break
                                                if is_too_close: break
                                            if is_too_close: break
                                        if is_too_close:
                                            can_place_due_to_context = False
                                            logger.debug(f"Blocked {struct_id} at ({r_cand},{c_cand}) by avoidance {avoid_id} at ({sr},{sc}).")
                                            break
                                    if not can_place_due_to_context: break
                            if not can_place_due_to_context: continue 

                        adjacent_to_ids = current_rules.get('adjacent_to_ids', [])
                        if adjacent_to_ids:
                            any_required_adjacent_type_placed_somewhere = False
                            for adj_id in adjacent_to_ids:
                                adj_id_int = int(adj_id)
                                if adj_id_int in _placed_structures_coords and _placed_structures_coords[adj_id_int]:
                                    any_required_adjacent_type_placed_somewhere = True
                                    break

                            if any_required_adjacent_type_placed_somewhere:
                                found_nearby_adjacent = False
                                for adj_id in adjacent_to_ids:
                                    adj_id_int = int(adj_id)
                                    if adj_id_int in _placed_structures_coords:
                                        for sr, sc, sh, sw in _placed_structures_coords[adj_id_int]:
                                            is_nearby = False
                                            for cur_r_offset in range(size[0]):
                                                for cur_c_offset in range(size[1]):
                                                    for adj_r_offset in range(sh):
                                                        for adj_c_offset in range(sw):
                                                            if calculate_manhattan_distance(r_cand + cur_r_offset, c_cand + cur_c_offset, sr + adj_r_offset, sc + adj_c_offset) <= 3: 
                                                                is_nearby = True
                                                                break
                                                        if is_nearby: break
                                                    if is_nearby: break
                                                if is_nearby: break
                                            if is_nearby:
                                                found_nearby_adjacent = True
                                                break
                                        if found_nearby_adjacent: break
                                if not found_nearby_adjacent:
                                    can_place_due_to_context = False
                                    logger.debug(f"Blocked {struct_id} at ({r_cand},{c_cand}) by missing nearby adjacency.")
                            else:
                                pass # Adjacency rule temporarily ignored in Phase 1 if partners don't exist yet
                        if not can_place_due_to_context: continue
                        
                        # 2. Priority zones
                        if not apply_priority_zones(r_cand, c_cand, grid_width, grid_height, current_rules.get("priority_zones", [])):
                            continue

                        possible_coords.append((r_cand, c_cand))
            
            if possible_coords:
                # Prioritize placement in priority_zones if defined
                current_rules = next((r for r in placement_rules if r.get('structure_id') == struct_id), {}) # Use structure_id
                priority_zones = current_rules.get('priority_zones', [])
                
                eligible_coords = []
                for r_coord, c_coord in possible_coords: # Use r_coord, c_coord to avoid confusion
                    if apply_priority_zones(r_coord, c_coord, grid_width, grid_height, priority_zones):
                        eligible_coords.append((r_coord, c_coord))
                
                if not eligible_coords and priority_zones:
                    eligible_coords = possible_coords
                    logger.debug(f"Falling back to any valid spot for {struct_id} due to no spots in priority zones.")

                if eligible_coords:
                    r_place, c_place = random.choice(eligible_coords)
                    place_structure(grid, r_place, c_place, struct_id, size)
                    placed_counts[struct_id] += 1
                else:
                    logger.warning(f"Could not place minimum {struct_id} (needed {min_count}, placed {placed_counts[struct_id]}) due to rule constraints or no eligible priority spots.")
                    break
            else:
                logger.warning(f"No valid placement found for minimum {struct_id} (needed {min_count}, placed {placed_counts[struct_id]}) with current grid state.")
                break
    
    logger.info(f"Phase 1 complete. Structures placed after min_count attempts: {placed_counts}")

    logger.info("Phase 2: Filling remaining empty cells up to max_count and general density.")
    # Phase 2: Fill remaining empty cells up to max_count and general density
    total_cells = grid_width * grid_height
    target_filled_cells = int(total_cells * general_density)
    current_filled_cells = sum(1 for row in grid for cell in row if cell != 0)

    # Shuffle rules to give a more random distribution when filling,
    # or shuffle structures_to_place directly
    random.shuffle(structures_to_place) 
    
    attempts = 0
    max_attempts = total_cells * 2
    
    while current_filled_cells < target_filled_cells and attempts < max_attempts:
        attempts += 1
        
        if not structures_to_place:
            logger.debug("No structures left to try placing in Phase 2. Breaking loop.")
            break
            
        struct_id, struct_def = random.choice(structures_to_place)
        
        if placed_counts[struct_id] >= struct_def.get('max_count', 1):
            continue

        size = struct_def.get('size', [1,1])

        r_try = random.randint(0, grid_height - 1)
        c_try = random.randint(0, grid_width - 1)

        if is_valid_placement(grid, r_try, c_try, size[0], size[1], grid_width, grid_height):
            can_place_due_to_context_phase2 = True
            
            current_rules = next((r for r in placement_rules if r.get('structure_id') == struct_id), {}) # Use structure_id
            
            # Check avoid_ids rule for this potential placement (strict in Phase 2)
            avoid_ids = current_rules.get('avoid_ids', [])
            if avoid_ids:
                for avoid_id in avoid_ids:
                    avoid_id_int = int(avoid_id) 
                    if avoid_id_int in _placed_structures_coords and _placed_structures_coords[avoid_id_int]:
                        for sr, sc, sh, sw in _placed_structures_coords[avoid_id_int]:
                            is_too_close = False
                            for cur_r_offset in range(size[0]):
                                for cur_c_offset in range(size[1]):
                                    for avoid_r_offset in range(sh):
                                        for avoid_c_offset in range(sw):
                                            if calculate_manhattan_distance(r_try + cur_r_offset, c_try + cur_c_offset, sr + avoid_r_offset, sc + avoid_c_offset) <= 2:
                                                is_too_close = True
                                                break
                                        if is_too_close: break
                                    if is_too_close: break
                                if is_too_close:
                                    can_place_due_to_context_phase2 = False
                                    break
                            if not can_place_due_to_context_phase2: break

            # Check adjacent_to_ids rule (enforced in Phase 2 if partners exist)
            adjacent_to_ids = current_rules.get('adjacent_to_ids', [])
            if can_place_due_to_context_phase2 and adjacent_to_ids:
                any_required_adjacent_type_placed_somewhere = False
                for adj_id in adjacent_to_ids:
                    adj_id_int = int(adj_id)
                    if adj_id_int in _placed_structures_coords and _placed_structures_coords[adj_id_int]:
                        any_required_adjacent_type_placed_somewhere = True
                        break

                if any_required_adjacent_type_placed_somewhere:
                    found_nearby_adjacent = False
                    for adj_id in adjacent_to_ids:
                        adj_id_int = int(adj_id)
                        if adj_id_int in _placed_structures_coords:
                            for sr, sc, sh, sw in _placed_structures_coords[adj_id_int]:
                                is_nearby = False
                                for cur_r_offset in range(size[0]):
                                    for cur_c_offset in range(size[1]):
                                        for adj_r_offset in range(sh):
                                            for adj_c_offset in range(sw):
                                                if calculate_manhattan_distance(r_try + cur_r_offset, c_try + cur_c_offset, sr + adj_r_offset, sc + adj_c_offset) <= 3: 
                                                    is_nearby = True
                                                    break
                                            if is_nearby: break
                                        if is_nearby: break
                                    if is_nearby: break
                                if is_nearby:
                                    found_nearby_adjacent = True
                                    break
                            if not found_nearby_adjacent:
                                can_place_due_to_context_phase2 = False
            
            if can_place_due_to_context_phase2:
                place_structure(grid, r_try, c_try, struct_id, size)
                placed_counts[struct_id] += 1
                current_filled_cells += (size[0] * size[1])
                logger.debug(f"Placed {struct_id} at ({r_try},{c_try}). Total {struct_id}: {placed_counts[struct_id]}")
        else:
            logger.debug(f"Failed to place {struct_id} at ({r_try},{c_try}) due to bounds/overlap.")

    logger.info(f"Grid generation complete ({grid_height}x{grid_width}). Final structures placed (ID: count): " +
                ", ".join([f"{sid}: {len(_placed_structures_coords.get(sid, []))}" for sid in augmented_buildings.keys()])) # Use _placed_structures_coords for final counts
    
    logger.info("Final generated grid layout:")
    for row in grid:
        logger.info(row)

    return grid

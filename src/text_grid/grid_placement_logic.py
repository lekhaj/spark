# src/text_grid/grid_placement_logic.py
import random
import logging

logger = logging.getLogger("GridGenerator")

def is_valid_placement(grid: list[list[int]], r: int, c: int, structure_width: int, structure_height: int, grid_width: int, grid_height: int) -> bool:
    """
    Checks if a multi-cell structure can be placed at (r, c) without overlap or going out of bounds.
    """
    # Check if placement is out of bounds
    if r < 0 or r + structure_height > grid_height or c < 0 or c + structure_width > grid_width:
        # logger.debug(f"Invalid placement: Out of bounds for ({r},{c}) size {structure_width}x{structure_height} in {grid_width}x{grid_height}")
        return False

    # Check for overlap with existing structures (non-zero cells)
    for row_offset in range(structure_height):
        for col_offset in range(structure_width):
            if grid[r + row_offset][c + col_offset] != 0:
                # logger.debug(f"Invalid placement: Overlap at ({r+row_offset},{c+col_offset})")
                return False # Cell is already occupied
    return True

def apply_priority_zones(r: int, c: int, grid_width: int, grid_height: int, priority_zones: list[str]) -> bool:
    """
    Checks if a coordinate (r, c) satisfies the priority zone rules provided by the LLM.
    """
    if not priority_zones or "any" in priority_zones:
        # logger.debug(f"Priority zone 'any' or no zones specified, allowing placement at ({r},{c}).")
        return True # Default if no specific zones or 'any' is specified

    is_on_edge = (r == 0 or r == grid_height - 1 or c == 0 or c == grid_width - 1)
    is_on_corner = ((r == 0 and (c == 0 or c == grid_width - 1)) or
                    (r == grid_height - 1 and (c == 0 or c == grid_width - 1)))
    # Define center region as inner 50% for 10x10, inner 3x3 or 4x4
    # A simplified center check
    is_in_center_region = (r >= grid_height * 0.3 and r < grid_height * 0.7 and
                            c >= grid_width * 0.3 and c < grid_width * 0.7)

    for zone in priority_zones:
        if zone == "corner" and is_on_corner:
            # logger.debug(f"Priority zone 'corner' met at ({r},{c}).")
            return True
        if zone == "edge" and is_on_edge and not is_on_corner: # Check for 'edge' only if it's not a corner
            # logger.debug(f"Priority zone 'edge' met at ({r},{c}).")
            return True
        if zone == "center" and is_in_center_region:
            # logger.debug(f"Priority zone 'center' met at ({r},{c}).")
            return True
        # Custom "coastal strip" examples - assuming coastal strips are just edges for simplicity
        if zone == "coastal_strip_0" and (r == 0 or c == 0): # Near top or left edge
            # logger.debug(f"Priority zone 'coastal_strip_0' met at ({r},{c}).")
            return True
        if zone == f"coastal_strip_{grid_height-1}" and (r == grid_height - 1 or c == grid_width - 1): # Near bottom or right edge
            # logger.debug(f"Priority zone 'coastal_strip_{grid_height-1}' met at ({r},{c}).")
            return True

    # logger.debug(f"No priority zone met for ({r},{c}) with zones: {priority_zones}")
    return False # No matching priority zone found


def calculate_manhattan_distance(r1: int, c1: int, r2: int, c2: int) -> int:
    """Calculates Manhattan distance between two points."""
    return abs(r1 - r2) + abs(c1 - c2)


def generate_grid_from_hints(llm_hints: dict, structured_structures: dict) -> list[list[int]]:
    """
    Generates a 2D grid layout based on LLM-provided placement hints and structure definitions.
    
    Args:
        llm_hints (dict): Parsed hints from the LLM, including grid_dimensions, placement_rules, general_density.
        structured_structures (dict): Dictionary mapping structure IDs (as strings) to their definitions.

    Returns:
        list[list[int]]: The generated 2D grid layout.
    """
    grid_width = llm_hints["grid_dimensions"]["width"]
    grid_height = llm_hints["grid_dimensions"]["height"]
    grid = [[0 for _ in range(grid_width)] for _ in range(grid_height)]
    
    # Store coordinates of placed structures for adjacency checks
    # Keys are integer IDs, values are lists of (r, c, width, height) tuples
    placed_structures_coords: dict[int, list[tuple[int, int, int, int]]] = {int(k): [] for k in structured_structures.keys()} 
    
    rand = random.Random() # Use a local random instance for reproducibility if needed

    # Sort rules: Prioritize rules without adjacent_to_ids (these are "seed" structures)
    # Then by higher minimum counts, then by stricter priority zones, then by size.
    def rule_sort_key(rule: dict) -> int:
        priority_score = 0
        
        # High bonus for rules with no adjacent_to_ids requirements (these are crucial to start populating)
        if not rule.get("adjacent_to_ids"):
            priority_score += 10000 # Make these very high priority

        priority_score += rule.get("min_count", 0) * 100 

        zones = rule.get("priority_zones", [])
        if "corner" in zones:
            priority_score += 50
        elif "edge" in zones:
            priority_score += 30
        elif "center" in zones:
            priority_score += 10
        
        size = rule.get("size", [1,1])
        priority_score += (size[0] * size[1]) # Add area as a tie-breaker

        return priority_score

    sorted_rules = sorted(llm_hints["placement_rules"], key=rule_sort_key, reverse=True)
    logger.info(f"Grid dimensions: {grid_width}x{grid_height}. Total rules: {len(sorted_rules)}")
    logger.debug(f"Sorted placement rules: {sorted_rules}")

    # Phase 1: Place minimum required structures adhering to rules
    logger.info("Phase 1: Attempting to place minimum required structures.")
    
    for rule in sorted_rules:
        struct_id = rule["structure_id"] 
        struct_type_name = rule["type"] 
        min_count = rule["min_count"] 
        
        structure_width = rule["size"][0]
        structure_height = rule["size"][1]
        
        current_placed_count_for_this_rule = len(placed_structures_coords.get(struct_id, []))
        
        if current_placed_count_for_this_rule >= min_count:
            logger.debug(f"Already met min_count for {struct_type_name} ({struct_id}). Placed: {current_placed_count_for_this_rule}")
            continue

        num_to_place_in_phase1 = min_count - current_placed_count_for_this_rule
        logger.info(f"Attempting to place {num_to_place_in_phase1} instances of {struct_type_name} ({struct_id}) for min_count.")
        
        attempts_per_instance = 500 # Increased attempts significantly for challenging placements

        for i in range(num_to_place_in_phase1):
            placed_current_structure = False
            
            all_potential_spots = []
            for r_cand in range(grid_height):
                for c_cand in range(grid_width):
                    all_potential_spots.append((r_cand, c_cand))
            rand.shuffle(all_potential_spots) 

            attempt_count = 0
            for r, c in all_potential_spots:
                attempt_count += 1
                if attempt_count > attempts_per_instance:
                    logger.warning(f"Exceeded max attempts ({attempts_per_instance}) for instance {i+1} of {struct_type_name} ({struct_id}). Giving up on this instance.")
                    break # Give up on this instance if too many attempts

                # 1. Basic validity (bounds and no overlap)
                if not is_valid_placement(grid, r, c, structure_width, structure_height, grid_width, grid_height):
                    logger.debug(f"Skipping ({r},{c}) for {struct_id}: Invalid placement/overlap.")
                    continue

                # 2. Priority zones
                if not apply_priority_zones(r, c, grid_width, grid_height, rule["priority_zones"]):
                    logger.debug(f"Skipping ({r},{c}) for {struct_id}: Fails priority zone.")
                    continue

                # 3. Contextual rules (avoidance and adjacency)
                can_place_due_to_context = True

                # Check avoid_ids rule (always hard block)
                if rule["avoid_ids"]:
                    for avoid_id in rule["avoid_ids"]:
                        if int(avoid_id) in placed_structures_coords and placed_structures_coords[int(avoid_id)]:
                            for sr, sc, sw, sh in placed_structures_coords[int(avoid_id)]:
                                for sr_offset in range(sh):
                                    for sc_offset in range(sw):
                                        if calculate_manhattan_distance(r, c, sr + sr_offset, sc + sc_offset) <= 2: 
                                            can_place_due_to_context = False
                                            logger.debug(f"Blocked {struct_id} at ({r},{c}) by avoidance {avoid_id} at ({sr},{sc}).")
                                            break
                                    if not can_place_due_to_context: break
                                if not can_place_due_to_context: break
                    if not can_place_due_to_context: continue 

                # Check adjacent_to_ids rule for Phase 1 (adaptive enforcement)
                if rule["adjacent_to_ids"]:
                    # Determine if *any* of the required adjacent types have *any* instances already placed on the map.
                    any_required_adjacent_type_placed_somewhere = False
                    for adj_id in rule["adjacent_to_ids"]:
                        if int(adj_id) in placed_structures_coords and placed_structures_coords[int(adj_id)]:
                            any_required_adjacent_type_placed_somewhere = True
                            break

                    if any_required_adjacent_type_placed_somewhere:
                        # If required adjacent types ARE present somewhere, then this structure MUST be nearby one.
                        found_nearby_adjacent = False
                        for adj_id in rule["adjacent_to_ids"]:
                            if int(adj_id) in placed_structures_coords:
                                for sr, sc, sw, sh in placed_structures_coords[int(adj_id)]:
                                    for sr_offset in range(sh):
                                        for sc_offset in range(sw):
                                            if calculate_manhattan_distance(r, c, sr + sr_offset, sc + sc_offset) <= 3: 
                                                found_nearby_adjacent = True
                                                break
                                        if found_nearby_adjacent: break
                                if found_nearby_adjacent: break
                        
                        if not found_nearby_adjacent:
                            can_place_due_to_context = False
                            logger.debug(f"Blocked {struct_id} at ({r},{c}) by missing nearby adjacency.")
                    else:
                        # If NONE of the required adjacent types are placed anywhere yet,
                        # this adjacency rule is TEMPORARILY IGNORED in Phase 1 to allow initial seeding.
                        pass 
                
                if can_place_due_to_context:
                    # Place the structure cells in the grid
                    for row_offset in range(structure_height):
                        for col_offset in range(structure_width):
                            grid[r + row_offset][c + col_offset] = struct_id
                    
                    # Record the placement for future adjacency checks
                    placed_structures_coords.setdefault(struct_id, []).append((r, c, structure_width, structure_height))
                    placed_current_structure = True
                    logger.info(f"Placed {struct_type_name} ({struct_id}) at ({r},{c}). Current count: {len(placed_structures_coords[struct_id])}")
                    break # Structure placed, move to next instance for this rule
            
            if not placed_current_structure:
                logger.warning(f"Could not place instance {i+1}/{num_to_place_in_phase1} of {struct_type_name} ({struct_id}) for min_count (no valid spot found after {attempt_count} attempts). Placed: {len(placed_structures_coords.get(struct_id, []))}.")
                # If an instance cannot be placed, it implies current constraints are too strict for remaining minimum
                # Continue with next rule, as we tried enough for this one, to avoid getting stuck.
                break 

    logger.info(f"Phase 1 complete. Structures placed after min_count attempts: {placed_structures_coords}")

    # Phase 2: Fill remaining empty cells up to max_count based on general density
    logger.info("Phase 2: Filling remaining empty cells up to max_count and general density.")
    
    all_grid_coords_for_phase2 = [(r, c) for r in range(grid_height) for c in range(grid_width)]
    rand.shuffle(all_grid_coords_for_phase2)

    for r, c in all_grid_coords_for_phase2:
        if grid[r][c] == 0: 
            if rand.random() < llm_hints["general_density"]: 
                
                available_rules_for_filling = [
                    rule for rule in llm_hints["placement_rules"]
                    if len(placed_structures_coords.get(rule["structure_id"], [])) < rule["max_count"]
                    and rule["size"] == [1,1] 
                    and is_valid_placement(grid, r, c, 1, 1, grid_width, grid_height) 
                    and apply_priority_zones(r, c, grid_width, grid_height, rule["priority_zones"])
                ]
                
                if available_rules_for_filling:
                    # Sort by least common first to try to fill diverse structures
                    available_rules_for_filling.sort(key=lambda rule: len(placed_structures_coords.get(rule["structure_id"], [])))
                    
                    chosen_rule = None
                    # Try to pick a rule that also satisfies adjacency/avoidance if possible
                    for rule_option in available_rules_for_filling:
                        struct_id_option = rule_option["structure_id"]
                        
                        can_place_due_to_context_phase2 = True
                        # Check avoid_ids rule for this potential placement (strict in Phase 2)
                        if rule_option["avoid_ids"]:
                            for avoid_id in rule_option["avoid_ids"]:
                                if int(avoid_id) in placed_structures_coords and placed_structures_coords[int(avoid_id)]:
                                    for sr, sc, sw, sh in placed_structures_coords[int(avoid_id)]:
                                        for sr_offset in range(sh):
                                            for sc_offset in range(sw):
                                                if calculate_manhattan_distance(r, c, sr + sr_offset, sc + sc_offset) <= 2:
                                                    can_place_due_to_context_phase2 = False
                                                    break
                                            if not can_place_due_to_context_phase2: break
                                    if not can_place_due_to_context_phase2: break
                        
                        # Check adjacent_to_ids rule (enforced in Phase 2 if partners exist)
                        if can_place_due_to_context_phase2 and rule_option["adjacent_to_ids"]:
                            any_required_adjacent_type_placed_somewhere = False
                            for adj_id in rule_option["adjacent_to_ids"]:
                                if int(adj_id) in placed_structures_coords and placed_structures_coords[int(adj_id)]:
                                    any_required_adjacent_type_placed_somewhere = True
                                    break

                            if any_required_adjacent_type_placed_somewhere:
                                found_nearby_adjacent = False
                                for adj_id in rule_option["adjacent_to_ids"]:
                                    if int(adj_id) in placed_structures_coords:
                                        for sr, sc, sw, sh in placed_structures_coords[int(adj_id)]:
                                            for sr_offset in range(sh):
                                                for sc_offset in range(sw):
                                                    if calculate_manhattan_distance(r, c, sr + sr_offset, sc + sc_offset) <= 3: 
                                                        found_nearby_adjacent = True
                                                        break
                                                if found_nearby_adjacent: break
                                        if found_nearby_adjacent: break
                                if not found_nearby_adjacent:
                                    can_place_due_to_context_phase2 = False

                        if can_place_due_to_context_phase2:
                            chosen_rule = rule_option
                            break 
                    
                    if chosen_rule: 
                        struct_id = chosen_rule["structure_id"]
                        grid[r][c] = struct_id
                        placed_structures_coords.setdefault(struct_id, []).append((r, c, 1, 1))

    logger.info(f"Grid generation complete ({grid_height}x{grid_width}). Final structures placed (ID: count): " +
                ", ".join([f"{sid}: {len(coords)}" for sid, coords in placed_structures_coords.items()]))
    
    logger.info("Final generated grid layout:")
    for row in grid:
        logger.info(row)

    return grid

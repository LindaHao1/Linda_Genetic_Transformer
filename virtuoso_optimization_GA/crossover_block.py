"""
crossover_block.py — Block-based crossover operator.

Algorithm
---------
1. Sample a random bounding-box size h×w (each dimension in [block_min_size, block_max_size]).
2. Sample Box A (r1, c1) inside parent1 and Box B (r2, c2) inside parent2.
3. Edge constraint: columns 0 and cols-1 are feed lines.
   If Box A touches the left edge (c1==0), force c2=0.
   If Box A touches the right edge (c1+w-1==cols-1), force c2=cols-w.
   If width spans the full grid, return None.
4. Sample transfer direction uniformly:
   dir=0 → base=parent2, overwrite [r2:r2+h, c2:c2+w] with parent1's [r1:r1+h, c1:c1+w]
   dir=1 → base=parent1, overwrite [r1:r1+h, c1:c1+w] with parent2's [r2:r2+h, c2:c2+w]
5. Track pixel_source {(r,c): 1 or 2} for value inheritance.
6. Run repair pipeline: fix_feed_cols → fix_inductors → fix_switches → fix_vias → fix_connectivity.
7. Inherit continuous/discrete component values from originating parent.
8. Final feasibility check; return None if invalid.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from virtuoso_optimization_SA.design import DesignState
from virtuoso_optimization_SA.feasibility import is_feasible

from .crossover import (
    _build_pos_map,
    _fix_connectivity,
    _fix_inductors,
    _fix_switches,
    _fix_vias,
)
from virtuoso_optimization_SA.feasibility import filter_valid_pairs
from .crossover_weighted_clustering import (
    _fix_feed_cols,
    _inherit_values_clustered,
)

_MAX_PLACEMENT_ATTEMPTS = 20


def _inherit_cap_pairs_block(
    child_matrix: List[List[int]],
    parent_base: DesignState,
    parent_donor: DesignState,
    base_row: int,
    base_col: int,
    donor_row: int,
    donor_col: int,
    h: int,
    w: int,
    max_cap_pairs: int,
    min_dist: int,
    max_dist: int,
    rng: random.Random,
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Inherit cap pixel pairs for a block crossover child, respecting the spatial
    transformation applied by the block swap.

    The block region [base_row:base_row+h, base_col:base_col+w] in the child
    was overwritten by the donor's block [donor_row:donor_row+h, donor_col:donor_col+w].
    The offset that maps donor block coords → child coords is:
        (base_row - donor_row, base_col - donor_col)

    Strategy
    --------
    Base parent caps:
      - Both endpoints outside the base block → keep as-is.
      - One endpoint inside the base block (partial) → translate that endpoint
        by the *reverse* offset (donor_row - base_row, donor_col - base_col);
        i.e. map the base-block cell to where the donor version lives, then
        apply the forward offset.  Net effect: the inside endpoint stays at the
        same child position (the donor filled it), so it is translated by 0 —
        which means we translate it by (donor_row - base_row, donor_col - base_col)
        then back by (base_row - donor_row, base_col - donor_col) = no change.
        Simpler formulation: translate the inside endpoint by (0, 0) — i.e. keep
        it at its base position, but it is now occupied by donor content so it
        may be invalid.  We keep the pair and let filter_valid_pairs decide.
      - Both endpoints inside the base block → drop (overwritten by donor).

    Donor parent caps:
      - Endpoints inside the donor block → translate by (base_row - donor_row,
        base_col - donor_col) to map to the child position.
      - Endpoints outside the donor block → keep at their original position
        (they don't exist in the child, so drop them).
      - Mixed (one in, one out) → translate the inside endpoint, keep the
        outside endpoint at its original position.  If either translated
        endpoint goes out of bounds, drop the pair.

    After collecting all candidate pairs, deduplicate by canonical endpoint key
    (average value if same pair appears from both parents), validate via
    filter_valid_pairs, shuffle, trim to max_cap_pairs.
    """
    if max_cap_pairs == 0:
        return [], []

    rows = len(child_matrix)
    cols = len(child_matrix[0])

    dr_offset = base_row - donor_row
    dc_offset = base_col - donor_col

    def _in_base_block(r: int, c: int) -> bool:
        return base_row <= r < base_row + h and base_col <= c < base_col + w

    def _in_donor_block(r: int, c: int) -> bool:
        return donor_row <= r < donor_row + h and donor_col <= c < donor_col + w

    def _in_bounds(r: int, c: int) -> bool:
        return 0 <= r < rows and 0 <= c < cols

    # Canonical key for deduplication
    def _key(ep1: List[int], ep2: List[int]) -> Tuple:
        a, b = tuple(ep1), tuple(ep2)
        return (min(a, b), max(a, b))

    pair_pool: Dict[Tuple, float] = {}

    # --- Base parent caps ---
    for pair, val in zip(parent_base.cap_pixel_pairs, parent_base.cap_pair_values):
        (r1, c1), (r2, c2) = pair[0], pair[1]
        in1 = _in_base_block(r1, c1)
        in2 = _in_base_block(r2, c2)
        if in1 and in2:
            continue  # fully inside overwritten region — drop
        # Keep partial or fully-outside pairs; let filter_valid_pairs validate
        k = _key(pair[0], pair[1])
        if k not in pair_pool:
            pair_pool[k] = val

    # --- Donor parent caps ---
    for pair, val in zip(parent_donor.cap_pixel_pairs, parent_donor.cap_pair_values):
        (r1, c1), (r2, c2) = pair[0], pair[1]
        in1 = _in_donor_block(r1, c1)
        in2 = _in_donor_block(r2, c2)
        if not in1 and not in2:
            continue  # fully outside donor block — not transplanted into child

        # Translate endpoints that are inside the donor block
        nr1 = r1 + dr_offset if in1 else r1
        nc1 = c1 + dc_offset if in1 else c1
        nr2 = r2 + dr_offset if in2 else r2
        nc2 = c2 + dc_offset if in2 else c2

        if not _in_bounds(nr1, nc1) or not _in_bounds(nr2, nc2):
            continue  # translated out of grid — drop

        k = _key([nr1, nc1], [nr2, nc2])
        if k in pair_pool:
            pair_pool[k] = (pair_pool[k] + val) / 2.0
        else:
            pair_pool[k] = val

    if not pair_pool:
        return [], []

    candidate_pairs = [[[k[0][0], k[0][1]], [k[1][0], k[1][1]]] for k in pair_pool]
    candidate_values = [pair_pool[k] for k in pair_pool]

    valid_pairs, valid_values = filter_valid_pairs(
        child_matrix, candidate_pairs, candidate_values, min_dist, max_dist
    )

    combined = list(zip(valid_pairs, valid_values))
    rng.shuffle(combined)
    combined = combined[:max_cap_pairs]
    if not combined:
        return [], []
    result_pairs, result_values = zip(*combined)
    return list(result_pairs), list(result_values)


def crossover_block(
    parent1: DesignState,
    parent2: DesignState,
    rng: random.Random,
    min_switches: int,
    max_switches: int,
    max_inductors: int,
    n_states: int,
    cap_min: float,
    cap_max: float,
    n_inductor_values: int = 4,
    max_vias: int = 0,
    block_min_size: int = 5,
    block_max_size: int = 15,
    max_cap_pairs: int = 0,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    single_inductor_value: bool = False,
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
) -> Tuple[Optional[DesignState], dict]:
    """
    Produce one child by block-based crossover of two parent designs.

    A rectangular block from one parent replaces the corresponding region in
    the other parent. Block size is drawn uniformly in [block_min_size, block_max_size]
    for each dimension. The transfer direction (which parent donates the block)
    is chosen at random each call.

    Returns (child, params) where child is a valid DesignState (or None if repair
    fails) and params describes the block crossover parameters used.
    """
    rows = parent1.rows
    cols = parent1.cols

    block_min_size = max(1, block_min_size)
    block_max_size = min(block_max_size, min(rows, cols))
    if block_min_size > block_max_size:
        return None, {}

    # Build position maps before any changes (for value inheritance)
    sw_map_p1  = _build_pos_map(parent1.matrix, 3)
    sw_map_p2  = _build_pos_map(parent2.matrix, 3)
    ind_map_p1 = _build_pos_map(parent1.matrix, 2)
    ind_map_p2 = _build_pos_map(parent2.matrix, 2)

    # Sample block dimensions
    h = rng.randint(block_min_size, block_max_size)
    w = rng.randint(block_min_size, block_max_size)

    # A block spanning the full width cannot satisfy the edge constraint
    if w >= cols:
        return None, {}

    # Sample Box A position in parent1
    for _ in range(_MAX_PLACEMENT_ATTEMPTS):
        r1 = rng.randint(0, rows - h)
        c1 = rng.randint(0, cols - w)

        # Determine valid column range for Box B based on edge constraint
        c2_min = 0
        c2_max = cols - w

        touches_left  = (c1 == 0)
        touches_right = (c1 + w - 1 == cols - 1)

        if touches_left and touches_right:
            # Box A spans full width — already blocked above, but guard here too
            continue
        elif touches_left:
            c2_min = c2_max = 0
        elif touches_right:
            c2_min = c2_max = cols - w

        if c2_min > c2_max:
            continue

        # Sample Box B position in parent2
        r2 = rng.randint(0, rows - h)
        c2 = rng.randint(c2_min, c2_max)

        # Sample transfer direction
        direction = rng.randint(0, 1)
        xover_params = {
            "block_h": h,
            "block_w": w,
            "direction": "p1_into_p2" if direction == 0 else "p2_into_p1",
        }

        if direction == 0:
            # Base = parent2; insert parent1's block into Box B position
            base_parent  = parent2
            donor_parent = parent1
            base_row, base_col     = r2, c2
            donor_row, donor_col   = r1, c1
            base_src_val  = 2   # pixel_source value for base pixels
            donor_src_val = 1   # pixel_source value for donor pixels
        else:
            # Base = parent1; insert parent2's block into Box A position
            base_parent  = parent1
            donor_parent = parent2
            base_row, base_col     = r1, c1
            donor_row, donor_col   = r2, c2
            base_src_val  = 1
            donor_src_val = 2

        # Build child matrix: deep copy of base, then overwrite the block region
        child_matrix: List[List[int]] = [row[:] for row in base_parent.matrix]
        pixel_source: Dict[Tuple[int, int], int] = {}

        for r in range(rows):
            for c in range(cols):
                pixel_source[(r, c)] = base_src_val

        for dr in range(h):
            for dc in range(w):
                r_dst = base_row  + dr
                c_dst = base_col  + dc
                r_src = donor_row + dr
                c_src = donor_col + dc
                child_matrix[r_dst][c_dst] = donor_parent.matrix[r_src][c_src]
                pixel_source[(r_dst, c_dst)] = donor_src_val

        # Repair pipeline
        _fix_feed_cols(child_matrix, rng)
        _fix_inductors(child_matrix, max_inductors, rng)
        _fix_switches(child_matrix, min_switches, max_switches, rng)
        if max_vias > 0:
            _fix_vias(child_matrix, max_vias, rng)
        if not _fix_connectivity(child_matrix, rng):
            continue   # try a new placement

        # Feasibility gate
        if not is_feasible(
            child_matrix, 0, max_inductors,
            min_switches=min_switches, max_switches=max_switches, max_vias=max_vias,
        ):
            continue

        # Value inheritance
        switch_states, cap_values, inductor_values, inductor_switch_states, inductor_cap_values = (
            _inherit_values_clustered(
                child_matrix,
                pixel_source,
                parent1,
                parent2,
                sw_map_p1,
                sw_map_p2,
                ind_map_p1,
                ind_map_p2,
                n_states,
                cap_min,
                cap_max,
                n_inductor_values,
                rng,
                ind_sw_states_p1=parent1.inductor_switch_states,
                ind_sw_states_p2=parent2.inductor_switch_states,
                inductor_cap_min=inductor_cap_min,
                inductor_cap_max=inductor_cap_max,
            )
        )

        # Enforce single inductor value: randomly inherit one parent's shared value
        if single_inductor_value and inductor_values:
            donor = rng.choice([parent1, parent2])
            val = donor.inductor_values[0] if donor.inductor_values else rng.randint(0, n_inductor_values - 1)
            inductor_values = [val] * len(inductor_values)

        # Inherit cap pixel pairs, translating them with the block transformation
        cap_pixel_pairs, cap_pair_values = _inherit_cap_pairs_block(
            child_matrix,
            base_parent,
            donor_parent,
            base_row,
            base_col,
            donor_row,
            donor_col,
            h,
            w,
            max_cap_pairs,
            cap_pair_min_dist,
            cap_pair_max_dist,
            rng,
        )

        return DesignState(
            matrix                 = child_matrix,
            switch_states          = switch_states,
            cap_values             = cap_values,
            inductor_values        = inductor_values,
            inductor_switch_states = inductor_switch_states,
            cap_pixel_pairs        = cap_pixel_pairs,
            cap_pair_values        = cap_pair_values,
            inductor_cap_values    = inductor_cap_values,
        ), xover_params

    return None, {"block_h": h, "block_w": w}  # all placement attempts exhausted

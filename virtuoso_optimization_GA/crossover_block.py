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

import copy
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
from .crossover_weighted_clustering import (
    _fix_feed_cols,
    _inherit_values_clustered,
)

_MAX_PLACEMENT_ATTEMPTS = 20


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
) -> Optional[DesignState]:
    """
    Produce one child by block-based crossover of two parent designs.

    A rectangular block from one parent replaces the corresponding region in
    the other parent. Block size is drawn uniformly in [block_min_size, block_max_size]
    for each dimension. The transfer direction (which parent donates the block)
    is chosen at random each call.

    Returns a valid DesignState, or None if the repair pipeline fails.
    """
    rows = parent1.rows
    cols = parent1.cols

    block_min_size = max(1, block_min_size)
    block_max_size = min(block_max_size, min(rows, cols))
    if block_min_size > block_max_size:
        return None

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
        return None

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
        switch_states, cap_values, inductor_values, inductor_switch_states = (
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
            )
        )

        return DesignState(
            matrix                 = child_matrix,
            switch_states          = switch_states,
            cap_values             = cap_values,
            inductor_values        = inductor_values,
            inductor_switch_states = inductor_switch_states,
        )

    return None  # all placement attempts exhausted

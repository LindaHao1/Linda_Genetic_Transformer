"""
crossover_weighted_clustering.py — Weighted clustering hybrid crossover operator.

Algorithm
---------
1. Initialize n_centroids centroids at random positions in the pixel grid.
2. Assign each centroid randomly to parent1 or parent2 (at least one per parent).
3. Each centroid has a random weight in [0.5, 1.0] that scales its distance metric.
4. Assign each pixel (r, c) to the centroid with minimum weighted Euclidean distance.
5. Build the child matrix by taking each pixel from the centroid's assigned parent.
6. Repair feed columns (I/O ports): keep exactly one non-zero pixel per feed column.
7. Run the standard repair pipeline (inductors → switches → vias → connectivity).
8. Final feasibility check; return None if still invalid.

Value inheritance
-----------------
For each component in the child matrix, values (cap, switch states, inductor index)
are inherited from whichever parent's cluster covered that pixel position. If the
position didn't exist in that parent's original matrix, a random value is used.
"""

from __future__ import annotations

import math
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
    _inherit_cap_pairs,
)
from virtuoso_optimization_SA.feasibility import filter_valid_pairs


# ---------------------------------------------------------------------------
# Feed column repair
# ---------------------------------------------------------------------------

def _fix_feed_cols(matrix: List[List[int]], rng: random.Random) -> None:
    """
    Ensure exactly one non-zero pixel per feed column (col 0 and col cols-1).

    - If a feed column has more than one non-zero pixel, keep one at random
      and zero out the rest.
    - If a feed column has zero non-zero pixels, place a type-1 pixel at a
      random row (connectivity repair handles actual path-finding afterwards).
    """
    rows = len(matrix)
    cols = len(matrix[0])

    for feed_col in (0, cols - 1):
        nonzero_rows = [r for r in range(rows) if matrix[r][feed_col] != 0]
        if len(nonzero_rows) > 1:
            keep = rng.choice(nonzero_rows)
            for r in nonzero_rows:
                if r != keep:
                    matrix[r][feed_col] = 0
        elif len(nonzero_rows) == 0:
            r = rng.randint(0, rows - 1)
            matrix[r][feed_col] = 1


# ---------------------------------------------------------------------------
# Value inheritance (cluster-based)
# ---------------------------------------------------------------------------

def _inherit_values_clustered(
    child_matrix: List[List[int]],
    pixel_source: Dict[Tuple[int, int], int],   # {(r,c): 1 or 2}
    parent1: DesignState,
    parent2: DesignState,
    sw_map_p1: Dict[Tuple[int, int], int],
    sw_map_p2: Dict[Tuple[int, int], int],
    ind_map_p1: Dict[Tuple[int, int], int],
    ind_map_p2: Dict[Tuple[int, int], int],
    n_states: int,
    cap_min: float,
    cap_max: float,
    n_inductor_values: int,
    rng: random.Random,
    ind_sw_states_p1: List[List[int]] = (),
    ind_sw_states_p2: List[List[int]] = (),
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
) -> Tuple[List[List[List[int]]], List[List[float]], List[int], List[List[int]], List[float]]:
    """
    Walk the child matrix in row-major order and build:
      - switch_states       [n_states][n_child_switches][n_caps_in_bank]
      - cap_values          [n_child_switches][n_caps_in_bank]
      - inductor_values     [n_child_inductors]
      - inductor_switch_states [n_states][n_child_inductors]
      - inductor_cap_values [n_child_inductors]

    Inheritance rule: pixel_source[(r, c)] == 1 → inherit from parent1,
                      pixel_source[(r, c)] == 2 → inherit from parent2.
    If the position did not exist in the originating parent, assign random value.
    """
    rows = len(child_matrix)
    cols = len(child_matrix[0])

    _new_sw_bank_size = (len(parent1.cap_values[0]) if parent1.cap_values else
                         len(parent2.cap_values[0]) if parent2.cap_values else 1)

    switch_states: List[List[List[int]]] = [[] for _ in range(n_states)]
    cap_values:    List[List[float]]     = []
    inductor_values: List[int]           = []
    inductor_cap_values: List[float]     = []

    p1_has_ind_sw = len(ind_sw_states_p1) > 0
    p2_has_ind_sw = len(ind_sw_states_p2) > 0
    inductor_switch_states: List[List[int]] = [[] for _ in range(n_states)]

    for r in range(rows):
        for c in range(cols):
            v = child_matrix[r][c]

            if v == 3:  # switch
                from_p1 = pixel_source.get((r, c), 1) == 1
                if from_p1 and (r, c) in sw_map_p1:
                    p_idx = sw_map_p1[(r, c)]
                    cap_values.append(list(parent1.cap_values[p_idx]))
                    bank_sz = len(parent1.cap_values[p_idx])
                    for si in range(n_states):
                        if p_idx < len(parent1.switch_states[si]):
                            switch_states[si].append(list(parent1.switch_states[si][p_idx]))
                        else:
                            switch_states[si].append([rng.randint(0, 1) for _ in range(bank_sz)])
                elif (not from_p1) and (r, c) in sw_map_p2:
                    p_idx = sw_map_p2[(r, c)]
                    cap_values.append(list(parent2.cap_values[p_idx]))
                    bank_sz = len(parent2.cap_values[p_idx])
                    for si in range(n_states):
                        if p_idx < len(parent2.switch_states[si]):
                            switch_states[si].append(list(parent2.switch_states[si][p_idx]))
                        else:
                            switch_states[si].append([rng.randint(0, 1) for _ in range(bank_sz)])
                else:
                    cap_values.append([rng.uniform(cap_min, cap_max) for _ in range(_new_sw_bank_size)])
                    for si in range(n_states):
                        switch_states[si].append([rng.randint(0, 1) for _ in range(_new_sw_bank_size)])

            elif v == 2:  # inductor
                from_p1 = pixel_source.get((r, c), 1) == 1
                if from_p1 and (r, c) in ind_map_p1:
                    p_idx = ind_map_p1[(r, c)]
                    val = (parent1.inductor_values[p_idx]
                           if p_idx < len(parent1.inductor_values)
                           else rng.randint(0, n_inductor_values - 1))
                    inductor_values.append(val)
                    ind_cap = (parent1.inductor_cap_values[p_idx]
                               if p_idx < len(parent1.inductor_cap_values)
                               else rng.uniform(inductor_cap_min, inductor_cap_max))
                    inductor_cap_values.append(ind_cap)
                    for si in range(n_states):
                        if p1_has_ind_sw and p_idx < len(ind_sw_states_p1[si]):
                            inductor_switch_states[si].append(ind_sw_states_p1[si][p_idx])
                        else:
                            inductor_switch_states[si].append(rng.randint(0, 1))
                elif (not from_p1) and (r, c) in ind_map_p2:
                    p_idx = ind_map_p2[(r, c)]
                    val = (parent2.inductor_values[p_idx]
                           if p_idx < len(parent2.inductor_values)
                           else rng.randint(0, n_inductor_values - 1))
                    inductor_values.append(val)
                    ind_cap = (parent2.inductor_cap_values[p_idx]
                               if p_idx < len(parent2.inductor_cap_values)
                               else rng.uniform(inductor_cap_min, inductor_cap_max))
                    inductor_cap_values.append(ind_cap)
                    for si in range(n_states):
                        if p2_has_ind_sw and p_idx < len(ind_sw_states_p2[si]):
                            inductor_switch_states[si].append(ind_sw_states_p2[si][p_idx])
                        else:
                            inductor_switch_states[si].append(rng.randint(0, 1))
                else:
                    inductor_values.append(rng.randint(0, n_inductor_values - 1))
                    inductor_cap_values.append(rng.uniform(inductor_cap_min, inductor_cap_max))
                    for si in range(n_states):
                        inductor_switch_states[si].append(rng.randint(0, 1))

    if not inductor_values:
        inductor_switch_states = []

    return switch_states, cap_values, inductor_values, inductor_switch_states, inductor_cap_values


# ---------------------------------------------------------------------------
# Position-aware cap pair inheritance for clustering crossover
# ---------------------------------------------------------------------------

def _inherit_cap_pairs_clustered(
    pixel_source: Dict[Tuple[int, int], int],
    parent1: DesignState,
    parent2: DesignState,
    child_matrix: List[List[int]],
    max_cap_pairs: int,
    min_dist: int,
    max_dist: int,
    rng: random.Random,
) -> Tuple[List, List[float]]:
    """
    Inherit cap pixel pairs from parents using the cluster pixel_source map.

    A pair from parent1 is kept only if both of its endpoints fall in a region
    assigned to parent1 (pixel_source == 1).  Likewise for parent2.  This mirrors
    the position-aware logic used by the block crossover.

    Returns (cap_pixel_pairs, cap_pair_values).
    """
    if max_cap_pairs == 0:
        return [], []

    def _key(pair: List) -> Tuple:
        ep1, ep2 = tuple(pair[0]), tuple(pair[1])
        return (min(ep1, ep2), max(ep1, ep2))

    pair_pool: Dict[Tuple, float] = {}

    for pair, val in zip(parent1.cap_pixel_pairs, parent1.cap_pair_values):
        r1, c1 = pair[0]
        r2, c2 = pair[1]
        if pixel_source.get((r1, c1), 1) == 1 and pixel_source.get((r2, c2), 1) == 1:
            k = _key(pair)
            pair_pool[k] = val

    for pair, val in zip(parent2.cap_pixel_pairs, parent2.cap_pair_values):
        r1, c1 = pair[0]
        r2, c2 = pair[1]
        if pixel_source.get((r1, c1), 2) == 2 and pixel_source.get((r2, c2), 2) == 2:
            k = _key(pair)
            if k not in pair_pool:
                pair_pool[k] = val
            else:
                pair_pool[k] = (pair_pool[k] + val) / 2.0  # both parents had same pair

    if not pair_pool:
        return [], []

    candidate_pairs  = [[[k[0][0], k[0][1]], [k[1][0], k[1][1]]] for k in pair_pool]
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crossover_weighted_clustering(
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
    n_centroids_max: int = 4,
    max_cap_pairs: int = 0,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    single_inductor_value: bool = False,
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
    grid_size: Optional[float] = None,
    connect_margin: float = 2.0,
) -> Tuple[Optional[DesignState], dict]:
    """
    Produce one child by weighted-clustering crossover of two parent designs.

    The number of centroids is drawn uniformly from [2, n_centroids_max] each
    call, so boundary complexity varies across offspring. Each centroid is
    assigned randomly to one of the two parents with the guarantee that both
    parents contribute at least one centroid. A per-centroid weight in [0.5, 1.0]
    scales the Euclidean distance, producing richer, non-linear cluster boundaries.

    Cap pixel pairs are inherited position-aware: a pair is kept only if both
    of its endpoints belong to a cluster region assigned to the same parent.

    Returns (child, params) where child is a valid DesignState (or None if repair
    fails) and params describes the crossover parameters used.
    """
    rows = parent1.rows
    cols = parent1.cols

    sizing_on = bool(parent1.pixel_sizes) and bool(parent2.pixel_sizes)

    # --- Randomly draw number of centroids for this call ---
    n_centroids_max = max(2, n_centroids_max)
    n_centroids = rng.randint(2, n_centroids_max)
    xover_params = {"n_centroids": n_centroids}

    # --- Build position maps BEFORE any changes ---
    sw_map_p1  = _build_pos_map(parent1.matrix, 3)
    sw_map_p2  = _build_pos_map(parent2.matrix, 3)
    ind_map_p1 = _build_pos_map(parent1.matrix, 2)
    ind_map_p2 = _build_pos_map(parent2.matrix, 2)

    # --- Initialize centroids and per-centroid weights ---
    centroids = [
        (rng.uniform(0, rows - 1), rng.uniform(0, cols - 1))
        for _ in range(n_centroids)
    ]
    weights = [rng.uniform(0.5, 1.0) for _ in range(n_centroids)]

    # --- Assign centroids to parents (random, but guarantee both appear) ---
    # Start with a fully random assignment, then fix it if one parent is missing.
    centroid_parent = [rng.choice([1, 2]) for _ in range(n_centroids)]
    if 1 not in centroid_parent:
        centroid_parent[rng.randrange(n_centroids)] = 1
    if 2 not in centroid_parent:
        centroid_parent[rng.randrange(n_centroids)] = 2

    # --- Pixel-to-cluster assignment ---
    pixel_source: Dict[Tuple[int, int], int] = {}
    for r in range(rows):
        for c in range(cols):
            best_idx = 0
            best_dist = math.inf
            for i, (cr, cc) in enumerate(centroids):
                d = math.sqrt((r - cr) ** 2 + (c - cc) ** 2) * weights[i]
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            pixel_source[(r, c)] = centroid_parent[best_idx]

    # --- Build child matrix (+ sizing matrix mirrored on the same source map) ---
    child_matrix: List[List[int]] = []
    child_pixel_sizes: List[List[float]] = [] if sizing_on else []
    for r in range(rows):
        row = []
        size_row = []
        for c in range(cols):
            src = pixel_source[(r, c)]
            row.append(parent1.matrix[r][c] if src == 1 else parent2.matrix[r][c])
            if sizing_on:
                size_row.append(
                    parent1.pixel_sizes[r][c] if src == 1 else parent2.pixel_sizes[r][c]
                )
        child_matrix.append(row)
        if sizing_on:
            child_pixel_sizes.append(size_row)

    # --- Fix feed columns (I/O port repair) ---
    _fix_feed_cols(child_matrix, rng)

    # --- Repair pipeline ---
    _fix_inductors(child_matrix, max_inductors, rng)
    _fix_switches(child_matrix, min_switches, max_switches, rng)
    if max_vias > 0:
        _fix_vias(child_matrix, max_vias, rng)
    if not _fix_connectivity(child_matrix, rng):
        return None, xover_params

    # --- Final feasibility gate ---
    if not is_feasible(
        child_matrix, 0, max_inductors,
        min_switches=min_switches, max_switches=max_switches, max_vias=max_vias,
        pixel_sizes=(child_pixel_sizes if sizing_on else None),
        grid_size=grid_size, connect_margin=connect_margin,
    ):
        return None, xover_params

    # --- Inherit continuous/discrete values from parents ---
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

    # --- Inherit cap pixel pairs: position-aware using cluster assignment ---
    cap_pixel_pairs, cap_pair_values = _inherit_cap_pairs_clustered(
        pixel_source,
        parent1,
        parent2,
        child_matrix,
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
        pixel_sizes            = child_pixel_sizes if sizing_on else [],
    ), xover_params

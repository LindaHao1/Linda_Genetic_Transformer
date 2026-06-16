"""
crossover.py — Zig-zag crossover operator for the switchable-filter GA.

Algorithm
---------
1. Randomly choose orientation: "vertical" (column cut) or "horizontal" (row cut).
2. Pick a base cut position uniformly at random within safe bounds.
3. Build a Markovian jagged boundary: each step has 0.95 probability of continuing
   in the same direction, 0.05 probability of choosing a new random direction.
4. Build the child matrix by splitting at the jagged boundary.
5. Repair the child in order:
      a. Fix inductors  — remove spacing violators, then trim to max_inductors
      b. Fix switches   — remove spacing violators, then adjust count to exact target
      c. Fix connectivity — bridge disconnected halves along a free interior row/col
6. Final full feasibility check; return None if still invalid.

Cap / inductor value inheritance
----------------------------------
Before any repair, we build position→index lookup tables for both parents.
After repair, every component position in the child is resolved back to its
parent's list and the corresponding value is inherited verbatim.
Components that were added by the repair receive random values.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List, Optional, Tuple

from virtuoso_optimization_SA.design import DesignState
from virtuoso_optimization_SA.feasibility import (
    _can_place_switch_at,
    _neighbours_8,
    _special_pixel_too_close,
    check_connectivity,
    filter_valid_pairs,
    is_feasible,
)



# ---------------------------------------------------------------------------
# Position → index lookup helpers
# ---------------------------------------------------------------------------

def _build_pos_map(matrix: List[List[int]], pixel_type: int) -> Dict[Tuple[int, int], int]:
    """Return {(r, c): row-major index} for all cells of pixel_type."""
    idx = 0
    result: Dict[Tuple[int, int], int] = {}
    for r, row in enumerate(matrix):
        for c, v in enumerate(row):
            if v == pixel_type:
                result[(r, c)] = idx
                idx += 1
    return result


# ---------------------------------------------------------------------------
# Repair helpers
# ---------------------------------------------------------------------------

def _fix_inductors(
    matrix: List[List[int]],
    max_inductors: int,
    rng: random.Random,
) -> None:
    """
    Repair inductors in-place:
    1. Remove all inductors (type 2) that violate the spacing rule
       (Chebyshev distance < 3 to any other type-2 or type-3 pixel).
    2. If the count still exceeds max_inductors, randomly remove the excess.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Pass 1: spacing violations — inductor footprint overlap
    violators = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 2:
                if _special_pixel_too_close(matrix, r, c, new_type=2):
                    violators.append((r, c))
    # Remove in random order to avoid order bias
    rng.shuffle(violators)
    for r, c in violators:
        if matrix[r][c] == 2:          # may already be cleared
            matrix[r][c] = 1

    # Pass 2: count trim
    ind_positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if matrix[r][c] == 2
    ]
    excess = len(ind_positions) - max_inductors
    if excess > 0:
        rng.shuffle(ind_positions)
        for r, c in ind_positions[:excess]:
            matrix[r][c] = 1


def _fix_switches(
    matrix: List[List[int]],
    min_switches: int,
    max_switches: int,
    rng: random.Random,
) -> None:
    """
    Repair switches in-place:
    1. Remove switch pixels (type 3) that violate the spacing rule
       (Chebyshev distance < 3 to any other type-2 or type-3 pixel).
    2. Adjust switch count to be within [min_switches, max_switches]:
       - If above max: randomly remove the excess switches.
       - If below min: randomly place switches on eligible type-1 interior pixels.
       - If within range: leave the count as-is.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Pass 1: spacing violations — switch footprint overlap
    violators = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 3:
                if _special_pixel_too_close(matrix, r, c, new_type=3):
                    violators.append((r, c))
    rng.shuffle(violators)
    for r, c in violators:
        if matrix[r][c] == 3:
            matrix[r][c] = 1

    # Pass 2: count adjustment
    sw_positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if matrix[r][c] == 3
    ]
    n_current = len(sw_positions)

    if n_current > max_switches:
        # Remove excess randomly
        rng.shuffle(sw_positions)
        for r, c in sw_positions[:n_current - max_switches]:
            matrix[r][c] = 1

    elif n_current < min_switches:
        # Add switches up to min_switches
        need = min_switches - n_current
        eligible = [
            (r, c)
            for r in range(rows)
            for c in range(1, cols - 1)
            if matrix[r][c] == 1 and _can_place_switch_at(matrix, r, c)
        ]
        rng.shuffle(eligible)
        placed = 0
        for r, c in eligible:
            if placed >= need:
                break
            if _can_place_switch_at(matrix, r, c):
                matrix[r][c] = 3
                placed += 1


def _fix_vias(
    matrix: List[List[int]],
    max_vias: int,
    rng: random.Random,
) -> None:
    """
    Repair via pixels (type 4) in-place:
    1. Remove type-4 pixels whose 8-neighbourhood contains any non-{0,1} pixel.
    2. If the count still exceeds max_vias, randomly remove the excess.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Pass 1: spacing violations
    violators = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 4:
                for nr, nc in _neighbours_8(r, c, rows, cols):
                    if matrix[nr][nc] not in (0, 1):
                        violators.append((r, c))
                        break
    rng.shuffle(violators)
    for r, c in violators:
        if matrix[r][c] == 4:
            matrix[r][c] = 1

    # Pass 2: count trim
    via_positions = [
        (r, c) for r in range(rows) for c in range(cols) if matrix[r][c] == 4
    ]
    excess = len(via_positions) - max_vias
    if excess > 0:
        rng.shuffle(via_positions)
        for r, c in via_positions[:excess]:
            matrix[r][c] = 1


def _fix_shunt_caps(
    matrix: List[List[int]],
    max_shunt_caps: int,
    rng: random.Random,
) -> None:
    """
    Repair shunt-capacitor pixels (type 5) in-place:
    1. Remove type-5 pixels that violate the 3×3 footprint-spacing rule
       (Chebyshev distance < required to any other special-footprint pixel,
       i.e. inductor/switch/shunt-cap) — same rule a switch gets.
    2. Trim the remaining count down to ``max_shunt_caps``.

    A combined (crossover) matrix can place two parents' shunt caps within each
    other's 3×3 footprint; without pass 1 those children would be rejected by
    is_feasible instead of repaired.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Pass 1: spacing violations — shunt-cap footprint overlap
    violators = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if matrix[r][c] == 5 and _special_pixel_too_close(matrix, r, c, new_type=5)
    ]
    rng.shuffle(violators)
    for r, c in violators:
        if matrix[r][c] == 5:
            matrix[r][c] = 1

    # Pass 2: count trim
    positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if matrix[r][c] == 5
    ]
    excess = len(positions) - max_shunt_caps
    if excess > 0:
        rng.shuffle(positions)
        for r, c in positions[:excess]:
            matrix[r][c] = 1


def _inherit_shunt_caps(
    child_matrix: List[List[int]],
    parent1: DesignState,
    parent2: DesignState,
    rng: random.Random,
    shunt_cap_min: float = 50e-15,
    shunt_cap_max: float = 1000e-15,
) -> List[float]:
    """
    Build ``shunt_cap_values`` for a child matrix in row-first type-5 order.

    Each type-5 cell in the child came directly from a parent half (the repair
    pipeline never creates type-5 pixels), so its value is inherited from
    whichever parent had a shunt cap at that exact (row, col).  Parent1 takes
    precedence on the rare collision where both parents have one at the same
    cell; a genuinely new position (should not occur) falls back to a random
    value in range.
    """
    vm1 = {pos: idx for pos, idx in _build_pos_map(parent1.matrix, 5).items()}
    vm2 = {pos: idx for pos, idx in _build_pos_map(parent2.matrix, 5).items()}

    values: List[float] = []
    for r in range(len(child_matrix)):
        for c in range(len(child_matrix[0])):
            if child_matrix[r][c] != 5:
                continue
            val: Optional[float] = None
            if (r, c) in vm1 and vm1[(r, c)] < len(parent1.shunt_cap_values):
                val = parent1.shunt_cap_values[vm1[(r, c)]]
            elif (r, c) in vm2 and vm2[(r, c)] < len(parent2.shunt_cap_values):
                val = parent2.shunt_cap_values[vm2[(r, c)]]
            if val is None:
                val = rng.uniform(shunt_cap_min, shunt_cap_max)
            values.append(val)
    return values


def _fix_connectivity(
    matrix: List[List[int]],
    rng: random.Random,
) -> bool:
    """
    Repair connectivity in-place using a bridging strategy:

    1. BFS from the left feed pixel → record max column reached.
    2. BFS from the right feed pixel → record min column reached.
    3. If the two frontiers already touch or overlap, connectivity is fine.
    4. Otherwise pick the bridging row that minimises the gap and set all
       interior cells between the two frontiers to type 1 (metal).
    5. Re-check; return False if connectivity cannot be established.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    def _bfs_reachable(start_r: int, start_c: int) -> set:
        visited: set = set()
        q: deque = deque()
        q.append((start_r, start_c))
        visited.add((start_r, start_c))
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbours_8(r, c, rows, cols):
                is_diagonal = (nr != r) and (nc != c)
                in_feed_col = (c == 0 or c == cols - 1 or nc == 0 or nc == cols - 1)
                if is_diagonal and in_feed_col:
                    continue
                if (nr, nc) not in visited and matrix[nr][nc] != 0:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return visited

    if check_connectivity(matrix):
        return True

    # Find feed pixels
    left_row  = next((r for r in range(rows) if matrix[r][0]        != 0), None)
    right_row = next((r for r in range(rows) if matrix[r][cols - 1] != 0), None)
    if left_row is None or right_row is None:
        return False

    left_reachable  = _bfs_reachable(left_row,  0)
    right_reachable = _bfs_reachable(right_row, cols - 1)

    # Rightmost column reached from left, leftmost from right
    right_frontier = max(c for _, c in left_reachable)
    left_frontier  = min(c for _, c in right_reachable)

    if right_frontier >= left_frontier:
        # Frontiers overlap — already connected (shouldn't happen, but guard)
        return check_connectivity(matrix)

    # Choose bridging row: prefer one that already has some non-zero pixels
    # between the two frontiers (minimises newly introduced pixels).
    gap_start = right_frontier + 1
    gap_end   = left_frontier - 1   # inclusive
    if gap_end < gap_start:
        # Frontiers are adjacent — just need a single connecting pixel
        gap_start = right_frontier + 1
        gap_end   = gap_start

    def _row_non_zero_in_gap(r: int) -> int:
        return sum(1 for c in range(gap_start, gap_end + 1) if matrix[r][c] != 0)

    best_row = max(range(rows), key=_row_non_zero_in_gap)

    # Bridge: set all cells in [gap_start, gap_end] of best_row to type 1
    for c in range(gap_start, gap_end + 1):
        if matrix[best_row][c] == 0:
            matrix[best_row][c] = 1

    return check_connectivity(matrix)


# ---------------------------------------------------------------------------
# Value inheritance helper
# ---------------------------------------------------------------------------

def _inherit_values(
    child_matrix: List[List[int]],
    cut_cols: List[int],          # cut_cols[r] = first column taken from parent2
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
    ind_sw_states_p1: List[List[int]] = (),  # parent1.inductor_switch_states
    ind_sw_states_p2: List[List[int]] = (),  # parent2.inductor_switch_states
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
) -> Tuple[List[List[List[int]]], List[List[float]], List[int], List[List[int]], List[float]]:
    """
    Walk the child matrix in row-major order and build:
      - switch_states       [n_states][n_child_switches][n_caps_in_bank]
      - cap_values          [n_child_switches][n_caps_in_bank]
      - inductor_values     [n_child_inductors]
      - inductor_cap_values [n_child_inductors]

    Inheritance rule:
      - Component at (r, c) with c <  cut_cols[r]  → was in parent1's half
      - Component at (r, c) with c >= cut_cols[r]  → was in parent2's half
    If the position existed in the originating parent's matrix, inherit its
    value directly.  Otherwise assign a random value.
    """
    rows = len(child_matrix)
    cols = len(child_matrix[0])

    # Determine cap bank size from parents (variable per switch, but use representative size for new switches).
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
                from_p1 = c < cut_cols[r]
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
                    # Newly placed switch (from repair)
                    cap_values.append([rng.uniform(cap_min, cap_max) for _ in range(_new_sw_bank_size)])
                    for si in range(n_states):
                        switch_states[si].append([rng.randint(0, 1) for _ in range(_new_sw_bank_size)])

            elif v == 2:  # inductor
                from_p1 = c < cut_cols[r]
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

    # If no inductors in child, use empty list (all-ON convention)
    if not inductor_values:
        inductor_switch_states = []

    return switch_states, cap_values, inductor_values, inductor_switch_states, inductor_cap_values


def _inherit_values_horizontal(
    child_matrix: List[List[int]],
    cut_rows: List[int],           # cut_rows[c] = first row taken from parent2
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
    Like _inherit_values() but for a horizontal (row-direction) cut.

    Inheritance rule:
      - Component at (r, c) with r <  cut_rows[c]  → was in parent1's half
      - Component at (r, c) with r >= cut_rows[c]  → was in parent2's half
    """
    rows = len(child_matrix)
    cols = len(child_matrix[0])

    # Determine cap bank size from parents (use representative size for new switches).
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
                from_p1 = r < cut_rows[c]
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
                from_p1 = r < cut_rows[c]
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
# Cap pair inheritance helper
# ---------------------------------------------------------------------------

def _inherit_cap_pairs(
    child_matrix: List[List[int]],
    parent1: DesignState,
    parent2: DesignState,
    max_cap_pairs: int,
    min_dist: int,
    max_dist: int,
    rng: random.Random,
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Combine cap pixel pairs from both parents, validate them against the child
    matrix, and return up to max_cap_pairs valid pairs with inherited values.

    Strategy:
      1. Pool all pairs from parent1 and parent2 (deduplicated by endpoints).
      2. For duplicates (same endpoints in both parents), use the average value.
      3. Run filter_valid_pairs() to drop pairs invalidated by the new matrix.
      4. Shuffle and trim to max_cap_pairs.

    Returns (cap_pixel_pairs, cap_pair_values).
    If both parents have no pairs, returns ([], []).
    """
    if max_cap_pairs == 0:
        return [], []

    # Build a dict: canonical endpoint tuple → value (average if both parents have it)
    pair_pool: Dict[Tuple, float] = {}

    def _key(pair: List[List[int]]) -> Tuple:
        ep1, ep2 = tuple(pair[0]), tuple(pair[1])
        return (min(ep1, ep2), max(ep1, ep2))

    for pair, val in zip(parent1.cap_pixel_pairs, parent1.cap_pair_values):
        k = _key(pair)
        pair_pool[k] = val

    for pair, val in zip(parent2.cap_pixel_pairs, parent2.cap_pair_values):
        k = _key(pair)
        if k in pair_pool:
            pair_pool[k] = (pair_pool[k] + val) / 2.0  # average
        else:
            pair_pool[k] = val

    if not pair_pool:
        return [], []

    # Convert back to list form for filter_valid_pairs
    candidate_pairs = [[[k[0][0], k[0][1]], [k[1][0], k[1][1]]] for k in pair_pool]
    candidate_values = [pair_pool[k] for k in pair_pool]

    # Validate against the child matrix
    valid_pairs, valid_values = filter_valid_pairs(
        child_matrix, candidate_pairs, candidate_values, min_dist, max_dist
    )

    # Shuffle and trim
    combined = list(zip(valid_pairs, valid_values))
    rng.shuffle(combined)
    combined = combined[:max_cap_pairs]

    if not combined:
        return [], []
    result_pairs, result_values = zip(*combined)
    return list(result_pairs), list(result_values)


# ---------------------------------------------------------------------------
# Markovian path helper
# ---------------------------------------------------------------------------

_MARKOV_P_SAME = 0.95  # probability of continuing in the same direction


def _zig_zag_markov_path(
    n_steps: int,
    base_cut: int,
    lo: int,
    hi: int,
    rng: random.Random,
) -> List[int]:
    """
    Generate a jagged cut-position path of length n_steps using a Markovian walk.

    Starting from base_cut, each step has _MARKOV_P_SAME probability of continuing
    in the same direction and (1 - _MARKOV_P_SAME) probability of choosing a new
    random direction from {-1, 0, +1}.  All positions are clamped to [lo, hi].
    """
    direction = rng.choice([-1, 0, 1])
    pos = base_cut
    positions: List[int] = []
    for _ in range(n_steps):
        positions.append(max(lo, min(hi, pos)))
        if rng.random() < _MARKOV_P_SAME:
            pos += direction
        else:
            direction = rng.choice([-1, 0, 1])
            pos += direction
    return positions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crossover_zig_zag(
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
    max_cap_pairs: int = 0,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    single_inductor_value: bool = False,
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
    grid_size: Optional[float] = None,
    connect_margin: float = 2.0,
    max_shunt_caps: int = 0,
    shunt_cap_min: float = 50e-15,
    shunt_cap_max: float = 1000e-15,
) -> Tuple[Optional[DesignState], dict]:
    """
    Produce one child by zig-zag crossover of two parent designs.

    Randomly chooses between a vertical cut (column-based, left/right halves) and a
    horizontal cut (row-based, top/bottom halves).  The jagged boundary is generated
    using a Markovian walk with probability _MARKOV_P_SAME of continuing direction.

    Returns (child, params) where child is a valid DesignState (or None if the repair
    pipeline fails) and params is a dict describing the crossover parameters used.
    """
    rows = parent1.rows
    cols = parent1.cols

    # Per-pixel sizing is active only when both parents carry a sizing matrix.
    sizing_on = bool(parent1.pixel_sizes) and bool(parent2.pixel_sizes)

    # --- Build position maps BEFORE any changes ---
    sw_map_p1  = _build_pos_map(parent1.matrix, 3)
    sw_map_p2  = _build_pos_map(parent2.matrix, 3)
    ind_map_p1 = _build_pos_map(parent1.matrix, 2)
    ind_map_p2 = _build_pos_map(parent2.matrix, 2)

    # --- Choose orientation ---
    orientation = rng.choice(["vertical", "horizontal"])

    if orientation == "vertical":
        # Vertical cut: split each row at a jagged column boundary
        if cols < 3:
            return None, {"orientation": orientation}
        base_cut = rng.randint(1, cols - 2)
        cut_cols = _zig_zag_markov_path(rows, base_cut, 1, cols - 2, rng)

        child_matrix: List[List[int]] = []
        for r in range(rows):
            cut = cut_cols[r]
            child_matrix.append(
                list(parent1.matrix[r][:cut]) + list(parent2.matrix[r][cut:])
            )

        # Mirror the same jagged cut on the sizing matrix so each child cell
        # keeps the size of the parent it came from.
        child_pixel_sizes: List[List[float]] = []
        if sizing_on:
            for r in range(rows):
                cut = cut_cols[r]
                child_pixel_sizes.append(
                    list(parent1.pixel_sizes[r][:cut]) + list(parent2.pixel_sizes[r][cut:])
                )

        xover_params = {"orientation": "vertical", "base_cut": base_cut}

    else:
        # Horizontal cut: split each column at a jagged row boundary
        if rows < 3:
            return None, {"orientation": orientation}
        base_cut = rng.randint(1, rows - 2)
        cut_rows = _zig_zag_markov_path(cols, base_cut, 1, rows - 2, rng)

        child_matrix = [[0] * cols for _ in range(rows)]
        child_pixel_sizes = [[0.0] * cols for _ in range(rows)] if sizing_on else []
        for r in range(rows):
            for c in range(cols):
                from_p1 = r < cut_rows[c]
                child_matrix[r][c] = (
                    parent1.matrix[r][c] if from_p1 else parent2.matrix[r][c]
                )
                if sizing_on:
                    child_pixel_sizes[r][c] = (
                        parent1.pixel_sizes[r][c] if from_p1 else parent2.pixel_sizes[r][c]
                    )

        xover_params = {"orientation": "horizontal", "base_cut": base_cut}

    # --- Repair pipeline ---
    _fix_inductors(child_matrix, max_inductors, rng)
    _fix_switches(child_matrix, min_switches, max_switches, rng)
    if max_vias > 0:
        _fix_vias(child_matrix, max_vias, rng)
    if max_shunt_caps > 0:
        _fix_shunt_caps(child_matrix, max_shunt_caps, rng)
    if not _fix_connectivity(child_matrix, rng):
        return None, xover_params

    # --- Final feasibility gate ---
    if not is_feasible(child_matrix, 0, max_inductors, min_switches=min_switches, max_switches=max_switches, max_vias=max_vias,
                       pixel_sizes=(child_pixel_sizes if sizing_on else None),
                       grid_size=grid_size, connect_margin=connect_margin):
        return None, xover_params

    # --- Inherit continuous/discrete values from parents ---
    if orientation == "vertical":
        switch_states, cap_values, inductor_values, inductor_switch_states, inductor_cap_values = _inherit_values(
            child_matrix,
            cut_cols,
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
    else:
        switch_states, cap_values, inductor_values, inductor_switch_states, inductor_cap_values = _inherit_values_horizontal(
            child_matrix,
            cut_rows,
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

    # Enforce single inductor value: randomly inherit one parent's shared value
    if single_inductor_value and inductor_values:
        donor = rng.choice([parent1, parent2])
        val = donor.inductor_values[0] if donor.inductor_values else rng.randint(0, n_inductor_values - 1)
        inductor_values = [val] * len(inductor_values)

    # --- Inherit cap pixel pairs from parents ---
    cap_pixel_pairs, cap_pair_values = _inherit_cap_pairs(
        child_matrix, parent1, parent2, max_cap_pairs, cap_pair_min_dist, cap_pair_max_dist, rng
    )

    # --- Inherit shunt-capacitor values (type-5 pixels) ---
    shunt_cap_values = _inherit_shunt_caps(
        child_matrix, parent1, parent2, rng, shunt_cap_min, shunt_cap_max
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
        shunt_cap_values       = shunt_cap_values,
        pixel_sizes            = child_pixel_sizes if sizing_on else [],
    ), xover_params

"""
crossover.py — Zig-zag crossover operator for the switchable-filter GA.

Algorithm
---------
1. Pick a base cut column c in [1, cols-2] uniformly at random.
2. For each row apply a zig-zag offset (±1), producing a jagged boundary.
3. Build the child matrix:
      child[r][:cut_r]   ← parent1.matrix[r][:cut_r]   (left half)
      child[r][cut_r:]   ← parent2.matrix[r][cut_r:]   (right half)
4. Repair the child in order:
      a. Fix inductors  — remove spacing violators, then trim to max_inductors
      b. Fix switches   — remove spacing violators, then adjust count to exact target
      c. Fix connectivity — bridge disconnected halves along a free interior row
5. Final full feasibility check; return None if still invalid.

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
    check_connectivity,
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
       (any type-2 or type-3 neighbour).
    2. If the count still exceeds max_inductors, randomly remove the excess.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Pass 1: spacing violations
    violators = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 2:
                for nr, nc in _neighbours_8(r, c, rows, cols):
                    if matrix[nr][nc] in (2, 3):
                        violators.append((r, c))
                        break
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
    n_switches_required: int,
    rng: random.Random,
) -> None:
    """
    Repair switches in-place:
    1. Remove switch pixels (type 3) that violate the spacing rule
       (any type-2 or type-3 neighbour).
    2. Adjust switch count to exactly n_switches_required:
       - If below: randomly place switches on eligible type-1 interior pixels.
       - If above: randomly remove the excess switches.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Pass 1: spacing violations
    violators = []
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 3:
                for nr, nc in _neighbours_8(r, c, rows, cols):
                    if matrix[nr][nc] in (2, 3):
                        violators.append((r, c))
                        break
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

    if n_current > n_switches_required:
        # Remove excess randomly
        rng.shuffle(sw_positions)
        for r, c in sw_positions[:n_current - n_switches_required]:
            matrix[r][c] = 1

    elif n_current < n_switches_required:
        # Add missing switches at eligible type-1 interior pixels
        need = n_switches_required - n_current
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
) -> Tuple[List[List[int]], List[float], List[int], List[List[int]]]:
    """
    Walk the child matrix in row-major order and build:
      - switch_states  [n_states][n_child_switches]
      - cap_values     [n_child_switches]
      - inductor_values[n_child_inductors]

    Inheritance rule:
      - Component at (r, c) with c <  cut_cols[r]  → was in parent1's half
      - Component at (r, c) with c >= cut_cols[r]  → was in parent2's half
    If the position existed in the originating parent's matrix, inherit its
    value directly.  Otherwise assign a random value.
    """
    rows = len(child_matrix)
    cols = len(child_matrix[0])

    switch_states: List[List[int]] = [[] for _ in range(n_states)]
    cap_values:    List[float]     = []
    inductor_values: List[int]     = []

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
                    cap_values.append(parent1.cap_values[p_idx])
                    for si in range(n_states):
                        if p_idx < len(parent1.switch_states[si]):
                            switch_states[si].append(parent1.switch_states[si][p_idx])
                        else:
                            switch_states[si].append(rng.randint(0, 1))
                elif (not from_p1) and (r, c) in sw_map_p2:
                    p_idx = sw_map_p2[(r, c)]
                    cap_values.append(parent2.cap_values[p_idx])
                    for si in range(n_states):
                        if p_idx < len(parent2.switch_states[si]):
                            switch_states[si].append(parent2.switch_states[si][p_idx])
                        else:
                            switch_states[si].append(rng.randint(0, 1))
                else:
                    # Newly placed switch (from repair)
                    cap_values.append(rng.uniform(cap_min, cap_max))
                    for si in range(n_states):
                        switch_states[si].append(rng.randint(0, 1))

            elif v == 2:  # inductor
                from_p1 = c < cut_cols[r]
                if from_p1 and (r, c) in ind_map_p1:
                    p_idx = ind_map_p1[(r, c)]
                    val = (parent1.inductor_values[p_idx]
                           if p_idx < len(parent1.inductor_values)
                           else rng.randint(0, n_inductor_values - 1))
                    inductor_values.append(val)
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
                    for si in range(n_states):
                        if p2_has_ind_sw and p_idx < len(ind_sw_states_p2[si]):
                            inductor_switch_states[si].append(ind_sw_states_p2[si][p_idx])
                        else:
                            inductor_switch_states[si].append(rng.randint(0, 1))
                else:
                    inductor_values.append(rng.randint(0, n_inductor_values - 1))
                    for si in range(n_states):
                        inductor_switch_states[si].append(rng.randint(0, 1))

    # If no inductors in child, use empty list (all-ON convention)
    if not inductor_values:
        inductor_switch_states = []

    return switch_states, cap_values, inductor_values, inductor_switch_states


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def crossover_zig_zag(
    parent1: DesignState,
    parent2: DesignState,
    rng: random.Random,
    n_switches_required: int,
    max_inductors: int,
    n_states: int,
    cap_min: float,
    cap_max: float,
    n_inductor_values: int = 4,
) -> Optional[DesignState]:
    """
    Produce one child by zig-zag crossover of two parent designs.

    Returns a valid DesignState, or None if the repair pipeline fails to
    produce a feasible design.
    """
    rows = parent1.rows
    cols = parent1.cols

    # --- Build position maps BEFORE any changes ---
    sw_map_p1  = _build_pos_map(parent1.matrix, 3)
    sw_map_p2  = _build_pos_map(parent2.matrix, 3)
    ind_map_p1 = _build_pos_map(parent1.matrix, 2)
    ind_map_p2 = _build_pos_map(parent2.matrix, 2)

    # --- Zig-zag cut ---
    if cols < 3:
        # Not enough interior columns for a meaningful cut
        return None
    base_cut = rng.randint(1, cols - 2)
    cut_cols = [
        max(1, min(cols - 2, base_cut + rng.randint(-1, 1)))
        for _ in range(rows)
    ]

    # --- Hybrid matrix ---
    child_matrix: List[List[int]] = []
    for r in range(rows):
        cut = cut_cols[r]
        child_matrix.append(
            list(parent1.matrix[r][:cut]) + list(parent2.matrix[r][cut:])
        )

    # --- Repair pipeline ---
    _fix_inductors(child_matrix, max_inductors, rng)
    _fix_switches(child_matrix, n_switches_required, rng)
    if not _fix_connectivity(child_matrix, rng):
        return None

    # --- Final feasibility gate ---
    if not is_feasible(child_matrix, n_switches_required, max_inductors):
        return None

    # --- Inherit continuous/discrete values from parents ---
    switch_states, cap_values, inductor_values, inductor_switch_states = _inherit_values(
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
    )

    return DesignState(
        matrix                 = child_matrix,
        switch_states          = switch_states,
        cap_values             = cap_values,
        inductor_values        = inductor_values,
        inductor_switch_states = inductor_switch_states,
    )

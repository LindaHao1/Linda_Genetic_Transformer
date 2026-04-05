"""
feasibility.py — Design-rule checks and random design generation.

Design rules
------------
a. Connectivity:     8-connected path (no 0 pixels) from the feed pixel in col 0
                     to the feed pixel in col N-1.
b. Switch spacing:   No other switch (3) or inductor (2) may be in the 8-neighbourhood
                     of any switch pixel.
c. Inductor spacing: No other inductor (2) or switch (3) may be in the 8-neighbourhood
                     of any inductor pixel.
d. Inductor count:   Total number of type-2 pixels ≤ max_inductors.
e. Switch count:     Total number of type-3 pixels == n_switches_required.
f. Feed columns:     Exactly one non-zero pixel in col 0 AND in col N-1.
"""

from __future__ import annotations

import random
from collections import deque
from typing import List, Optional, Tuple

from .design import DesignState


# ---------------------------------------------------------------------------
# Neighbourhood helper
# ---------------------------------------------------------------------------

def _neighbours_8(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Return all valid (row, col) 8-neighbours of (r, c)."""
    result = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                result.append((nr, nc))
    return result


def _neighbours_4(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
    """Return all valid (row, col) 4-neighbours (up/down/left/right) of (r, c)."""
    result = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

def check_connectivity(matrix: List[List[int]], four_connected: bool = False) -> bool:
    """
    Rule a: BFS from the feed pixel in col 0 to the feed pixel in col N-1,
    traversing only cells with value != 0.

    When ``four_connected=False`` (default) 8-connectivity is used, but
    diagonal steps are suppressed at the feed columns.
    When ``four_connected=True`` strict 4-connectivity (up/down/left/right
    only) is enforced throughout the entire grid.
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Find feed pixels
    left_row  = next((r for r in range(rows) if matrix[r][0]        != 0), None)
    right_row = next((r for r in range(rows) if matrix[r][cols - 1] != 0), None)
    if left_row is None or right_row is None:
        return False
    if left_row == right_row and cols == 1:
        return True

    # BFS
    visited = [[False] * cols for _ in range(rows)]
    q: deque = deque()
    q.append((left_row, 0))
    visited[left_row][0] = True

    while q:
        r, c = q.popleft()
        if r == right_row and c == cols - 1:
            return True
        if four_connected:
            neighbours = _neighbours_4(r, c, rows, cols)
        else:
            neighbours = _neighbours_8(r, c, rows, cols)
            # Diagonal steps are not allowed when either endpoint is in a feed column
            neighbours = [
                (nr, nc) for nr, nc in neighbours
                if not (
                    (nr != r) and (nc != c)
                    and (c == 0 or c == cols - 1 or nc == 0 or nc == cols - 1)
                )
            ]
        for nr, nc in neighbours:
            if not visited[nr][nc] and matrix[nr][nc] != 0:
                visited[nr][nc] = True
                q.append((nr, nc))

    return False


def check_connectivity_4way(matrix: List[List[int]]) -> bool:
    """Convenience wrapper: 4-connected path from left to right feed pixel."""
    return check_connectivity(matrix, four_connected=True)


def check_spacing(matrix: List[List[int]], pixel_type: int) -> bool:
    """
    Rules b/c: for every pixel of ``pixel_type``, none of its 8-neighbours
    may be a switch (3) or an inductor (2).
    """
    rows = len(matrix)
    cols = len(matrix[0])
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == pixel_type:
                for nr, nc in _neighbours_8(r, c, rows, cols):
                    if matrix[nr][nc] in (2, 3):
                        return False
    return True


def check_switch_spacing(matrix: List[List[int]]) -> bool:
    """Rule b: switches must not neighbour other switches or inductors."""
    return check_spacing(matrix, 3)


def check_inductor_spacing(matrix: List[List[int]]) -> bool:
    """Rule c: inductors must not neighbour other inductors or switches."""
    return check_spacing(matrix, 2)


def check_inductor_count(matrix: List[List[int]], max_inductors: int) -> bool:
    """Rule d: total type-2 pixels ≤ max_inductors."""
    n = sum(v == 2 for row in matrix for v in row)
    return n <= max_inductors


def check_switch_count(matrix: List[List[int]], n_switches_required: int) -> bool:
    """Rule e: total type-3 pixels == n_switches_required."""
    n = sum(v == 3 for row in matrix for v in row)
    return n == n_switches_required


def check_switch_count_range(matrix: List[List[int]], min_sw: int, max_sw: int) -> bool:
    """Rule e (relaxed): total type-3 pixels in [min_sw, max_sw]."""
    n = sum(v == 3 for row in matrix for v in row)
    return min_sw <= n <= max_sw


def check_switch_states_distinct(switch_states: List[List[int]]) -> bool:
    """Check that no two switch state vectors are identical.

    Ensures each filter state has a unique switch configuration so the two
    states are genuinely different operating points.
    """
    n = len(switch_states)
    for i in range(n):
        for j in range(i + 1, n):
            if switch_states[i] == switch_states[j]:
                return False
    return True


def check_feed_columns(matrix: List[List[int]]) -> bool:
    """Rule f: exactly one non-zero pixel in col 0 and col N-1."""
    rows = len(matrix)
    cols = len(matrix[0])
    left_count  = sum(1 for r in range(rows) if matrix[r][0]        != 0)
    right_count = sum(1 for r in range(rows) if matrix[r][cols - 1] != 0)
    return left_count == 1 and right_count == 1


def is_feasible(
    matrix: List[List[int]],
    n_switches_required: int,
    max_inductors: int,
    *,
    min_switches: Optional[int] = None,
    max_switches: Optional[int] = None,
    four_connected: bool = False,
) -> bool:
    """Full feasibility check (all rules a–f).

    When ``min_switches`` / ``max_switches`` are provided the switch count is
    checked against [min_switches, max_switches] instead of the exact value
    ``n_switches_required``.  SA and GA callers omit these keyword arguments
    and get the original strict-count behaviour unchanged.

    When ``four_connected=True`` the connectivity rule uses 4-way (Manhattan)
    connectivity instead of the default 8-way.
    """
    if min_switches is not None or max_switches is not None:
        lo = min_switches if min_switches is not None else n_switches_required
        hi = max_switches if max_switches is not None else n_switches_required
        sw_ok = check_switch_count_range(matrix, lo, hi)
    else:
        sw_ok = check_switch_count(matrix, n_switches_required)

    return (
        check_feed_columns(matrix)
        and check_connectivity(matrix, four_connected=four_connected)
        and check_switch_spacing(matrix)
        and check_inductor_spacing(matrix)
        and check_inductor_count(matrix, max_inductors)
        and sw_ok
    )


def is_feasible_no_switch_count(
    matrix: List[List[int]],
    max_inductors: int,
    *,
    four_connected: bool = False,
) -> bool:
    """Feasibility check excluding the exact switch count rule (used during generation)."""
    return (
        check_feed_columns(matrix)
        and check_connectivity(matrix, four_connected=four_connected)
        and check_inductor_spacing(matrix)
        and check_inductor_count(matrix, max_inductors)
    )


# ---------------------------------------------------------------------------
# Can a given cell host a switch (type-3) safely?
# ---------------------------------------------------------------------------

def _can_place_switch_at(
    matrix: List[List[int]],
    r: int,
    c: int,
) -> bool:
    """
    Return True if placing a switch at (r, c) would not violate rule b.
    Assumes the cell is currently type 1 (metal).
    Also checks that this cell is not in the feed columns.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    # Feed columns are reserved; do not place switches there
    if c == 0 or c == cols - 1:
        return False
    for nr, nc in _neighbours_8(r, c, rows, cols):
        if matrix[nr][nc] in (2, 3):
            return False
    return True


def _can_place_inductor_at(
    matrix: List[List[int]],
    r: int,
    c: int,
) -> bool:
    """
    Return True if placing an inductor at (r, c) would not violate rule c.
    Assumes the cell is currently type 1 (metal).
    """
    rows = len(matrix)
    cols = len(matrix[0])
    if c == 0 or c == cols - 1:
        return False
    for nr, nc in _neighbours_8(r, c, rows, cols):
        if matrix[nr][nc] in (2, 3):
            return False
    return True


# ---------------------------------------------------------------------------
# Random design generation
# ---------------------------------------------------------------------------

def generate_random_design(
    rows: int,
    cols: int,
    n_switches: int,
    max_inductors: int,
    n_states: int,
    cap_min: float,
    cap_max: float,
    rng: random.Random,
    n_inductor_values: int = 4,
    p0: float = 0.399,   # probability of type 0 (ground) for interior cells
    p1: float = 0.60,   # probability of type 1 (metal)
    p2: float = 0.001,   # probability of type 2 (inductor)
    max_attempts: int = 10_000,
    require_4_connected: bool = False,
) -> Optional[DesignState]:
    """
    Generate a random feasible DesignState.

    Algorithm
    ---------
    1. Pick one random feed row for left column (col 0) and one for right column (col N-1).
    2. Fill the interior (cols 1 to N-2) with random types {0, 1, 2} using weights
       (p0, p1, p2).  Feed columns are all-0 except their feed row (which is type 1).
    3. Check connectivity + inductor count + spacing.  Repeat until valid.
    4. Greedily place ``n_switches`` switches by randomly flipping eligible type-1
       interior pixels to type 3.  Retry matrix if not enough eligible positions.
    5. Generate random switch state vectors and cap values.

    Returns None if no valid design is found within ``max_attempts``.
    """
    if cols < 2:
        raise ValueError("cols must be >= 2 (feed columns required).")

    types  = [0, 1, 2]
    weights = [p0, p1, p2]

    for _attempt in range(max_attempts):
        # --- Step 1: choose feed rows ---
        left_feed_row  = rng.randint(0, rows - 1)
        right_feed_row = rng.randint(0, rows - 1)

        # --- Step 2: build matrix ---
        matrix = [[0] * cols for _ in range(rows)]

        # Feed columns: single non-zero pixel each
        matrix[left_feed_row][0]          = 1
        matrix[right_feed_row][cols - 1]  = 1

        # Interior columns: random {0, 1, 2}
        for r in range(rows):
            for c in range(1, cols - 1):
                matrix[r][c] = rng.choices(types, weights=weights)[0]

        # --- Step 3: connectivity + inductor count check ---
        if not check_connectivity(matrix, four_connected=require_4_connected):
            continue
        if not check_inductor_count(matrix, max_inductors):
            continue
        # Also verify inductor spacing before placing switches
        if not check_inductor_spacing(matrix):
            continue

        # --- Step 4: place switches ---
        # Collect all type-1 interior cells that can host a switch
        eligible = [
            (r, c)
            for r in range(rows)
            for c in range(1, cols - 1)
            if matrix[r][c] == 1 and _can_place_switch_at(matrix, r, c)
        ]
        if len(eligible) < n_switches:
            continue  # not enough room for all switches

        rng.shuffle(eligible)
        placed = 0
        for (r, c) in eligible:
            if placed == n_switches:
                break
            # Re-check because earlier placements may invalidate later candidates
            if _can_place_switch_at(matrix, r, c):
                matrix[r][c] = 3
                placed += 1

        if placed < n_switches:
            continue

        # --- Final feasibility check ---
        if not is_feasible(matrix, n_switches, max_inductors,
                           four_connected=require_4_connected):
            continue

        # --- Step 5: generate switch states, cap values, and inductor values ---
        # Retry until every pair of states differs by at least 1 switch bit.
        switch_states: Optional[List[List[int]]] = None
        for _ in range(1_000):
            candidate_states = [
                [rng.randint(0, 1) for _ in range(n_switches)]
                for _ in range(n_states)
            ]
            # Check all pairs are distinct by at least one element
            all_distinct = all(
                any(a != b for a, b in zip(candidate_states[i], candidate_states[j]))
                for i in range(n_states)
                for j in range(i + 1, n_states)
            )
            if all_distinct:
                switch_states = candidate_states
                break

        if switch_states is None:
            continue  # couldn't find distinct states — retry the whole matrix

        cap_values = [
            rng.uniform(cap_min, cap_max)
            for _ in range(n_switches)
        ]
        n_ind = sum(v == 2 for row in matrix for v in row)
        inductor_values = [
            rng.randint(0, n_inductor_values - 1)
            for _ in range(n_ind)
        ]

        return DesignState(
            matrix          = matrix,
            switch_states   = switch_states,
            cap_values      = cap_values,
            inductor_values = inductor_values,
        )

    return None  # failed to generate within budget

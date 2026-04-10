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


def check_inductor_states_distinct(inductor_switch_states: List[List[int]]) -> bool:
    """Check that no two inductor state vectors are identical.

    An empty list (all-ON convention) is trivially valid.
    """
    n = len(inductor_switch_states)
    for i in range(n):
        for j in range(i + 1, n):
            if inductor_switch_states[i] == inductor_switch_states[j]:
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
    max_vias: int = 0,
) -> bool:
    """Full feasibility check (all rules a–f, plus optional via rules).

    When ``min_switches`` / ``max_switches`` are provided the switch count is
    checked against [min_switches, max_switches] instead of the exact value
    ``n_switches_required``.  SA and GA callers omit these keyword arguments
    and get the original strict-count behaviour unchanged.

    When ``four_connected=True`` the connectivity rule uses 4-way (Manhattan)
    connectivity instead of the default 8-way.

    When ``max_vias > 0`` via spacing and count rules are enforced.
    """
    if min_switches is not None or max_switches is not None:
        lo = min_switches if min_switches is not None else n_switches_required
        hi = max_switches if max_switches is not None else n_switches_required
        sw_ok = check_switch_count_range(matrix, lo, hi)
    else:
        sw_ok = check_switch_count(matrix, n_switches_required)

    via_ok = (max_vias == 0) or (check_via_spacing(matrix) and check_via_count(matrix, max_vias))

    return (
        check_feed_columns(matrix)
        and check_connectivity(matrix, four_connected=four_connected)
        and check_switch_spacing(matrix)
        and check_inductor_spacing(matrix)
        and check_inductor_count(matrix, max_inductors)
        and sw_ok
        and via_ok
    )


# ---------------------------------------------------------------------------
# Cap pixel pair feasibility
# ---------------------------------------------------------------------------

def _pair_interior_cells(pair: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Return the cells strictly between the two endpoints of an axis-aligned pair
    (exclusive of the endpoints themselves).
    """
    (r1, c1), (r2, c2) = pair[0], pair[1]
    if r1 == r2:
        c_lo, c_hi = min(c1, c2), max(c1, c2)
        return [(r1, c) for c in range(c_lo + 1, c_hi)]
    else:
        r_lo, r_hi = min(r1, r2), max(r1, r2)
        return [(r, c1) for r in range(r_lo + 1, r_hi)]


def check_cap_pair_valid(
    matrix: List[List[int]],
    pair: List[List[int]],
    min_dist: int = 2,
    max_dist: int = 3,
    other_pair_interior_cells: Optional[set] = None,
) -> bool:
    """
    Return True if a cap pixel pair is valid:

    1. Both endpoints must be type-1 (metal) pixels.
    2. The pair must be axis-aligned (horizontal or vertical only).
    3. Manhattan distance between endpoints must be in [min_dist, max_dist].
    4. The cells strictly between the two endpoints (on the axis, exclusive of
       the endpoints themselves) must ALL be type-0 (ground).  The capacitor
       must bridge across a pure ground gap — no metal, inductors, or switches
       in between.
    5. The cells adjacent to the interior path (±1 row for horizontal,
       ±1 col for vertical) must contain no inductors (2), switches (3), or
       interior path cells of any other existing cap pair.

    Parameters
    ----------
    other_pair_interior_cells : set of (row, col) tuples that are interior path
        cells of all *other* existing cap pairs.  When provided, rule 5 also
        rejects clearance cells that coincide with another pair's path.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    (r1, c1), (r2, c2) = pair[0], pair[1]
    blocked = other_pair_interior_cells or set()

    # Rule 1: both endpoints must be type 1
    if matrix[r1][c1] != 1 or matrix[r2][c2] != 1:
        return False

    # Rule 2: axis-aligned only
    if r1 != r2 and c1 != c2:
        return False

    # Rule 3: distance bounds
    dist = abs(r2 - r1) + abs(c2 - c1)
    if dist < min_dist or dist > max_dist:
        return False

    if r1 == r2:
        # Horizontal pair
        r = r1
        c_lo, c_hi = min(c1, c2), max(c1, c2)
        # Rule 4: interior axis cells must be type 0 (ground only)
        for c in range(c_lo + 1, c_hi):
            if matrix[r][c] != 0:
                return False
        # Rule 5: adjacent rows must have no type 2/3 and no other pair paths
        for c in range(c_lo + 1, c_hi):
            for dr in (-1, 1):
                nr = r + dr
                if 0 <= nr < rows:
                    if matrix[nr][c] in (2, 3):
                        return False
                    if (nr, c) in blocked:
                        return False
    else:
        # Vertical pair
        c = c1
        r_lo, r_hi = min(r1, r2), max(r1, r2)
        # Rule 4: interior axis cells must be type 0 (ground only)
        for r in range(r_lo + 1, r_hi):
            if matrix[r][c] != 0:
                return False
        # Rule 5: adjacent cols must have no type 2/3 and no other pair paths
        for r in range(r_lo + 1, r_hi):
            for dc in (-1, 1):
                nc = c + dc
                if 0 <= nc < cols:
                    if matrix[r][nc] in (2, 3):
                        return False
                    if (r, nc) in blocked:
                        return False

    return True


def _build_other_interior_cells(
    pairs: List[List[List[int]]],
    exclude_index: int = -1,
) -> set:
    """
    Build the union of interior path cells for all pairs *except* the one at
    ``exclude_index``.  Used to check clearance against other pair paths.
    """
    cells: set = set()
    for i, p in enumerate(pairs):
        if i == exclude_index:
            continue
        for cell in _pair_interior_cells(p):
            cells.add(cell)
    return cells


def filter_valid_pairs(
    matrix: List[List[int]],
    pairs: List[List[List[int]]],
    pair_values: List[float],
    min_dist: int = 2,
    max_dist: int = 3,
) -> Tuple[List[List[List[int]]], List[float]]:
    """
    Remove any cap pixel pairs that are no longer valid after a matrix mutation.
    Each pair is checked against the interior cells of all other remaining pairs.
    Returns the filtered (pairs, pair_values) lists.
    """
    kept_pairs: List[List[List[int]]] = []
    kept_values: List[float] = []
    for i, (pair, val) in enumerate(zip(pairs, pair_values)):
        other_cells = _build_other_interior_cells(pairs, exclude_index=i)
        if check_cap_pair_valid(matrix, pair, min_dist, max_dist, other_cells):
            kept_pairs.append(pair)
            kept_values.append(val)
    return kept_pairs, kept_values


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


def _can_place_via_at(
    matrix: List[List[int]],
    r: int,
    c: int,
) -> bool:
    """
    Return True if placing a via-ground pixel (type 4) at (r, c) would not
    violate the spacing rule: all 8-neighbours must be type 0 or 1 only.
    Assumes the cell is currently type 1 (metal).
    """
    rows = len(matrix)
    cols = len(matrix[0])
    if c == 0 or c == cols - 1:
        return False
    for nr, nc in _neighbours_8(r, c, rows, cols):
        if matrix[nr][nc] not in (0, 1):
            return False
    return True


def check_via_spacing(matrix: List[List[int]]) -> bool:
    """Via spacing rule: all 8-neighbours of every type-4 pixel must be 0 or 1."""
    rows = len(matrix)
    cols = len(matrix[0])
    for r in range(rows):
        for c in range(cols):
            if matrix[r][c] == 4:
                for nr, nc in _neighbours_8(r, c, rows, cols):
                    if matrix[nr][nc] not in (0, 1):
                        return False
    return True


def check_via_count(matrix: List[List[int]], max_vias: int) -> bool:
    """Return True if the number of type-4 pixels does not exceed max_vias."""
    return sum(v == 4 for row in matrix for v in row) <= max_vias


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
    n_initial_inductors: int = 0,
    p0: float = 0.40,   # probability of type 0 (ground) for interior cells
    p1: float = 0.60,   # probability of type 1 (metal)
    max_attempts: int = 10_000,
    require_4_connected: bool = False,
    max_cap_pairs: int = 0,
    cap_pair_min: float = 50e-15,
    cap_pair_max: float = 500e-15,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    n_init_vias: int = 0,
    max_vias: int = 0,
) -> Optional[DesignState]:
    """
    Generate a random feasible DesignState.

    Algorithm
    ---------
    1. Pick one random feed row for left column (col 0) and one for right column (col N-1).
    2. Fill the interior (cols 1 to N-2) with random types {0, 1} using weights (p0, p1).
       Feed columns are all-0 except their feed row (which is type 1).
    3. Check connectivity.  Repeat until valid.
    4. Greedily place ``n_switches`` switches by randomly flipping eligible type-1
       interior pixels to type 3.  Retry matrix if not enough eligible positions.
    5. Greedily place ``n_initial_inductors`` inductors (capped at max_inductors) by
       randomly flipping eligible type-1 interior pixels to type 2, similar to switches.
    6. Generate random switch states, inductor switch states, cap values, inductor values.
    7. Optionally add up to max_cap_pairs capacitor pixel pairs.

    Returns None if no valid design is found within ``max_attempts``.
    """
    if cols < 2:
        raise ValueError("cols must be >= 2 (feed columns required).")

    n_ind_target = min(n_initial_inductors, max_inductors)

    for _attempt in range(max_attempts):
        # --- Step 1: choose feed rows ---
        left_feed_row  = rng.randint(0, rows - 1)
        right_feed_row = rng.randint(0, rows - 1)

        # --- Step 2: build matrix with only 0/1 types ---
        matrix = [[0] * cols for _ in range(rows)]

        # Feed columns: single non-zero pixel each
        matrix[left_feed_row][0]          = 1
        matrix[right_feed_row][cols - 1]  = 1

        # Interior columns: random {0, 1}
        for r in range(rows):
            for c in range(1, cols - 1):
                matrix[r][c] = rng.choices([0, 1], weights=[p0, p1])[0]

        # --- Step 3: connectivity check ---
        if not check_connectivity(matrix, four_connected=require_4_connected):
            continue

        # --- Step 4: place switches ---
        # Constrain initial switches to the middle columns (mid ± 3) to avoid
        # wasted switches near the feed edges.
        mid_col = cols // 2
        sw_col_min = max(1, mid_col - 3)
        sw_col_max = min(cols - 2, mid_col + 3)
        eligible_sw = [
            (r, c)
            for r in range(rows)
            for c in range(sw_col_min, sw_col_max + 1)
            if matrix[r][c] == 1 and _can_place_switch_at(matrix, r, c)
        ]
        if len(eligible_sw) < n_switches:
            continue  # not enough room for all switches

        rng.shuffle(eligible_sw)
        placed_sw = 0
        for (r, c) in eligible_sw:
            if placed_sw == n_switches:
                break
            if _can_place_switch_at(matrix, r, c):
                matrix[r][c] = 3
                placed_sw += 1

        if placed_sw < n_switches:
            continue

        # --- Step 5: place inductors ---
        if n_ind_target > 0:
            eligible_ind = [
                (r, c)
                for r in range(rows)
                for c in range(1, cols - 1)
                if matrix[r][c] == 1 and _can_place_inductor_at(matrix, r, c)
            ]
            rng.shuffle(eligible_ind)
            placed_ind = 0
            for (r, c) in eligible_ind:
                if placed_ind == n_ind_target:
                    break
                if _can_place_inductor_at(matrix, r, c):
                    matrix[r][c] = 2
                    placed_ind += 1
            # Don't retry matrix if we couldn't place all — just use fewer

        # --- Step 5b: place initial via-ground pixels (type 4) ---
        if n_init_vias > 0 and max_vias > 0:
            eligible_via = [
                (r, c)
                for r in range(rows)
                for c in range(1, cols - 1)
                if matrix[r][c] == 1 and _can_place_via_at(matrix, r, c)
            ]
            rng.shuffle(eligible_via)
            placed_via = 0
            for (r, c) in eligible_via:
                if placed_via >= min(n_init_vias, max_vias):
                    break
                if _can_place_via_at(matrix, r, c):
                    matrix[r][c] = 4
                    placed_via += 1
            # Don't retry matrix if we couldn't place all — just use fewer

        # --- Final feasibility check ---
        if not is_feasible(matrix, n_switches, max_inductors,
                           four_connected=require_4_connected,
                           max_vias=max_vias):
            continue

        # --- Step 6: generate switch states, inductor switch states, cap / inductor values ---
        # Retry until every pair of switch states differs by at least 1 bit.
        switch_states: Optional[List[List[int]]] = None
        for _ in range(1_000):
            candidate_states = [
                [rng.randint(0, 1) for _ in range(n_switches)]
                for _ in range(n_states)
            ]
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

        cap_values = [rng.uniform(cap_min, cap_max) for _ in range(n_switches)]

        n_ind = sum(v == 2 for row in matrix for v in row)
        inductor_values = [rng.randint(0, n_inductor_values - 1) for _ in range(n_ind)]

        # Inductor switch states: random ON/OFF per inductor per state.
        # Enforce all-distinct rows when 2^n_ind >= n_states.
        # Empty when there are no inductors (all-ON convention is vacuously satisfied).
        if n_ind > 0:
            can_be_distinct = (1 << n_ind) >= n_states
            inductor_switch_states: Optional[List[List[int]]] = None
            for _ in range(1_000):
                candidate_ind_states = [
                    [rng.randint(0, 1) for _ in range(n_ind)]
                    for _ in range(n_states)
                ]
                if not can_be_distinct or check_inductor_states_distinct(candidate_ind_states):
                    inductor_switch_states = candidate_ind_states
                    break
            if inductor_switch_states is None:
                continue  # couldn't find distinct inductor states — retry the whole matrix
        else:
            inductor_switch_states = []

        # --- Step 7: attempt to add cap pixel pairs ---
        cap_pixel_pairs: List[List[List[int]]] = []
        cap_pair_values_list: List[float] = []
        if max_cap_pairs > 0:
            metal_pixels = [
                (r, c)
                for r in range(rows)
                for c in range(1, cols - 1)
                if matrix[r][c] == 1
            ]
            pair_attempts = 0
            max_pair_attempts = max_cap_pairs * 200
            while len(cap_pixel_pairs) < max_cap_pairs and pair_attempts < max_pair_attempts:
                pair_attempts += 1
                if len(metal_pixels) < 2:
                    break
                a, b = rng.sample(metal_pixels, 2)
                r1, c1 = a
                r2, c2 = b
                if r1 != r2 and c1 != c2:
                    continue
                candidate = [[r1, c1], [r2, c2]]
                if candidate in cap_pixel_pairs or [[r2, c2], [r1, c1]] in cap_pixel_pairs:
                    continue
                other_cells = _build_other_interior_cells(cap_pixel_pairs)
                if not check_cap_pair_valid(matrix, candidate, cap_pair_min_dist, cap_pair_max_dist, other_cells):
                    continue
                cap_pixel_pairs.append(candidate)
                cap_pair_values_list.append(rng.uniform(cap_pair_min, cap_pair_max))

        return DesignState(
            matrix                 = matrix,
            switch_states          = switch_states,
            cap_values             = cap_values,
            inductor_values        = inductor_values,
            inductor_switch_states = inductor_switch_states,
            cap_pixel_pairs        = cap_pixel_pairs,
            cap_pair_values        = cap_pair_values_list,
        )

    return None  # failed to generate within budget

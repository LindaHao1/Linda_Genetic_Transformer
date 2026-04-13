"""
feasibility.py — Design-rule checks and random design generation.

Design rules
------------
a. Connectivity:     8-connected path (no 0 pixels) from the feed pixel in col 0
                     to the feed pixel in col N-1.
b/c. Special spacing: Every pair of switch (3) and/or inductor (2) pixels must have
                     Chebyshev distance ≥ 3, i.e. max(|Δrow|, |Δcol|) ≥ 3.
                     Diagonal counts as 1, so this forbids a 5×5 box around each
                     special pixel from containing any other special pixel.
                     The old 8-neighbour check (distance ≥ 2) was too weak — it
                     allowed two switches separated by a single pixel, which still
                     leaves insufficient physical clearance.
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


def check_special_pixel_spacing(matrix: List[List[int]], min_dist: int = 3) -> bool:
    """
    Rules b/c: every pair of switch (3) and/or inductor (2) pixels must have
    Chebyshev distance ≥ min_dist, where Chebyshev distance = max(|Δrow|, |Δcol|).

    This is stricter than the old 8-neighbour check (which only enforced distance ≥ 2).
    With min_dist=3 the forbidden zone around each special pixel is a 5×5 box.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    # Collect all special pixel positions once
    specials = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if matrix[r][c] in (2, 3)
    ]
    for i, (r1, c1) in enumerate(specials):
        for r2, c2 in specials[i + 1:]:
            if max(abs(r1 - r2), abs(c1 - c2)) < min_dist:
                return False
    return True


def check_switch_spacing(matrix: List[List[int]]) -> bool:
    """Rule b: switches must be at Chebyshev distance ≥ 3 from all other special pixels."""
    return check_special_pixel_spacing(matrix)


def check_inductor_spacing(matrix: List[List[int]]) -> bool:
    """Rule c: inductors must be at Chebyshev distance ≥ 3 from all other special pixels."""
    return check_special_pixel_spacing(matrix)


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
        and check_special_pixel_spacing(matrix)   # covers both switch and inductor spacing
        and check_inductor_count(matrix, max_inductors)
        and sw_ok
        and via_ok
    )


# ---------------------------------------------------------------------------
# Cap pixel pair feasibility
# ---------------------------------------------------------------------------

def _pair_interior_cells(pair: List[List[int]]) -> List[Tuple[int, int]]:
    """
    Return the cells strictly between the two endpoints of a cap pair
    (exclusive of the endpoints themselves).

    For axis-aligned pairs: cells along the shared row or column.
    For diagonal / off-diagonal pairs: cells along the Bresenham path
    between the two endpoints (stepping by sign(dr), sign(dc) each step
    along the dominant axis, rounding the minor axis).
    """
    (r1, c1), (r2, c2) = pair[0], pair[1]
    if r1 == r2:
        c_lo, c_hi = min(c1, c2), max(c1, c2)
        return [(r1, c) for c in range(c_lo + 1, c_hi)]
    if c1 == c2:
        r_lo, r_hi = min(r1, r2), max(r1, r2)
        return [(r, c1) for r in range(r_lo + 1, r_hi)]

    # General (diagonal / off-diagonal) case
    dr = r2 - r1
    dc = c2 - c1
    steps = max(abs(dr), abs(dc))
    cells: List[Tuple[int, int]] = []
    for i in range(1, steps):
        r = r1 + round(i * dr / steps)
        c = c1 + round(i * dc / steps)
        if (r, c) not in cells:
            cells.append((r, c))
    return cells


def _pair_all_cells(pair: List[List[int]]) -> List[Tuple[int, int]]:
    """Return the full path of a cap pair: both endpoints plus all interior cells."""
    (r1, c1), (r2, c2) = pair[0], pair[1]
    return [(r1, c1), (r2, c2)] + _pair_interior_cells(pair)


def check_cap_pair_valid(
    matrix: List[List[int]],
    pair: List[List[int]],
    min_dist: int = 2,
    max_dist: int = 3,
    other_pair_path_cells: Optional[set] = None,
) -> bool:
    """
    Return True if a cap pixel pair is valid:

    1. Both endpoints must be type-1 (metal) pixels.
    2. (removed) — diagonal and off-diagonal pairs are now permitted.
    3. Chebyshev distance max(|dr|, |dc|) between endpoints must be in
       [min_dist, max_dist].
    4. The cells strictly between the two endpoints (exclusive of the
       endpoints) must contain AT LEAST ONE type-0 (ground) cell.
    5. Every cell in the full path (both endpoints + all interior cells)
       must have no special pixels (type 2 or 3) within Chebyshev distance
       ≤ 1 (i.e. no type-2/3 pixel inside the 3×3 box centred on each
       path cell).
    6. The 3×3 corridor swept along the full path must not intersect the
       full path (endpoints + interior) of any other existing cap pair.
       This prevents port overlap and ensures clear separation between caps.
       Applies to all pair orientations (axis-aligned and diagonal).

    Parameters
    ----------
    other_pair_path_cells : set of (row, col) tuples covering the full path
        (endpoints + interior) of all *other* existing cap pairs.  When
        provided, rule 6 rejects any pair whose 3×3 corridor contains a cell
        from this set.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    (r1, c1), (r2, c2) = pair[0], pair[1]
    blocked = other_pair_path_cells or set()

    # Rule 1: both endpoints must be type 1
    if matrix[r1][c1] != 1 or matrix[r2][c2] != 1:
        return False

    # Rule 3: Chebyshev distance bounds
    dist = max(abs(r2 - r1), abs(c2 - c1))
    if dist < min_dist or dist > max_dist:
        return False

    interior = _pair_interior_cells(pair)

    # Rule 4: interior must contain at least one type-0 (ground) cell
    if not any(matrix[ir][ic] == 0 for (ir, ic) in interior):
        return False

    all_path_cells = [(r1, c1), (r2, c2)] + interior

    # Rule 5: every cell in the full path must have no special pixel (type 2
    # or 3) within Chebyshev distance ≤ 1 (3×3 box).
    for (pr, pc) in all_path_cells:
        r_lo = max(0, pr - 1)
        r_hi = min(rows - 1, pr + 1)
        c_lo = max(0, pc - 1)
        c_hi = min(cols - 1, pc + 1)
        for nr in range(r_lo, r_hi + 1):
            for nc in range(c_lo, c_hi + 1):
                if matrix[nr][nc] in (2, 3):
                    return False

    # Rule 6: the 3×3 corridor around the full path of this pair must not
    # contain any path cell (endpoint or interior) of another existing pair.
    if blocked:
        for (pr, pc) in all_path_cells:
            r_lo = max(0, pr - 1)
            r_hi = min(rows - 1, pr + 1)
            c_lo = max(0, pc - 1)
            c_hi = min(cols - 1, pc + 1)
            for nr in range(r_lo, r_hi + 1):
                for nc in range(c_lo, c_hi + 1):
                    if (nr, nc) in blocked:
                        return False

    return True


def _build_other_path_cells(
    pairs: List[List[List[int]]],
    exclude_index: int = -1,
) -> set:
    """
    Build the union of full path cells (endpoints + interior) for all pairs
    *except* the one at ``exclude_index``.  Used to enforce Rule 6 clearance.
    """
    cells: set = set()
    for i, p in enumerate(pairs):
        if i == exclude_index:
            continue
        for cell in _pair_all_cells(p):
            cells.add(cell)
    return cells


def _build_other_interior_cells(
    pairs: List[List[List[int]]],
    exclude_index: int = -1,
) -> set:
    """
    Build the union of interior path cells (excluding endpoints) for all pairs
    *except* the one at ``exclude_index``.

    .. deprecated::
        Prefer :func:`_build_other_path_cells` which also includes endpoints,
        giving stronger overlap protection.
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
    Each pair is checked against the full path cells of all other remaining pairs.
    Returns the filtered (pairs, pair_values) lists.
    """
    kept_pairs: List[List[List[int]]] = []
    kept_values: List[float] = []
    for i, (pair, val) in enumerate(zip(pairs, pair_values)):
        other_cells = _build_other_path_cells(pairs, exclude_index=i)
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

def _special_pixel_too_close(
    matrix: List[List[int]],
    r: int,
    c: int,
    min_dist: int = 3,
) -> bool:
    """
    Return True if any existing special pixel (type 2 or 3) is within Chebyshev
    distance < min_dist of (r, c).  Used by placement helpers.
    """
    rows = len(matrix)
    cols = len(matrix[0])
    r_lo = max(0, r - (min_dist - 1))
    r_hi = min(rows - 1, r + (min_dist - 1))
    c_lo = max(0, c - (min_dist - 1))
    c_hi = min(cols - 1, c + (min_dist - 1))
    for nr in range(r_lo, r_hi + 1):
        for nc in range(c_lo, c_hi + 1):
            if nr == r and nc == c:
                continue
            if matrix[nr][nc] in (2, 3):
                return True
    return False


def _can_place_switch_at(
    matrix: List[List[int]],
    r: int,
    c: int,
) -> bool:
    """
    Return True if placing a switch at (r, c) would not violate rule b.
    Enforces Chebyshev distance ≥ 3 from all other special pixels (types 2, 3).
    Assumes the cell is currently type 1 (metal).
    Also checks that this cell is not in the feed columns.
    """
    cols = len(matrix[0])
    if c == 0 or c == cols - 1:
        return False
    return not _special_pixel_too_close(matrix, r, c)


def _can_place_inductor_at(
    matrix: List[List[int]],
    r: int,
    c: int,
) -> bool:
    """
    Return True if placing an inductor at (r, c) would not violate rule c.
    Enforces Chebyshev distance ≥ 3 from all other special pixels (types 2, 3).
    Assumes the cell is currently type 1 (metal).
    """
    cols = len(matrix[0])
    if c == 0 or c == cols - 1:
        return False
    return not _special_pixel_too_close(matrix, r, c)


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
    single_inductor_value: bool = False,
    n_initial_inductors: int = 0,
    p0: float = 0.50,   # probability of type 0 (ground) for interior cells
    p1: float = 0.50,   # probability of type 1 (metal)
    max_attempts: int = 10_000,
    require_4_connected: bool = False,
    max_cap_pairs: int = 0,
    cap_pair_init_min: int = 0,
    cap_pair_init_max: int = -1,
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
        if single_inductor_value and n_ind > 0:
            shared_val = rng.randint(0, n_inductor_values - 1)
            inductor_values = [shared_val] * n_ind
        else:
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
            # Sample a target count from [cap_pair_init_min, cap_pair_init_max]
            # cap_pair_init_max defaults to max_cap_pairs when -1 (unset)
            init_min = max(0, min(cap_pair_init_min, max_cap_pairs))
            init_max = max_cap_pairs if cap_pair_init_max < 0 else min(cap_pair_init_max, max_cap_pairs)
            target_n_pairs = rng.randint(init_min, init_max)

            if target_n_pairs > 0:
                metal_set = {
                    (r, c)
                    for r in range(rows)
                    for c in range(1, cols - 1)
                    if matrix[r][c] == 1
                }
                candidates: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
                for (r1, c1) in metal_set:
                    # Enumerate all neighbours within Chebyshev distance [min_dist, max_dist].
                    # Only iterate offsets with dr >= 0 (or dr==0 and dc>0) to avoid
                    # adding each pair twice.
                    for dr in range(0, cap_pair_max_dist + 1):
                        dc_range = range(
                            -cap_pair_max_dist if dr > 0 else 1,
                            cap_pair_max_dist + 1,
                        )
                        for dc in dc_range:
                            if max(dr, abs(dc)) < cap_pair_min_dist:
                                continue
                            if max(dr, abs(dc)) > cap_pair_max_dist:
                                continue
                            if (r1 + dr, c1 + dc) in metal_set:
                                candidates.append(((r1, c1), (r1 + dr, c1 + dc)))

                rng.shuffle(candidates)
                for (r1, c1), (r2, c2) in candidates:
                    if len(cap_pixel_pairs) >= target_n_pairs:
                        break
                    candidate = [[r1, c1], [r2, c2]]
                    other_cells = _build_other_path_cells(cap_pixel_pairs)
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

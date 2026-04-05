"""
feasibility.py — Design-rule checks and random design generation (ML-local copy).

Decoupled from virtuoso_optimization_SA.  Adds a configurable feasibility
check (``check_design_feasibility``) driven by ``FeasibilityCheckConfig`` so
individual rules can be toggled via the ML config file.

Design rules
------------
a. Connectivity:     8-connected path (no 0 pixels) from the feed pixel in col 0
                     to the feed pixel in col N-1.
b. Switch spacing:   No other switch (3) or inductor (2) may be in the 8-neighbourhood
                     of any switch pixel.
c. Inductor spacing: No other inductor (2) or switch (3) may be in the 8-neighbourhood
                     of any inductor pixel.
d. Inductor count:   Total number of type-2 pixels <= max_inductors.
e. Switch count:     Total number of type-3 pixels in [n_switches_min, n_switches_max].
f. Feed columns:     Exactly one non-zero pixel in col 0 AND in col N-1.
g. Switch states:    All switch state vectors are pairwise distinct.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .design import DesignState


# ---------------------------------------------------------------------------
# Configurable feasibility flags
# ---------------------------------------------------------------------------

@dataclass
class FeasibilityCheckConfig:
    """
    Controls which individual feasibility rules are enforced.

    All flags default to True (same strict behaviour as SA).  Set any flag to
    False to relax the corresponding rule.
    """
    check_connectivity:         bool = True   # rule a
    check_switch_spacing:       bool = True   # rule b
    check_inductor_spacing:     bool = True   # rule c
    check_inductor_count:       bool = True   # rule d
    check_switch_count:         bool = True   # rule e (switch count range)
    check_feed_columns:         bool = True   # rule f
    check_switch_states_distinct: bool = True # rule g

    @classmethod
    def from_dict(cls, d: dict) -> "FeasibilityCheckConfig":
        return cls(
            check_connectivity          = bool(d.get("check_connectivity",          True)),
            check_switch_spacing        = bool(d.get("check_switch_spacing",        True)),
            check_inductor_spacing      = bool(d.get("check_inductor_spacing",      True)),
            check_inductor_count        = bool(d.get("check_inductor_count",        True)),
            check_switch_count          = bool(d.get("check_switch_count",          True)),
            check_feed_columns          = bool(d.get("check_feed_columns",          True)),
            check_switch_states_distinct= bool(d.get("check_switch_states_distinct",True)),
        )

    @classmethod
    def all_enabled(cls) -> "FeasibilityCheckConfig":
        return cls()

    @classmethod
    def all_disabled(cls) -> "FeasibilityCheckConfig":
        return cls(
            check_connectivity=False,
            check_switch_spacing=False,
            check_inductor_spacing=False,
            check_inductor_count=False,
            check_switch_count=False,
            check_feed_columns=False,
            check_switch_states_distinct=False,
        )


# ---------------------------------------------------------------------------
# Neighbourhood helpers
# ---------------------------------------------------------------------------

def _neighbours_8(r: int, c: int, rows: int, cols: int) -> List[Tuple[int, int]]:
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
    """
    rows = len(matrix)
    cols = len(matrix[0])

    left_row  = next((r for r in range(rows) if matrix[r][0]        != 0), None)
    right_row = next((r for r in range(rows) if matrix[r][cols - 1] != 0), None)
    if left_row is None or right_row is None:
        return False
    if left_row == right_row and cols == 1:
        return True

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
    """Rule d: total type-2 pixels <= max_inductors."""
    n = sum(v == 2 for row in matrix for v in row)
    return n <= max_inductors


def check_switch_count(matrix: List[List[int]], n_switches_required: int) -> bool:
    """Rule e (exact): total type-3 pixels == n_switches_required."""
    n = sum(v == 3 for row in matrix for v in row)
    return n == n_switches_required


def check_switch_count_range(matrix: List[List[int]], min_sw: int, max_sw: int) -> bool:
    """Rule e (relaxed): total type-3 pixels in [min_sw, max_sw]."""
    n = sum(v == 3 for row in matrix for v in row)
    return min_sw <= n <= max_sw


def check_switch_states_distinct(switch_states: List[List[int]]) -> bool:
    """Rule g: all switch state vectors are pairwise distinct."""
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


# ---------------------------------------------------------------------------
# Composite feasibility (SA-compatible, not config-driven)
# ---------------------------------------------------------------------------

def is_feasible(
    matrix: List[List[int]],
    n_switches_required: int,
    max_inductors: int,
    *,
    min_switches: Optional[int] = None,
    max_switches: Optional[int] = None,
    four_connected: bool = False,
) -> bool:
    """Full feasibility check (all rules a-f), SA-compatible signature."""
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
    """Feasibility check excluding the switch count rule (used during generation)."""
    return (
        check_feed_columns(matrix)
        and check_connectivity(matrix, four_connected=four_connected)
        and check_inductor_spacing(matrix)
        and check_inductor_count(matrix, max_inductors)
    )


# ---------------------------------------------------------------------------
# Configurable composite feasibility (ML-specific, flag-driven)
# ---------------------------------------------------------------------------

def check_design_feasibility(
    matrix: List[List[int]],
    switch_states: List[List[int]],
    n_switches_min: int,
    n_switches_max: int,
    max_inductors: int,
    feas_cfg: FeasibilityCheckConfig,
    four_connected: bool = False,
) -> Tuple[bool, str]:
    """
    Configurable feasibility check for the ML pipeline.

    Each rule is guarded by the corresponding flag in ``feas_cfg``.  Returns
    ``(is_feasible, failure_reason)`` where ``failure_reason`` is an empty
    string on success and a short label naming the violated rule otherwise.

    This is the single entry point for all ML feasibility decisions, replacing
    the scattered individual check calls in online_trainer.py.
    """
    if feas_cfg.check_feed_columns and not check_feed_columns(matrix):
        return False, "feed_columns"

    if feas_cfg.check_switch_count and not check_switch_count_range(
        matrix, n_switches_min, n_switches_max
    ):
        return False, "switch_count"

    if feas_cfg.check_inductor_count and not check_inductor_count(matrix, max_inductors):
        return False, "inductor_count"

    if feas_cfg.check_switch_spacing and not check_switch_spacing(matrix):
        return False, "spacing"

    if feas_cfg.check_inductor_spacing and not check_inductor_spacing(matrix):
        return False, "spacing"

    if feas_cfg.check_connectivity and not check_connectivity(
        matrix, four_connected=four_connected
    ):
        return False, "connectivity"

    if feas_cfg.check_switch_states_distinct and not check_switch_states_distinct(switch_states):
        return False, "state_distinct"

    return True, ""


# ---------------------------------------------------------------------------
# Cell placement helpers (used by generate_random_design)
# ---------------------------------------------------------------------------

def _can_place_switch_at(matrix: List[List[int]], r: int, c: int) -> bool:
    rows = len(matrix)
    cols = len(matrix[0])
    if c == 0 or c == cols - 1:
        return False
    for nr, nc in _neighbours_8(r, c, rows, cols):
        if matrix[nr][nc] in (2, 3):
            return False
    return True


def _can_place_inductor_at(matrix: List[List[int]], r: int, c: int) -> bool:
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
    p0: float = 0.399,
    p1: float = 0.60,
    p2: float = 0.001,
    max_attempts: int = 10_000,
    require_4_connected: bool = False,
) -> Optional[DesignState]:
    """
    Generate a random feasible DesignState.

    Uses the same algorithm as the SA version but returns an ML-local
    DesignState (from virtuoso_optimization_ML.design).
    """
    if cols < 2:
        raise ValueError("cols must be >= 2 (feed columns required).")

    types   = [0, 1, 2]
    weights = [p0, p1, p2]

    for _attempt in range(max_attempts):
        left_feed_row  = rng.randint(0, rows - 1)
        right_feed_row = rng.randint(0, rows - 1)

        matrix = [[0] * cols for _ in range(rows)]
        matrix[left_feed_row][0]         = 1
        matrix[right_feed_row][cols - 1] = 1

        for r in range(rows):
            for c in range(1, cols - 1):
                matrix[r][c] = rng.choices(types, weights=weights)[0]

        if not check_connectivity(matrix, four_connected=require_4_connected):
            continue
        if not check_inductor_count(matrix, max_inductors):
            continue
        if not check_inductor_spacing(matrix):
            continue

        eligible = [
            (r, c)
            for r in range(rows)
            for c in range(1, cols - 1)
            if matrix[r][c] == 1 and _can_place_switch_at(matrix, r, c)
        ]
        if len(eligible) < n_switches:
            continue

        rng.shuffle(eligible)
        placed = 0
        for (r, c) in eligible:
            if placed == n_switches:
                break
            if _can_place_switch_at(matrix, r, c):
                matrix[r][c] = 3
                placed += 1

        if placed < n_switches:
            continue

        if not is_feasible(matrix, n_switches, max_inductors,
                           four_connected=require_4_connected):
            continue

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
            continue

        cap_values = [rng.uniform(cap_min, cap_max) for _ in range(n_switches)]
        n_ind = sum(v == 2 for row in matrix for v in row)
        inductor_values = [rng.randint(0, n_inductor_values - 1) for _ in range(n_ind)]

        return DesignState(
            matrix          = matrix,
            switch_states   = switch_states,
            cap_values      = cap_values,
            inductor_values = inductor_values,
        )

    return None

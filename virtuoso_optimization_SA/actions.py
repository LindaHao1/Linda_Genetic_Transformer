"""
actions.py — SA mutation actions for the switchable-filter design.

Six actions are available:

1. flip_01              — flip up to N_regular_flip type-0 or type-1 pixels.
2. move_switch          — relocate one switch to a different type-1 pixel.
3. flip_inductor        — add or remove one inductor pixel.
4. flip_inductor_value  — change the inductance value of one inductor pixel
                          (does not alter inductor count).
5. perturb_caps         — add Gaussian noise to cap values (clipped to valid range).
6. flip_switch_state    — flip one bit in the switch-state matrix.

``random_action`` picks one of the six uniformly at random (or weighted by
``action_weights``) and returns a mutated copy of the design.  If the chosen
action fails to produce a feasible design after ``max_retries`` attempts it
returns None, signalling the caller to retry.
"""

from __future__ import annotations

import random
from typing import List, Optional

from .design import DesignState
from .feasibility import (
    _can_place_inductor_at,
    _can_place_switch_at,
    check_connectivity,
    check_inductor_count,
    check_inductor_spacing,
    check_switch_spacing,
    is_feasible,
)


# ---------------------------------------------------------------------------
# Action 1: flip 0 ↔ 1 for up to N pixels
# ---------------------------------------------------------------------------

def action_flip_01(
    state: DesignState,
    n_flip: int,
    rng: random.Random,
) -> Optional[DesignState]:
    """
    Randomly flip up to ``n_flip`` type-0 or type-1 interior pixels
    (feed columns are excluded).

    Returns a mutated copy, or None if the result is not feasible.
    """
    rows  = state.rows
    cols  = state.cols

    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(1, cols - 1)          # exclude feed columns
        if state.matrix[r][c] in (0, 1)
    ]
    if not candidates:
        return None

    k = rng.randint(1, min(n_flip, len(candidates)))
    chosen = rng.sample(candidates, k)

    new = state.copy()
    for r, c in chosen:
        new.matrix[r][c] = 1 - new.matrix[r][c]   # 0 → 1, 1 → 0

    # Validate
    if not check_connectivity(new.matrix):
        return None
    # Spacing constraints still satisfied (we only touched 0/1 cells)
    return new


# ---------------------------------------------------------------------------
# Action 2: move one switch
# ---------------------------------------------------------------------------

def action_move_switch(
    state: DesignState,
    rng: random.Random,
) -> Optional[DesignState]:
    """
    Pick a random switch pixel (type 3), remove it (set to 1), then place it
    at a randomly chosen different type-1 pixel that satisfies spacing rules.

    Returns a mutated copy, or None if no valid relocation exists.
    """
    rows  = state.rows
    cols  = state.cols

    sw_positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if state.matrix[r][c] == 3
    ]
    if not sw_positions:
        return None

    new = state.copy()

    # Remove a random switch
    rm_r, rm_c = rng.choice(sw_positions)
    new.matrix[rm_r][rm_c] = 1

    # Find eligible type-1 cells (excluding the one we just freed)
    eligible = [
        (r, c)
        for r in range(rows)
        for c in range(1, cols - 1)
        if new.matrix[r][c] == 1 and _can_place_switch_at(new.matrix, r, c)
        and (r, c) != (rm_r, rm_c)
    ]
    if not eligible:
        return None

    rng.shuffle(eligible)
    for pr, pc in eligible:
        if _can_place_switch_at(new.matrix, pr, pc):
            new.matrix[pr][pc] = 3
            if check_connectivity(new.matrix):
                return new
            # undo and try next
            new.matrix[pr][pc] = 1

    return None


# ---------------------------------------------------------------------------
# Action 3: add or remove one inductor
# ---------------------------------------------------------------------------

def _inductor_row_first_index(matrix: list, r: int, c: int) -> int:
    """Return the row-first position of inductor pixel (r, c) among all type-2 pixels."""
    idx = 0
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 2:
                if row == r and col == c:
                    return idx
                idx += 1
    raise ValueError(f"No type-2 pixel at ({r}, {c})")


def action_flip_inductor(
    state: DesignState,
    max_inductors: int,
    rng: random.Random,
    n_inductor_values: int = 4,
) -> Optional[DesignState]:
    """
    Either add an inductor (type 1 → 2) or remove one (type 2 → 1).

    - If the current inductor count equals ``max_inductors``, only removal is
      attempted.
    - Otherwise the action is chosen uniformly at random.
    - ``inductor_values`` is kept in sync: a random value is inserted on add,
      and the corresponding entry is removed on removal.

    Returns a mutated copy, or None if the action cannot be applied.
    """
    rows  = state.rows
    cols  = state.cols
    n_ind = state.n_inductors

    if n_ind >= max_inductors:
        action = "remove"
    elif n_ind == 0:
        action = "add"
    else:
        action = rng.choice(["add", "remove"])

    new = state.copy()

    if action == "add":
        candidates = [
            (r, c)
            for r in range(rows)
            for c in range(1, cols - 1)
            if new.matrix[r][c] == 1 and _can_place_inductor_at(new.matrix, r, c)
        ]
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        new.matrix[r][c] = 2
        if not check_connectivity(new.matrix):
            return None
        # Insert a random inductor value at the correct row-first position
        idx = _inductor_row_first_index(new.matrix, r, c)
        new.inductor_values.insert(idx, rng.randint(0, n_inductor_values - 1))
        return new

    else:  # remove
        candidates = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if new.matrix[r][c] == 2
        ]
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        # Find and remove the corresponding inductor_values entry before modifying matrix
        idx = _inductor_row_first_index(new.matrix, r, c)
        new.matrix[r][c] = 1
        if not check_connectivity(new.matrix):
            return None
        if idx < len(new.inductor_values):
            new.inductor_values.pop(idx)
        return new


# ---------------------------------------------------------------------------
# Action 4 (new): flip the inductance value of one inductor pixel
# ---------------------------------------------------------------------------

def action_flip_inductor_value(
    state: DesignState,
    n_inductor_values: int,
    rng: random.Random,
) -> Optional[DesignState]:
    """
    Pick a random type-2 (inductor) pixel and change its entry in
    ``inductor_values`` to a different index sampled from
    [0, n_inductor_values).

    This does NOT add or remove inductor pixels — the inductor count is
    unchanged.

    Returns a mutated copy, or None if there are no inductor pixels or only
    one possible value.
    """
    if state.n_inductors == 0 or n_inductor_values < 2:
        return None

    # Collect inductor positions in row-first order
    ind_positions = [
        (r, c)
        for r in range(state.rows)
        for c in range(state.cols)
        if state.matrix[r][c] == 2
    ]

    # Pick a random inductor
    local_idx = rng.randrange(len(ind_positions))

    # Choose a different value from the available options
    current_val = state.inductor_values[local_idx] if local_idx < len(state.inductor_values) else 0
    choices = [v for v in range(n_inductor_values) if v != current_val]
    if not choices:
        return None

    new = state.copy()
    # Ensure inductor_values has an entry for every inductor
    while len(new.inductor_values) < len(ind_positions):
        new.inductor_values.append(0)
    new.inductor_values[local_idx] = rng.choice(choices)
    return new


# ---------------------------------------------------------------------------
# Action 5 (was 4): perturb capacitance values
# ---------------------------------------------------------------------------

def action_perturb_caps(
    state: DesignState,
    sigma: float,
    cap_min: float,
    cap_max: float,
    rng: random.Random,
) -> Optional[DesignState]:
    """
    Add independent Gaussian noise (std = ``sigma``) to every cap value,
    then clip to [cap_min, cap_max].

    Returns a mutated copy (always valid, so never returns None).
    """
    if not state.cap_values:
        return None

    new = state.copy()
    for i in range(len(new.cap_values)):
        noise = rng.gauss(0.0, sigma)
        new.cap_values[i] = max(cap_min, min(cap_max, new.cap_values[i] + noise))
    return new


# ---------------------------------------------------------------------------
# Action 6 (was 5): flip one switch-state bit
# ---------------------------------------------------------------------------

def action_flip_switch_state(
    state: DesignState,
    rng: random.Random,
) -> Optional[DesignState]:
    """
    Flip one randomly chosen bit in the switch_states matrix.

    Returns a mutated copy, or None if there are no switch states.
    """
    if not state.switch_states or not state.switch_states[0]:
        return None

    n_states  = state.n_states
    n_switches = len(state.switch_states[0])

    si = rng.randint(0, n_states  - 1)
    sw = rng.randint(0, n_switches - 1)

    new = state.copy()
    new.switch_states[si][sw] = 1 - new.switch_states[si][sw]
    return new


# ---------------------------------------------------------------------------
# Dispatcher: pick a random action and apply it
# ---------------------------------------------------------------------------

_ACTION_NAMES = [
    "flip_01",
    "move_switch",
    "flip_inductor",
    "flip_inductor_value",
    "perturb_caps",
    "flip_switch_state",
]


def random_action(
    state: DesignState,
    n_regular_flip: int,
    max_inductors: int,
    cap_min: float,
    cap_max: float,
    cap_sigma: float,
    rng: random.Random,
    n_inductor_values: int = 4,
    action_weights: Optional[List[float]] = None,
    max_retries: int = 50,
) -> Optional[DesignState]:
    """
    Pick one of the 6 SA actions (or weighted by ``action_weights``) and apply
    it to produce a feasible mutated design.

    Retries up to ``max_retries`` times (re-drawing the action each time)
    before returning None.

    Parameters
    ----------
    n_inductor_values : number of discrete inductance choices in the translation table.
    action_weights : optional list of 6 floats (probabilities / unnormalised weights).
        Order matches _ACTION_NAMES:
        [flip_01, move_switch, flip_inductor, flip_inductor_value,
         perturb_caps, flip_switch_state]
    """
    weights = action_weights or [1.0] * 6

    for _ in range(max_retries):
        action_name = rng.choices(_ACTION_NAMES, weights=weights)[0]

        if action_name == "flip_01":
            result = action_flip_01(state, n_regular_flip, rng)
        elif action_name == "move_switch":
            result = action_move_switch(state, rng)
        elif action_name == "flip_inductor":
            result = action_flip_inductor(state, max_inductors, rng, n_inductor_values)
        elif action_name == "flip_inductor_value":
            result = action_flip_inductor_value(state, n_inductor_values, rng)
        elif action_name == "perturb_caps":
            result = action_perturb_caps(state, cap_sigma, cap_min, cap_max, rng)
        elif action_name == "flip_switch_state":
            result = action_flip_switch_state(state, rng)
        else:
            continue

        if result is not None:
            return result

    return None

"""
actions.py — SA mutation actions for the switchable-filter design.

Actions available:

1. flip_01              — flip up to N_regular_flip type-0 or type-1 pixels.
2. modify_switch        — with equal probability: move, add, or remove one switch.
                          Add inserts a new switch_states column with values that keep
                          all states distinct (avoids all-0/all-1 if n_states > 2).
                          Remove drops the switch and its switch_states column.
3. flip_inductor        — add or remove one inductor pixel.
4. flip_inductor_value  — change the inductance value of one inductor pixel
                          (does not alter inductor count).
5. perturb_caps         — add Gaussian noise to cap values (clipped to valid range).
6. flip_switch_state    — flip one bit in the switch-state matrix.

``random_action`` picks one action by weight, retries it up to
``max_retries_per_action`` times, then falls back to the next weighted action
if still infeasible — cycling through all actions before returning (None, None).
"""

from __future__ import annotations

import random
from typing import List, Optional

from .design import DesignState
from .feasibility import (
    _build_other_path_cells,
    _can_place_inductor_at,
    _can_place_shunt_cap_at,
    _can_place_switch_at,
    _can_place_via_at,
    check_cap_pair_valid,
    check_connectivity,
    check_connectivity_sized,
    check_inductor_count,
    check_inductor_spacing,
    check_inductor_states_distinct,
    check_switch_spacing,
    check_switch_states_distinct,
    check_switch_states_distinct_banked,
    filter_valid_pairs,
    is_feasible,
)


def _states_can_be_distinct(n_bits: int, n_states: int) -> bool:
    """Return True if 2^n_bits >= n_states (enough distinct binary vectors exist)."""
    return (1 << n_bits) >= n_states


# ---------------------------------------------------------------------------
# Action 1: flip 0 ↔ 1 for up to N pixels
# ---------------------------------------------------------------------------

def action_flip_01(
    state: DesignState,
    n_flip: int,
    rng: random.Random,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    flip_inner_region=None,
    flip_inner_prob: float = 1.0,
    flip_outer_prob: float = 1.0,
    require_axis_aligned: bool = True,
    flip_region_size: int = 8,
    grid_size: float = 25.0,
    connect_margin: float = 2.0,
) -> tuple:
    """
    Randomly flip up to ``n_flip`` type-0 or type-1 interior pixels
    (feed columns are excluded).

    A random ``flip_region_size`` × ``flip_region_size`` window is chosen
    first; only pixels within that window are candidates.  The inner/outer
    probability weighting (``flip_inner_region``) is applied within the window.

    Returns (mutated copy, info_dict), or (None, None) if infeasible.
    """
    rows  = state.rows
    cols  = state.cols

    # All valid window top-left positions (window must fit within the grid).
    r0_max = max(0, rows - flip_region_size)
    c0_max = max(0, cols - flip_region_size)

    # Choose window position: biased towards the outer region when
    # flip_inner_region is configured, uniform otherwise.
    # Each window position is weighted by the blended expected flip probability
    # of its cells: (n_inner * flip_inner_prob + n_outer * flip_outer_prob) / n_total.
    # The overlap counts are computed analytically via rectangle intersection.
    if flip_inner_region is not None and flip_inner_prob != flip_outer_prob:
        (ri1, ci1), (ri2, ci2) = flip_inner_region
        all_positions = [
            (r0, c0)
            for r0 in range(r0_max + 1)
            for c0 in range(c0_max + 1)
        ]
        pos_weights = []
        for r0, c0 in all_positions:
            r_end   = min(r0 + flip_region_size, rows)
            c_start = max(1, c0)
            c_end   = min(c0 + flip_region_size, cols - 1)
            total   = (r_end - r0) * max(0, c_end - c_start)
            if total == 0:
                pos_weights.append(0.0)
                continue
            # Analytical inner-region overlap (rectangle intersection)
            r_inner = max(0, min(r_end - 1, ri2) - max(r0, ri1) + 1)
            c_inner = max(0, min(c_end - 1, ci2) - max(c_start, ci1) + 1)
            n_inner = r_inner * c_inner
            n_outer = total - n_inner
            pos_weights.append(
                (n_inner * flip_inner_prob + n_outer * flip_outer_prob) / total
            )
        (r0, c0) = rng.choices(all_positions, weights=pos_weights)[0]
    else:
        r0 = rng.randint(0, r0_max)
        c0 = rng.randint(0, c0_max)

    r1_win = r0 + flip_region_size - 1
    c1_win = c0 + flip_region_size - 1

    candidates = [
        (r, c)
        for r in range(r0, min(r1_win + 1, rows))
        for c in range(max(1, c0), min(c1_win + 1, cols - 1))   # exclude feed columns
        if state.matrix[r][c] in (0, 1)
    ]
    if not candidates:
        return None, None

    k = rng.randint(1, min(n_flip, len(candidates)))

    # Weighted sampling without replacement (Efraimidis-Spirakis) when a
    # non-uniform inner/outer region is configured; fall back to uniform otherwise.
    if flip_inner_region is not None and flip_inner_prob != flip_outer_prob:
        (ri1, ci1), (ri2, ci2) = flip_inner_region
        def _w(r, c):
            return flip_inner_prob if (ri1 <= r <= ri2 and ci1 <= c <= ci2) else flip_outer_prob
        cand_weights = [_w(r, c) for r, c in candidates]
        keys = [rng.random() ** (1.0 / w) if w > 0 else 0.0 for w in cand_weights]
        sorted_idx = sorted(range(len(candidates)), key=lambda i: keys[i], reverse=True)
        chosen = [candidates[i] for i in sorted_idx[:k]]
    else:
        chosen = rng.sample(candidates, k)

    new = state.copy()
    flipped_cells = []
    for r, c in chosen:
        old_val = new.matrix[r][c]
        new.matrix[r][c] = 1 - old_val   # 0 → 1, 1 → 0
        flipped_cells.append({"row": r, "col": c, "from": old_val, "to": new.matrix[r][c]})

    # Validate — overlap-aware connectivity when per-pixel sizing is enabled.
    if new.pixel_sizes:
        conn_ok = check_connectivity_sized(
            new.matrix, new.pixel_sizes, grid_size, connect_margin
        )
    else:
        conn_ok = check_connectivity(new.matrix)
    if not conn_ok:
        return None, None
    # Spacing constraints still satisfied (we only touched 0/1 cells)
    # Prune any cap pairs whose endpoints or corridor were disturbed
    new.cap_pixel_pairs, new.cap_pair_values = filter_valid_pairs(
        new.matrix, new.cap_pixel_pairs, new.cap_pair_values,
        cap_pair_min_dist, cap_pair_max_dist,
        require_axis_aligned=require_axis_aligned,
    )
    info = {
        "action": "flip_01",
        "description": f"Flipped {len(flipped_cells)} pixel(s) between type-0 (empty) and type-1 (metal)",
        "flipped_cells": flipped_cells,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 2: move one switch
# ---------------------------------------------------------------------------

def action_move_switch(
    state: DesignState,
    rng: random.Random,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    require_axis_aligned: bool = True,
) -> tuple:
    """
    Pick a random switch pixel (type 3), remove it (set to 1), then place it
    at a randomly chosen different type-1 pixel that satisfies spacing rules.

    Returns (mutated copy, info_dict), or (None, None) if no valid relocation exists.
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
        return None, None

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
        return None, None

    rng.shuffle(eligible)
    for pr, pc in eligible:
        if _can_place_switch_at(new.matrix, pr, pc):
            new.matrix[pr][pc] = 3
            if check_connectivity(new.matrix):
                # Prune pairs whose corridor now overlaps the new switch
                new.cap_pixel_pairs, new.cap_pair_values = filter_valid_pairs(
                    new.matrix, new.cap_pixel_pairs, new.cap_pair_values,
                    cap_pair_min_dist, cap_pair_max_dist,
                    require_axis_aligned=require_axis_aligned,
                )
                info = {
                    "action": "move_switch",
                    "description": f"Moved switch from ({rm_r},{rm_c}) to ({pr},{pc})",
                    "removed_from": {"row": rm_r, "col": rm_c},
                    "placed_at": {"row": pr, "col": pc},
                }
                return new, info
            # undo and try next
            new.matrix[pr][pc] = 1

    return None, None


# ---------------------------------------------------------------------------
# Action 2b: modify switch (move / add / remove) — replaces move_switch
# ---------------------------------------------------------------------------

def _switch_row_first_index(matrix: list, r: int, c: int) -> int:
    """Return the row-first position of switch pixel (r, c) among all type-3 pixels."""
    idx = 0
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 3:
                if row == r and col == c:
                    return idx
                idx += 1
    raise ValueError(f"No type-3 pixel at ({r}, {c})")


def action_modify_switch(
    state: DesignState,
    rng: random.Random,
    min_switches: int = 1,
    max_switches: int = 99,
    cap_min: float = 50e-15,
    cap_max: float = 500e-15,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    require_axis_aligned: bool = True,
) -> tuple:
    """
    Modify the switch layout with equal probability across three sub-actions:

    - move:   relocate one switch to a different type-1 pixel (same constraints
              as the old action_move_switch).
    - add:    place a new switch on an eligible type-1 pixel; insert a new column
              in switch_states whose values keep all states distinct.  When
              n_states > 2, all-0 and all-1 columns are additionally rejected.
              A new cap value is inserted at the matching row-first position.
    - remove: pick a random switch, remove it (type 3 → 1), drop the corresponding
              column from switch_states and entry from cap_values.  Rejected if the
              remaining states are no longer distinct.

    Sub-actions that would violate min_switches / max_switches bounds are excluded
    from the random draw.

    Returns (mutated copy, info_dict), or (None, None) if all eligible sub-actions
    fail.
    """
    rows = state.rows
    cols = state.cols

    sw_positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if state.matrix[r][c] == 3
    ]
    n_switches = len(sw_positions)
    n_states = state.n_states

    can_move   = n_switches > 0
    can_add    = n_switches < max_switches
    can_remove = n_switches > min_switches

    choices = []
    if can_move:   choices.append("move")
    if can_add:    choices.append("add")
    if can_remove: choices.append("remove")

    if not choices:
        return None, None

    sub_action = rng.choice(choices)

    # ---- move ----
    if sub_action == "move":
        return action_move_switch(
            state, rng,
            cap_pair_min_dist=cap_pair_min_dist,
            cap_pair_max_dist=cap_pair_max_dist,
            require_axis_aligned=require_axis_aligned,
        )

    # ---- add ----
    elif sub_action == "add":
        new = state.copy()

        eligible = [
            (r, c)
            for r in range(rows)
            for c in range(1, cols - 1)
            if new.matrix[r][c] == 1 and _can_place_switch_at(new.matrix, r, c)
        ]
        if not eligible:
            return None, None

        rng.shuffle(eligible)
        for pr, pc in eligible:
            new2 = new.copy()
            new2.matrix[pr][pc] = 3
            if not check_connectivity(new2.matrix):
                continue

            idx = _switch_row_first_index(new2.matrix, pr, pc)

            # Generate a switch_states bank column that keeps states distinct.
            # When n_states > 2, additionally reject all-0 and all-1 columns.
            bank_size = len(state.cap_values[0]) if state.cap_values else 1
            placed = False
            total_bits_after = sum(len(b) for b in new2.switch_states[0]) + bank_size if new2.switch_states else bank_size
            for _ in range(200):
                col_banks = [[rng.randint(0, 1) for _ in range(bank_size)] for _ in range(n_states)]
                if n_states > 2 and (all(all(v == 0 for v in b) for b in col_banks)
                                     or all(all(v == 1 for v in b) for b in col_banks)):
                    continue
                new_ss = [
                    row[:idx] + [col_banks[si]] + row[idx:]
                    for si, row in enumerate(new2.switch_states)
                ]
                if (_states_can_be_distinct(total_bits_after, n_states)
                        and not check_switch_states_distinct_banked(new_ss)):
                    continue
                new2.switch_states = new_ss
                placed = True
                break

            if not placed:
                continue

            # Insert corresponding cap bank
            new_cap_bank = [rng.uniform(cap_min, cap_max) for _ in range(bank_size)]
            new2.cap_values.insert(idx, new_cap_bank)

            new2.cap_pixel_pairs, new2.cap_pair_values = filter_valid_pairs(
                new2.matrix, new2.cap_pixel_pairs, new2.cap_pair_values,
                cap_pair_min_dist, cap_pair_max_dist,
                require_axis_aligned=require_axis_aligned,
            )
            info = {
                "action":           "modify_switch",
                "sub_action":       "add",
                "description":      f"Added switch at ({pr},{pc}), switch_states column {idx}",
                "placed_at":        {"row": pr, "col": pc},
                "switch_list_index": idx,
                "cap_values_fF":    [v * 1e15 for v in new_cap_bank],
                "n_switches_before": n_switches,
                "n_switches_after":  n_switches + 1,
            }
            return new2, info

        return None, None

    # ---- remove ----
    else:
        sw_list = list(sw_positions)
        rng.shuffle(sw_list)

        for rm_r, rm_c in sw_list:
            idx = _switch_row_first_index(state.matrix, rm_r, rm_c)

            new = state.copy()
            new.matrix[rm_r][rm_c] = 1

            if not check_connectivity(new.matrix):
                continue

            # Drop column from switch_states
            new_ss = [row[:idx] + row[idx + 1:] for row in new.switch_states]
            n_sw_new = n_switches - 1
            total_bits_after = sum(len(b) for b in new_ss[0]) if new_ss and new_ss[0] else 0
            if (new_ss and new_ss[0]
                    and _states_can_be_distinct(total_bits_after, n_states)
                    and not check_switch_states_distinct_banked(new_ss)):
                continue
            new.switch_states = new_ss

            # Drop the corresponding cap value
            if idx < len(new.cap_values):
                new.cap_values.pop(idx)

            new.cap_pixel_pairs, new.cap_pair_values = filter_valid_pairs(
                new.matrix, new.cap_pixel_pairs, new.cap_pair_values,
                cap_pair_min_dist, cap_pair_max_dist,
                require_axis_aligned=require_axis_aligned,
            )
            info = {
                "action":            "modify_switch",
                "sub_action":        "remove",
                "description":       f"Removed switch at ({rm_r},{rm_c}), switch_states column {idx}",
                "removed_from":      {"row": rm_r, "col": rm_c},
                "switch_list_index": idx,
                "n_switches_before": n_switches,
                "n_switches_after":  n_switches - 1,
            }
            return new, info

        return None, None


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
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    single_inductor_value: bool = False,
    require_axis_aligned: bool = True,
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
) -> tuple:
    """
    Either add an inductor (type 1 → 2) or remove one (type 2 → 1).

    - If the current inductor count equals ``max_inductors``, only removal is
      attempted.
    - Otherwise the action is chosen uniformly at random.
    - ``inductor_values`` is kept in sync: a random value is inserted on add,
      and the corresponding entry is removed on removal.

    Returns (mutated copy, info_dict), or (None, None) if the action cannot be applied.
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
            return None, None
        r, c = rng.choice(candidates)
        new.matrix[r][c] = 2
        if not check_connectivity(new.matrix):
            return None, None
        # Insert a random inductor value at the correct row-first position
        idx = _inductor_row_first_index(new.matrix, r, c)
        if single_inductor_value and new.inductor_values:
            val = new.inductor_values[0]  # inherit the shared value from existing inductors
        else:
            val = rng.randint(0, n_inductor_values - 1)
        new.inductor_values.insert(idx, val)
        # Insert a random inductor cap value at the same position.
        new_ind_cap = rng.uniform(inductor_cap_min, inductor_cap_max)
        new.inductor_cap_values.insert(idx, new_ind_cap)
        # Keep inductor_switch_states in sync (new inductor defaults to ON in all states).
        # If the list was empty (all-ON convention), initialize it now so the explicit
        # per-inductor state is tracked going forward.
        n_states_count = len(new.switch_states)
        if not new.inductor_switch_states and n_states_count > 0:
            # n_ind is the count *before* this add, so new total = n_ind + 1
            new.inductor_switch_states = [[1] * (n_ind + 1) for _ in range(n_states_count)]
        elif new.inductor_switch_states:
            for state_row in new.inductor_switch_states:
                state_row.insert(idx, 1)
        # Reject if adding this inductor makes inductor states non-distinct
        n_ind_new = n_ind + 1
        if (new.inductor_switch_states
                and _states_can_be_distinct(n_ind_new, len(new.switch_states))
                and not check_inductor_states_distinct(new.inductor_switch_states)):
            return None, None
        # Prune pairs whose corridor now overlaps the new inductor
        new.cap_pixel_pairs, new.cap_pair_values = filter_valid_pairs(
            new.matrix, new.cap_pixel_pairs, new.cap_pair_values,
            cap_pair_min_dist, cap_pair_max_dist,
            require_axis_aligned=require_axis_aligned,
        )
        info = {
            "action": "flip_inductor",
            "description": f"Added inductor at ({r},{c}) with value index {val} (list index {idx})",
            "sub_action": "add",
            "position": {"row": r, "col": c},
            "inductor_list_index": idx,
            "inductor_value_index": val,
            "n_inductors_before": n_ind,
            "n_inductors_after": n_ind + 1,
        }
        return new, info

    else:  # remove
        candidates = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if new.matrix[r][c] == 2
        ]
        if not candidates:
            return None, None
        r, c = rng.choice(candidates)
        # Find and remove the corresponding inductor_values entry before modifying matrix
        idx = _inductor_row_first_index(new.matrix, r, c)
        old_val = new.inductor_values[idx] if idx < len(new.inductor_values) else None
        new.matrix[r][c] = 1
        if not check_connectivity(new.matrix):
            return None, None
        if idx < len(new.inductor_values):
            new.inductor_values.pop(idx)
        if idx < len(new.inductor_cap_values):
            new.inductor_cap_values.pop(idx)
        # Keep inductor_switch_states in sync.
        # After removing, if every state row is now empty, reset to [] (all-ON convention).
        if new.inductor_switch_states:
            for state_row in new.inductor_switch_states:
                if idx < len(state_row):
                    state_row.pop(idx)
            if all(len(sr) == 0 for sr in new.inductor_switch_states):
                new.inductor_switch_states = []
        # Reject if removing this inductor makes remaining inductor states non-distinct
        n_ind_new = n_ind - 1
        if (new.inductor_switch_states
                and _states_can_be_distinct(n_ind_new, len(new.switch_states))
                and not check_inductor_states_distinct(new.inductor_switch_states)):
            return None, None
        # Re-validate pairs (endpoint now type 1 again — generally safe, but
        # run filter in case the corridor was already invalid for other reasons)
        new.cap_pixel_pairs, new.cap_pair_values = filter_valid_pairs(
            new.matrix, new.cap_pixel_pairs, new.cap_pair_values,
            cap_pair_min_dist, cap_pair_max_dist,
            require_axis_aligned=require_axis_aligned,
        )
        info = {
            "action": "flip_inductor",
            "description": f"Removed inductor at ({r},{c}) (had value index {old_val}, list index {idx})",
            "sub_action": "remove",
            "position": {"row": r, "col": c},
            "inductor_list_index": idx,
            "inductor_value_index": old_val,
            "n_inductors_before": n_ind,
            "n_inductors_after": n_ind - 1,
        }
        return new, info


# ---------------------------------------------------------------------------
# Action (via): add/remove via-ground pixels (type 4)
# ---------------------------------------------------------------------------

def action_flip_via(
    state: DesignState,
    max_vias: int,
    via_flip_max_count: int,
    rng: random.Random,
) -> tuple:
    """
    Remove up to ``via_flip_max_count`` random via pixels (type 4 → 1) and
    add up to ``via_flip_max_count`` random via pixels (type 1 → 4) in one step.

    Both counts are drawn independently from [0, via_flip_max_count].  The
    result is rejected if the total number of vias after the operation would
    exceed ``max_vias``, or if the net change is zero.

    Via spacing rule: all 8-neighbours of a type-4 pixel must be type 0 or 1.

    Returns (mutated copy, info_dict), or (None, None) on failure.
    """
    if max_vias <= 0:
        return None, None

    n_remove = rng.randint(0, via_flip_max_count)
    n_add    = rng.randint(0, via_flip_max_count)
    if n_remove == 0 and n_add == 0:
        return None, None

    new  = state.copy()
    rows = new.rows
    cols = new.cols

    # --- Remove phase ---
    via_positions = [
        (r, c) for r in range(rows) for c in range(cols)
        if new.matrix[r][c] == 4
    ]
    n_remove = min(n_remove, len(via_positions))
    if n_remove > 0:
        rng.shuffle(via_positions)
        for r, c in via_positions[:n_remove]:
            new.matrix[r][c] = 1

    # --- Add phase ---
    actual_added = 0
    if n_add > 0:
        eligible = [
            (r, c) for r in range(rows) for c in range(1, cols - 1)
            if new.matrix[r][c] == 1 and _can_place_via_at(new.matrix, r, c)
        ]
        rng.shuffle(eligible)
        for r, c in eligible:
            if actual_added >= n_add:
                break
            if _can_place_via_at(new.matrix, r, c):
                new.matrix[r][c] = 4
                actual_added += 1
    n_add = actual_added

    # --- Enforce max ---
    n_via_after = sum(v == 4 for row in new.matrix for v in row)
    if n_via_after > max_vias:
        return None, None

    if n_remove == 0 and n_add == 0:
        return None, None  # effective no-op

    info = {
        "action":        "flip_via",
        "n_removed":     n_remove,
        "n_added":       n_add,
        "n_vias_after":  n_via_after,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 4 (new): flip the inductance value of one inductor pixel
# ---------------------------------------------------------------------------

def action_flip_inductor_value(
    state: DesignState,
    n_inductor_values: int,
    rng: random.Random,
    single_inductor_value: bool = False,
) -> tuple:
    """
    Pick a random type-2 (inductor) pixel and change its entry in
    ``inductor_values`` to a different index sampled from
    [0, n_inductor_values).

    This does NOT add or remove inductor pixels — the inductor count is
    unchanged.

    Returns (mutated copy, info_dict), or (None, None) if there are no
    inductor pixels or only one possible value.
    """
    if state.n_inductors == 0 or n_inductor_values < 2:
        return None, None

    # Collect inductor positions in row-first order
    ind_positions = [
        (r, c)
        for r in range(state.rows)
        for c in range(state.cols)
        if state.matrix[r][c] == 2
    ]

    # Pick a random inductor
    local_idx = rng.randrange(len(ind_positions))
    r, c = ind_positions[local_idx]

    # Choose a different value from the available options
    current_val = state.inductor_values[local_idx] if local_idx < len(state.inductor_values) else 0
    choices = [v for v in range(n_inductor_values) if v != current_val]
    if not choices:
        return None, None

    new = state.copy()
    # Ensure inductor_values has an entry for every inductor
    while len(new.inductor_values) < len(ind_positions):
        new.inductor_values.append(0)
    new_val = rng.choice(choices)
    if single_inductor_value:
        new.inductor_values = [new_val] * len(ind_positions)
        info = {
            "action": "flip_inductor_value",
            "description": f"Changed ALL inductor values to index {new_val} (single_inductor_value mode)",
            "old_value_index": current_val,
            "new_value_index": new_val,
        }
    else:
        new.inductor_values[local_idx] = new_val
        info = {
            "action": "flip_inductor_value",
            "description": f"Changed inductor at ({r},{c}) value index {current_val} → {new_val}",
            "position": {"row": r, "col": c},
            "inductor_list_index": local_idx,
            "old_value_index": current_val,
            "new_value_index": new_val,
        }
    return new, info


# ---------------------------------------------------------------------------
# Action 5 (was 4): perturb capacitance values
# ---------------------------------------------------------------------------

def action_perturb_caps(
    state: DesignState,
    sigma: float,
    cap_min: float,
    cap_max: float,
    rng: random.Random,
) -> tuple:
    """
    Add independent Gaussian noise (std = ``sigma``) to every cap value,
    then clip to [cap_min, cap_max].

    Returns (mutated copy, info_dict), or (None, None) if no cap values exist.
    """
    if not state.cap_values:
        return None, None

    new = state.copy()
    sw_idx = rng.randint(0, len(new.cap_values) - 1)
    cap_changes = []
    for j in range(len(new.cap_values[sw_idx])):
        old_val = new.cap_values[sw_idx][j]
        noise = rng.gauss(0.0, sigma)
        new.cap_values[sw_idx][j] = max(cap_min, min(cap_max, old_val + noise))
        cap_changes.append({
            "switch": sw_idx,
            "cap": j,
            "old_fF": round(old_val * 1e15, 4),
            "new_fF": round(new.cap_values[sw_idx][j] * 1e15, 4),
            "delta_fF": round((new.cap_values[sw_idx][j] - old_val) * 1e15, 4),
        })
    info = {
        "action": "perturb_caps",
        "description": f"Perturbed switch {sw_idx} bank ({len(cap_changes)} cap(s)) with Gaussian noise (sigma={sigma*1e15:.1f} fF)",
        "switch_index": sw_idx,
        "sigma_fF": sigma * 1e15,
        "cap_changes": cap_changes,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 5b: perturb inductor series capacitance values
# ---------------------------------------------------------------------------

def action_perturb_inductor_caps(
    state: DesignState,
    sigma: float,
    inductor_cap_min: float,
    inductor_cap_max: float,
    rng: random.Random,
) -> tuple:
    """
    Add independent Gaussian noise (std = ``sigma``) to every inductor cap value,
    then clip to [inductor_cap_min, inductor_cap_max].

    Returns (mutated copy, info_dict), or (None, None) if no inductor cap values exist.
    """
    if not state.inductor_cap_values:
        return None, None

    new = state.copy()
    cap_changes = []
    for i in range(len(new.inductor_cap_values)):
        old_val = new.inductor_cap_values[i]
        noise = rng.gauss(0.0, sigma)
        new.inductor_cap_values[i] = max(inductor_cap_min, min(inductor_cap_max, old_val + noise))
        cap_changes.append({
            "index": i,
            "old_fF": round(old_val * 1e15, 4),
            "new_fF": round(new.inductor_cap_values[i] * 1e15, 4),
            "delta_fF": round((new.inductor_cap_values[i] - old_val) * 1e15, 4),
        })
    info = {
        "action": "perturb_inductor_caps",
        "description": f"Perturbed {len(cap_changes)} inductor cap(s) with Gaussian noise (sigma={sigma*1e15:.1f} fF)",
        "sigma_fF": sigma * 1e15,
        "cap_changes": cap_changes,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 6 (was 5): flip one switch-state bit
# ---------------------------------------------------------------------------

def action_flip_switch_state(
    state: DesignState,
    rng: random.Random,
    max_flips: int = 1,
) -> tuple:
    """
    Flip between 1 and ``max_flips`` randomly chosen bits in the switch_states matrix.

    The number of bits to flip is chosen uniformly from [1, min(max_flips, total_bits)].
    Bits are sampled without replacement from the full (n_states × n_switches) grid.

    Returns (mutated copy, info_dict), or (None, None) if there are no switch states.
    """
    if not state.switch_states or not state.switch_states[0]:
        return None, None

    n_states   = state.n_states
    n_switches = len(state.switch_states[0])

    # Always flip bits within a single randomly chosen state row.
    # Each "position" is (switch_idx, cap_in_bank_idx).
    si = rng.randint(0, n_states - 1)
    positions_in_state = [
        (sw, j)
        for sw in range(n_switches)
        for j in range(len(state.switch_states[si][sw]))
    ]
    if not positions_in_state:
        return None, None
    k = rng.randint(1, min(max_flips, len(positions_in_state)))
    chosen = rng.sample(positions_in_state, k)

    new = state.copy()
    flips = []
    for sw, j in chosen:
        old_bit = new.switch_states[si][sw][j]
        new.switch_states[si][sw][j] = 1 - old_bit
        flips.append({"state": si, "switch": sw, "cap": j, "old": old_bit, "new": 1 - old_bit})

    # Reject if the flip creates duplicate switch state vectors (when enough
    # distinct vectors exist to satisfy the constraint).
    total_bits = sum(len(b) for b in new.switch_states[0]) if new.switch_states else 0
    if (_states_can_be_distinct(total_bits, n_states)
            and not check_switch_states_distinct_banked(new.switch_states)):
        return None, None

    info = {
        "action":      "flip_switch_state",
        "description": f"Flipped {k} switch-state bit(s) in state {si}: {flips}",
        "n_flips":     k,
        "flips":       flips,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 10: flip one inductor ON/OFF bit in one state
# ---------------------------------------------------------------------------

def action_flip_inductor_state(
    state: DesignState,
    rng: random.Random,
    max_flips: int = 1,
) -> tuple:
    """
    Flip between 1 and ``max_flips`` randomly chosen bits in the inductor_switch_states matrix.

    The number of bits to flip is chosen uniformly from [1, min(max_flips, total_bits)].
    Bits are sampled without replacement from the full (n_states × n_inductors) grid.

    If ``inductor_switch_states`` is empty (all-ON by convention), it is first
    initialized to an all-ones matrix of shape (n_states × n_inductors).

    Returns (mutated copy, info_dict), or (None, None) if there are no inductor
    pixels or no states.
    """
    n_ind    = state.n_inductors
    n_states = state.n_states

    if n_ind == 0 or n_states == 0:
        return None, None

    new = state.copy()

    # Initialize to all-ON if missing
    if not new.inductor_switch_states:
        new.inductor_switch_states = [[1] * n_ind for _ in range(n_states)]

    # Always flip bits within a single randomly chosen state row.
    si = rng.randint(0, n_states - 1)
    positions_in_state = [(si, ii) for ii in range(n_ind)]
    k = rng.randint(1, min(max_flips, len(positions_in_state)))
    chosen = rng.sample(positions_in_state, k)

    flips = []
    for si, ii in chosen:
        old_bit = new.inductor_switch_states[si][ii]
        new.inductor_switch_states[si][ii] = 1 - old_bit
        flips.append({"state": si, "inductor": ii, "old": old_bit, "new": 1 - old_bit})

    # Reject if the flip creates duplicate inductor state vectors (when enough
    # distinct vectors exist to satisfy the constraint).
    if (_states_can_be_distinct(n_ind, n_states)
            and not check_inductor_states_distinct(new.inductor_switch_states)):
        return None, None

    info = {
        "action":      "flip_inductor_state",
        "description": f"Flipped {k} inductor-state bit(s): {flips}",
        "n_flips":     k,
        "flips":       flips,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 7: add one cap pixel pair
# ---------------------------------------------------------------------------

def action_add_cap_pair(
    state: DesignState,
    max_cap_pairs: int,
    cap_pair_min: float,
    cap_pair_max: float,
    cap_pair_min_dist: int,
    cap_pair_max_dist: int,
    rng: random.Random,
    max_search: int = 300,
    require_axis_aligned: bool = True,
) -> tuple:
    """
    Find two type-1 interior pixels that form a valid cap pair and add it to
    the design.  When ``require_axis_aligned`` is True (default), only
    horizontal or vertical pairs are considered; otherwise diagonal pairs are
    also permitted.

    Returns (mutated copy, info_dict), or (None, None) if already at the cap
    pair limit or no valid pair can be found within ``max_search`` attempts.
    """
    if state.n_cap_pairs >= max_cap_pairs:
        return None, None

    rows = state.rows
    cols = state.cols

    # Pre-enumerate all type-1 pixel pairs within Chebyshev distance range,
    # including diagonal and off-diagonal directions.
    metal_set = {
        (r, c)
        for r in range(rows)
        for c in range(1, cols - 1)
        if state.matrix[r][c] == 1
    }
    if len(metal_set) < 2:
        return None, None

    existing = state.cap_pixel_pairs
    existing_path_cells = _build_other_path_cells(existing)

    candidates = []
    for (r1, c1) in metal_set:
        if require_axis_aligned:
            # Only horizontal (dr=0) and vertical (dc=0) directions
            for dist in range(cap_pair_min_dist, cap_pair_max_dist + 1):
                for (dr, dc) in ((0, dist), (0, -dist), (dist, 0), (-dist, 0)):
                    if (r1 + dr, c1 + dc) in metal_set:
                        candidates.append(((r1, c1), (r1 + dr, c1 + dc)))
        else:
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
    for (r1, c1), (r2, c2) in candidates[:max_search]:
        candidate = [[r1, c1], [r2, c2]]
        rev_candidate = [[r2, c2], [r1, c1]]
        if candidate in existing or rev_candidate in existing:
            continue
        if not check_cap_pair_valid(
            state.matrix, candidate,
            cap_pair_min_dist, cap_pair_max_dist,
            existing_path_cells,
            require_axis_aligned=require_axis_aligned,
        ):
            continue
        new = state.copy()
        new_val = rng.uniform(cap_pair_min, cap_pair_max)
        new.cap_pixel_pairs.append(candidate)
        new.cap_pair_values.append(new_val)
        info = {
            "action": "add_cap_pair",
            "description": f"Added cap pair [({r1},{c1})↔({r2},{c2})] = {new_val*1e15:.1f} fF",
            "pair": candidate,
            "value_fF": new_val * 1e15,
            "n_pairs_before": state.n_cap_pairs,
            "n_pairs_after": state.n_cap_pairs + 1,
        }
        return new, info

    return None, None


# ---------------------------------------------------------------------------
# Action 8: remove one cap pixel pair
# ---------------------------------------------------------------------------

def action_remove_cap_pair(
    state: DesignState,
    rng: random.Random,
) -> tuple:
    """
    Remove a randomly selected cap pixel pair.

    Returns (mutated copy, info_dict), or (None, None) if no pairs exist.
    """
    if state.n_cap_pairs == 0:
        return None, None

    new = state.copy()
    idx = rng.randrange(len(new.cap_pixel_pairs))
    removed_pair = new.cap_pixel_pairs.pop(idx)
    removed_val = new.cap_pair_values.pop(idx)
    info = {
        "action": "remove_cap_pair",
        "description": (
            f"Removed cap pair [{removed_pair[0]}↔{removed_pair[1]}] "
            f"({removed_val*1e15:.1f} fF)"
        ),
        "pair": removed_pair,
        "value_fF": removed_val * 1e15,
        "n_pairs_before": state.n_cap_pairs,
        "n_pairs_after": state.n_cap_pairs - 1,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 9: perturb cap pair capacitance values
# ---------------------------------------------------------------------------

def action_perturb_pair_caps(
    state: DesignState,
    sigma: float,
    cap_pair_min: float,
    cap_pair_max: float,
    rng: random.Random,
) -> tuple:
    """
    Add independent Gaussian noise (std = ``sigma``) to every cap pair value,
    then clip to [cap_pair_min, cap_pair_max].

    Returns (mutated copy, info_dict), or (None, None) if no pair values exist.
    """
    if not state.cap_pair_values:
        return None, None

    new = state.copy()
    pair_changes = []
    for i in range(len(new.cap_pair_values)):
        old_val = new.cap_pair_values[i]
        noise = rng.gauss(0.0, sigma)
        new.cap_pair_values[i] = max(cap_pair_min, min(cap_pair_max, old_val + noise))
        pair_changes.append({
            "index": i,
            "old_fF": round(old_val * 1e15, 4),
            "new_fF": round(new.cap_pair_values[i] * 1e15, 4),
            "delta_fF": round((new.cap_pair_values[i] - old_val) * 1e15, 4),
        })
    info = {
        "action": "perturb_pair_caps",
        "description": (
            f"Perturbed {len(pair_changes)} pair cap(s) with Gaussian noise "
            f"(sigma={sigma*1e15:.1f} fF)"
        ),
        "sigma_fF": sigma * 1e15,
        "pair_changes": pair_changes,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 12: perturb per-pixel sizes
# ---------------------------------------------------------------------------

def action_perturb_pixel_sizes(
    state: DesignState,
    n_pixels: int,
    sigma: float,
    size_min: float,
    size_max: float,
    grid_size: float,
    connect_margin: float,
    rng: random.Random,
) -> tuple:
    """
    Add Gaussian noise (std = ``sigma`` um) to the size of up to ``n_pixels``
    randomly-selected non-ground interior pixels, then clip to
    [size_min, size_max].

    No-op (returns (None, None)) when per-pixel sizing is disabled
    (``state.pixel_sizes`` empty).  After perturbation the design must remain
    connected under the overlap rule — if shrinking a pixel breaks the metal
    path the mutation is rejected.

    Returns (mutated copy, info_dict), or (None, None) if infeasible / disabled.
    """
    if not state.pixel_sizes:
        return None, None

    rows, cols = state.rows, state.cols
    # Only cells that actually carry a metal patch (non-zero) and are interior
    # (feed columns are feedline traces, not sized patches).
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(1, cols - 1)
        if state.matrix[r][c] != 0
    ]
    if not candidates:
        return None, None

    k = rng.randint(1, min(n_pixels, len(candidates)))
    chosen = rng.sample(candidates, k)

    new = state.copy()
    size_changes = []
    for r, c in chosen:
        old_val = new.pixel_sizes[r][c]
        noise = rng.gauss(0.0, sigma)
        new.pixel_sizes[r][c] = max(size_min, min(size_max, old_val + noise))
        size_changes.append({
            "row": r,
            "col": c,
            "old_um": round(old_val, 4),
            "new_um": round(new.pixel_sizes[r][c], 4),
        })

    # A size reduction may sever the metal path — reject if so.
    if not check_connectivity_sized(new.matrix, new.pixel_sizes, grid_size, connect_margin):
        return None, None

    info = {
        "action": "perturb_pixel_sizes",
        "description": (
            f"Perturbed {len(size_changes)} pixel size(s) with Gaussian noise "
            f"(sigma={sigma:.2f} um)"
        ),
        "sigma_um": sigma,
        "size_changes": size_changes,
    }
    return new, info


# ---------------------------------------------------------------------------
# Action 13: modify shunt-capacitor pixels (add / remove type-5)
# ---------------------------------------------------------------------------

def _shunt_cap_row_first_index(matrix: list, r: int, c: int) -> int:
    """Return the row-first position of shunt-cap pixel (r, c) among all type-5 pixels."""
    idx = 0
    for row in range(len(matrix)):
        for col in range(len(matrix[0])):
            if matrix[row][col] == 5:
                if row == r and col == c:
                    return idx
                idx += 1
    raise ValueError(f"No type-5 pixel at ({r}, {c})")


def action_modify_shunt_cap(
    state: DesignState,
    rng: random.Random,
    max_shunt_caps: int = 99,
    min_shunt_caps: int = 0,
    shunt_cap_min: float = 50e-15,
    shunt_cap_max: float = 1000e-15,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    require_axis_aligned: bool = True,
) -> tuple:
    """
    Add or remove one shunt-capacitor pixel (type 5).

    A shunt cap is a switch pixel without the switch: a metal pixel permanently
    tied to ground through a MIM cap.  It has no per-state ON/OFF — only a
    capacitance value.  This action is the type-5 analogue of the switch
    add/remove sub-actions, but simpler (no switch_states bookkeeping).

    - add:    convert an eligible interior type-1 pixel to type 5 and insert a
              random capacitance at its row-first position.
    - remove: convert a random type-5 pixel back to type 1 and drop its value.

    Sub-actions that would violate min_shunt_caps / max_shunt_caps are excluded.
    Converting metal ↔ shunt-cap never removes metal, so connectivity is
    preserved by construction.

    Returns (mutated copy, info_dict), or (None, None) if no sub-action applies.
    """
    rows = state.rows
    cols = state.cols

    scap_positions = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if state.matrix[r][c] == 5
    ]
    n_scap = len(scap_positions)

    can_add    = n_scap < max_shunt_caps
    can_remove = n_scap > min_shunt_caps

    choices = []
    if can_add:    choices.append("add")
    if can_remove: choices.append("remove")
    if not choices:
        return None, None

    sub_action = rng.choice(choices)

    # ---- add ----
    if sub_action == "add":
        eligible = [
            (r, c)
            for r in range(rows)
            for c in range(1, cols - 1)
            if state.matrix[r][c] == 1 and _can_place_shunt_cap_at(state.matrix, r, c)
        ]
        if not eligible:
            return None, None
        pr, pc = rng.choice(eligible)
        new = state.copy()
        new.matrix[pr][pc] = 5
        idx = _shunt_cap_row_first_index(new.matrix, pr, pc)
        new_val = rng.uniform(shunt_cap_min, shunt_cap_max)
        new.shunt_cap_values.insert(idx, new_val)
        # Prune any cap pairs whose 3×3 corridor now overlaps the new shunt cap
        # (same treatment switches get — keeps the type-5 footprint clear).
        new.cap_pixel_pairs, new.cap_pair_values = filter_valid_pairs(
            new.matrix, new.cap_pixel_pairs, new.cap_pair_values,
            cap_pair_min_dist, cap_pair_max_dist,
            require_axis_aligned=require_axis_aligned,
        )
        info = {
            "action":              "modify_shunt_cap",
            "sub_action":          "add",
            "description":         f"Added shunt cap at ({pr},{pc}) = {new_val*1e15:.1f} fF (list index {idx})",
            "placed_at":           {"row": pr, "col": pc},
            "shunt_list_index":    idx,
            "value_fF":            new_val * 1e15,
            "n_shunt_before":      n_scap,
            "n_shunt_after":       n_scap + 1,
        }
        return new, info

    # ---- remove ----
    else:
        rm_r, rm_c = rng.choice(scap_positions)
        idx = _shunt_cap_row_first_index(state.matrix, rm_r, rm_c)
        new = state.copy()
        new.matrix[rm_r][rm_c] = 1
        removed_val = (new.shunt_cap_values.pop(idx)
                       if idx < len(new.shunt_cap_values) else None)
        info = {
            "action":           "modify_shunt_cap",
            "sub_action":       "remove",
            "description":      f"Removed shunt cap at ({rm_r},{rm_c}) (list index {idx})",
            "removed_from":     {"row": rm_r, "col": rm_c},
            "shunt_list_index": idx,
            "value_fF":         (removed_val * 1e15) if removed_val is not None else None,
            "n_shunt_before":   n_scap,
            "n_shunt_after":    n_scap - 1,
        }
        return new, info


# ---------------------------------------------------------------------------
# Action 14: perturb shunt-capacitor values
# ---------------------------------------------------------------------------

def action_perturb_shunt_caps(
    state: DesignState,
    sigma: float,
    shunt_cap_min: float,
    shunt_cap_max: float,
    rng: random.Random,
) -> tuple:
    """
    Add independent Gaussian noise (std = ``sigma``) to every shunt-cap value,
    then clip to [shunt_cap_min, shunt_cap_max].

    Returns (mutated copy, info_dict), or (None, None) if no shunt caps exist.
    """
    if not state.shunt_cap_values:
        return None, None

    new = state.copy()
    cap_changes = []
    for i in range(len(new.shunt_cap_values)):
        old_val = new.shunt_cap_values[i]
        noise = rng.gauss(0.0, sigma)
        new.shunt_cap_values[i] = max(shunt_cap_min, min(shunt_cap_max, old_val + noise))
        cap_changes.append({
            "index": i,
            "old_fF": round(old_val * 1e15, 4),
            "new_fF": round(new.shunt_cap_values[i] * 1e15, 4),
            "delta_fF": round((new.shunt_cap_values[i] - old_val) * 1e15, 4),
        })
    info = {
        "action": "perturb_shunt_caps",
        "description": (
            f"Perturbed {len(cap_changes)} shunt cap(s) with Gaussian noise "
            f"(sigma={sigma*1e15:.1f} fF)"
        ),
        "sigma_fF": sigma * 1e15,
        "cap_changes": cap_changes,
    }
    return new, info


# ---------------------------------------------------------------------------
# Dispatcher: pick a random action and apply it
# ---------------------------------------------------------------------------

_ACTION_NAMES = [
    "flip_01",
    "modify_switch",
    "flip_inductor",
    "flip_inductor_value",
    "perturb_caps",
    "flip_switch_state",
    "add_cap_pair",
    "remove_cap_pair",
    "perturb_pair_caps",
    "flip_inductor_state",
    "flip_via",
    "perturb_inductor_caps",
    "perturb_pixel_sizes",
    "modify_shunt_cap",
    "perturb_shunt_caps",
]


def _dispatch_action(
    action_name: str,
    state: DesignState,
    n_regular_flip: int,
    max_inductors: int,
    cap_min: float,
    cap_max: float,
    cap_sigma: float,
    rng: random.Random,
    n_inductor_values: int,
    max_cap_pairs: int,
    cap_pair_min: float,
    cap_pair_max: float,
    cap_pair_sigma: float,
    cap_pair_min_dist: int,
    cap_pair_max_dist: int,
    max_switch_state_flips: int,
    max_inductor_state_flips: int,
    max_vias: int = 0,
    via_flip_max_count: int = 3,
    single_inductor_value: bool = False,
    require_axis_aligned: bool = True,
    flip_inner_region=None,
    flip_inner_prob: float = 1.0,
    flip_outer_prob: float = 1.0,
    flip_region_size: int = 8,
    min_switches: int = 1,
    max_switches: int = 99,
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
    inductor_cap_sigma: float = 50e-15,
    pixel_size_min: float = 25.0,
    pixel_size_max: float = 37.0,
    pixel_size_sigma: float = 2.0,
    grid_size: float = 25.0,
    connect_margin: float = 2.0,
    n_pixel_size_perturb: int = 8,
    max_shunt_caps: int = 0,
    min_shunt_caps: int = 0,
    shunt_cap_min: float = 50e-15,
    shunt_cap_max: float = 1000e-15,
    shunt_cap_sigma: float = 50e-15,
) -> tuple:
    """Apply a single named action once. Returns (result, info) or (None, None)."""
    if action_name == "flip_01":
        return action_flip_01(
            state, n_regular_flip, rng,
            cap_pair_min_dist=cap_pair_min_dist,
            cap_pair_max_dist=cap_pair_max_dist,
            flip_inner_region=flip_inner_region,
            flip_inner_prob=flip_inner_prob,
            flip_outer_prob=flip_outer_prob,
            require_axis_aligned=require_axis_aligned,
            flip_region_size=flip_region_size,
            grid_size=grid_size,
            connect_margin=connect_margin,
        )
    elif action_name == "modify_switch":
        return action_modify_switch(
            state, rng,
            min_switches=min_switches,
            max_switches=max_switches,
            cap_min=cap_min,
            cap_max=cap_max,
            cap_pair_min_dist=cap_pair_min_dist,
            cap_pair_max_dist=cap_pair_max_dist,
            require_axis_aligned=require_axis_aligned,
        )
    elif action_name == "flip_inductor":
        return action_flip_inductor(
            state, max_inductors, rng, n_inductor_values,
            cap_pair_min_dist=cap_pair_min_dist,
            cap_pair_max_dist=cap_pair_max_dist,
            single_inductor_value=single_inductor_value,
            require_axis_aligned=require_axis_aligned,
            inductor_cap_min=inductor_cap_min,
            inductor_cap_max=inductor_cap_max,
        )
    elif action_name == "flip_inductor_value":
        return action_flip_inductor_value(state, n_inductor_values, rng,
                                          single_inductor_value=single_inductor_value)
    elif action_name == "perturb_caps":
        return action_perturb_caps(state, cap_sigma, cap_min, cap_max, rng)
    elif action_name == "flip_switch_state":
        return action_flip_switch_state(state, rng, max_flips=max_switch_state_flips)
    elif action_name == "add_cap_pair":
        return action_add_cap_pair(
            state, max_cap_pairs, cap_pair_min, cap_pair_max,
            cap_pair_min_dist, cap_pair_max_dist, rng,
            require_axis_aligned=require_axis_aligned,
        )
    elif action_name == "remove_cap_pair":
        return action_remove_cap_pair(state, rng)
    elif action_name == "perturb_pair_caps":
        return action_perturb_pair_caps(
            state, cap_pair_sigma, cap_pair_min, cap_pair_max, rng,
        )
    elif action_name == "flip_inductor_state":
        return action_flip_inductor_state(state, rng, max_flips=max_inductor_state_flips)
    elif action_name == "flip_via":
        return action_flip_via(state, max_vias, via_flip_max_count, rng)
    elif action_name == "perturb_inductor_caps":
        return action_perturb_inductor_caps(
            state, inductor_cap_sigma, inductor_cap_min, inductor_cap_max, rng,
        )
    elif action_name == "perturb_pixel_sizes":
        return action_perturb_pixel_sizes(
            state, n_pixel_size_perturb, pixel_size_sigma,
            pixel_size_min, pixel_size_max, grid_size, connect_margin, rng,
        )
    elif action_name == "modify_shunt_cap":
        return action_modify_shunt_cap(
            state, rng,
            max_shunt_caps=max_shunt_caps,
            min_shunt_caps=min_shunt_caps,
            shunt_cap_min=shunt_cap_min,
            shunt_cap_max=shunt_cap_max,
            cap_pair_min_dist=cap_pair_min_dist,
            cap_pair_max_dist=cap_pair_max_dist,
            require_axis_aligned=require_axis_aligned,
        )
    elif action_name == "perturb_shunt_caps":
        return action_perturb_shunt_caps(
            state, shunt_cap_sigma, shunt_cap_min, shunt_cap_max, rng,
        )
    return None, None


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
    max_retries_per_action: int = 200,
    max_cap_pairs: int = 0,
    cap_pair_min: float = 50e-15,
    cap_pair_max: float = 500e-15,
    cap_pair_sigma: float = 20e-15,
    cap_pair_min_dist: int = 2,
    cap_pair_max_dist: int = 3,
    max_switch_state_flips: int = 1,
    max_inductor_state_flips: int = 1,
    max_vias: int = 0,
    via_flip_max_count: int = 3,
    single_inductor_value: bool = False,
    require_axis_aligned: bool = True,
    flip_inner_region=None,
    flip_inner_prob: float = 1.0,
    flip_outer_prob: float = 1.0,
    flip_region_size: int = 8,
    min_switches: int = 1,
    max_switches: int = 99,
    inductor_cap_min: float = 200e-15,
    inductor_cap_max: float = 1000e-15,
    inductor_cap_sigma: float = 50e-15,
    pixel_size_min: float = 25.0,
    pixel_size_max: float = 37.0,
    pixel_size_sigma: float = 2.0,
    grid_size: float = 25.0,
    connect_margin: float = 2.0,
    n_pixel_size_perturb: int = 8,
    max_shunt_caps: int = 0,
    min_shunt_caps: int = 0,
    shunt_cap_min: float = 50e-15,
    shunt_cap_max: float = 1000e-15,
    shunt_cap_sigma: float = 50e-15,
) -> tuple:
    """
    Pick one action by weight, retry it up to ``max_retries_per_action`` times,
    then fall back to another action if it stays infeasible — and so on until
    either a feasible mutation is found or all actions are exhausted.

    This preserves the intended weight distribution: an action is chosen once
    by its weight, then exhausted before the next action is drawn.  The old
    approach re-drew from the full pool on every failure, which distorted the
    effective weights whenever some actions are frequently infeasible.

    Returns
    -------
    (DesignState, info_dict) on success, or (None, None) on failure.

    Parameters
    ----------
    action_weights : optional list of floats (unnormalised weights).
        Order matches _ACTION_NAMES:
        [flip_01, modify_switch, flip_inductor, flip_inductor_value,
         perturb_caps, flip_switch_state,
         add_cap_pair, remove_cap_pair, perturb_pair_caps, flip_inductor_state,
         flip_via, perturb_inductor_caps, perturb_pixel_sizes,
         modify_shunt_cap, perturb_shunt_caps]
        Lists shorter than len(_ACTION_NAMES) are zero-padded; longer lists are truncated.
    max_retries_per_action : number of attempts for the chosen action before
        giving up on it and sampling a different one.
    max_cap_pairs : maximum number of cap pixel pairs allowed (0 disables pair actions).
    """
    # Pad or truncate weights to match action list length for backward compat.
    weights = list(action_weights) if action_weights else [1.0] * len(_ACTION_NAMES)
    if len(weights) < len(_ACTION_NAMES):
        weights += [0.0] * (len(_ACTION_NAMES) - len(weights))
    elif len(weights) > len(_ACTION_NAMES):
        weights = weights[:len(_ACTION_NAMES)]

    dispatch_kwargs = dict(
        n_regular_flip=n_regular_flip,
        max_inductors=max_inductors,
        cap_min=cap_min,
        cap_max=cap_max,
        cap_sigma=cap_sigma,
        rng=rng,
        n_inductor_values=n_inductor_values,
        max_cap_pairs=max_cap_pairs,
        cap_pair_min=cap_pair_min,
        cap_pair_max=cap_pair_max,
        cap_pair_sigma=cap_pair_sigma,
        cap_pair_min_dist=cap_pair_min_dist,
        cap_pair_max_dist=cap_pair_max_dist,
        max_switch_state_flips=max_switch_state_flips,
        max_inductor_state_flips=max_inductor_state_flips,
        max_vias=max_vias,
        via_flip_max_count=via_flip_max_count,
        single_inductor_value=single_inductor_value,
        require_axis_aligned=require_axis_aligned,
        flip_inner_region=flip_inner_region,
        flip_inner_prob=flip_inner_prob,
        flip_outer_prob=flip_outer_prob,
        flip_region_size=flip_region_size,
        min_switches=min_switches,
        max_switches=max_switches,
        inductor_cap_min=inductor_cap_min,
        inductor_cap_max=inductor_cap_max,
        inductor_cap_sigma=inductor_cap_sigma,
        pixel_size_min=pixel_size_min,
        pixel_size_max=pixel_size_max,
        pixel_size_sigma=pixel_size_sigma,
        grid_size=grid_size,
        connect_margin=connect_margin,
        n_pixel_size_perturb=n_pixel_size_perturb,
        max_shunt_caps=max_shunt_caps,
        min_shunt_caps=min_shunt_caps,
        shunt_cap_min=shunt_cap_min,
        shunt_cap_max=shunt_cap_max,
        shunt_cap_sigma=shunt_cap_sigma,
    )

    # Build a pool of (name, weight) pairs with positive weight.
    pool_names   = [n for n, w in zip(_ACTION_NAMES, weights) if w > 0]
    pool_weights = [w for w in weights if w > 0]

    while pool_names:
        # 1. Choose one action from the remaining pool by weight.
        chosen = rng.choices(pool_names, weights=pool_weights)[0]
        chosen_idx = pool_names.index(chosen)

        # 2. Retry that exact action up to max_retries_per_action times.
        for _ in range(max_retries_per_action):
            result, info = _dispatch_action(chosen, state, **dispatch_kwargs)
            if result is not None:
                return result, info

        # 3. Action exhausted — remove it and try the next one.
        pool_names.pop(chosen_idx)
        pool_weights.pop(chosen_idx)

    return None, None

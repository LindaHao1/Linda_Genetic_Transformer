"""
expand_design.py — Physically expand a smaller DesignState to a larger grid by
inserting rows and columns one at a time.

How expansion works
-------------------
We need to add exactly delta_cols columns and delta_rows rows to reach the target
size.  The order of col/row insertions is shuffled so they are interleaved.

For each insertion we pick one of three modes:

  "duplicate"  — copy the content of the adjacent row or column.
                 Insertion position is chosen randomly in the interior.
                 Special pixels (inductor/switch) in the source are demoted to
                 metal (type 1) so Chebyshev spacing constraints are not broken.

  "random_fill" — fill the new row/column with random 0/1 pixels.
                  Insertion position is chosen randomly in the interior.
                  No special pixels are ever placed.

  "zero_pad"   — insert a row or column of all zeros (ground).
                 Insertion position is always at a boundary (leftmost/rightmost
                 interior column, or top/bottom row), so zero padding expands
                 the design outward rather than slicing through the interior.

The relative frequency of each mode is controlled by p_duplicate, p_random_fill,
and p_zero_pad weights in the config.  Higher p_duplicate preserves more of the
original design's RF performance topology.
"""

from __future__ import annotations

import copy
import random as _random_module
from typing import List, Tuple

from virtuoso_optimization_SA.design import DesignState
from virtuoso_optimization_SA.feasibility import filter_valid_pairs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _choose_insertion_mode(
    rng: _random_module.Random,
    p_duplicate: float,
    p_random_fill: float,
    p_zero_pad: float,
) -> str:
    total = p_duplicate + p_random_fill + p_zero_pad
    r = rng.random() * total
    if r < p_duplicate:
        return "duplicate"
    if r < p_duplicate + p_random_fill:
        return "random_fill"
    return "zero_pad"


def _insert_column(
    matrix: List[List[int]],
    col_pos: int,
    mode: str,
    rng: _random_module.Random,
) -> List[List[int]]:
    """
    Insert a new column at col_pos.

    For duplicate/random_fill: col_pos is a random interior position [1, cols-2].
    For zero_pad: col_pos is a boundary position (1 = left edge, cols-1 = right edge).
    Feed columns (0 and N-1) are never touched.
    """
    new_matrix = []
    for row in matrix:
        if mode == "zero_pad":
            new_val = 0
        elif mode == "random_fill":
            new_val = rng.randint(0, 1)
        else:  # "duplicate"
            src = row[col_pos - 1]
            # Demote special pixels so Chebyshev ≥ 3 spacing is not immediately violated
            new_val = 1 if src in (2, 3) else src
        new_row = list(row)
        new_row.insert(col_pos, new_val)
        new_matrix.append(new_row)
    return new_matrix


def _insert_row(
    matrix: List[List[int]],
    row_pos: int,
    mode: str,
    rng: _random_module.Random,
) -> List[List[int]]:
    """
    Insert a new row at row_pos.

    For duplicate/random_fill: row_pos is a random position in [0, rows].
    For zero_pad: row_pos is a boundary position (0 = bottom edge, rows = top edge).
    Feed columns (0 and cols-1) are always forced to 0 in new rows.
    """
    cols = len(matrix[0])
    if mode == "zero_pad":
        new_row = [0] * cols
    elif mode == "random_fill":
        # Feed columns must stay 0 to avoid creating a second feed pixel.
        new_row = [
            (rng.randint(0, 1) if 0 < c < cols - 1 else 0)
            for c in range(cols)
        ]
    else:  # "duplicate"
        src_idx = max(0, row_pos - 1)
        src_row = matrix[src_idx]
        new_row = []
        for c, v in enumerate(src_row):
            if c == 0 or c == cols - 1:
                new_row.append(0)  # never duplicate feed pixels
            elif v in (2, 3):
                new_row.append(1)  # demote special pixels
            else:
                new_row.append(v)
    result = [list(r) for r in matrix]
    result.insert(row_pos, new_row)
    return result


def _shift_cap_pairs_col(
    pairs: List[List[List[int]]],
    col_pos: int,
) -> List[List[List[int]]]:
    """Shift cap pair column coordinates >= col_pos by +1."""
    new_pairs = []
    for pair in pairs:
        new_pair = [[ep[0], ep[1] + 1 if ep[1] >= col_pos else ep[1]] for ep in pair]
        new_pairs.append(new_pair)
    return new_pairs


def _shift_cap_pairs_row(
    pairs: List[List[List[int]]],
    row_pos: int,
) -> List[List[List[int]]]:
    """Shift cap pair row coordinates >= row_pos by +1."""
    new_pairs = []
    for pair in pairs:
        new_pair = [[ep[0] + 1 if ep[0] >= row_pos else ep[0], ep[1]] for ep in pair]
        new_pairs.append(new_pair)
    return new_pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expand_design_to_size(
    design: DesignState,
    target_rows: int,
    target_cols: int,
    cfg,
    rng: _random_module.Random,
) -> DesignState:
    """
    Physically expand *design* to (target_rows x target_cols) by inserting
    exactly (target_cols - design.cols) columns and (target_rows - design.rows)
    rows.  The insertion order is shuffled so rows and cols are interleaved.

    Each insertion uses one of three modes (chosen by weighted random):
      - "duplicate"   — copy adjacent content at a random interior position
      - "random_fill" — random 0/1 content at a random interior position
      - "zero_pad"    — zeros inserted at the boundary (edges only)

    Weights are cfg.finetune_expand_p_duplicate / p_random_fill / p_zero_pad.

    cap_pixel_pairs coordinates are shifted after every insertion and invalid
    pairs are pruned.  switch_states, cap_values, inductor_values, and
    inductor_switch_states are unchanged (no new special pixels are inserted).

    Returns a new DesignState; the input is not modified.
    """
    delta_rows = target_rows - design.rows
    delta_cols = target_cols - design.cols

    if delta_rows < 0 or delta_cols < 0:
        raise ValueError(
            f"expand_design_to_size: target ({target_rows}x{target_cols}) is smaller "
            f"than source ({design.rows}x{design.cols}) in at least one dimension."
        )
    if delta_rows == 0 and delta_cols == 0:
        return design.copy()

    p_dup  = cfg.finetune_expand_p_duplicate
    p_rand = cfg.finetune_expand_p_random_fill
    p_pad  = cfg.finetune_expand_p_zero_pad

    # Build a shuffled list of operations — exactly delta_cols cols + delta_rows rows
    ops = ["col"] * delta_cols + ["row"] * delta_rows
    rng.shuffle(ops)

    current = design.copy()

    for op in ops:
        mode  = _choose_insertion_mode(rng, p_dup, p_rand, p_pad)
        mat   = current.matrix
        pairs = current.cap_pixel_pairs
        vals  = current.cap_pair_values

        if op == "col":
            if mode == "zero_pad":
                # Boundary only: insert just inside the left or right feed column
                col_pos = rng.choice([1, current.cols - 1])
            else:
                # Random interior position (never at feed columns 0 or cols-1)
                col_pos = rng.randint(1, current.cols - 1)
            new_mat = _insert_column(mat, col_pos, mode, rng)
            shifted = _shift_cap_pairs_col(pairs, col_pos)
        else:
            if mode == "zero_pad":
                # Boundary only: insert at the bottom (0) or top (rows) edge
                row_pos = rng.choice([0, current.rows])
            else:
                row_pos = rng.randint(0, current.rows)
            new_mat = _insert_row(mat, row_pos, mode, rng)
            shifted = _shift_cap_pairs_row(pairs, row_pos)

        valid_pairs, valid_vals = filter_valid_pairs(
            new_mat, shifted, vals,
            cfg.cap_pair_min_dist,
            cfg.cap_pair_max_dist,
            cfg.cap_pair_axis_aligned,
        )

        current = DesignState(
            matrix                  = new_mat,
            switch_states           = copy.deepcopy(current.switch_states),
            cap_values              = list(current.cap_values),
            inductor_values         = list(current.inductor_values),
            inductor_switch_states  = copy.deepcopy(current.inductor_switch_states),
            cap_pixel_pairs         = valid_pairs,
            cap_pair_values         = valid_vals,
        )

    return current

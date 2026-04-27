"""
expand_design.py — Physically expand a smaller DesignState to a larger grid.

Two top-level expansion modes are selected by weighted random (p_embed vs
p_random_insertion):

  p_embed         — Embed the seed design at the center of the target grid.
                    Outside pixels are filled with a random density d drawn
                    once per expansion call from Uniform(0, 1): each outside
                    pixel is independently set to 1 with probability d (so
                    d=0 gives pure zero-padding, d=1 gives all-ones).
                    Feed rows are always extended with solid metal regardless
                    of density.

  p_random_insertion — Insert rows and columns one at a time in shuffled
                    order.  Each insertion picks one of three sub-modes by
                    weighted random (p_duplicate / p_random_fill / p_zero_pad):

    "duplicate"   — copy the adjacent row/col at a random interior position.
                    Special pixels (inductor/switch) are demoted to metal (type
                    1) so Chebyshev spacing constraints are not broken.

    "random_fill" — random 0/1 content at a random interior position.

    "zero_pad"    — zeros inserted at the boundary edges only (not interior).
"""

from __future__ import annotations

import copy
import random as _random_module
from typing import List, Tuple

from virtuoso_engine.pixel_matrix import get_feed_rows
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


def _enforce_feed_adjacency(
    matrix: List[List[int]],
    col_pos: int,
    left_feed_row: int,
    right_feed_row: int,
    pre_insert_cols: int,
) -> List[List[int]]:
    """
    After inserting a column at col_pos, guarantee the bridging pixel between
    each feed column and the interior is non-zero.

    The connectivity checker forbids diagonal steps from feed columns, so the
    only valid exit from (left_feed_row, 0) is straight-right to col 1, and
    the only valid exit from (right_feed_row, N-1) is straight-left to col N-2.
    If either bridging pixel is 0 the design is unconditionally disconnected.
    """
    result = [list(row) for row in matrix]
    if col_pos == 1 and result[left_feed_row][1] == 0:
        result[left_feed_row][1] = 1
    if col_pos == pre_insert_cols - 1 and result[right_feed_row][col_pos] == 0:
        result[right_feed_row][col_pos] = 1
    return result


def embed_centered_with_density(
    design: DesignState,
    target_rows: int,
    target_cols: int,
    density: float,
    rng: _random_module.Random,
) -> DesignState:
    """
    Embed the original design at the center of a (target_rows x target_cols) grid.

    Outside pixels (not covered by the seed and not in feed-row extensions) are
    filled with independent Bernoulli(density) draws: each pixel is 1 with
    probability `density` and 0 otherwise.  density=0.0 gives pure zero-padding;
    density=1.0 gives an all-ones background.

    Feed connectivity is maintained by extending a solid metal trace (type 1)
    from each new edge column to the original feed pixels, regardless of density.

    Cap-pair coordinates are shifted by (pad_bottom, pad_left); distances are
    unchanged so all pairs remain valid without filtering.
    """
    src = design.matrix
    src_rows = len(src)
    src_cols = len(src[0])

    delta_rows = target_rows - src_rows
    delta_cols = target_cols - src_cols

    pad_bottom = delta_rows // 2
    pad_left   = delta_cols // 2

    left_feed_row, right_feed_row = get_feed_rows(src)
    new_left_feed_row  = left_feed_row  + pad_bottom
    new_right_feed_row = right_feed_row + pad_bottom

    # Fill background with random density (skip feed columns 0 and N-1),
    # then embed original design on top.
    # Feed columns are kept all-zero here; feed extensions below set exactly one
    # metal pixel in each, satisfying the one-non-zero-per-feed-column rule.
    new_matrix = [
        [
            0 if (c == 0 or c == target_cols - 1) else (1 if rng.random() < density else 0)
            for c in range(target_cols)
        ]
        for _ in range(target_rows)
    ]
    for r in range(src_rows):
        for c in range(src_cols):
            new_matrix[r + pad_bottom][c + pad_left] = src[r][c]

    # Left feed extension: solid metal from new col 0 to the embedded left feed
    for c in range(pad_left):
        new_matrix[new_left_feed_row][c] = 1

    # Right feed extension: solid metal from embedded right feed to new col N-1
    for c in range(pad_left + src_cols, target_cols):
        new_matrix[new_right_feed_row][c] = 1

    # Shift cap pairs by the embedding offset (distances unchanged, no filter needed)
    shifted_pairs = [
        [[ep[0] + pad_bottom, ep[1] + pad_left] for ep in pair]
        for pair in design.cap_pixel_pairs
    ]

    return DesignState(
        matrix                 = new_matrix,
        switch_states          = copy.deepcopy(design.switch_states),
        cap_values             = list(design.cap_values),
        inductor_values        = list(design.inductor_values),
        inductor_switch_states = copy.deepcopy(design.inductor_switch_states),
        cap_pixel_pairs        = shifted_pairs,
        cap_pair_values        = list(design.cap_pair_values),
    )


def zero_pad_centered_with_feed(
    design: DesignState,
    target_rows: int,
    target_cols: int,
) -> DesignState:
    """Convenience wrapper: embed with zero background (density=0)."""
    return embed_centered_with_density(design, target_rows, target_cols, 0.0, _random_module.Random())


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
    Physically expand *design* to (target_rows x target_cols).

    Two top-level modes are selected by weighted random draw:

      p_embed            — embed the seed at center; outside pixels filled with
                           density d ~ Uniform(0, 1) drawn fresh each call.
      p_random_insertion — insert rows/cols one at a time in shuffled order,
                           each using duplicate / random_fill / zero_pad sub-modes.

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

    p_embed = getattr(cfg, "finetune_expand_p_embed", 0.3)
    p_ins   = getattr(cfg, "finetune_expand_p_random_insertion", 0.7)
    total_top = p_embed + p_ins
    if rng.random() * total_top < p_embed:
        d_min = getattr(cfg, "finetune_expand_embed_density_min", 0.0)
        d_max = getattr(cfg, "finetune_expand_embed_density_max", 1.0)
        density = d_min + rng.random() * (d_max - d_min)
        return embed_centered_with_density(design, target_rows, target_cols, density, rng)

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
            left_feed_row, right_feed_row = get_feed_rows(mat)
            pre_insert_cols = current.cols
            if mode == "zero_pad":
                # Boundary only: insert just inside the left or right feed column
                col_pos = rng.choice([1, current.cols - 1])
            else:
                # Random interior position (never at feed columns 0 or cols-1)
                col_pos = rng.randint(1, current.cols - 1)
            new_mat = _insert_column(mat, col_pos, mode, rng)
            new_mat = _enforce_feed_adjacency(
                new_mat, col_pos, left_feed_row, right_feed_row, pre_insert_cols
            )
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

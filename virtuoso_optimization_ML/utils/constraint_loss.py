"""
constraint_loss.py — Differentiable soft constraint losses for flow matching training.

Core idea
---------
During flow matching training the model predicts a velocity field v_θ(x_t, t, c).
From this we can reconstruct a differentiable estimate of the target design image:

    x1_pred = x_t + (1 - t) * v_pred

This is exact at convergence and fully differentiable w.r.t. model parameters.
Soft constraint losses applied to x1_pred backpropagate through the velocity network,
pushing generated designs toward feasible regions without needing sentinel-labeled data.

Temporal weighting
------------------
At small t (near 0), x_t ≈ pure noise and x1_pred is unreliable.  The caller should
weight the returned loss by t.clamp(min=0.2) to suppress the constraint signal at
high-noise timesteps.

Image layout (as produced by DesignEncoder)
--------------------------------------------
Shape: (B, 1, rows, cols_total)  where  cols_total = cols + n_states
  - Columns 0 .. cols-1      : pixel matrix (float values in [0, 1])
  - Columns cols .. cols_total-1 : switch-state columns (one per filter state)

Pixel type boundaries (from design_encoder.py)
-----------------------------------------------
  [0,     0.125)  → type 0 : ground
  [0.125, 0.275)  → type 1 : metal
  [0.275, 0.550)  → type 2 : inductor
  [0.550, 1.0 ]   → type 3 : switch
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from ..encoding.design_encoder import _BOUND_01 as _B01, _BOUND_12 as _B12, _BOUND_23 as _B23


# ---------------------------------------------------------------------------
# Hardcoded hyperparameters
# ---------------------------------------------------------------------------

_TAU          = 0.02   # sigmoid temperature — controls soft/hard transition
_W_COUNT      = 1.0    # weight for count constraint loss
_W_SPACING    = 1.0    # weight for spacing constraint loss
_W_FEED       = 1.0    # weight for feed-column constraint loss
_W_STATE      = 0.5    # weight for switch-state distinctness loss


# ---------------------------------------------------------------------------
# Soft type probabilities
# ---------------------------------------------------------------------------

def soft_type_probs(v: Tensor, tau: float = _TAU) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Compute soft type-membership probabilities for each pixel value.

    Parameters
    ----------
    v   : any-shape float tensor of pixel values (nominally in [0, 1]).
    tau : sigmoid temperature.  Smaller → harder boundaries.

    Returns
    -------
    (p0, p1, p2, p3) — each same shape as v, values in (0, 1).

    Note: these are NOT a softmax partition.  Pixels in the gaps between
    regions get low probability for all types, which naturally penalises
    out-of-range values.
    """
    p0 = torch.sigmoid((_B01 - v) / tau)
    p1 = torch.sigmoid((v - _B01) / tau) * torch.sigmoid((_B12 - v) / tau)
    p2 = torch.sigmoid((v - _B12) / tau) * torch.sigmoid((_B23 - v) / tau)
    p3 = torch.sigmoid((v - _B23) / tau)
    return p0, p1, p2, p3


# ---------------------------------------------------------------------------
# Individual constraint losses
# ---------------------------------------------------------------------------

def count_loss(
    x1_pred:      Tensor,
    cols:         int,
    n_sw_min:     int,
    n_sw_max:     int,
    max_inductors: int,
) -> Tensor:
    """
    Penalise switch and inductor count violations.

    Switch count:   n_sw_min ≤ #switches ≤ n_sw_max
    Inductor count: #inductors ≤ max_inductors
    """
    matrix = x1_pred[:, 0, :, :cols]          # (B, rows, cols)
    rows_n, cols_n = matrix.shape[1], matrix.shape[2]
    norm = float(rows_n * cols_n) ** 2         # scale-invariant normalisation

    _, _, p2, p3 = soft_type_probs(matrix)

    soft_sw  = p3.sum(dim=[1, 2])              # (B,)
    soft_ind = p2.sum(dim=[1, 2])              # (B,)

    loss_sw  = (F.relu(n_sw_min - soft_sw) + F.relu(soft_sw - n_sw_max)).pow(2)
    loss_ind = F.relu(soft_ind - max_inductors).pow(2)

    return ((loss_sw + loss_ind) / norm).mean()


def spacing_loss(x1_pred: Tensor, cols: int) -> Tensor:
    """
    Penalise switches or inductors that are adjacent (8-connected) to another
    switch or inductor.

    Uses F.pad + slicing to implement each of the 8 directional shifts without
    circular boundary wrap-around.
    """
    matrix = x1_pred[:, 0, :, :cols]          # (B, rows, cols)
    _, _, p2, p3 = soft_type_probs(matrix)
    active = p2 + p3                           # (B, rows, cols) — prob of ind or sw

    rows_n, cols_n = matrix.shape[1], matrix.shape[2]
    norm = float(rows_n * cols_n)

    total = active.new_zeros(1)

    # All 8 (dr, dc) shifts; each direction only contributes one side of each pair
    shifts = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    for dr, dc in shifts:
        # Shift `active` by (dr, dc) using pad+slice so boundaries are zero (not wrapped)
        if dr == -1:
            # Neighbour is one row above: pad bottom, remove first row
            nbr = F.pad(active, (0, 0, 0, 1))[:, 1:, :]
        elif dr == 1:
            nbr = F.pad(active, (0, 0, 1, 0))[:, :-1, :]
        else:
            nbr = active

        if dc == -1:
            nbr = F.pad(nbr, (0, 1, 0, 0))[:, :, 1:]
        elif dc == 1:
            nbr = F.pad(nbr, (1, 0, 0, 0))[:, :, :-1]

        # Penalise: probability of being ind/sw × probability of neighbour being ind/sw
        total = total + (active * nbr).sum()

    # Each undirected pair counted twice (once for each direction) → divide by 2
    return (total / 2.0 / norm / x1_pred.shape[0]).squeeze()


def feed_column_loss(x1_pred: Tensor, cols: int) -> Tensor:
    """
    Penalise feed columns (col 0 and col cols-1) having ≠ 1 active (non-ground) pixel.

    The decoder enforces exactly one non-zero pixel in each feed column; this loss
    teaches the generative model to produce images that already satisfy this before
    decoding, reducing decode failures.
    """
    matrix = x1_pred[:, 0, :, :cols]          # (B, rows, cols)
    p0, _, _, _ = soft_type_probs(matrix)

    loss = x1_pred.new_zeros(1)
    for feed_col in (0, cols - 1):
        p_active = 1.0 - p0[:, :, feed_col]   # (B, rows)
        # Penalise: sum of active probabilities should equal exactly 1
        loss = loss + (p_active.sum(dim=1) - 1.0).pow(2).mean()

    return (loss / 2.0).squeeze()             # average over the two feed columns


def switch_state_loss(x1_pred: Tensor, cols: int, n_states: int) -> Tensor:
    """
    Penalise switch-state columns that are too similar (non-distinct states).

    State columns are the last n_states columns of the image.  They contain
    continuous values in [0, 1], thresholded at 0.5 by the hard decoder.
    We penalise pairwise cosine similarity between state vectors.
    """
    if n_states < 2:
        return x1_pred.new_zeros(1).squeeze()

    state_cols = x1_pred[:, 0, :, cols:]      # (B, rows, n_states)

    loss = x1_pred.new_zeros(1)
    n_pairs = 0
    for s1 in range(n_states):
        for s2 in range(s1 + 1, n_states):
            v1 = state_cols[:, :, s1]          # (B, rows)
            v2 = state_cols[:, :, s2]
            dot  = (v1 * v2).sum(dim=1)        # (B,)
            norm = v1.norm(dim=1) * v2.norm(dim=1) + 1e-8
            sim  = dot / norm                  # cosine similarity in (-1, 1)
            loss = loss + sim.pow(2).mean()
            n_pairs += 1

    return (loss / n_pairs).squeeze()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

def combined_constraint_loss(
    x1_pred:       Tensor,
    cols:          int,
    n_states:      int,
    n_sw_min:      int,
    n_sw_max:      int,
    max_inductors: int,
) -> tuple[Tensor, dict]:
    """
    Compute the combined differentiable constraint loss.

    Parameters
    ----------
    x1_pred       : (B, 1, rows, cols + n_states) predicted design image.
    cols          : number of pixel-matrix columns (excludes switch-state columns).
    n_states      : number of filter states (= number of switch-state columns).
    n_sw_min/max  : switch count range constraint.
    max_inductors : maximum inductor count constraint.

    Returns
    -------
    (total_loss, breakdown_dict)
        total_loss    : scalar Tensor, weighted sum of all sub-losses.
        breakdown_dict: {"count", "spacing", "feed", "state"} for logging.
    """
    l_count   = count_loss(x1_pred, cols, n_sw_min, n_sw_max, max_inductors)
    l_spacing = spacing_loss(x1_pred, cols)
    l_feed    = feed_column_loss(x1_pred, cols)
    l_state   = switch_state_loss(x1_pred, cols, n_states)

    total = (
        _W_COUNT   * l_count
        + _W_SPACING * l_spacing
        + _W_FEED    * l_feed
        + _W_STATE   * l_state
    )

    breakdown = {
        "count":   l_count.item(),
        "spacing": l_spacing.item(),
        "feed":    l_feed.item(),
        "state":   l_state.item(),
    }
    return total, breakdown

"""
cost_utils.py — Tensor-based cost function for ML optimization.

Mirrors the asymmetric penalty logic of the SA cost function but operates
directly on S-param tensors (no file I/O), making it suitable for scoring
world-model predictions during online training.

Penalty rules (same convention as SA):
  passband  : max(0, passband_min_db  - S21(f))  averaged over passband points
  stopband  : max(0, S21(f)  - stopband_max_db)  averaged over stopband points
  cross-stop: max(0, S21(f)  - cross_stopband_max_db) * cross_weight
              averaged over cross-stopband points

All penalties are summed over states and scaled by state weight.
"""

from __future__ import annotations

from typing import List, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from virtuoso_optimization_SA.cost import FilterStateSpec


def compute_cost_from_tensor(
    s_params: torch.Tensor,
    freq_points: np.ndarray,
    specs: "List[FilterStateSpec]",
) -> float:
    """
    Compute the ML proxy cost for a design given its predicted S-param tensor.

    Parameters
    ----------
    s_params    : (n_states, n_freq) float tensor — S21 in dB per state.
    freq_points : (n_freq,) numpy array of frequencies in Hz.
    specs       : list of FilterStateSpec objects (one per state).

    Returns
    -------
    float — total cost (lower is better; 0.0 means fully within spec).
    """
    s_np  = s_params.detach().cpu().numpy()   # (n_states, n_freq)
    freq  = freq_points
    total = 0.0

    spec_by_idx = {s.state_index: s for s in specs}

    for si in range(s_np.shape[0]):
        spec = spec_by_idx.get(si)
        if spec is None:
            continue
        row = s_np[si]   # (n_freq,)
        state_cost = 0.0
        n_pts = 0

        # Passband: penalise when S21 is below the minimum acceptable level
        pb_lo, pb_hi = spec.passband
        pb_mask = (freq >= pb_lo) & (freq <= pb_hi)
        if pb_mask.any():
            shortfall = np.maximum(0.0, spec.passband_min_db - row[pb_mask])
            state_cost += 5.0 * shortfall.sum()   # 5× weight matches SA convention
            n_pts += pb_mask.sum()

        # Stopband: penalise when S21 exceeds the maximum acceptable level
        for sb_lo, sb_hi in spec.stopband_regions:
            sb_mask = (freq >= sb_lo) & (freq <= sb_hi)
            if sb_mask.any():
                excess = np.maximum(0.0, row[sb_mask] - spec.stopband_max_db)
                state_cost += excess.sum()
                n_pts += sb_mask.sum()

        # Cross-state decoupling penalty
        for cs_lo, cs_hi in spec.cross_stopband_regions:
            cs_mask = (freq >= cs_lo) & (freq <= cs_hi)
            if cs_mask.any():
                excess = np.maximum(0.0, row[cs_mask] - spec.cross_stopband_max_db)
                state_cost += (excess * spec.cross_stopband_weight).sum()
                n_pts += cs_mask.sum()

        mean_cost = state_cost / n_pts if n_pts > 0 else 0.0
        total += mean_cost * spec.weight

    return float(total)


def compute_score_tensor(
    s_params: torch.Tensor,
    freq_points: np.ndarray,
    specs: "List[FilterStateSpec]",
) -> torch.Tensor:
    """
    Decompose the per-state cost into four interpretable score components.

    Parameters
    ----------
    s_params    : (n_states, n_freq) float tensor — S21 in dB per state.
    freq_points : (n_freq,) numpy array of frequencies in Hz.
    specs       : list of FilterStateSpec objects (one per state).

    Returns
    -------
    (n_states, 4) float32 tensor where the four components per state are:
      [0] lower_reject  — mean excess penalty over stopband_regions[0]
      [1] passband      — mean shortfall penalty over the passband (5× weight)
      [2] higher_reject — mean excess penalty over stopband_regions[1:]
      [3] cross_band    — mean excess penalty over all cross_stopband_regions

    All values are >= 0 (0.0 means fully within spec for that region).
    """
    s_np  = s_params.detach().cpu().numpy()   # (n_states, n_freq)
    freq  = freq_points
    n_states = s_np.shape[0]
    out   = np.zeros((n_states, 4), dtype=np.float32)

    spec_by_idx = {s.state_index: s for s in specs}

    for si in range(n_states):
        spec = spec_by_idx.get(si)
        if spec is None:
            continue
        row = s_np[si]   # (n_freq,)

        # --- [1] Passband shortfall ---
        pb_lo, pb_hi = spec.passband
        pb_mask = (freq >= pb_lo) & (freq <= pb_hi)
        if pb_mask.any():
            shortfall = np.maximum(0.0, spec.passband_min_db - row[pb_mask])
            out[si, 1] = float(5.0 * shortfall.mean())

        # --- [0] Lower reject: first stopband region ---
        if spec.stopband_regions:
            sb_lo, sb_hi = spec.stopband_regions[0]
            sb_mask = (freq >= sb_lo) & (freq <= sb_hi)
            if sb_mask.any():
                excess = np.maximum(0.0, row[sb_mask] - spec.stopband_max_db)
                out[si, 0] = float(excess.mean())

        # --- [2] Higher reject: remaining stopband regions ---
        if len(spec.stopband_regions) > 1:
            vals = []
            for sb_lo, sb_hi in spec.stopband_regions[1:]:
                sb_mask = (freq >= sb_lo) & (freq <= sb_hi)
                if sb_mask.any():
                    excess = np.maximum(0.0, row[sb_mask] - spec.stopband_max_db)
                    vals.append(excess.mean())
            if vals:
                out[si, 2] = float(np.mean(vals))

        # --- [3] Cross-band: all cross_stopband_regions ---
        if spec.cross_stopband_regions:
            vals = []
            for cs_lo, cs_hi in spec.cross_stopband_regions:
                cs_mask = (freq >= cs_lo) & (freq <= cs_hi)
                if cs_mask.any():
                    excess = np.maximum(0.0, row[cs_mask] - spec.cross_stopband_max_db)
                    vals.append((excess * spec.cross_stopband_weight).mean())
            if vals:
                out[si, 3] = float(np.mean(vals))

    return torch.tensor(out, dtype=torch.float32)

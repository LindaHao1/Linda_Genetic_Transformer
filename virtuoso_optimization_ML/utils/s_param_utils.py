"""
s_param_utils.py — S-parameter loading, interpolation, and target curve helpers.

All ML modules use a canonical fixed frequency grid (defined in MLConfig) so
that tensors have consistent shapes.  Touchstone files from the simulator may
use a different grid, so we interpolate S21 onto the canonical grid here.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

import numpy as np

try:
    import skrf as rf
except ImportError as exc:
    raise ImportError(
        "scikit-rf is required.  Install with:  pip install scikit-rf"
    ) from exc

try:
    import torch
except ImportError as exc:
    raise ImportError(
        "PyTorch is required.  Install with:  pip install torch"
    ) from exc

if TYPE_CHECKING:
    from virtuoso_engine.engine import RunResult
    from virtuoso_optimization_SA.cost import FilterStateSpec


# ---------------------------------------------------------------------------
# Single-file loader
# ---------------------------------------------------------------------------

def load_and_interpolate_s21(
    s2p_path: str | Path,
    freq_points: np.ndarray,
) -> torch.Tensor:
    """
    Load a 2-port Touchstone file and return S21 in dB interpolated onto
    the canonical frequency grid.

    Parameters
    ----------
    s2p_path   : path to a .s2p file produced by Spectre.
    freq_points: 1-D numpy array of target frequencies in Hz, shape (n_freq,).

    Returns
    -------
    Tensor of shape (n_freq,), float32, S21 in dB.
    """
    ntwk  = rf.Network(str(s2p_path))
    f_sim = ntwk.f                  # (N,)
    s21   = ntwk.s_db[:, 1, 0]     # (N,)

    # Interpolate onto canonical grid; clamp to edge values outside range
    s21_interp = np.interp(freq_points, f_sim, s21)
    return torch.tensor(s21_interp, dtype=torch.float32)


# ---------------------------------------------------------------------------
# RunResult → stacked tensor
# ---------------------------------------------------------------------------

def result_to_s_param_tensor(
    result: "RunResult",
    freq_points: np.ndarray,
    n_states: int,
) -> Optional[torch.Tensor]:
    """
    Convert a RunResult's s_param_files list into a (n_states, n_freq) tensor.

    Returns None if the result failed or any state file is missing.
    """
    if not result.success:
        return None

    files = result.s_param_files          # list[Path | None], one per state
    if files is None or len(files) < n_states:
        return None

    rows: List[torch.Tensor] = []
    for i in range(n_states):
        p = files[i]
        if p is None or not Path(p).exists():
            return None
        try:
            rows.append(load_and_interpolate_s21(p, freq_points))
        except Exception:  # noqa: BLE001
            return None

    return torch.stack(rows, dim=0)       # (n_states, n_freq)


# ---------------------------------------------------------------------------
# Target curve builder
# ---------------------------------------------------------------------------

def build_target_s_param_tensor(
    filter_specs: List["FilterStateSpec"],
    freq_points: np.ndarray,
) -> torch.Tensor:
    """
    Build an ideal target S21 curve from FilterStateSpec objects.

    The target value at each frequency point is:
      - passband_min_db  inside the passband
      - stopband_max_db  inside each stopband region
      - linearly interpolated in transition bands

    The result is the conditioning input fed to the generative model.

    Parameters
    ----------
    filter_specs : list of FilterStateSpec, one per state (in state_index order).
    freq_points  : canonical grid, shape (n_freq,).

    Returns
    -------
    Tensor of shape (n_states, n_freq), float32.
    """
    n_states = len(filter_specs)
    n_freq   = len(freq_points)
    target   = np.zeros((n_states, n_freq), dtype=np.float32)

    spec_by_idx = {s.state_index: s for s in filter_specs}

    for state_idx in range(n_states):
        spec = spec_by_idx.get(state_idx)
        if spec is None:
            continue

        row = np.full(n_freq, spec.stopband_max_db, dtype=np.float32)

        pb_lo, pb_hi = spec.passband
        for j, f in enumerate(freq_points):
            if pb_lo <= f <= pb_hi:
                row[j] = spec.passband_min_db

        target[state_idx] = row

    return torch.tensor(target, dtype=torch.float32)

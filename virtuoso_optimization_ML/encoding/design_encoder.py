"""
design_encoder.py — Encode a DesignState into a single-channel float image
and decode it back.

Image layout
------------
Shape: (1, rows, cols + n_states)

The first `cols` columns contain the pixel matrix encoded into [0, 1]:

  Pixel type 0 (ground)   → region [0.00, 0.10],  center 0.05  (constant)
  Pixel type 1 (metal)    → region [0.15, 0.25],  center 0.20  (constant)
  Pixel type 2 (inductor) → region [0.30, 0.50],  value interpolated over
                             [0.30, 0.50] by discrete inductor_index / (n-1)
  Pixel type 3 (switch)   → region [0.60, 0.90],  value interpolated over
                             [0.60, 0.90] by (cap - cap_min) / (cap_max - cap_min)

Region boundaries for decoding are determined by manually defined dividing
points (midpoints of the gaps between adjacent regions):
  _BOUND_01 = 0.125  — splits ground  / metal
  _BOUND_12 = 0.275  — splits metal   / inductor
  _BOUND_23 = 0.550  — splits inductor / switch

Once the region is identified, the raw value is clipped to [R_LO, R_HI]
before recovering the inductor index or cap value.  This gives region-specific
snapping rather than nearest-centre ambiguity.

The last `n_states` columns encode switch states using switch-index order:
  Switches are sorted in row-first scan order (same order as cap_values /
  switch_states).  For state s, row i of column (cols + s) holds:
    1.0  if switch i is ON  in state s
    0.0  if switch i is OFF in state s
  Rows beyond the number of switches present are left at 0.0.

  This means the state columns always use the first n_switches rows as a
  compact lookup table — independent of where each switch sits spatially —
  so multiple switches in the same spatial row are handled correctly.

The decoder identifies state columns by position (last n_states cols), not by
value, so there is no ambiguity with pixel type 1 (metal, value 0.20).

Inductor / switch positions are tracked in row-first scan order, matching
the order used by DesignState.inductor_values and cap_values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Region boundaries (all in [0, 1])
# ---------------------------------------------------------------------------

_R0_LO, _R0_HI = 0.00, 0.10  # ground
_R1_LO, _R1_HI = 0.15, 0.25  # metal
_R2_LO, _R2_HI = 0.30, 0.50  # inductor
_R3_LO, _R3_HI = 0.60, 0.90  # switch

_R0_CTR = 0.05
_R1_CTR = 0.20
_R2_CTR = 0.40   # mid of inductor range (for unknown/fallback)
_R3_CTR = 0.75   # mid of switch range

# Manually defined dividing points between adjacent regions.
# A float value is classified into the region whose interval it falls in,
# removing the nearest-centre ambiguity that occurs near region gaps.
_BOUND_01 = (_R0_HI + _R1_LO) / 2   # 0.125  — ground  / metal
_BOUND_12 = (_R1_HI + _R2_LO) / 2   # 0.275  — metal   / inductor
_BOUND_23 = (_R2_HI + _R3_LO) / 2   # 0.550  — inductor / switch


# ---------------------------------------------------------------------------
# Encoder / Decoder class
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    rows:              int
    cols:              int
    n_states:          int
    cap_min:           float
    cap_max:           float
    n_inductor_values: int   # number of discrete inductor indices (e.g. 4)
    n_switches:        int   # expected number of switch pixels
    max_inductors:     int   # max expected inductor pixels


class DesignEncoder:
    """
    Bidirectional mapping between DesignState and a float image tensor.

    Parameters
    ----------
    cfg : EncoderConfig
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Encode
    # ------------------------------------------------------------------

    def encode(self, design) -> torch.Tensor:
        """
        Convert a DesignState to a float32 tensor of shape
        (1, rows, cols + n_states).

        Parameters
        ----------
        design : DesignState

        Returns
        -------
        torch.Tensor, float32, shape (1, rows, cols + n_states)
        """
        cfg    = self.cfg
        rows   = cfg.rows
        cols   = cfg.cols
        n_s    = cfg.n_states

        img = np.zeros((rows, cols + n_s), dtype=np.float32)

        # --- Collect inductor/switch positions in row-first order ---
        ind_positions: List[tuple] = []
        sw_positions:  List[tuple] = []
        for r in range(rows):
            for c in range(cols):
                v = design.matrix[r][c]
                if v == 2:
                    ind_positions.append((r, c))
                elif v == 3:
                    sw_positions.append((r, c))

        ind_idx_map = {pos: i for i, pos in enumerate(ind_positions)}
        sw_idx_map  = {pos: i for i, pos in enumerate(sw_positions)}

        # --- Fill pixel matrix columns ---
        for r in range(rows):
            for c in range(cols):
                v = design.matrix[r][c]
                img[r, c] = self._pixel_to_float(
                    ptype    = v,
                    ind_idx  = ind_idx_map.get((r, c), 0),
                    sw_idx   = sw_idx_map.get((r, c), 0),
                    cap_values      = design.cap_values,
                    inductor_values = design.inductor_values,
                )

        # --- Fill switch-state columns ---
        # Row i of each state column encodes switch i's ON/OFF state.
        # This is independent of the switch's spatial position, so multiple
        # switches in the same grid row are handled correctly.
        for s in range(n_s):
            col_idx = cols + s
            state_row = design.switch_states[s]  # length n_switches, 0/1
            for sw_i in range(len(state_row)):
                img[sw_i, col_idx] = float(state_row[sw_i])

        return torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # (1, H, W)

    def _pixel_to_float(
        self,
        ptype: int,
        ind_idx: int,
        sw_idx: int,
        cap_values: List[float],
        inductor_values: List[int],
    ) -> float:
        cfg = self.cfg
        if ptype == 0:
            return _R0_CTR
        elif ptype == 1:
            return _R1_CTR
        elif ptype == 2:
            # Discrete inductor index → linear mapping within [R2_LO, R2_HI]
            n = max(cfg.n_inductor_values - 1, 1)
            idx = inductor_values[ind_idx] if ind_idx < len(inductor_values) else 0
            idx = max(0, min(idx, cfg.n_inductor_values - 1))
            return _R2_LO + (idx / n) * (_R2_HI - _R2_LO)
        else:  # ptype == 3
            cap_min = cfg.cap_min
            cap_max = cfg.cap_max
            cap = cap_values[sw_idx] if sw_idx < len(cap_values) else cap_min
            cap = max(cap_min, min(cap, cap_max))
            t = (cap - cap_min) / (cap_max - cap_min) if cap_max > cap_min else 0.0
            return _R3_LO + t * (_R3_HI - _R3_LO)

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def decode(self, image: torch.Tensor):
        """
        Convert a float32 tensor of shape (1, rows, cols + n_states) back to
        a DesignState.  The result may or may not pass is_feasible() — the
        caller is responsible for validation.

        Parameters
        ----------
        image : torch.Tensor, shape (1, rows, cols + n_states)  OR
                              shape (rows, cols + n_states)

        Returns
        -------
        DesignState
        """
        from ..design import DesignState  # deferred

        cfg  = self.cfg
        rows = cfg.rows
        cols = cfg.cols
        n_s  = cfg.n_states

        arr = image.squeeze(0).detach().cpu().numpy()  # (rows, cols + n_states)

        # Clip to [0, 1]
        arr = np.clip(arr, 0.0, 1.0)

        # --- Decode pixel matrix ---
        matrix: List[List[int]] = []
        cap_values_list: List[float] = []
        inductor_values_list: List[int] = []

        # First pass: determine pixel types
        ptype_grid = np.zeros((rows, cols), dtype=int)
        for r in range(rows):
            for c in range(cols):
                ptype_grid[r, c] = self._float_to_ptype(arr[r, c])

        # Enforce feed column constraint: col 0 and col N-1 must have exactly
        # one non-zero pixel (type 1 / metal).  Keep the row with the highest
        # raw value; set all others to 0 (ground).
        for feed_col in (0, cols - 1):
            best_row = int(np.argmax(arr[:, feed_col]))
            for r in range(rows):
                ptype_grid[r, feed_col] = 1 if r == best_row else 0

        # Second pass: build output lists in row-first order
        ind_count = 0
        sw_count  = 0
        sw_positions: List[tuple] = []

        for r in range(rows):
            row_out: List[int] = []
            for c in range(cols):
                pt = int(ptype_grid[r, c])
                row_out.append(pt)
                if pt == 2:
                    # Clip to inductor region then recover discrete index
                    val = float(np.clip(arr[r, c], _R2_LO, _R2_HI))
                    n   = max(cfg.n_inductor_values - 1, 1)
                    t   = (val - _R2_LO) / (_R2_HI - _R2_LO)
                    idx = int(round(t * n))
                    idx = max(0, min(idx, cfg.n_inductor_values - 1))
                    inductor_values_list.append(idx)
                    ind_count += 1
                elif pt == 3:
                    # Clip to switch region then recover continuous cap value
                    val = float(np.clip(arr[r, c], _R3_LO, _R3_HI))
                    t   = (val - _R3_LO) / (_R3_HI - _R3_LO)
                    cap = cfg.cap_min + t * (cfg.cap_max - cfg.cap_min)
                    cap_values_list.append(cap)
                    sw_positions.append((r, c))
                    sw_count += 1
            matrix.append(row_out)

        # --- Decode switch states from appended columns ---
        # Row i of each state column holds switch i's ON/OFF value.
        switch_states: List[List[int]] = []
        for s in range(n_s):
            col_idx = cols + s
            state_row: List[int] = []
            for sw_i in range(sw_count):
                val = arr[sw_i, col_idx]
                state_row.append(1 if val >= 0.5 else 0)
            switch_states.append(state_row)

        return DesignState(
            matrix          = matrix,
            switch_states   = switch_states,
            cap_values      = cap_values_list,
            inductor_values = inductor_values_list,
        )

    def _float_to_ptype(self, val: float) -> int:
        """Classify a float in [0,1] to a pixel type using explicit region dividers."""
        if val < _BOUND_01:
            return 0
        elif val < _BOUND_12:
            return 1
        elif val < _BOUND_23:
            return 2
        else:
            return 3

    # ------------------------------------------------------------------
    # Image dimensions helper
    # ------------------------------------------------------------------

    @property
    def img_h(self) -> int:
        return self.cfg.rows

    @property
    def img_w(self) -> int:
        return self.cfg.cols + self.cfg.n_states

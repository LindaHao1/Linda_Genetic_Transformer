"""
design_encoder.py — Encode a DesignState into a single-channel float image
and decode it back.

Image layout
------------
Shape: (1, rows, cols + n_states)

The first `cols` columns contain the pixel matrix encoded into [0, 1]:

  Pixel type 0 (ground)   → region [0.00, 0.15],  center 0.05  (constant)
  Pixel type 1 (metal)    → region [0.20, 0.35],  center 0.25  (constant)
  Pixel type 2 (inductor) → region [0.45, 0.60],  value interpolated over
                             [0.45, 0.60] by discrete inductor_index / (n-1)
  Pixel type 3 (switch)   → region [0.70, 0.85],  value interpolated over
                             [0.70, 0.85] by (cap - cap_min) / (cap_max - cap_min)

The 0.05 gaps between regions prevent smooth transitions and make the four
types clearly separable in the pixel value histogram.

The last `n_states` columns encode switch states:
  column (cols + s) = 1.0 at the row of each switch that is ON in state s,
                      0.0 elsewhere.
The decoder identifies state columns by position (last n_states cols), not by
value, so there is no ambiguity with pixel type 1 (metal, value 0.25).

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

_R0_LO, _R0_HI = 0.00, 0.15   # ground
_R1_LO, _R1_HI = 0.20, 0.35   # metal
_R2_LO, _R2_HI = 0.45, 0.60   # inductor
_R3_LO, _R3_HI = 0.70, 0.85   # switch

_R0_CTR = 0.05
_R1_CTR = 0.275
_R2_CTR = 0.525   # mid of inductor range (for unknown/fallback)
_R3_CTR = 0.775   # mid of switch range

# Snap threshold: value → nearest region centre
_REGION_CENTERS = np.array([_R0_CTR, _R1_CTR, _R2_CTR, _R3_CTR])


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
        for s in range(n_s):
            col_idx = cols + s
            state_row = design.switch_states[s]  # length n_switches, 0/1
            for sw_pos, sw_i in sw_idx_map.items():
                r_sw = sw_pos[0]
                img[r_sw, col_idx] = float(state_row[sw_i])

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
        from virtuoso_optimization_SA.design import DesignState  # deferred

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

        # Second pass: build output lists in row-first order
        ind_count = 0
        sw_count  = 0
        sw_positions: List[tuple] = []

        for r in range(rows):
            row_out: List[int] = []
            for c in range(cols):
                pt = ptype_grid[r, c]
                row_out.append(pt)
                if pt == 2:
                    # Recover discrete inductor index from float value
                    val = arr[r, c]
                    n   = max(cfg.n_inductor_values - 1, 1)
                    t   = (val - _R2_LO) / (_R2_HI - _R2_LO + 1e-9)
                    idx = int(round(t * n))
                    idx = max(0, min(idx, cfg.n_inductor_values - 1))
                    inductor_values_list.append(idx)
                    ind_count += 1
                elif pt == 3:
                    # Recover continuous cap value from float value
                    val = arr[r, c]
                    t   = (val - _R3_LO) / (_R3_HI - _R3_LO + 1e-9)
                    t   = max(0.0, min(t, 1.0))
                    cap = cfg.cap_min + t * (cfg.cap_max - cfg.cap_min)
                    cap_values_list.append(cap)
                    sw_positions.append((r, c))
                    sw_count += 1
            matrix.append(row_out)

        # --- Decode switch states from appended columns ---
        n_switches_found = sw_count
        switch_states: List[List[int]] = []
        for s in range(n_s):
            col_idx = cols + s
            state_row: List[int] = []
            for sw_pos in sw_positions:
                r_sw = sw_pos[0]
                val  = arr[r_sw, col_idx]
                state_row.append(1 if val >= 0.5 else 0)
            switch_states.append(state_row)

        # Pad cap_values / inductor_values if shorter than expected
        # (can happen if decoded design has fewer switches/inductors than config)
        while len(cap_values_list) < cfg.n_switches:
            cap_values_list.append(cfg.cap_min)
        while len(switch_states) > 0 and len(switch_states[0]) < cfg.n_switches:
            for s in range(n_s):
                switch_states[s].append(0)

        return DesignState(
            matrix          = matrix,
            switch_states   = switch_states,
            cap_values      = cap_values_list[:cfg.n_switches],
            inductor_values = inductor_values_list,
        )

    def _float_to_ptype(self, val: float) -> int:
        """Snap a float in [0,1] to the nearest pixel type region."""
        dists = np.abs(_REGION_CENTERS - val)
        return int(np.argmin(dists))

    # ------------------------------------------------------------------
    # Image dimensions helper
    # ------------------------------------------------------------------

    @property
    def img_h(self) -> int:
        return self.cfg.rows

    @property
    def img_w(self) -> int:
        return self.cfg.cols + self.cfg.n_states

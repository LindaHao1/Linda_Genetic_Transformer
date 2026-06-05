"""
design.py — DesignState: the complete description of one switchable-filter candidate.

A design is fully described by three things:
  matrix               (rows x cols) of int  — pixel types 0/1/2/3
  switch_states        (n_states x n_switches x n_caps_in_bank) of int — 0=OFF / 1=ON per cap
  cap_values           (n_switches x n_caps_in_bank) of float — capacitance per bank cap (F)
  inductor_switch_states (n_states x n_inductors) of int — 0=OFF (open) / 1=ON per state
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DesignState:
    """
    Immutable-ish container for one switchable-filter candidate.

    Attributes
    ----------
    matrix : 2-D list of int
        Pixel grid.  Values: 0=ground, 1=metal, 2=inductor, 3=switch.
        Row 0 is the bottom of the layout; col 0 / col N-1 are the feed columns.
    switch_states : list of list of list of int
        Outer length = n_states, middle length = n_switches, inner length = n_caps_in_bank.
        1 = cap ON (switch closed), 0 = cap OFF (switch open) for that state.
    cap_values : list of list of float
        Capacitor bank values (F) per switch pixel, in row-first order.
        Outer length = n_switches, inner length = n_caps_in_bank.
    inductor_values : list of int
        Index into the global ``inductor_translation_table`` for each inductor
        pixel (type 2), in row-first order.  Length == n_inductors.
        Empty list means all inductors use the engine's ``inductor_s1p`` fallback.
    """

    matrix:                List[List[int]]
    switch_states:         List[List[List[int]]]   # [n_states][n_switches][n_caps_in_bank]
    cap_values:            List[List[float]]       # [n_switches][n_caps_in_bank]
    inductor_values:       List[int] = field(default_factory=list)  # [n_inductors]
    inductor_switch_states: List[List[int]] = field(default_factory=list)
    # [n_states][n_inductors] — 1=ON (connect s1p nport), 0=OFF (open circuit, 1 TΩ to GND).
    # Empty outer list means all inductors are ON in all states (backward-compatible default).
    cap_pixel_pairs: List[List[List[int]]] = field(default_factory=list)  # [[[r1,c1],[r2,c2]], ...]
    cap_pair_values: List[float] = field(default_factory=list)            # [n_cap_pairs]
    inductor_cap_values: List[float] = field(default_factory=list)        # [n_inductors] series MIM cap per inductor
    pixel_sizes: List[List[float]] = field(default_factory=list)          # [rows][cols] per-pixel metal patch size (um)
    # Empty list  → uniform mode: engine uses scalar layout.pixel_size, connectivity uses
    #               pure grid adjacency (backward-compatible default).
    # Populated   → per-pixel sizes (same shape as ``matrix``).  The value at ground (type 0)
    #               and feed-column cells is ignored (kept for shape consistency).

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def rows(self) -> int:
        return len(self.matrix)

    @property
    def cols(self) -> int:
        return len(self.matrix[0]) if self.matrix else 0

    @property
    def n_switches(self) -> int:
        """Count switch pixels (type 3) in the matrix."""
        return sum(v == 3 for row in self.matrix for v in row)

    @property
    def n_inductors(self) -> int:
        """Count inductor pixels (type 2) in the matrix."""
        return sum(v == 2 for row in self.matrix for v in row)

    @property
    def n_vias(self) -> int:
        """Count via-ground pixels (type 4) in the matrix."""
        return sum(v == 4 for row in self.matrix for v in row)

    @property
    def n_states(self) -> int:
        return len(self.switch_states)

    @property
    def n_cap_pairs(self) -> int:
        return len(self.cap_pixel_pairs)

    # ------------------------------------------------------------------
    # Copy helper
    # ------------------------------------------------------------------

    def copy(self) -> "DesignState":
        return DesignState(
            matrix                 = copy.deepcopy(self.matrix),
            switch_states          = copy.deepcopy(self.switch_states),
            cap_values             = copy.deepcopy(self.cap_values),
            inductor_values        = list(self.inductor_values),
            inductor_switch_states = copy.deepcopy(self.inductor_switch_states),
            cap_pixel_pairs        = copy.deepcopy(self.cap_pixel_pairs),
            cap_pair_values        = list(self.cap_pair_values),
            inductor_cap_values    = list(self.inductor_cap_values),
            pixel_sizes            = copy.deepcopy(self.pixel_sizes),
        )

    # ------------------------------------------------------------------
    # Serialisation helpers (for config override building)
    # ------------------------------------------------------------------

    def to_pixelated_config(self) -> dict:
        """
        Return the ``pixelated`` sub-dict understood by VirtuosoPixelEngine.

        switch_states encodes which switches are ON for each state.
        Each row in state_list corresponds to one filter state.
        inductor_values holds the translation-table index per inductor pixel.
        """
        cfg: dict = {
            "pixel_state_matrix": self.matrix,
            "state_list":         self.switch_states,
            "cap_values":         self.cap_values,
        }
        cfg["inductor_values"] = self.inductor_values  # always emit, even [] — prevents base config bleed-through via deep_merge
        if self.inductor_switch_states:
            cfg["inductor_switch_states"] = self.inductor_switch_states
        if self.inductor_cap_values:
            cfg["inductor_cap_values"] = self.inductor_cap_values
        if self.cap_pixel_pairs:
            cfg["cap_pixel_pairs"] = self.cap_pixel_pairs
            cfg["cap_pair_values"] = self.cap_pair_values
        if self.pixel_sizes:
            cfg["pixel_sizes"] = self.pixel_sizes  # only emit when sizing enabled — keeps uniform path untouched
        return cfg

    def __repr__(self) -> str:
        rows_str = "\n  ".join(str(row) for row in self.matrix)
        return (
            f"DesignState(rows={self.rows}, cols={self.cols}, "
            f"n_sw={self.n_switches}, n_ind={self.n_inductors}, "
            f"n_pairs={self.n_cap_pairs}, n_states={self.n_states})\n"
            f"  Matrix:\n  {rows_str}\n"
            f"  SwitchStates:         {self.switch_states}\n"
            f"  CapValues:            {self.cap_values}\n"
            f"  InductorValues:       {self.inductor_values}\n"
            f"  InductorSwitchStates: {self.inductor_switch_states}\n"
            f"  InductorCapValues:    {self.inductor_cap_values}\n"
            f"  CapPixelPairs:        {self.cap_pixel_pairs}\n"
            f"  CapPairValues:        {self.cap_pair_values}\n"
            f"  PixelSizes(set):      {bool(self.pixel_sizes)}"
        )

"""
design.py — DesignState: the complete description of one switchable-filter candidate.

A design is fully described by three things:
  matrix        (rows x cols) of int  — pixel types 0/1/2/3
  switch_states (n_states x n_switches) of int — 0=OFF / 1=ON per state
  cap_values    (n_switches,) of float — series coupling capacitance per switch (F)
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
    switch_states : list of list of int
        Outer length = n_states, inner length = n_switches.
        1 = switch ON, 0 = switch OFF for that state.
    cap_values : list of float
        Series capacitance (F) for each switch pixel, in row-first order.
    inductor_values : list of int
        Index into the global ``inductor_translation_table`` for each inductor
        pixel (type 2), in row-first order.  Length == n_inductors.
        Empty list means all inductors use the engine's ``inductor_s1p`` fallback.
    """

    matrix:          List[List[int]]
    switch_states:   List[List[int]]   # [n_states][n_switches]
    cap_values:      List[float]       # [n_switches]
    inductor_values: List[int] = field(default_factory=list)  # [n_inductors]

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
    def n_states(self) -> int:
        return len(self.switch_states)

    # ------------------------------------------------------------------
    # Copy helper
    # ------------------------------------------------------------------

    def copy(self) -> "DesignState":
        return DesignState(
            matrix          = copy.deepcopy(self.matrix),
            switch_states   = copy.deepcopy(self.switch_states),
            cap_values      = list(self.cap_values),
            inductor_values = list(self.inductor_values),
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
        if self.inductor_values:
            cfg["inductor_values"] = self.inductor_values
        return cfg

    def __repr__(self) -> str:
        rows_str = "\n  ".join(str(row) for row in self.matrix)
        return (
            f"DesignState(rows={self.rows}, cols={self.cols}, "
            f"n_sw={self.n_switches}, n_ind={self.n_inductors}, "
            f"n_states={self.n_states})\n"
            f"  Matrix:\n  {rows_str}\n"
            f"  SwitchStates:    {self.switch_states}\n"
            f"  CapValues:       {self.cap_values}\n"
            f"  InductorValues:  {self.inductor_values}"
        )

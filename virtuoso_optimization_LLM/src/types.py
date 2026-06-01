"""
Core data structures.

`Design` is the canonical in-memory description of one switchable filter
candidate.  `Design.to_record()` emits exactly the dict that
`world_model_test/encode.py:encode_sample` consumes, so any Design produced by
the tokenizer round-trips through the surrogate without adaptation.

Pixel codes (matrix) follow encode.py: 0=ground, 1=metal, 2=inductor,
3=switch, 4=via.  Capacitor pairs are NOT a pixel code -- they are an overlay
of metal-pixel coordinate pairs (see docs/architecture.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

PIXEL_GROUND = 0
PIXEL_METAL = 1
PIXEL_INDUCTOR = 2
PIXEL_SWITCH = 3
PIXEL_VIA = 4


@dataclass
class Design:
    """
    One switchable-filter candidate.

    Per-component lists are in raster (row-major) order of the matching pixel
    type in `pixel_grid`.  This ordering is the single convention shared by the
    tokenizer, the surrogate encoder, and `to_record()`.
    """

    pixel_grid: List[List[int]]                       # H x W, codes 0..4
    switch_caps: List[float] = field(default_factory=list)          # per switch pixel (F)
    inductor_values: List[int] = field(default_factory=list)        # per inductor pixel
    inductor_cap_values: List[float] = field(default_factory=list)  # per inductor pixel (F)
    cap_pixel_pairs: List[List[List[int]]] = field(default_factory=list)  # [[[r1,c1],[r2,c2]],...]
    cap_pair_values: List[float] = field(default_factory=list)      # per pair (F)
    switch_states: List[List[int]] = field(default_factory=list)    # K x n_switches (0/1)
    inductor_switch_states: List[List[int]] = field(default_factory=list)  # K x n_inductors

    # ---- derived geometry -------------------------------------------------

    @property
    def rows(self) -> int:
        return len(self.pixel_grid)

    @property
    def cols(self) -> int:
        return len(self.pixel_grid[0]) if self.pixel_grid else 0

    @property
    def n_switches(self) -> int:
        return sum(v == PIXEL_SWITCH for row in self.pixel_grid for v in row)

    @property
    def n_inductors(self) -> int:
        return sum(v == PIXEL_INDUCTOR for row in self.pixel_grid for v in row)

    @property
    def n_states(self) -> int:
        return len(self.switch_states)

    # ---- surrogate contract ----------------------------------------------

    @classmethod
    def from_record(cls, rec: Dict[str, Any]) -> "Design":
        return cls(
            pixel_grid=[list(r) for r in rec["matrix"]],
            switch_caps=list(rec.get("cap_values", [])),
            inductor_values=list(rec.get("inductor_values", [])),
            inductor_cap_values=list(rec.get("inductor_cap_values", [])),
            cap_pixel_pairs=[[list(p[0]), list(p[1])]
                             for p in rec.get("cap_pixel_pairs", [])],
            cap_pair_values=list(rec.get("cap_pair_values", [])),
            switch_states=[list(s) for s in rec.get("switch_states", [])],
            inductor_switch_states=[list(s) for s in
                                    rec.get("inductor_switch_states", [])],
        )

    def to_record(self) -> Dict[str, Any]:
        """
        Build the dict consumed by `world_model_test/encode.py:encode_sample`.

        `cap_values` is FLAT (one float per switch pixel) -- this matches
        encode.py, which assumes switch cap-bank size 1.  Banked switches are a
        TODO(user) (see docs/world_model_notes.md).
        """
        return {
            "matrix": [list(row) for row in self.pixel_grid],
            "cap_values": list(self.switch_caps),
            "inductor_values": list(self.inductor_values),
            "inductor_cap_values": list(self.inductor_cap_values),
            "cap_pixel_pairs": [[list(p[0]), list(p[1])] for p in self.cap_pixel_pairs],
            "cap_pair_values": list(self.cap_pair_values),
            "switch_states": [list(s) for s in self.switch_states],
            "inductor_switch_states": [list(s) for s in self.inductor_switch_states],
        }


@dataclass
class SParams:
    """
    Per-state S-parameters.  The world model predicts S21 magnitude in dB only,
    so that is what we store and score against (s11/etc. are not modelled).
    """

    freq_hz: np.ndarray          # (n_freq,)
    s21_db: np.ndarray           # (K, n_freq)

    @property
    def n_states(self) -> int:
        return int(self.s21_db.shape[0])

    @property
    def n_freq(self) -> int:
        return int(self.s21_db.shape[1])


@dataclass
class TrainingExample:
    """One (design, achieved S-params) pair for SFT / hindsight relabeling."""

    design: Design
    sparams: SParams                       # true if simulated, else surrogate-predicted
    source: str = "surrogate_labeled"      # "simulated" | "surrogate_labeled"
    iteration_added: int = 0
